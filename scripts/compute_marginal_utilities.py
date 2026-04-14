"""
Compute marginal utility scores for <input, retrieved example> pairs.

This script:
1. Loads a dataset (e.g., Stanford Cars) with pre-computed CLIP embeddings
2. For each query example, retrieves top-k semantically similar candidates
3. Computes marginal utility: how much each retrieved example improves prediction
4. Saves results with proper checkpointing and experiment tracking

Usage:
    python scripts/compute_marginal_utilities.py
    python scripts/compute_marginal_utilities.py --config-name marginal_utility
    python scripts/compute_marginal_utilities.py checkpoint.resume_from=path/to/checkpoint.pkl

Kaggle Usage:
    Set environment variable: KAGGLE_CHECKPOINT_DATASET=username/dataset-name
    The script will automatically download existing checkpoints and upload new ones
"""

import sys
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import subprocess
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataclasses import MarginalUtilityResult
from data.stanford_cars import StanfordCarsDataset
from data.mini_imagenet import MiniImageNetDataset
from models.llava_wrapper import LLaVAWrapper
from utils.kaggle_utils import (
    is_kaggle_environment,
    get_kaggle_checkpoint_dataset,
    kaggle_download_checkpoints,
    kaggle_upload_checkpoints
)


def load_dataset(cfg: DictConfig):
    """Load the dataset based on config."""
    print(f"Loading dataset: {cfg.dataset.name} ({cfg.dataset.split} split)...")

    if cfg.dataset.name == "stanford_cars":
        dataset = StanfordCarsDataset(
            split=cfg.dataset.split,
            data_dir=cfg.dataset.cache_dir,
            class_split_seed=cfg.dataset.class_split_seed
        )
    elif cfg.dataset.name == "mini_imagenet":
        dataset = MiniImageNetDataset(
            split=cfg.dataset.split,
            data_dir=cfg.dataset.cache_dir,
            class_split_seed=cfg.dataset.class_split_seed
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    print(f"✓ Loaded {len(dataset)} examples\n")
    return dataset


def load_clip_embeddings(dataset, cfg: DictConfig):
    """Load pre-computed CLIP embeddings."""
    print(f"Loading CLIP embeddings...")
    success = dataset.load_clip_embeddings()
    if not success:
        raise FileNotFoundError(
            f"CLIP embeddings not found. Please run: "
            f"python scripts/build_clip_embeddings.py --splits {cfg.dataset.split}"
        )
    print(f"✓ Loaded embeddings: shape {dataset.clip_embeddings.shape}")
    print(f"  Model: {dataset.clip_model_name}\n")


def initialize_model(cfg: DictConfig):
    """Initialize LLaVA model."""
    print("Initializing LLaVA model...")
    print(f"  Model: {cfg.model.name}")
    print(f"  Quantization: {'8-bit' if cfg.model.load_in_8bit else '4-bit' if cfg.model.load_in_4bit else 'None (fp16/fp32)'}")
    print(f"\nNote: First time may take a while to download model (~14GB)...")
    print(f"If download fails with SSL errors, you can:")
    print(f"  1. Wait and retry (network issues are usually temporary)")
    print(f"  2. Pre-download: huggingface-cli download {cfg.model.name}")
    print(f"  3. Use offline mode: export HF_DATASETS_OFFLINE=1\n")

    # Determine actual device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"  Using device: {device}\n")

    try:
        # Check if vision caching should be disabled (for memory-constrained environments)
        enable_vision_cache = cfg.model.get('cache_vision_embeddings', False)  # Default to False for safety

        model = LLaVAWrapper(
            model_name=cfg.model.name,
            device=device,
            load_in_8bit=cfg.model.load_in_8bit,
            load_in_4bit=cfg.model.load_in_4bit,
            use_cache=True,
            cache_vision_embeddings=enable_vision_cache,
            max_vision_cache_size=5000,
        )
        return model
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Check internet connection")
        print(f"  - Try: huggingface-cli download {cfg.model.name}")
        print(f"  - Check HuggingFace status: https://status.huggingface.co/")
        raise


def compute_or_load_baselines(model, dataset, cfg: DictConfig) -> Dict[int, float]:
    """
    Compute or load cached baseline (0-shot) probabilities.

    Returns:
        Dictionary mapping example index -> 0-shot log probability
    """
    if not cfg.computation.cache_baselines:
        print("Computing baseline probabilities (caching disabled)...")
        return model.compute_baseline_probabilities(
            dataset=dataset,
            batch_size=cfg.computation.baseline_batch_size
        )

    # Check for cached baselines
    cache_path = Path(cfg.computation.baseline_cache_path.format(split=cfg.dataset.split))

    if cache_path.exists():
        print(f"Loading cached baseline probabilities from {cache_path}")
        with open(cache_path, 'rb') as f:
            baseline_probs = pickle.load(f)
        print(f"✓ Loaded {len(baseline_probs)} cached baseline probabilities")

        # Verify cache is complete for the current dataset
        expected_size = len(dataset)
        if len(baseline_probs) < expected_size:
            print(f"⚠️  WARNING: Cached baselines incomplete ({len(baseline_probs)}/{expected_size})")
            print(f"  Removing incomplete cache and recomputing...\n")
            cache_path.unlink()
        else:
            print()
            return baseline_probs

    # Compute baselines
    print("Computing baseline probabilities (this may take a while)...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_probs = model.compute_baseline_probabilities(
        dataset=dataset,
        batch_size=cfg.computation.baseline_batch_size,
        save_path=str(cache_path)
    )
    print()
    return baseline_probs


def retrieve_candidates(dataset, query_idx: int, top_k: int, cfg: DictConfig) -> Tuple[List[int], np.ndarray]:
    """
    Retrieve top-k similar examples for a query.

    Args:
        dataset: Dataset with CLIP embeddings
        query_idx: Index of query example
        top_k: Number of candidates to retrieve
        cfg: Configuration with retrieval settings

    Returns:
        (candidate_indices, similarity_scores)
    """
    # Check if stratified sampling is enabled
    if cfg.retrieval.get('use_stratified_sampling', False):
        return retrieve_candidates_stratified(dataset, query_idx, top_k, cfg)

    # Default: Get top-k similar examples
    # exclude_query=True will automatically filter out the query itself
    similar_indices, similarities = dataset.get_top_k_similar(
        query_idx=query_idx,
        k=top_k,
        exclude_query=True,
        exclude_same_class=cfg.retrieval.exclude_same_class
    )

    # Verification assertion: ensure query is excluded from candidates
    assert query_idx not in similar_indices, f"Query {query_idx} found in candidates!"

    # Debug logging for first few queries to confirm exclusion
    if query_idx < 3:
        print(f"[DEBUG] Query {query_idx}: Retrieved {len(similar_indices)} candidates, query excluded: True")

    return similar_indices, similarities


def retrieve_candidates_stratified(
    dataset,
    query_idx: int,
    top_k: int,
    cfg: DictConfig
) -> Tuple[List[int], np.ndarray]:
    """
    Retrieve candidates using stratified sampling by similarity ranges.

    This ensures diverse similarity coverage, which helps the reranker model
    learn patterns across the full similarity spectrum rather than just
    high-similarity examples.

    Args:
        dataset: Dataset with CLIP embeddings
        query_idx: Index of query example
        top_k: Number of candidates to retrieve total
        cfg: Configuration with retrieval settings

    Returns:
        (candidate_indices, similarity_scores)
    """
    # Get stratification config
    strat_cfg = cfg.retrieval.stratification
    pool_size = cfg.retrieval.get('pool_size', 100)

    # Get larger pool of candidates to sample from
    all_indices, all_similarities = dataset.get_top_k_similar(
        query_idx=query_idx,
        k=pool_size,
        exclude_query=True,
        exclude_same_class=cfg.retrieval.exclude_same_class
    )

    # Verification assertion: ensure query is excluded from candidate pool
    assert query_idx not in all_indices, f"Query {query_idx} found in candidate pool!"

    # Debug logging for first few queries to confirm exclusion in stratified sampling
    if query_idx < 3:
        print(f"[DEBUG] Query {query_idx} (stratified): Retrieved pool of {len(all_indices)} candidates, query excluded: True")

    if len(all_indices) == 0:
        print(f"Warning: No valid candidates for query {query_idx}")
        return [], np.array([])

    # Define similarity ranges based on percentiles and sample ratios
    ranges = []
    for range_cfg in strat_cfg:
        percentile_low = range_cfg['percentile'][0]
        percentile_high = range_cfg['percentile'][1]
        sample_ratio = range_cfg['sample_ratio']
        ranges.append((percentile_low, percentile_high, sample_ratio))

    # Sample from each range
    selected_indices = []
    selected_similarities = []

    for percentile_low, percentile_high, sample_ratio in ranges:
        # Calculate how many samples from this range based on ratio
        n_samples = int(top_k * sample_ratio)

        # Calculate index range (percentiles map to positions in sorted list)
        idx_low = int(len(all_indices) * percentile_low / 100)
        idx_high = int(len(all_indices) * percentile_high / 100)
        idx_high = max(idx_high, idx_low + 1)  # Ensure at least 1 element

        # Get candidates in this range
        range_indices = all_indices[idx_low:idx_high]
        range_similarities = all_similarities[idx_low:idx_high]

        if len(range_indices) == 0:
            continue

        # Sample n_samples from this range
        n_to_sample = min(n_samples, len(range_indices))
        if n_to_sample == 0:
            continue

        sampled_positions = np.random.choice(
            len(range_indices),
            size=n_to_sample,
            replace=False
        )

        selected_indices.extend([range_indices[i] for i in sampled_positions])
        selected_similarities.extend([range_similarities[i] for i in sampled_positions])

    # Convert to arrays
    selected_indices = np.array(selected_indices)
    selected_similarities = np.array(selected_similarities)

    # If we somehow got more than top_k (due to rounding), randomly sample down
    if len(selected_indices) > top_k:
        sampled_positions = np.random.choice(
            len(selected_indices),
            size=top_k,
            replace=False
        )
        selected_indices = selected_indices[sampled_positions]
        selected_similarities = selected_similarities[sampled_positions]

    # If we got fewer (due to rounding), add more from the pool
    elif len(selected_indices) < top_k:
        n_needed = top_k - len(selected_indices)
        # Get remaining candidates not already selected
        all_indices_set = set(all_indices)
        selected_set = set(selected_indices)
        remaining = list(all_indices_set - selected_set)

        if len(remaining) >= n_needed:
            additional = np.random.choice(remaining, size=n_needed, replace=False)
            additional_sims = np.array([
                all_similarities[list(all_indices).index(idx)] for idx in additional
            ])
            selected_indices = np.concatenate([selected_indices, additional])
            selected_similarities = np.concatenate([selected_similarities, additional_sims])

    return selected_indices.tolist(), selected_similarities


def compute_utilities_for_query(
    model,
    dataset,
    query_idx: int,
    candidate_indices: List[int],
    similarity_scores: np.ndarray,
    baseline_probs: Dict[int, float],
    cfg: DictConfig
) -> List[MarginalUtilityResult]:
    """
    Compute marginal utilities for all candidates of a query.

    Args:
        model: LLaVA model
        dataset: Dataset
        query_idx: Index of query example
        candidate_indices: Indices of candidate examples
        similarity_scores: CLIP similarity scores
        baseline_probs: Pre-computed baseline probabilities
        cfg: Configuration

    Returns:
        List of MarginalUtilityResult objects
    """
    query_example, query_image = dataset[query_idx]
    baseline_log_prob = baseline_probs[query_idx]

    results = []
    batch_size = cfg.computation.batch_size

    # Process candidates in batches
    for batch_start in range(0, len(candidate_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(candidate_indices))
        batch_indices = candidate_indices[batch_start:batch_end]
        batch_similarities = similarity_scores[batch_start:batch_end]

        # Prepare batch data
        query_images = []
        query_labels = []
        example_images = []
        example_labels = []
        baseline_log_probs = []

        for candidate_idx in batch_indices:
            candidate_example, candidate_image = dataset[candidate_idx]

            query_images.append(query_image)
            query_labels.append(query_example.label_name)
            example_images.append(candidate_image)
            example_labels.append(candidate_example.label_name)
            baseline_log_probs.append(baseline_log_prob)

        # Compute utilities for batch with error handling
        try:
            utilities = model.compute_marginal_utilities_batch(
                query_images=query_images,
                query_labels=query_labels,
                example_images=example_images,
                example_labels=example_labels,
                baseline_log_probs=baseline_log_probs
            )
        except Exception as e:
            # Log error and skip this batch
            print(f"\n⚠️  Error computing utilities for query {query_idx}, batch {batch_start}-{batch_end}: {e}")
            print(f"   Skipping {len(batch_indices)} pairs for this batch")
            continue

        # Create results
        for i, (candidate_idx, similarity, utility) in enumerate(
            zip(batch_indices, batch_similarities, utilities)
        ):
            candidate_example, _ = dataset[candidate_idx]

            # Reconstruct 1-shot log prob from utility
            # utility = (u1 - u0) / max(|u1|, |u0|)
            denominator = max(abs(baseline_log_prob), 1e-10)
            oneshot_log_prob = baseline_log_prob + utility * denominator

            result = MarginalUtilityResult(
                query_idx=query_idx,
                example_idx=candidate_idx,
                query_label=query_example.label_name,
                example_label=candidate_example.label_name,
                baseline_log_prob=baseline_log_prob,
                oneshot_log_prob=oneshot_log_prob,
                marginal_utility=utility,
                similarity_score=float(similarity),
                same_class=(query_example.label == candidate_example.label)
            )
            results.append(result)

    return results


def load_checkpoint(cfg: DictConfig) -> Tuple[List[MarginalUtilityResult], int]:
    """
    Load existing checkpoint if available.

    If running in Kaggle and KAGGLE_CHECKPOINT_DATASET is set,
    downloads checkpoints from Kaggle dataset first.

    Returns:
        (existing_results, last_completed_query_idx)
    """
    if not cfg.checkpoint.enabled:
        return [], -1

    checkpoint_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "checkpoints"

    # Download checkpoints from Kaggle if configured
    if is_kaggle_environment():
        kaggle_dataset = get_kaggle_checkpoint_dataset()
        if kaggle_dataset:
            kaggle_download_checkpoints(checkpoint_dir, kaggle_dataset)

    if not checkpoint_dir.exists():
        return [], -1

    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))

    if not checkpoint_files:
        return [], -1

    # Load the latest checkpoint
    latest_checkpoint = checkpoint_files[-1]
    print(f"Found existing checkpoint: {latest_checkpoint}")

    with open(latest_checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)

    results = checkpoint_data['results']
    last_query = checkpoint_data['last_query_idx']

    print(f"✓ Loaded checkpoint with {len(results)} results")
    print(f"  Last completed query: {last_query}")
    print(f"  Resuming from query {last_query + 1}\n")

    return results, last_query


def save_checkpoint(results: List[MarginalUtilityResult], query_idx: int, cfg: DictConfig, upload_to_kaggle: bool = True):
    """
    Save checkpoint with current results.

    Args:
        results: List of marginal utility results
        query_idx: Current query index
        cfg: Configuration
        upload_to_kaggle: Whether to upload to Kaggle (only done on save_interval)
    """
    if not cfg.checkpoint.enabled:
        return

    checkpoint_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{query_idx:06d}.pkl"

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'last_query_idx': query_idx,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'num_queries': len(set(r.query_idx for r in results)),
            'num_pairs': len(results)
        }, f)

    # Upload to Kaggle if configured and requested
    if upload_to_kaggle and is_kaggle_environment():
        kaggle_dataset = get_kaggle_checkpoint_dataset()
        if kaggle_dataset:
            kaggle_upload_checkpoints(checkpoint_dir, kaggle_dataset, cfg.experiment.name)


def run_experiment(model, dataset, baseline_probs: Dict[int, float], cfg: DictConfig) -> List[MarginalUtilityResult]:
    """
    Run the full marginal utility computation experiment.

    Args:
        model: LLaVA model
        dataset: Dataset
        baseline_probs: Pre-computed baseline probabilities
        cfg: Configuration

    Returns:
        List of all MarginalUtilityResult objects
    """
    # Load checkpoint if exists
    all_results, start_query = load_checkpoint(cfg)
    start_query += 1  # Start from next query

    # Determine which queries to process
    num_queries = len(dataset)
    if cfg.limits.max_queries:
        num_queries = min(num_queries, cfg.limits.max_queries)

    print(f"\n{'='*70}")
    print(f"Computing marginal utilities for {num_queries} queries")
    print(f"Retrieving top-{cfg.retrieval.top_k} candidates per query")
    if start_query > 0:
        print(f"Resuming from query {start_query} (already completed {start_query})")
    print(f"{'='*70}\n")

    # Track skipped queries
    skipped_queries = []
    expected_pairs_per_query = cfg.retrieval.top_k

    # Process each query
    for query_idx in tqdm(range(start_query, num_queries), desc="Queries", initial=start_query, total=num_queries):
        # Retrieve candidates
        candidate_indices, similarities = retrieve_candidates(
            dataset=dataset,
            query_idx=query_idx,
            top_k=cfg.retrieval.top_k,
            cfg=cfg
        )

        # Track results before this query
        results_before = len(all_results)

        # Compute utilities
        query_results = compute_utilities_for_query(
            model=model,
            dataset=dataset,
            query_idx=query_idx,
            candidate_indices=candidate_indices,
            similarity_scores=similarities,
            baseline_probs=baseline_probs,
            cfg=cfg
        )

        # Store results
        all_results.extend(query_results)

        # Check if query was partially/fully skipped
        results_added = len(all_results) - results_before
        if results_added < expected_pairs_per_query:
            skipped_queries.append((query_idx, expected_pairs_per_query - results_added))

        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and (query_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

        # Save checkpoint periodically
        if cfg.checkpoint.enabled and (query_idx + 1) % cfg.checkpoint.save_interval == 0:
            save_checkpoint(all_results, query_idx, cfg)
            msg = f"✓ Checkpoint saved: query {query_idx}"
            if skipped_queries:
                msg += f" ({len(skipped_queries)} queries skipped)"
            print(f"\n{msg}")

    # Save final checkpoint to capture any remaining queries
    if cfg.checkpoint.enabled and len(all_results) > 0:
        save_checkpoint(all_results, query_idx, cfg, upload_to_kaggle=True)
        print(f"\n✓ Final checkpoint saved: query {query_idx}")

    # Report skipped queries
    if skipped_queries:
        print(f"\n{'='*70}")
        print(f"⚠️  Warning: {len(skipped_queries)} queries had errors")
        print(f"   Total pairs skipped: {sum(count for _, count in skipped_queries)}")
        if len(skipped_queries) <= 10:
            print(f"   Query indices: {[idx for idx, _ in skipped_queries]}")
        else:
            print(f"   First 10 query indices: {[idx for idx, _ in skipped_queries[:10]]}")
        print(f"{'='*70}\n")

    return all_results


def save_results(results: List[MarginalUtilityResult], cfg: DictConfig):
    """Save results to disk and upload final checkpoint to Kaggle."""
    output_dir = Path(cfg.output.save_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"marginal_utilities_{cfg.dataset.split}.pkl"

    # Save results
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'num_queries': len(set(r.query_idx for r in results)),
            'num_pairs': len(results)
        }, f)

    print(f"\n{'='*70}")
    print(f"✓ Saved {len(results)} results to {output_path}")
    print(f"  Queries processed: {len(set(r.query_idx for r in results))}")
    print(f"  Total pairs: {len(results)}")
    print(f"{'='*70}\n")

    # Upload final checkpoint to Kaggle
    if is_kaggle_environment():
        kaggle_dataset = get_kaggle_checkpoint_dataset()
        if kaggle_dataset:
            checkpoint_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "checkpoints"
            kaggle_upload_checkpoints(checkpoint_dir, kaggle_dataset, cfg.experiment.name)


@hydra.main(version_base=None, config_path="../configs", config_name="marginal_utility")
def main(cfg: DictConfig):
    """Main entry point."""
    # Print configuration
    print(f"\n{'='*70}")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")
    print(f"{'='*70}\n")

    # Set random seeds
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Load dataset
    dataset = load_dataset(cfg)

    # Load CLIP embeddings
    load_clip_embeddings(dataset, cfg)

    # Initialize LLaVA model
    model = initialize_model(cfg)

    # Compute or load baseline probabilities
    baseline_probs = compute_or_load_baselines(model, dataset, cfg)

    # Run experiment
    results = run_experiment(model, dataset, baseline_probs, cfg)

    # Save results
    save_results(results, cfg)

    print("✓ Experiment complete!")


if __name__ == "__main__":
    main()
