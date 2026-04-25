"""
Multi-GPU version of compute_marginal_utilities.py

This script distributes query processing across multiple GPUs for faster computation.
Each GPU processes a subset of queries independently, and results are merged at the end.

Usage:
    python scripts/compute_marginal_utilities_multigpu.py \
        --config-name=marginal_utility_mini_imagenet \
        model.load_in_8bit=true \
        computation.batch_size=16 \
        retrieval.top_k=20
"""

import sys
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import subprocess
import json
import multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

# Use tqdm.auto which automatically selects notebook or terminal version
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataclasses import MarginalUtilityResult
from data.stanford_cars import StanfordCarsDataset
from data.mini_imagenet import MiniImageNetDataset
from models.idefics2_wrapper import Idefics2Wrapper
from utils.kaggle_utils import (
    is_kaggle_environment,
    get_kaggle_checkpoint_dataset,
    kaggle_download_checkpoints,
    kaggle_upload_checkpoints
)
from utils.multigpu_utils import get_available_gpus, split_work_across_gpus




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
        # Disable local caching in Kaggle to save disk space
        # HuggingFace's built-in cache is sufficient
        cache_locally = cfg.dataset.get('cache_dataset_locally', False)

        dataset = MiniImageNetDataset(
            split=cfg.dataset.split,
            data_dir=cfg.dataset.cache_dir,
            class_split_seed=cfg.dataset.class_split_seed,
            train_ratio=cfg.dataset.get('train_ratio', 0.8),
            val_ratio=cfg.dataset.get('val_ratio', 0.1),
            cache_dataset_locally=cache_locally
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


def initialize_model(cfg: DictConfig, gpu_id: int):
    """Initialize Idefics2 model on specific GPU."""
    print(f"[GPU {gpu_id}] Initializing Idefics2 model...")
    print(f"  Model: {cfg.model.name}")
    print(f"  Quantization: {'8-bit' if cfg.model.load_in_8bit else '4-bit' if cfg.model.load_in_4bit else 'None (fp16/fp32)'}")

    try:
        # Check if vision caching should be disabled (for memory-constrained environments)
        # Default to False for multi-GPU to prevent OOM on T4 GPUs
        enable_vision_cache = cfg.model.get('cache_vision_embeddings', False)

        model = Idefics2Wrapper(
            model_name=cfg.model.name,
            device=f"cuda:{gpu_id}",
            load_in_8bit=cfg.model.load_in_8bit,
            load_in_4bit=cfg.model.load_in_4bit,
            use_cache=True,
            cache_vision_embeddings=enable_vision_cache,
            max_vision_cache_size=5000,  # Limit cache to prevent OOM
        )
        return model
    except Exception as e:
        print(f"\n❌ [GPU {gpu_id}] Error loading model: {e}")
        raise


def load_baseline_probs(cfg: DictConfig) -> Dict[int, float]:
    """Load cached baseline probabilities."""
    cache_path = Path(cfg.computation.baseline_cache_path.format(split=cfg.dataset.split))

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Baseline probabilities not found at {cache_path}. "
            f"Please run compute_marginal_utilities.py first to generate baseline probs."
        )

    print(f"Loading cached baseline probabilities from {cache_path}")
    with open(cache_path, 'rb') as f:
        baseline_probs = pickle.load(f)
    print(f"✓ Loaded {len(baseline_probs)} cached baseline probabilities\n")

    return baseline_probs


def retrieve_candidates(dataset, query_idx: int, top_k: int, cfg: DictConfig) -> Tuple[List[int], np.ndarray]:
    """Retrieve top-k similar examples for a query."""
    if cfg.retrieval.get('use_stratified_sampling', False):
        # Import stratified sampling function from original script
        from compute_marginal_utilities import retrieve_candidates_stratified
        return retrieve_candidates_stratified(dataset, query_idx, top_k, cfg)

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


def compute_utilities_for_query(
    model,
    dataset,
    query_idx: int,
    candidate_indices: List[int],
    similarity_scores: np.ndarray,
    baseline_probs: Dict[int, float],
    cfg: DictConfig
) -> List[MarginalUtilityResult]:
    """Compute marginal utilities for all candidates of a query."""
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

        # Compute utilities for batch
        try:
            utilities = model.compute_marginal_utilities_batch(
                query_images=query_images,
                query_labels=query_labels,
                example_images=example_images,
                example_labels=example_labels,
                baseline_log_probs=baseline_log_probs
            )
        except Exception as e:
            print(f"\n⚠️  Error computing utilities for query {query_idx}: {e}")
            continue

        # Create results
        for i, (candidate_idx, similarity, utility) in enumerate(
            zip(batch_indices, batch_similarities, utilities)
        ):
            candidate_example, _ = dataset[candidate_idx]

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


def save_gpu_checkpoint(
    gpu_id: int,
    results: List[MarginalUtilityResult],
    query_idx: int,
    cfg: DictConfig,
    upload_to_kaggle: bool = False,
    checkpoint_lock = None,
    num_gpus: int = 1
):
    """
    Save checkpoint for a specific GPU.

    Args:
        gpu_id: GPU device ID
        results: Results so far
        query_idx: Last completed query index
        cfg: Configuration
        upload_to_kaggle: Whether to upload to Kaggle (only done periodically)
        checkpoint_lock: Optional multiprocessing lock to prevent concurrent uploads
        num_gpus: Total number of GPUs being used (for multi-GPU uploads)
    """
    if not cfg.checkpoint.enabled:
        return

    # All GPUs save to same directory (filenames won't clash due to query ranges)
    checkpoint_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{query_idx:06d}.pkl"

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'gpu_id': gpu_id,
            'results': results,
            'last_query_idx': query_idx,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'num_queries': len(set(r.query_idx for r in results)),
            'num_pairs': len(results)
        }, f)

    # Upload to Kaggle if requested
    # Use lock to prevent concurrent uploads from multiple GPUs
    if upload_to_kaggle and is_kaggle_environment():
        kaggle_dataset = get_kaggle_checkpoint_dataset()
        if kaggle_dataset:
            if checkpoint_lock:
                with checkpoint_lock:
                    # Upload latest checkpoint per GPU range to avoid excessive storage
                    kaggle_upload_checkpoints(
                        checkpoint_dir,
                        kaggle_dataset,
                        cfg.experiment.name,
                        latest_only=False,
                        latest_per_range=True,
                        num_ranges=num_gpus
                    )
            else:
                kaggle_upload_checkpoints(
                    checkpoint_dir,
                    kaggle_dataset,
                    cfg.experiment.name,
                    latest_only=False,
                    latest_per_range=True,
                    num_ranges=num_gpus
                )


def load_gpu_checkpoint(gpu_id: int, cfg: DictConfig, query_start: int, query_end: int, checkpoint_lock=None) -> Tuple[List[MarginalUtilityResult], int]:
    """
    Load checkpoint for a specific GPU if it exists.

    If running in Kaggle and KAGGLE_CHECKPOINT_DATASET is set,
    downloads checkpoints from Kaggle dataset first (only GPU 0 downloads to avoid race conditions).

    Args:
        gpu_id: GPU device ID
        cfg: Configuration
        query_start: Starting query index for this GPU
        query_end: Ending query index for this GPU
        checkpoint_lock: Optional multiprocessing lock to synchronize downloads

    Returns:
        (existing_results, last_completed_query_idx)
    """
    if not cfg.checkpoint.enabled:
        return [], -1

    checkpoint_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "checkpoints"

    # Download checkpoints from Kaggle if configured
    # Only GPU 0 downloads to avoid race conditions
    if gpu_id == 0 and is_kaggle_environment():
        kaggle_dataset = get_kaggle_checkpoint_dataset()
        if kaggle_dataset:
            if checkpoint_lock:
                with checkpoint_lock:
                    print(f"[GPU {gpu_id}] Downloading checkpoints from Kaggle dataset: {kaggle_dataset}")
                    kaggle_download_checkpoints(checkpoint_dir, kaggle_dataset)
            else:
                print(f"[GPU {gpu_id}] Downloading checkpoints from Kaggle dataset: {kaggle_dataset}")
                kaggle_download_checkpoints(checkpoint_dir, kaggle_dataset)
    elif gpu_id > 0 and is_kaggle_environment():
        # Other GPUs wait a bit for GPU 0 to finish downloading
        import time
        print(f"[GPU {gpu_id}] Waiting for GPU 0 to download checkpoints...")
        time.sleep(30)  # Wait 30 seconds for GPU 0 to download

    if not checkpoint_dir.exists():
        return [], -1

    # Find checkpoints for this GPU's query range
    all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))

    # Filter checkpoints within this GPU's query range
    relevant_checkpoints = []
    for ckpt in all_checkpoints:
        # Extract query index from filename
        try:
            query_idx = int(ckpt.stem.split('_')[1])
            if query_start <= query_idx < query_end:
                relevant_checkpoints.append(ckpt)
        except (IndexError, ValueError):
            continue

    if not relevant_checkpoints:
        print(f"[GPU {gpu_id}] No checkpoints found in range {query_start}-{query_end}")
        return [], -1

    # Load the latest checkpoint for this GPU's range
    latest_checkpoint = relevant_checkpoints[-1]
    print(f"[GPU {gpu_id}] Found checkpoint: {latest_checkpoint}")

    with open(latest_checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)

    results = checkpoint_data['results']
    last_query = checkpoint_data['last_query_idx']

    print(f"[GPU {gpu_id}] ✓ Loaded {len(results)} results, resuming from query {last_query + 1}")

    return results, last_query


def _worker_with_gpu_isolation(
    gpu_id: int,
    query_start: int,
    query_end: int,
    cfg: DictConfig,
    return_dict: dict,
    checkpoint_lock=None,
    num_gpus: int = 1,
) -> None:
    """
    Wrapper that sets CUDA_VISIBLE_DEVICES before importing CUDA.
    This MUST be the entry point for multiprocessing to avoid OOM.
    """
    # CRITICAL: Set CUDA_VISIBLE_DEVICES FIRST before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # After setting CUDA_VISIBLE_DEVICES, the assigned GPU becomes cuda:0
    # Pass the original gpu_id for logging, but use device_id=0 for actual device
    process_query_range(
        original_gpu_id=gpu_id,  # For logging and return_dict key
        device_id=0,  # Always 0 since we only see one GPU
        query_start=query_start,
        query_end=query_end,
        cfg=cfg,
        return_dict=return_dict,
        checkpoint_lock=checkpoint_lock,
        num_gpus=num_gpus
    )


def process_query_range(
    original_gpu_id: int,
    device_id: int,
    query_start: int,
    query_end: int,
    cfg: DictConfig,
    return_dict: dict,
    checkpoint_lock=None,
    num_gpus: int = 1,
) -> None:
    """
    Process a range of queries on a specific GPU.

    Args:
        original_gpu_id: Original GPU ID (for logging and return_dict key)
        device_id: Local device ID (always 0 after CUDA_VISIBLE_DEVICES is set)
        query_start: Starting query index
        query_end: Ending query index (exclusive)
        cfg: Configuration
        return_dict: Shared dictionary to store results
        checkpoint_lock: Lock for coordinating checkpoint uploads
        num_gpus: Total number of GPUs being used
    """
    try:
        # CUDA_VISIBLE_DEVICES is already set by wrapper
        # This process only sees one GPU (cuda:0)
        torch.cuda.set_device(device_id)

        print(f"[GPU {original_gpu_id}] Processing queries {query_start} to {query_end-1}")
        print(f"[GPU {original_gpu_id}] Using local device cuda:{device_id}")

        # Load dataset (each process needs its own copy)
        dataset = load_dataset(cfg)
        load_clip_embeddings(dataset, cfg)

        # Initialize model on this GPU
        # Since CUDA_VISIBLE_DEVICES is set, use device_id (which is 0)
        model = initialize_model(cfg, device_id)

        # Load baseline probabilities
        baseline_probs = load_baseline_probs(cfg)

        # Try to load checkpoint for this GPU
        print(f"[GPU {original_gpu_id}] Checking for checkpoints in: {Path(cfg.checkpoint.save_dir) / cfg.experiment.name / 'checkpoints'}")
        all_results, last_completed = load_gpu_checkpoint(original_gpu_id, cfg, query_start, query_end, checkpoint_lock)

        # Adjust start position if resuming
        resume_start = max(query_start, last_completed + 1)
        if resume_start > query_start:
            print(f"[GPU {original_gpu_id}] ✓ Resuming from query {resume_start} (loaded {len(all_results)} existing results)")
        else:
            print(f"[GPU {original_gpu_id}] Starting fresh from query {query_start}")

        # Process queries
        # Configure tqdm based on environment
        tqdm_kwargs = {
            'desc': f"GPU {original_gpu_id}",
            'initial': resume_start - query_start,
            'total': query_end - query_start,
            'leave': True,  # Keep the progress bar after completion
        }

        # Only use position parameter in terminal (not in Kaggle notebooks)
        if not is_kaggle_environment():
            tqdm_kwargs['position'] = original_gpu_id

        for query_idx in tqdm(range(resume_start, query_end), **tqdm_kwargs):
            # Retrieve candidates
            candidate_indices, similarities = retrieve_candidates(
                dataset=dataset,
                query_idx=query_idx,
                top_k=cfg.retrieval.top_k,
                cfg=cfg
            )

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

            all_results.extend(query_results)

            # Clear GPU cache periodically to prevent OOM
            if (query_idx - resume_start + 1) % 50 == 0:
                torch.cuda.empty_cache()

                # Print cache stats
                cache_stats = model.get_cache_stats()
                if cache_stats['cache_enabled']:
                    print(f"\n[GPU {original_gpu_id}] Vision cache size: {cache_stats['cache_size']} images")

            # Clear vision cache periodically to prevent OOM (every 1000 queries)
            if (query_idx - resume_start + 1) % 1000 == 0:
                print(f"\n[GPU {original_gpu_id}] Clearing vision cache to free memory...")
                model.clear_vision_cache()
                torch.cuda.empty_cache()

            # Save checkpoint periodically (each GPU saves independently)
            if cfg.checkpoint.enabled and (query_idx + 1) % cfg.checkpoint.save_interval == 0:
                # Upload to Kaggle every checkpoint save (every save_interval queries, default 100)
                save_gpu_checkpoint(original_gpu_id, all_results, query_idx, cfg, upload_to_kaggle=True, checkpoint_lock=checkpoint_lock, num_gpus=num_gpus)
                print(f"\n[GPU {original_gpu_id}] ✓ Checkpoint saved: query {query_idx}")

        # Save final checkpoint for this GPU and upload to Kaggle
        if cfg.checkpoint.enabled and len(all_results) > 0:
            save_gpu_checkpoint(original_gpu_id, all_results, query_end - 1, cfg, upload_to_kaggle=True, checkpoint_lock=checkpoint_lock, num_gpus=num_gpus)
            print(f"\n[GPU {original_gpu_id}] ✓ Final checkpoint saved")

        # Store results in shared dictionary
        return_dict[original_gpu_id] = all_results

        print(f"\n[GPU {original_gpu_id}] ✓ Completed {len(all_results)} results")

    except Exception as e:
        print(f"\n[GPU {original_gpu_id}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[original_gpu_id] = []


def save_results(results: List[MarginalUtilityResult], cfg: DictConfig):
    """Save results to disk."""
    output_dir = Path(cfg.output.save_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"marginal_utilities_{cfg.dataset.split}.pkl"

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


@hydra.main(version_base=None, config_path="../configs", config_name="marginal_utility")
def main(cfg: DictConfig):
    """Main entry point for multi-GPU processing."""
    print(f"\n{'='*70}")
    print(f"Multi-GPU Marginal Utility Computation")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"{'='*70}\n")

    # Get available GPUs
    num_gpus = get_available_gpus()

    if num_gpus == 0:
        print("❌ No GPUs available. This script requires at least one GPU.")
        return

    gpu_ids = list(range(num_gpus))
    print(f"✓ Found {num_gpus} GPU(s): {gpu_ids}\n")

    # Set random seeds
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    # Load dataset once to determine query range
    dataset = load_dataset(cfg)
    load_clip_embeddings(dataset, cfg)

    num_queries = len(dataset)
    if cfg.limits.max_queries:
        num_queries = min(num_queries, cfg.limits.max_queries)

    print(f"Total queries to process: {num_queries}")
    print(f"GPUs to use: {num_gpus}")

    # Divide queries among GPUs using utility function
    query_splits = split_work_across_gpus(num_queries, num_gpus)

    query_ranges = []
    for i, (start, end) in enumerate(query_splits):
        query_ranges.append((i, start, end))
        print(f"  GPU {i}: queries {start} to {end-1} ({end-start} queries)")

    print()

    # Use multiprocessing to run on multiple GPUs
    # Each process will handle one GPU
    manager = mp.Manager()
    return_dict = manager.dict()
    checkpoint_lock = manager.Lock()  # Lock for coordinating checkpoint downloads/uploads

    processes = []
    for gpu_id, start, end in query_ranges:
        p = mp.Process(
            target=_worker_with_gpu_isolation,
            args=(gpu_id, start, end, cfg, return_dict, checkpoint_lock, len(gpu_ids))
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Merge results from all GPUs
    print("\n" + "="*70)
    print("Merging results from all GPUs...")

    all_results = []
    for gpu_id in gpu_ids:
        gpu_results = return_dict.get(gpu_id, [])
        print(f"  GPU {gpu_id}: {len(gpu_results)} results")
        all_results.extend(gpu_results)

    print(f"\nTotal results: {len(all_results)}")
    print("="*70)

    # Save merged results
    save_results(all_results, cfg)

    print("✓ Multi-GPU experiment complete!")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
