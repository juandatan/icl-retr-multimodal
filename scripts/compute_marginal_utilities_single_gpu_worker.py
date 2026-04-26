"""
Single-GPU worker script for marginal utility computation.
Called by the multi-GPU launcher as a separate subprocess.
"""

import sys
import os
from pathlib import Path
import pickle
import argparse

# Set CUDA device BEFORE any imports
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be set before running this script")

print(f"Worker starting with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compute_marginal_utilities_multigpu import (
    load_dataset,
    load_clip_embeddings,
    initialize_model,
    load_baseline_probs,
    retrieve_candidates,
    compute_utilities_for_query,
    load_gpu_checkpoint,
    save_gpu_checkpoint,
    is_kaggle_environment
)
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True, help="Original GPU ID for logging")
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-end", type=int, required=True)
    parser.add_argument("--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save results")
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'rb') as f:
        cfg = pickle.load(f)

    gpu_id = args.gpu_id
    print(f"[GPU {gpu_id}] Processing queries {args.query_start} to {args.query_end-1}")
    print(f"[GPU {gpu_id}] Using local device cuda:0 (physical GPU {gpu_id})")

    # Set device
    torch.cuda.set_device(0)

    # Load dataset
    dataset = load_dataset(cfg)
    load_clip_embeddings(dataset, cfg)

    # Initialize model
    model = initialize_model(cfg, 0)

    # Load baseline probabilities
    baseline_probs = load_baseline_probs(cfg)

    # Try to load checkpoint
    all_results, last_completed = load_gpu_checkpoint(gpu_id, cfg, args.query_start, args.query_end, None)

    # Adjust start position if resuming
    resume_start = max(args.query_start, last_completed + 1)
    if resume_start > args.query_start:
        print(f"[GPU {gpu_id}] ✓ Resuming from query {resume_start}")
    else:
        print(f"[GPU {gpu_id}] Starting fresh from query {args.query_start}")

    # Process queries
    for query_idx in tqdm(range(resume_start, args.query_end), desc=f"GPU {gpu_id}"):
        try:
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

            # Clear GPU cache periodically
            if (query_idx - resume_start + 1) % 50 == 0:
                torch.cuda.empty_cache()

            # Save checkpoint periodically
            if cfg.checkpoint.enabled and (query_idx + 1) % cfg.checkpoint.save_interval == 0:
                # Enable Kaggle upload for crash recovery
                # No lock needed - each worker uploads its own checkpoint with unique filename
                save_gpu_checkpoint(gpu_id, all_results, query_idx, cfg, upload_to_kaggle=True, checkpoint_lock=None, num_gpus=2)

        except Exception as e:
            print(f"[GPU {gpu_id}] ⚠️  Error on query {query_idx}: {e}")
            continue

    # Save final results
    with open(args.output_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"[GPU {gpu_id}] ✓ Completed {len(all_results)} results, saved to {args.output_path}")


if __name__ == "__main__":
    main()
