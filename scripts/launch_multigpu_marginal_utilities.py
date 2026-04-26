"""
Multi-GPU launcher for marginal utility computation using subprocesses.
This avoids CUDA context leakage by running completely separate Python processes.
"""

import subprocess
import sys
import os
from pathlib import Path
import pickle
import tempfile
import time

import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.multigpu_utils import get_available_gpus, split_work_across_gpus


def get_total_queries(cfg: DictConfig) -> int:
    """Determine total number of queries to process."""
    if cfg.limits.max_queries:
        return cfg.limits.max_queries

    # Would need to load dataset to get actual size, for now estimate
    return 40000  # Mini-ImageNet train split size


@hydra.main(version_base=None, config_path="../configs", config_name="marginal_utility_mini_imagenet")
def main(cfg: DictConfig):
    print("="*80)
    print("Multi-GPU Marginal Utility Computation (Subprocess Mode)")
    print("="*80)

    # Get available GPUs
    num_gpus = get_available_gpus()
    gpu_ids = list(range(num_gpus))
    print(f"\nDetected {num_gpus} GPUs: {gpu_ids}")

    if num_gpus == 0:
        print("❌ No GPUs detected! Exiting.")
        return

    # Determine work split
    total_queries = get_total_queries(cfg)
    work_splits = split_work_across_gpus(total_queries, num_gpus)

    # Combine GPU IDs with work ranges
    query_ranges = [(gpu_ids[i], start, end) for i, (start, end) in enumerate(work_splits)]

    print("\nWork distribution:")
    for gpu_id, start, end in query_ranges:
        print(f"  GPU {gpu_id}: queries {start} to {end-1} ({end-start} queries)")

    # Save config to temp file (shared by all workers)
    temp_dir = Path(tempfile.mkdtemp())
    config_path = temp_dir / "config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(cfg, f)

    print(f"\nConfig saved to: {config_path}")

    # Create temp output paths for each worker
    output_paths = {}
    for gpu_id, _, _ in query_ranges:
        output_paths[gpu_id] = temp_dir / f"results_gpu{gpu_id}.pkl"

    # Launch workers as subprocesses
    processes = []
    worker_script = Path(__file__).parent / "compute_marginal_utilities_single_gpu_worker.py"

    print("\nLaunching workers...")
    for gpu_id, start, end in query_ranges:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        cmd = [
            sys.executable,
            str(worker_script),
            "--gpu-id", str(gpu_id),
            "--query-start", str(start),
            "--query-end", str(end),
            "--config-path", str(config_path),
            "--output-path", str(output_paths[gpu_id])
        ]

        print(f"  Launching GPU {gpu_id} worker...")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, proc))
        time.sleep(2)  # Stagger launches slightly

    print(f"\n✓ All {len(processes)} workers launched")
    print("\nWaiting for workers to complete...")

    # Wait for all workers to complete
    for gpu_id, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"  ✓ GPU {gpu_id} completed successfully")
        else:
            print(f"  ❌ GPU {gpu_id} failed with exit code {proc.returncode}")

    # Merge results
    print("\nMerging results from all GPUs...")
    all_results = []
    for gpu_id, _, _ in query_ranges:
        output_path = output_paths[gpu_id]
        if output_path.exists():
            with open(output_path, 'rb') as f:
                gpu_results = pickle.load(f)
            print(f"  GPU {gpu_id}: {len(gpu_results)} results")
            all_results.extend(gpu_results)
        else:
            print(f"  GPU {gpu_id}: No results file found")

    print(f"\nTotal results: {len(all_results)}")

    # Save merged results
    output_dir = Path(cfg.output.save_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "marginal_utilities.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"✓ Results saved to: {output_path}")

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*80)
    print("✓ Multi-GPU computation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
