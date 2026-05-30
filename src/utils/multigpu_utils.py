"""
Multi-GPU utilities for distributed processing.

Provides reusable patterns for:
- Splitting work across multiple GPUs
- Parallel processing with proper device isolation
- Result merging and aggregation
"""

import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Callable, Any, Dict, Optional
import torch


def get_available_gpus() -> int:
    """
    Get number of available GPUs.

    Returns:
        Number of CUDA GPUs available (0 if none)
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def split_work_across_gpus(
    total_items: int,
    num_gpus: int
) -> List[tuple]:
    """
    Split work items evenly across GPUs.

    Args:
        total_items: Total number of items to process
        num_gpus: Number of GPUs to use

    Returns:
        List of (start_idx, end_idx) tuples for each GPU
    """
    items_per_gpu = total_items // num_gpus
    splits = []

    for i in range(num_gpus):
        start_idx = i * items_per_gpu
        # Last GPU gets any remainder
        end_idx = start_idx + items_per_gpu if i < num_gpus - 1 else total_items
        splits.append((start_idx, end_idx))

    return splits


def run_parallel_on_gpus(
    worker_fn: Callable,
    work_items: List[Any],
    num_gpus: int,
    worker_kwargs: Optional[Dict] = None
) -> List[Any]:
    """
    Run a worker function in parallel across multiple GPUs.

    Args:
        worker_fn: Worker function with signature:
                   worker_fn(gpu_id, items, **kwargs) -> result
        work_items: List of items to process (will be split across GPUs)
        num_gpus: Number of GPUs to use
        worker_kwargs: Additional keyword arguments to pass to worker_fn

    Returns:
        List of results from each GPU worker

    Example:
        def my_worker(gpu_id, items, model_name):
            device = f"cuda:{gpu_id}"
            # Load model on specific GPU
            model = load_model(model_name, device)
            # Process items
            results = [model(item) for item in items]
            return results

        results = run_parallel_on_gpus(
            worker_fn=my_worker,
            work_items=list(range(1000)),
            num_gpus=2,
            worker_kwargs={'model_name': 'my-model'}
        )
    """
    if worker_kwargs is None:
        worker_kwargs = {}

    # Split work across GPUs
    splits = split_work_across_gpus(len(work_items), num_gpus)
    work_splits = [work_items[start:end] for start, end in splits]

    print(f"Splitting work across {num_gpus} GPUs:")
    for i, items in enumerate(work_splits):
        print(f"  GPU {i}: {len(items)} items")

    # Launch each worker as a fresh subprocess with CUDA_VISIBLE_DEVICES set in the
    # OS environment before Python starts — the only reliable way to isolate CUDA
    # when the parent process has already (or may have already) touched CUDA.
    worker_script = Path(__file__).parent.parent.parent / "scripts" / "evaluate_icl_single_gpu_worker.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Write shared config once; each worker gets its own queries and output file.
        config_path = tmp / "worker_kwargs.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(worker_kwargs, f)

        processes = []
        for gpu_id, items in enumerate(work_splits):
            queries_path = tmp / f"queries_{gpu_id}.pkl"
            output_path = tmp / f"result_{gpu_id}.pkl"

            with open(queries_path, 'wb') as f:
                pickle.dump(items, f)

            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
            cmd = [
                sys.executable,
                str(worker_script),
                "--worker-id", str(gpu_id),
                "--config-path", str(config_path),
                "--queries-path", str(queries_path),
                "--output-path", str(output_path),
            ]
            p = subprocess.Popen(cmd, env=env)
            processes.append((gpu_id, p, output_path))

        # Wait for all workers and collect results
        results = [None] * num_gpus
        for gpu_id, p, output_path in processes:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Worker GPU {gpu_id} exited with code {p.returncode}")
            with open(output_path, 'rb') as f:
                results[gpu_id] = pickle.load(f)

    return results


def merge_dict_results(
    results: List[Dict[str, Any]],
    sum_keys: Optional[List[str]] = None,
    mean_keys: Optional[List[str]] = None,
    concat_keys: Optional[List[str]] = None,
    nested_dict_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Merge dictionary results from multiple GPU workers.

    Args:
        results: List of result dictionaries from workers
        sum_keys: Keys to sum across results (e.g., 'correct', 'total')
        mean_keys: Keys to average across results (e.g., 'accuracy')
        concat_keys: Keys to concatenate across results (e.g., 'predictions')
        nested_dict_keys: Keys containing nested dicts to merge by summing values

    Returns:
        Merged dictionary

    Example:
        results = [
            {'correct': 10, 'total': 20, 'predictions': [1, 2, 3]},
            {'correct': 15, 'total': 20, 'predictions': [4, 5, 6]}
        ]

        merged = merge_dict_results(
            results,
            sum_keys=['correct', 'total'],
            concat_keys=['predictions']
        )
        # {'correct': 25, 'total': 40, 'predictions': [1, 2, 3, 4, 5, 6]}
    """
    if not results:
        return {}

    if sum_keys is None:
        sum_keys = []
    if mean_keys is None:
        mean_keys = []
    if concat_keys is None:
        concat_keys = []
    if nested_dict_keys is None:
        nested_dict_keys = []

    merged = {}

    # Sum specified keys
    for key in sum_keys:
        merged[key] = sum(r.get(key, 0) for r in results)

    # Average specified keys
    for key in mean_keys:
        values = [r.get(key, 0) for r in results]
        merged[key] = sum(values) / len(values) if values else 0

    # Concatenate specified keys
    for key in concat_keys:
        merged[key] = []
        for r in results:
            if key in r:
                merged[key].extend(r[key])

    # Merge nested dictionaries by summing values
    for key in nested_dict_keys:
        merged[key] = {}
        for r in results:
            if key in r:
                for sub_key, value in r[key].items():
                    merged[key][sub_key] = merged[key].get(sub_key, 0) + value

    # Copy any other keys from first result (assuming they're the same across workers)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    processed_keys = set(sum_keys + mean_keys + concat_keys + nested_dict_keys)
    for key in all_keys - processed_keys:
        if key in results[0]:
            merged[key] = results[0][key]

    return merged


class MultiGPUManager:
    """
    Context manager for multi-GPU processing.

    Handles device detection, work splitting, and result merging.

    Example:
        with MultiGPUManager(num_gpus=2) as mgr:
            results = mgr.run_parallel(
                worker_fn=my_worker,
                work_items=data,
                worker_kwargs={'model': 'llava'}
            )
    """

    def __init__(
        self,
        num_gpus: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize multi-GPU manager.

        Args:
            num_gpus: Number of GPUs to use (None = auto-detect all)
            verbose: Whether to print status messages
        """
        self.num_available_gpus = get_available_gpus()
        self.num_gpus = num_gpus if num_gpus is not None else self.num_available_gpus
        self.num_gpus = min(self.num_gpus, self.num_available_gpus)
        self.verbose = verbose

        if self.verbose:
            print(f"MultiGPU Manager initialized:")
            print(f"  Available GPUs: {self.num_available_gpus}")
            print(f"  Using GPUs: {self.num_gpus}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def should_use_multi_gpu(self) -> bool:
        """Check if multi-GPU should be used (more than 1 GPU available)."""
        return self.num_gpus > 1

    def run_parallel(
        self,
        worker_fn: Callable,
        work_items: List[Any],
        worker_kwargs: Optional[Dict] = None
    ) -> List[Any]:
        """
        Run work in parallel across GPUs.

        Args:
            worker_fn: Worker function
            work_items: Items to process
            worker_kwargs: Additional kwargs for worker

        Returns:
            List of results from each GPU
        """
        if not self.should_use_multi_gpu():
            # Single GPU mode - just run on GPU 0
            if self.verbose:
                print("Running in single-GPU mode")
            if worker_kwargs is None:
                worker_kwargs = {}
            return [worker_fn(0, work_items, **worker_kwargs)]

        return run_parallel_on_gpus(
            worker_fn=worker_fn,
            work_items=work_items,
            num_gpus=self.num_gpus,
            worker_kwargs=worker_kwargs
        )

    @staticmethod
    def merge_results(
        results: List[Dict],
        **merge_kwargs
    ) -> Dict:
        """
        Merge results from parallel workers.

        Args:
            results: List of result dicts
            **merge_kwargs: Arguments for merge_dict_results()

        Returns:
            Merged dictionary
        """
        return merge_dict_results(results, **merge_kwargs)
