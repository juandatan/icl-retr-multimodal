"""Shared runtime primitives for reproducible, resumable ML jobs."""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch


def atomic_pickle_dump(value: Any, path: Path) -> None:
    """Atomically replace a pickle checkpoint after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "wb") as file:
        pickle.dump(value, file)
    os.replace(temporary_path, path)


def file_sha256(path: Optional[str | Path]) -> Optional[str]:
    """Return a file digest, or ``None`` when the optional file is absent."""
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with open(resolved, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> Optional[str]:
    """Return the checked-out Git revision without failing outside a repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def stratified_query_indices(
    examples: Sequence[Any],
    num_queries: Optional[int],
    seed: int,
) -> list[int]:
    """Select a reproducible, image-balanced query sample."""
    if num_queries is None or num_queries >= len(examples):
        return list(range(len(examples)))
    if num_queries <= 0:
        raise ValueError("num_queries must be positive or null")

    by_class: dict[int, list[int]] = defaultdict(list)
    for dataset_idx, example in enumerate(examples):
        by_class[int(example.label)].append(dataset_idx)

    rng = np.random.default_rng(seed)
    for indices in by_class.values():
        rng.shuffle(indices)

    selected: list[int] = []
    active_classes = sorted(by_class)
    offsets = {class_idx: 0 for class_idx in active_classes}
    while len(selected) < num_queries and active_classes:
        for class_idx_value in rng.permutation(active_classes):
            class_idx = int(class_idx_value)
            offset = offsets[class_idx]
            if offset < len(by_class[class_idx]):
                selected.append(by_class[class_idx][offset])
                offsets[class_idx] += 1
                if len(selected) == num_queries:
                    break
        active_classes = [
            class_idx
            for class_idx in active_classes
            if offsets[class_idx] < len(by_class[class_idx])
        ]
    return selected


def _count_gpus_without_cuda_init() -> int:
    """Count GPUs without initializing CUDA in the parent process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([line for line in result.stdout.splitlines() if line.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def setup_device(
    num_gpus_requested: Optional[int] = None,
) -> tuple[str, int, bool]:
    """Return ``(device, gpu_count, use_multi_gpu)`` without parent CUDA init."""
    available = _count_gpus_without_cuda_init()
    if available:
        device = "cuda"
        num_gpus = min(
            num_gpus_requested if num_gpus_requested is not None else available,
            available,
        )
        print(f"Using device: cuda, GPUs: {num_gpus}/{available}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, num_gpus = "mps", 1
        print("Using device: mps")
    else:
        device, num_gpus = "cpu", 1
        print("Using device: cpu")
    return device, num_gpus, device == "cuda" and num_gpus > 1
