"""
Create the reproducible image-level train/val/test split used for CUB-200.

All classes appear in every split; images are divided per class so the
class distribution is balanced across splits.  The split is saved as a JSON
file mapping split name -> list of HF dataset indices, and optionally uploaded
to a Kaggle dataset so it can be reused across notebook runs.

The split file is the single source of truth: load the dataset with
  image_split_path=<path>
to use it.

Usage:
    python -m scripts.create_image_split \
        --dataset cub_200 \
        --output data/cub_200/image_split.json \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --seed 42

    # Also upload to Kaggle:
    python -m scripts.create_image_split \
        --dataset cub_200 \
        --output data/cub_200/image_split.json \
        --kaggle-dataset juandatan/cub-200-image-split
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.dataset_registry import FINE_GRAINED_DATASETS, get_dataset_spec
from src.utils.kaggle_utils import kaggle_publish_files


def create_image_split(
    dataset_name: str,
    data_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    """
    Divide a fine-grained dataset's images into train/val/test splits,
    stratified by class.

    Returns a dict:
        {
            "train": [hf_idx, ...],
            "val":   [hf_idx, ...],
            "test":  [hf_idx, ...],
            "metadata": {...}
        }
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    print(f"Loading full {dataset_name} dataset (no split filtering)...")
    ds = _FullFineGrainedDataset(dataset_name=dataset_name, data_dir=data_dir)

    print(f"Total images: {len(ds.all_hf_indices)}")
    print(f"Total classes: {len(ds.label_to_hf_indices)}")

    rng = np.random.default_rng(seed)

    train_indices, val_indices, test_indices = [], [], []

    for label in sorted(ds.label_to_hf_indices.keys()):
        indices = np.array(ds.label_to_hf_indices[label])
        rng.shuffle(indices)

        n = len(indices)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Give remainder to test
        n_test = n - n_train - n_val
        if n_test < 1:
            # Shrink val if necessary
            n_val = max(0, n_val - (1 - n_test))
            n_test = n - n_train - n_val

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())

    # Verify no overlap
    assert not (set(train_indices) & set(val_indices)), "train/val overlap"
    assert not (set(train_indices) & set(test_indices)), "train/test overlap"
    assert not (set(val_indices) & set(test_indices)), "val/test overlap"
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(ds.all_hf_indices)

    split = {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
        "metadata": {
            "dataset": dataset_name,
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(test_ratio, 6),
            "total_images": len(ds.all_hf_indices),
            "num_classes": len(ds.label_to_hf_indices),
            "split_counts": {
                "train": len(train_indices),
                "val": len(val_indices),
                "test": len(test_indices),
            },
        },
    }

    print(f"\nSplit summary:")
    print(f"  Train: {len(train_indices):,} images")
    print(f"  Val:   {len(val_indices):,} images")
    print(f"  Test:  {len(test_indices):,} images")

    return split


class _FullFineGrainedDataset:
    """Minimal loader that reads all HF items without any split filtering."""

    def __init__(self, dataset_name: str, data_dir: str):
        from datasets import concatenate_datasets, load_from_disk, load_dataset

        spec = get_dataset_spec(dataset_name)
        data_dir = Path(data_dir)
        cache_subdir = Path(spec.data_dir).name
        local_path = data_dir / "hf_cache" / f"{cache_subdir}_merged"

        if local_path.exists():
            hf_dataset = load_from_disk(str(local_path))
        else:
            parts = []
            for repo_id in spec.hf_repo_ids:
                repo_splits = load_dataset(repo_id, cache_dir=str(data_dir / "hf_cache"))
                # Each repo (e.g. CUB_train, CUB_test) exposes exactly one internal
                # split, under its own name ("train" or "test") -- concatenate
                # whatever split(s) each repo actually has rather than assuming "train".
                for hf_split in repo_splits.values():
                    parts.append(hf_split)
            hf_dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
            hf_dataset.save_to_disk(str(local_path))

        self.all_hf_indices = list(range(len(hf_dataset)))
        self.label_to_hf_indices: dict = defaultdict(list)

        for idx, item in enumerate(hf_dataset):
            self.label_to_hf_indices[item['label']].append(idx)


def upload_split_to_kaggle(split_path: Path, kaggle_dataset: str):
    """Upload the split JSON to a Kaggle dataset."""
    return kaggle_publish_files(
        [split_path],
        kaggle_dataset,
        version_message=f"Publish split: {split_path.name}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create image-level train/val/test split for a fine-grained dataset"
    )
    parser.add_argument("--dataset", required=True, choices=list(FINE_GRAINED_DATASETS.keys()),
                        help="Registered dataset name (see src/data/dataset_registry.py)")
    parser.add_argument("--data-dir", default=None,
                        help="Dataset data directory (default: from registry)")
    parser.add_argument("--output", default=None,
                        help="Output path for the split JSON (default: <data-dir>/image_split.json)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of images per class for training (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of images per class for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--kaggle-dataset", type=str, default=None,
                        help="Kaggle dataset to upload to (username/dataset-slug)")
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    data_dir = args.data_dir or spec.data_dir
    output = args.output or str(Path(data_dir) / "image_split.json")

    split = create_image_split(
        dataset_name=args.dataset,
        data_dir=data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split, f)
    print(f"\n✓ Split saved to {output_path}")

    if args.kaggle_dataset:
        upload_split_to_kaggle(output_path, args.kaggle_dataset)


if __name__ == "__main__":
    main()
