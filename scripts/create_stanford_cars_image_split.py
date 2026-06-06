"""
Create a reproducible image-level train/val/test split for Stanford Cars.

All 196 classes appear in every split; images are divided per class so the
class distribution is balanced across splits.  The split is saved as a JSON
file mapping split name -> list of HF dataset indices, and optionally uploaded
to a Kaggle dataset so it can be reused across notebook runs.

The split file is the single source of truth: load StanfordCarsDataset with
  image_split_path=<path>
to use it.

Usage:
    python scripts/create_stanford_cars_image_split.py \
        --output data/stanford_cars/image_split.json \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --seed 42

    # Also upload to Kaggle:
    python scripts/create_stanford_cars_image_split.py \
        --output data/stanford_cars/image_split.json \
        --kaggle-dataset juandatan/stanford-cars-image-split
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.stanford_cars import StanfordCarsDataset


def create_image_split(
    data_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    """
    Divide Stanford Cars images into train/val/test splits, stratified by class.

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

    # Load dataset without any split filtering to get all images.
    # We pass image_split_path=None (default) and use class-level split
    # with all classes in "train" — but we only need the raw HF indices,
    # so we load with train_ratio=1.0, val_ratio=0.0 to keep every example
    # in a single bucket, then ignore the class split entirely.
    print("Loading full Stanford Cars dataset (no split filtering)...")
    # Use a temporary subclass that skips filtering so we get every image.
    ds = _FullStanfordCars(data_dir=data_dir)

    print(f"Total images: {len(ds.all_hf_indices)}")
    print(f"Total classes: {len(ds.label_to_hf_indices)}")

    rng = np.random.default_rng(seed)

    train_indices, val_indices, test_indices = [], [], []

    for label in sorted(ds.label_to_hf_indices.keys()):
        indices = np.array(ds.label_to_hf_indices[label])
        rng.shuffle(indices)

        n = len(indices)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        # Give remainder to test
        n_test  = n - n_train - n_val
        if n_test < 1:
            # Shrink val if necessary
            n_val  = max(0, n_val - (1 - n_test))
            n_test = n - n_train - n_val

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(  indices[n_train:n_train + n_val].tolist())
        test_indices.extend( indices[n_train + n_val:].tolist())

    # Verify no overlap
    assert not (set(train_indices) & set(val_indices)),  "train/val overlap"
    assert not (set(train_indices) & set(test_indices)), "train/test overlap"
    assert not (set(val_indices)   & set(test_indices)), "val/test overlap"
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(ds.all_hf_indices)

    split = {
        "train": sorted(train_indices),
        "val":   sorted(val_indices),
        "test":  sorted(test_indices),
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(test_ratio, 6),
            "total_images": len(ds.all_hf_indices),
            "num_classes": len(ds.label_to_hf_indices),
            "split_counts": {
                "train": len(train_indices),
                "val":   len(val_indices),
                "test":  len(test_indices),
            },
        },
    }

    print(f"\nSplit summary:")
    print(f"  Train: {len(train_indices):,} images")
    print(f"  Val:   {len(val_indices):,} images")
    print(f"  Test:  {len(test_indices):,} images")

    return split


class _FullStanfordCars:
    """Minimal loader that reads all HF items without any split filtering."""

    def __init__(self, data_dir: str):
        from datasets import load_from_disk, load_dataset

        data_dir = Path(data_dir)
        local_path = data_dir / "hf_cache" / "stanford_cars_train"

        if local_path.exists():
            hf_dataset = load_from_disk(str(local_path))
        else:
            hf_dataset = load_dataset(
                "tanganke/stanford_cars",
                split="train",
                cache_dir=str(data_dir / "hf_cache")
            )
            hf_dataset.save_to_disk(str(local_path))

        self.all_hf_indices = list(range(len(hf_dataset)))
        self.label_to_hf_indices: dict = defaultdict(list)

        for idx, item in enumerate(hf_dataset):
            self.label_to_hf_indices[item['label']].append(idx)


def upload_split_to_kaggle(split_path: Path, kaggle_dataset: str):
    """Upload the split JSON to a Kaggle dataset."""
    import json as _json
    import subprocess
    import shutil
    import tempfile

    username, slug = kaggle_dataset.split('/')
    title = slug.replace('-', ' ').title()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(split_path, tmp / split_path.name)

        metadata = {
            "title": title,
            "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp / "dataset-metadata.json").write_text(_json.dumps(metadata, indent=2))

        # Check if dataset exists
        exists = subprocess.run(
            ["kaggle", "datasets", "status", kaggle_dataset],
            capture_output=True
        ).returncode == 0

        if exists:
            cmd = ["kaggle", "datasets", "version", "-p", str(tmp),
                   "-m", f"Update split: {split_path.name}", "--dir-mode", "zip"]
        else:
            cmd = ["kaggle", "datasets", "create", "-p", str(tmp), "--dir-mode", "zip"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Uploaded to https://www.kaggle.com/datasets/{kaggle_dataset}")
        else:
            print(f"✗ Upload failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Create image-level train/val/test split for Stanford Cars"
    )
    parser.add_argument("--data-dir", default="data/stanford_cars",
                        help="Stanford Cars data directory")
    parser.add_argument("--output", default="data/stanford_cars/image_split.json",
                        help="Output path for the split JSON")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of images per class for training (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of images per class for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--kaggle-dataset", type=str, default=None,
                        help="Kaggle dataset to upload to (username/dataset-slug)")
    args = parser.parse_args()

    split = create_image_split(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split, f)
    print(f"\n✓ Split saved to {output_path}")

    if args.kaggle_dataset:
        upload_split_to_kaggle(output_path, args.kaggle_dataset)


if __name__ == "__main__":
    main()
