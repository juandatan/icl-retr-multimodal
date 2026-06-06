"""
Upload reranker checkpoint to Kaggle dataset for remote evaluation.

Usage:
    python scripts/upload_checkpoint_to_kaggle.py \
        --checkpoint outputs/reranker_checkpoints/reranker_mini_imagenet_v2/best_model.pt \
        --dataset-name juandatan/mini-imagenet-reranker-checkpoint \
        --title "Mini-ImageNet Reranker Checkpoint"
"""

import argparse
import json
import subprocess
import shutil
from pathlib import Path
import tempfile


def upload_checkpoint_to_kaggle(
    checkpoint_path: str,
    dataset_name: str,
    title: str = None,
    is_public: bool = False
):
    """
    Upload a checkpoint file to Kaggle as a dataset.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        dataset_name: Kaggle dataset name (username/dataset-slug)
        title: Dataset title (defaults to dataset slug)
        is_public: Whether to make dataset public
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Parse dataset name
    if '/' not in dataset_name:
        raise ValueError("dataset_name must be in format 'username/dataset-slug'")

    username, slug = dataset_name.split('/')

    if title is None:
        title = slug.replace('-', ' ').title()

    print(f"Uploading checkpoint to Kaggle dataset: {dataset_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy checkpoint to temp directory
        dest_file = temp_path / checkpoint_path.name
        shutil.copy2(checkpoint_path, dest_file)
        print(f"✓ Copied checkpoint to temporary directory")

        # Create dataset metadata
        metadata = {
            "title": title,
            "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
            "resources": [
                {
                    "path": checkpoint_path.name,
                    "description": f"Reranker checkpoint: {checkpoint_path.name}"
                }
            ]
        }

        metadata_path = temp_path / "dataset-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Created metadata file")

        # Check if dataset exists
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "status", dataset_name],
                capture_output=True,
                text=True,
                check=False
            )
            dataset_exists = result.returncode == 0
        except FileNotFoundError:
            raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")

        # Upload (create new or update existing)
        if dataset_exists:
            print(f"Dataset exists, creating new version...")
            cmd = [
                "kaggle", "datasets", "version",
                "-p", str(temp_path),
                "-m", f"Updated checkpoint: {checkpoint_path.name}",
                "--dir-mode", "zip"
            ]
        else:
            print(f"Creating new dataset...")
            cmd = [
                "kaggle", "datasets", "create",
                "-p", str(temp_path),
                "--dir-mode", "zip"
            ]

            if is_public:
                cmd.append("--public")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error uploading dataset:")
            print(result.stderr)
            return False

        print(result.stdout)
        print(f"\n✓ Successfully uploaded checkpoint!")
        print(f"Dataset URL: https://www.kaggle.com/datasets/{dataset_name}")
        print(f"\nTo use in Kaggle notebook:")
        print(f"  Add as dataset: {dataset_name}")
        print(f"  Checkpoint path: /kaggle/input/{slug}/{checkpoint_path.name}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Kaggle dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Kaggle dataset name (username/dataset-slug)")
    parser.add_argument("--title", type=str, default=None,
                        help="Dataset title (optional)")
    parser.add_argument("--public", action="store_true",
                        help="Make dataset public")

    args = parser.parse_args()

    success = upload_checkpoint_to_kaggle(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset_name,
        title=args.title,
        is_public=args.public
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
