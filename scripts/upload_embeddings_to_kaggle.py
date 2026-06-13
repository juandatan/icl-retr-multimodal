"""
Upload CLIP embedding splits to a Kaggle dataset.

Usage:
    python scripts/upload_embeddings_to_kaggle.py \
        --embeddings-dir data/stanford_cars \
        --dataset-name juandatan/stanford-cars-clip \
        --title "Stanford Cars CLIP Embeddings"
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def upload_embeddings_to_kaggle(
    embeddings_dir: str,
    dataset_name: str,
    title: str = None,
):
    embeddings_dir = Path(embeddings_dir)
    embedding_files = sorted(embeddings_dir.glob("clip_embeddings_*.pkl"))

    if not embedding_files:
        raise FileNotFoundError(f"No clip_embeddings_*.pkl files found in {embeddings_dir}")

    if '/' not in dataset_name:
        raise ValueError("dataset_name must be in format 'username/dataset-slug'")

    username, slug = dataset_name.split('/')
    if title is None:
        title = slug.replace('-', ' ').title()

    print(f"Uploading CLIP embeddings to Kaggle dataset: {dataset_name}")
    for f in embedding_files:
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for f in embedding_files:
            shutil.copy2(f, temp_path / f.name)

        metadata = {
            "title": title,
            "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(temp_path / "dataset-metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        status = subprocess.run(
            ["kaggle", "datasets", "status", dataset_name],
            capture_output=True, text=True, check=False
        )
        dataset_exists = status.returncode == 0

        if dataset_exists:
            print("Dataset exists, creating new version...")
            cmd = ["kaggle", "datasets", "version", "-p", str(temp_path),
                   "-m", f"Upload {len(embedding_files)} embedding splits", "--dir-mode", "zip"]
        else:
            print("Creating new dataset...")
            cmd = ["kaggle", "datasets", "create", "-p", str(temp_path), "--dir-mode", "zip"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False

        print(result.stdout)
        print(f"✓ Uploaded {len(embedding_files)} embedding files to {dataset_name}")
        print(f"\nMount paths in Kaggle notebooks:")
        for f in embedding_files:
            print(f"  /kaggle/input/{slug}/{f.name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Upload CLIP embeddings to Kaggle dataset")
    parser.add_argument("--embeddings-dir", required=True,
                        help="Directory containing clip_embeddings_*.pkl files")
    parser.add_argument("--dataset-name", required=True,
                        help="Kaggle dataset name (username/dataset-slug)")
    parser.add_argument("--title", default=None,
                        help="Dataset title (optional)")

    args = parser.parse_args()
    success = upload_embeddings_to_kaggle(
        embeddings_dir=args.embeddings_dir,
        dataset_name=args.dataset_name,
        title=args.title,
    )
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
