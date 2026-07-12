"""
Upload SigLIP image embeddings, SigLIP text embeddings, and top-K I2T
distractor rankings for a dataset to a single Kaggle dataset.

Bundles the outputs of build_siglip_embeddings.py and
build_distractor_rankings.py:
    siglip_text_embeddings.pkl
    siglip_image_embeddings_{split}.pkl   (one per split)
    siglip_top{k}_rankings_{split}.pkl    (one per split)

Usage:
    python scripts/upload_siglip_distractor_set_to_kaggle.py \
        --data-dir data/cub_200 \
        --dataset-name juandatan/cub-200-siglip-embeddings-distractor-set \
        --splits train val test
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def upload_to_kaggle(
    data_dir: str,
    dataset_name: str,
    splits: list,
    title: str = None,
):
    data_dir = Path(data_dir)

    files_to_upload = [data_dir / 'siglip_text_embeddings.pkl']
    for split in splits:
        files_to_upload.append(data_dir / f'siglip_image_embeddings_{split}.pkl')
        # Top-K rankings filename is k-dependent; glob to pick up whatever k was used.
        ranking_matches = sorted(data_dir.glob(f'siglip_top*_rankings_{split}.pkl'))
        if not ranking_matches:
            raise FileNotFoundError(
                f"No siglip_top*_rankings_{split}.pkl found in {data_dir}. Run "
                f"scripts/build_distractor_rankings.py --dataset ... --splits {split} first."
            )
        files_to_upload.extend(ranking_matches)

    missing = [f for f in files_to_upload if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    if '/' not in dataset_name:
        raise ValueError("dataset_name must be in format 'username/dataset-slug'")

    username, slug = dataset_name.split('/')
    if title is None:
        title = slug.replace('-', ' ').title()

    print(f"Uploading SigLIP embeddings + distractor rankings to Kaggle dataset: {dataset_name}")
    for f in files_to_upload:
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for f in files_to_upload:
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
                   "-m", f"Upload {len(files_to_upload)} files", "--dir-mode", "zip"]
        else:
            print("Creating new dataset...")
            cmd = ["kaggle", "datasets", "create", "-p", str(temp_path), "--dir-mode", "zip"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False

        print(result.stdout)
        print(f"✓ Uploaded {len(files_to_upload)} files to {dataset_name}")
        print(f"\nMount paths in Kaggle notebooks:")
        for f in files_to_upload:
            print(f"  /kaggle/input/{slug}/{f.name}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload SigLIP embeddings + distractor rankings to a Kaggle dataset"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing siglip_*.pkl files (e.g. data/cub_200)")
    parser.add_argument("--dataset-name", required=True,
                        help="Kaggle dataset name (username/dataset-slug)")
    parser.add_argument("--splits", type=str, nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help="Which splits' image embeddings/rankings to upload (default: all)")
    parser.add_argument("--title", default=None,
                        help="Dataset title (optional)")

    args = parser.parse_args()
    success = upload_to_kaggle(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        splits=args.splits,
        title=args.title,
    )
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
