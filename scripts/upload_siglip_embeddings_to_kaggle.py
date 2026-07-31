"""Publish the SigLIP artifacts used for CUB hard-label construction."""

import argparse
from pathlib import Path

from src.utils.kaggle_utils import kaggle_publish_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--title")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = [data_dir / "siglip_text_embeddings.pkl"]
    files.extend(
        data_dir / f"siglip_image_embeddings_{split}.pkl"
        for split in args.splits
    )
    success = kaggle_publish_files(
        files,
        args.dataset_name,
        title=args.title,
        version_message=f"Publish SigLIP artifacts for {', '.join(args.splits)}",
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
