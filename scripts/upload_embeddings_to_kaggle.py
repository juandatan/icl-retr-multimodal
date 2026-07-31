"""Publish precomputed CUB CLIP embeddings as a Kaggle dataset."""

import argparse
from pathlib import Path

from src.utils.kaggle_utils import kaggle_publish_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--title")
    args = parser.parse_args()

    files = sorted(Path(args.embeddings_dir).glob("clip_embeddings_*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No clip_embeddings_*.pkl files found in {args.embeddings_dir}"
        )
    success = kaggle_publish_files(
        files,
        args.dataset_name,
        title=args.title,
        version_message=f"Publish {len(files)} CLIP embedding split(s)",
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
