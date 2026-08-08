"""Publish a completed frozen-Idefics2 probe cache to Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _cache_files(cache_dir: Path) -> tuple[dict, list[Path]]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
    with metadata_path.open() as file:
        metadata = json.load(file)
    if not bool(metadata.get("complete", False)):
        raise ValueError(
            "Refusing to publish an incomplete probe cache; metadata.complete "
            "is not true"
        )

    splits = metadata.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("Probe cache metadata has no splits")
    dtype = str(metadata.get("dtype"))
    if dtype not in {"int8", "float16"}:
        raise ValueError(f"Unsupported probe cache dtype: {dtype!r}")

    files = [metadata_path]
    for split in splits:
        files.extend([
            cache_dir / f"pair_representations_{split}.npy",
            cache_dir / f"pair_complete_{split}.npy",
            cache_dir / f"query_indices_{split}.npy",
            cache_dir / f"candidate_indices_{split}.npy",
        ])
        if dtype == "int8":
            files.append(cache_dir / f"pair_representation_scales_{split}.npy")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Completed cache is missing files: {missing}")
    return metadata, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Kaggle dataset identifier in owner/slug form",
    )
    parser.add_argument("--title")
    parser.add_argument(
        "--version-message",
        default="Publish completed frozen-Idefics2 probe cache",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    if args.dataset_name.count("/") != 1:
        raise ValueError("--dataset-name must use owner/slug format")
    owner, slug = args.dataset_name.split("/", maxsplit=1)
    if not owner or not slug:
        raise ValueError("--dataset-name must use owner/slug format")

    metadata, files = _cache_files(cache_dir)
    total_bytes = sum(path.stat().st_size for path in files)
    completed_pairs = sum(
        int(value) for value in metadata.get("completed_pairs", {}).values()
    )
    print(
        f"Validated {len(files)} cache files, {completed_pairs:,} pairs, "
        f"{total_bytes / 1_000_000_000:.2f} GB",
        flush=True,
    )

    # Keep staging on the cache filesystem so hard links do not consume another
    # copy of the ~1.23 GB representation arrays.
    with tempfile.TemporaryDirectory(
        dir=cache_dir.parent,
        prefix=".kaggle-probe-cache-",
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        for path in files:
            os.link(path, staging_dir / path.name)
        dataset_metadata = {
            "title": args.title or slug.replace("-", " ").title(),
            "id": f"{owner}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging_dir / "dataset-metadata.json").write_text(
            json.dumps(dataset_metadata, indent=2) + "\n"
        )

        status = subprocess.run(
            ["kaggle", "datasets", "status", args.dataset_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            command = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(staging_dir),
                "-m",
                args.version_message,
                "--dir-mode",
                "zip",
            ]
        else:
            command = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(staging_dir),
                "--dir-mode",
                "zip",
            ]
        subprocess.run(command, check=True)

    print(f"✓ Published completed probe cache to {args.dataset_name}")


if __name__ == "__main__":
    main()
