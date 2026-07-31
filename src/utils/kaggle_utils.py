"""Kaggle persistence for long-running evaluation artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def kaggle_publish_files(
    files: list[Path],
    dataset_name: str,
    title: Optional[str] = None,
    version_message: str = "Update artifacts",
) -> bool:
    """Create or replace a Kaggle artifact dataset from an explicit file set."""
    resolved_files = [Path(path) for path in files]
    missing = [path for path in resolved_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {missing}")
    if "/" not in dataset_name:
        raise ValueError("dataset_name must use the 'owner/slug' format")

    owner, slug = dataset_name.split("/", maxsplit=1)
    title = title or slug.replace("-", " ").title()
    with tempfile.TemporaryDirectory() as temporary_dir:
        staging_dir = Path(temporary_dir)
        for path in resolved_files:
            shutil.copy2(path, staging_dir / path.name)
        metadata = {
            "title": title,
            "id": f"{owner}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        status = subprocess.run(
            ["kaggle", "datasets", "status", dataset_name],
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
                version_message,
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
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Kaggle publish failed: {result.stderr}")
            return False
        print(f"✓ Published {len(resolved_files)} file(s) to {dataset_name}")
        return True


def kaggle_upload_eval_results(
    output_dir: Path,
    dataset_name: str,
    title: Optional[str] = None,
) -> bool:
    """Merge a local result directory into a versioned Kaggle dataset.

    Existing remote files are downloaded into the staging directory first so
    publishing one pipeline stage cannot erase artifacts from another stage.
    """
    if not dataset_name:
        return False

    output_dir = Path(output_dir)
    files = [path for path in output_dir.iterdir() if path.is_file()]
    if not files:
        print(f"⚠️  No files found in {output_dir} to upload")
        return False

    owner, slug = dataset_name.split("/", maxsplit=1)
    title = title or slug.replace("-", " ").title()

    try:
        with tempfile.TemporaryDirectory() as temporary_dir:
            staging_dir = Path(temporary_dir)
            status = subprocess.run(
                ["kaggle", "datasets", "status", dataset_name],
                capture_output=True,
                text=True,
                check=False,
            )
            dataset_exists = status.returncode == 0

            if dataset_exists:
                download = subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        dataset_name,
                        "-p",
                        str(staging_dir),
                        "--unzip",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,
                )
                if download.returncode != 0:
                    print(
                        "⚠️  Refusing to publish because the existing Kaggle "
                        f"dataset could not be preserved: {download.stderr}"
                    )
                    return False

            for path in files:
                shutil.copy2(path, staging_dir / path.name)

            metadata = {
                "title": title,
                "id": f"{owner}/{slug}",
                "licenses": [{"name": "CC0-1.0"}],
            }
            (staging_dir / "dataset-metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

            if dataset_exists:
                command = [
                    "kaggle",
                    "datasets",
                    "version",
                    "-p",
                    str(staging_dir),
                    "-m",
                    f"Upload {len(files)} result file(s)",
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

            upload = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if upload.returncode != 0:
                print(f"⚠️  Failed to upload results: {upload.stderr}")
                return False

            print(f"✓ Uploaded {len(files)} result file(s) to {dataset_name}")
            return True
    except subprocess.TimeoutExpired:
        print("⚠️  Kaggle operation timed out")
        return False
    except FileNotFoundError:
        print("⚠️  Kaggle CLI not found; local artifacts were preserved")
        return False
