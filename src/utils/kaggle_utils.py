"""Kaggle persistence for long-running evaluation artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import zipfile
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


def _list_dataset_files(dataset_name: str) -> list[str]:
    """Return every file name in a Kaggle dataset, following result pages."""
    names, page_token = [], None
    while True:
        command = ["kaggle", "datasets", "files", dataset_name, "--csv", "--page-size", "200"]
        if page_token:
            command += ["--page-token", page_token]
        listing = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
        if listing.returncode != 0:
            raise RuntimeError(f"could not list files: {listing.stderr.strip()}")
        page_token = None
        rows = []
        for line in listing.stdout.splitlines():
            if line.startswith("Next Page Token = "):
                page_token = line.removeprefix("Next Page Token = ").strip()
            elif line.strip() and not line.startswith("Warning:"):
                rows.append(line)
        if not rows or rows[0].split(",")[0] != "name":
            raise RuntimeError(f"unexpected file listing: {listing.stdout.strip()[:200]}")
        names += [next(csv.reader([row]))[0] for row in rows[1:]]
        if not page_token:
            return names


def _download_existing_files(dataset_name: str, staging_dir: Path) -> Optional[str]:
    """Download every file of an existing dataset; return an error or None.

    The whole-dataset archive endpoint returns 404 for some private datasets
    (seen with kaggle 2.1.2 after a ``create``), while per-file downloads
    still work, so fall back to those and confirm that nothing is missing.
    """
    archive = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(staging_dir), "--unzip"],
        capture_output=True, text=True, check=False, timeout=300,
    )
    try:
        names = _list_dataset_files(dataset_name)
    except RuntimeError as error:
        return f"{archive.stderr.strip()} {error}".strip()
    for name in names:
        target = staging_dir / name
        if target.is_file():
            continue
        single = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-f", name, "-p", str(staging_dir)],
            capture_output=True, text=True, check=False, timeout=300,
        )
        # Large single files arrive zipped.
        zipped = staging_dir / f"{name}.zip"
        if not target.is_file() and zipped.is_file():
            with zipfile.ZipFile(zipped) as archive_file:
                archive_file.extractall(staging_dir)
            zipped.unlink()
        if not target.is_file():
            return f"could not download {name}: {single.stderr.strip()}"
    return None


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
                error = _download_existing_files(dataset_name, staging_dir)
                if error:
                    print(
                        "⚠️  Refusing to publish because the existing Kaggle "
                        f"dataset could not be preserved: {error}"
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
