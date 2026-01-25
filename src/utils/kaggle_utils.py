"""
Utilities for managing checkpoints with Kaggle Datasets API.

This module provides functions to:
1. Detect if running in Kaggle environment
2. Download existing checkpoints from Kaggle dataset
3. Upload new checkpoints to Kaggle dataset

Usage:
    Set environment variable: KAGGLE_CHECKPOINT_DATASET=username/dataset-name
    The functions will automatically handle checkpoint persistence.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional


def is_kaggle_environment() -> bool:
    """Check if running in Kaggle environment."""
    return os.path.exists('/kaggle')


def get_kaggle_checkpoint_dataset() -> str:
    """Get Kaggle checkpoint dataset name from environment variable."""
    return os.environ.get('KAGGLE_CHECKPOINT_DATASET', '')


def kaggle_download_checkpoints(checkpoint_dir: Path, dataset_name: str) -> int:
    """
    Download existing checkpoints from Kaggle dataset.

    Note: Downloads to a temporary directory first, then extracts the latest checkpoint
    to avoid accumulating old checkpoints on each download.

    Args:
        checkpoint_dir: Local directory to download checkpoints to
        dataset_name: Kaggle dataset name (format: username/dataset-name)

    Returns:
        Number of checkpoints downloaded
    """
    if not dataset_name:
        print("ℹ️  KAGGLE_CHECKPOINT_DATASET not set. Starting fresh.")
        return 0

    print(f"\n{'='*70}")
    print(f"Downloading existing checkpoints from Kaggle dataset: {dataset_name}")
    print(f"{'='*70}\n")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tempfile
        import shutil

        # Download to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', temp_dir, '--unzip'],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                # Find all downloaded checkpoints
                temp_path = Path(temp_dir)
                downloaded_checkpoints = sorted(temp_path.glob("**/checkpoint_*.pkl"))

                if downloaded_checkpoints:
                    # Copy only the latest checkpoint
                    latest = downloaded_checkpoints[-1]
                    target = checkpoint_dir / latest.name
                    shutil.copy(latest, target)

                    print(f"✓ Downloaded latest checkpoint: {latest.name}")
                    print(f"  (Ignoring {len(downloaded_checkpoints) - 1} older checkpoints for efficiency)")
                    return 1
                else:
                    print("ℹ️  No checkpoints found in dataset. Starting fresh.")
                    return 0
            else:
                if "404" in result.stderr or "not found" in result.stderr.lower():
                    print(f"ℹ️  No existing dataset found. Will create new one on first checkpoint save.")
                else:
                    print(f"⚠️  Failed to download checkpoints: {result.stderr}")
                    print(f"   Starting fresh...")
                return 0

    except subprocess.TimeoutExpired:
        print("⚠️  Download timed out. Starting fresh...")
        return 0
    except FileNotFoundError:
        print("⚠️  Kaggle CLI not found. Install with: pip install kaggle")
        print("   Starting fresh...")
        return 0
    except Exception as e:
        print(f"⚠️  Error downloading checkpoints: {e}")
        print("   Starting fresh...")
        return 0


def kaggle_upload_checkpoints(checkpoint_dir: Path, dataset_name: str, experiment_name: str, latest_only: bool = True) -> bool:
    """
    Upload checkpoints to Kaggle dataset.

    Args:
        checkpoint_dir: Local directory containing checkpoints
        dataset_name: Kaggle dataset name (format: username/dataset-name)
        experiment_name: Name of the experiment (for dataset title)
        latest_only: If True, only upload the latest checkpoint (default: True for efficiency)

    Returns:
        True if upload successful, False otherwise
    """
    if not dataset_name:
        return False

    try:
        # Find checkpoints to upload
        if latest_only:
            # Upload latest checkpoint while preserving existing ones
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
            if not checkpoints:
                print("⚠️  No checkpoints found to upload")
                return False

            latest_checkpoint = checkpoints[-1]
            print(f"\n📤 Uploading latest checkpoint to Kaggle: {latest_checkpoint.name}")

            # Create a temporary directory with all checkpoints
            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                temp_checkpoint_dir = temp_path / "checkpoints"
                temp_checkpoint_dir.mkdir()

                # Check if dataset exists
                result = subprocess.run(
                    ['kaggle', 'datasets', 'status', dataset_name],
                    capture_output=True,
                    text=True
                )
                dataset_exists = result.returncode == 0

                if dataset_exists:
                    # Download existing checkpoints first to preserve them
                    print(f"  Downloading existing checkpoints to merge...")
                    download_result = subprocess.run(
                        ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(temp_checkpoint_dir), '--unzip'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if download_result.returncode != 0:
                        print(f"  ⚠️  Warning: Could not download existing checkpoints: {download_result.stderr}")
                        print(f"  Continuing with upload (may overwrite existing data)...")

                # Copy the latest checkpoint (will add to existing or create new)
                shutil.copy(latest_checkpoint, temp_checkpoint_dir / latest_checkpoint.name)

                # Count total checkpoints
                total_checkpoints = len(list(temp_checkpoint_dir.glob("checkpoint_*.pkl")))
                print(f"  Total checkpoints in dataset: {total_checkpoints}")

                # Create metadata
                username, dataset_slug = dataset_name.split('/')
                metadata = {
                    "title": f"{experiment_name} Checkpoints",
                    "id": f"{username}/{dataset_slug}",
                    "licenses": [{"name": "CC0-1.0"}]
                }

                metadata_path = temp_path / "dataset-metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Upload
                if dataset_exists:
                    result = subprocess.run(
                        ['kaggle', 'datasets', 'version', '-p', str(temp_path), '-m', f'Checkpoint {latest_checkpoint.stem}', '--dir-mode', 'zip'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                else:
                    result = subprocess.run(
                        ['kaggle', 'datasets', 'create', '-p', str(temp_path), '--dir-mode', 'zip'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                if result.returncode == 0:
                    print(f"✓ Uploaded {latest_checkpoint.name} to Kaggle successfully")
                    return True
                else:
                    print(f"⚠️  Failed to upload: {result.stderr}")
                    return False

        else:
            # Upload all checkpoints (original behavior)
            print(f"\n📤 Uploading all checkpoints to Kaggle dataset: {dataset_name}")

            # Check if dataset exists
            result = subprocess.run(
                ['kaggle', 'datasets', 'status', dataset_name],
                capture_output=True,
                text=True
            )
            dataset_exists = result.returncode == 0

            # Create dataset metadata
            username, dataset_slug = dataset_name.split('/')
            metadata = {
                "title": f"{experiment_name} Checkpoints",
                "id": f"{username}/{dataset_slug}",
                "licenses": [{"name": "CC0-1.0"}]
            }

            metadata_path = checkpoint_dir.parent / "dataset-metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if dataset_exists:
                # Update existing dataset
                result = subprocess.run(
                    ['kaggle', 'datasets', 'version', '-p', str(checkpoint_dir.parent), '-m', 'Updated checkpoints', '--dir-mode', 'zip'],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
            else:
                # Create new dataset
                result = subprocess.run(
                    ['kaggle', 'datasets', 'create', '-p', str(checkpoint_dir.parent), '--dir-mode', 'zip'],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

            # Clean up metadata file
            if metadata_path.exists():
                metadata_path.unlink()

            if result.returncode == 0:
                print(f"✓ Uploaded checkpoints to Kaggle successfully")
                return True
            else:
                print(f"⚠️  Failed to upload: {result.stderr}")
                return False

    except subprocess.TimeoutExpired:
        print(f"⚠️  Upload timed out")
        return False
    except FileNotFoundError:
        print("⚠️  Kaggle CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"⚠️  Error uploading checkpoints: {e}")
        return False


def setup_kaggle_credentials(kaggle_json_path: Optional[str] = None):
    """
    Setup Kaggle API credentials.

    Args:
        kaggle_json_path: Path to kaggle.json file. If None, assumes credentials are already set up.
    """
    if kaggle_json_path and Path(kaggle_json_path).exists():
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True, parents=True)

        target_path = kaggle_dir / 'kaggle.json'

        # Copy credentials
        import shutil
        shutil.copy(kaggle_json_path, target_path)

        # Set proper permissions
        target_path.chmod(0o600)

        print(f"✓ Kaggle credentials configured from {kaggle_json_path}")
    elif not (Path.home() / '.kaggle' / 'kaggle.json').exists():
        print("⚠️  Kaggle credentials not found. Please set up kaggle.json")
        print("   See: https://github.com/Kaggle/kaggle-api#api-credentials")
