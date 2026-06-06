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


def get_kaggle_checkpoint_dataset(cfg=None) -> str:
    """Get Kaggle checkpoint dataset name from config or environment variable.

    Config takes precedence so each experiment uses its own dataset without
    relying on environment variable state across runs.
    """
    if cfg is not None:
        from_cfg = cfg.get('checkpoint', {}).get('kaggle_dataset', '')
        if from_cfg:
            return from_cfg
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
                    # Copy all checkpoints to the checkpoint directory
                    for ckpt in downloaded_checkpoints:
                        target = checkpoint_dir / ckpt.name
                        shutil.copy(ckpt, target)

                    print(f"✓ Downloaded {len(downloaded_checkpoints)} checkpoint(s)")
                    return len(downloaded_checkpoints)
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


def kaggle_upload_checkpoints(checkpoint_dir: Path, dataset_name: str, experiment_name: str, latest_only: bool = True, latest_per_range: bool = False, num_ranges: int = 1) -> bool:
    """
    Upload checkpoints to Kaggle dataset.

    Args:
        checkpoint_dir: Local directory containing checkpoints
        dataset_name: Kaggle dataset name (format: username/dataset-name)
        experiment_name: Name of the experiment (for dataset title)
        latest_only: If True, only upload the latest checkpoint (default: True for single-GPU)
        latest_per_range: If True, upload latest checkpoint per query range (for multi-GPU)
        num_ranges: Number of query ranges (for multi-GPU, typically number of GPUs)

    Returns:
        True if upload successful, False otherwise
    """
    if not dataset_name:
        return False

    try:
        # Find checkpoints to upload
        all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
        if not all_checkpoints:
            print("⚠️  No checkpoints found to upload")
            return False

        # Select which checkpoints to upload
        if latest_per_range and num_ranges > 1:
            # For multi-GPU: upload latest checkpoint from each query range
            # Group checkpoints by query range
            checkpoints_by_range = [[] for _ in range(num_ranges)]

            # Estimate query range size from total queries
            # Assume checkpoints are evenly distributed
            max_query_idx = max(int(ckpt.stem.split('_')[1]) for ckpt in all_checkpoints)
            range_size = (max_query_idx + 1) // num_ranges

            for ckpt in all_checkpoints:
                query_idx = int(ckpt.stem.split('_')[1])
                range_id = min(query_idx // range_size, num_ranges - 1)
                checkpoints_by_range[range_id].append(ckpt)

            # Select latest from each range
            checkpoints = []
            for range_id, range_ckpts in enumerate(checkpoints_by_range):
                if range_ckpts:
                    checkpoints.append(sorted(range_ckpts)[-1])
        elif latest_only:
            checkpoints = [all_checkpoints[-1]]  # Only latest
        else:
            checkpoints = all_checkpoints  # All checkpoints

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Check if dataset exists
            result = subprocess.run(
                ['kaggle', 'datasets', 'status', dataset_name],
                capture_output=True,
                text=True
            )
            dataset_exists = result.returncode == 0

            # Copy selected checkpoints to temp directory
            for ckpt in checkpoints:
                shutil.copy(ckpt, temp_path / ckpt.name)

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
            if len(checkpoints) == 1:
                checkpoint_name = checkpoints[0].name
                message = f'Update: {checkpoint_name}'
            else:
                checkpoint_names = [ckpt.name for ckpt in checkpoints]
                message = f'Update: {len(checkpoints)} checkpoints'

            if dataset_exists:
                result = subprocess.run(
                    ['kaggle', 'datasets', 'version', '-p', str(temp_path), '-m', message, '--dir-mode', 'zip'],
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
                if len(checkpoints) == 1:
                    print(f"✓ Uploaded {checkpoint_name} to Kaggle")
                else:
                    print(f"✓ Uploaded {len(checkpoints)} checkpoints to Kaggle")
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


def resolve_data_paths(
    local_path: str,
    kaggle_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    required: bool = True
) -> str:
    """
    Resolve data paths for both local and Kaggle environments.

    Args:
        local_path: Path to use in local environment
        kaggle_path: Path to use in Kaggle environment (e.g., /kaggle/input/dataset/file.pkl)
        dataset_name: Name of Kaggle dataset for error messages
        required: If True, raise error if file not found

    Returns:
        Resolved path to use

    Raises:
        FileNotFoundError: If required=True and file not found
    """
    if is_kaggle_environment():
        # Use Kaggle path
        path = kaggle_path if kaggle_path else local_path

        if required and not Path(path).exists():
            error_msg = f"File not found at {path}."
            if dataset_name:
                error_msg += f"\n\nPlease add the '{dataset_name}' dataset to your Kaggle notebook inputs."
                error_msg += "\n\nTo add a dataset:"
                error_msg += "\n  1. Click 'Add Input' in the right sidebar"
                error_msg += "\n  2. Search for the dataset"
                error_msg += "\n  3. Click 'Add' to attach it to your notebook"
            raise FileNotFoundError(error_msg)

        return path
    else:
        # Use local path
        if required and not Path(local_path).exists():
            raise FileNotFoundError(
                f"File not found at {local_path}.\n"
                f"Please ensure the file exists or generate it first."
            )

        return local_path
