"""Dataset and embedding-artifact loading for the CUB-200 pipeline."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .dataset_registry import get_dataset_spec

if TYPE_CHECKING:
    from .fine_grained_hf_dataset import FineGrainedHFDataset


SUPPORTED_DATASET = "cub_200"


def download_kaggle_file(
    dataset_name: str,
    filename: str,
    cache_dir: str = "cache/embeddings",
) -> Path:
    """Download one file from a Kaggle dataset into a stable local cache."""
    cache_path = Path(cache_dir) / dataset_name.replace("/", "_") / filename
    if cache_path.exists():
        print(f"✓ Using cached file: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_name,
            "-p",
            temporary_dir,
            "--unzip",
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Failed to download {dataset_name}: {error.stderr}"
            ) from error
        except FileNotFoundError as error:
            raise RuntimeError(
                "Kaggle CLI not found. Install the project dependencies and "
                "configure Kaggle credentials."
            ) from error

        downloaded_path = Path(temporary_dir) / filename
        if not downloaded_path.exists():
            raise FileNotFoundError(
                f"{filename} was not present in Kaggle dataset {dataset_name}"
            )
        shutil.move(str(downloaded_path), cache_path)

    print(f"✓ Downloaded and cached: {cache_path}")
    return cache_path


def resolve_clip_cache_path(
    split: str,
    kaggle_dataset: Optional[str],
) -> Optional[Path]:
    """Resolve a CLIP embedding cache from Kaggle mounts or the CLI cache."""
    if not kaggle_dataset:
        return None

    owner, slug = kaggle_dataset.split("/", maxsplit=1)
    filename = f"clip_embeddings_{split}.pkl"
    mounted_paths = (
        Path(f"/kaggle/input/datasets/{owner}/{slug}/{filename}"),
        Path(f"/kaggle/input/{slug}/{filename}"),
    )
    for mounted_path in mounted_paths:
        if mounted_path.exists():
            print(f"✓ Using mounted Kaggle input: {mounted_path}")
            return mounted_path

    return download_kaggle_file(kaggle_dataset, filename)


def resolve_siglip_cache_path(
    dataset_name: str,
    filename: str,
    kaggle_dataset: Optional[str],
) -> Path:
    """Resolve a SigLIP artifact from a Kaggle mount or the local data tree."""
    if kaggle_dataset:
        owner, slug = kaggle_dataset.split("/", maxsplit=1)
        mounted_paths = (
            Path(f"/kaggle/input/datasets/{owner}/{slug}/{filename}"),
            Path(f"/kaggle/input/{slug}/{filename}"),
        )
        for mounted_path in mounted_paths:
            if mounted_path.exists():
                print(f"✓ Using mounted Kaggle input: {mounted_path}")
                return mounted_path
    return Path("data") / dataset_name / filename


def load_siglip_inputs(
    dataset_name: str,
    split: str,
    kaggle_dataset: Optional[str],
    *,
    expected_class_names: Optional[Sequence[str]] = None,
    expected_example_ids: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list[str], np.ndarray, Path, Path]:
    """Load aligned SigLIP class-text and split-image embedding tables."""
    text_path = resolve_siglip_cache_path(
        dataset_name, "siglip_text_embeddings.pkl", kaggle_dataset
    )
    image_path = resolve_siglip_cache_path(
        dataset_name, f"siglip_image_embeddings_{split}.pkl", kaggle_dataset
    )
    for path in (text_path, image_path):
        if not path.exists():
            raise FileNotFoundError(f"Required SigLIP artifact not found: {path}")
    with open(text_path, "rb") as file:
        text_data = pickle.load(file)
    with open(image_path, "rb") as file:
        image_data = pickle.load(file)
    text_model = text_data.get("model_name")
    image_model = image_data.get("model_name")
    if text_model and image_model and text_model != image_model:
        raise ValueError(
            "SigLIP text and image artifacts were produced by different models"
        )
    class_names = list(text_data["class_names"])
    if expected_class_names is not None and class_names != list(expected_class_names):
        raise ValueError("SigLIP class ordering differs from the loaded dataset")
    example_ids = image_data.get("example_ids")
    if (
        expected_example_ids is not None
        and example_ids is not None
        and list(example_ids) != list(expected_example_ids)
    ):
        raise ValueError("SigLIP image rows differ from the loaded dataset split")
    return (
        np.asarray(text_data["embeddings"]),
        class_names,
        np.asarray(image_data["embeddings"]),
        text_path,
        image_path,
    )


def load_dataset(
    dataset_name: str,
    split: str,
    image_split_path: Optional[str],
    embeddings_kaggle_dataset: Optional[str] = None,
) -> "FineGrainedHFDataset":
    """Load one CUB-200 split and its precomputed CLIP embeddings."""
    if dataset_name != SUPPORTED_DATASET:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}; this pipeline supports "
            f"{SUPPORTED_DATASET!r} only"
        )

    from .fine_grained_hf_dataset import FineGrainedHFDataset

    spec = get_dataset_spec(dataset_name)
    print(f"\nLoading {spec.display_name} ({split} split)...")
    dataset = FineGrainedHFDataset(
        hf_repo_ids=list(spec.hf_repo_ids),
        split=split,
        data_dir=spec.data_dir,
        class_split_seed=42,
        image_split_path=image_split_path,
    )

    cache_path = resolve_clip_cache_path(split, embeddings_kaggle_dataset)
    if not dataset.load_clip_embeddings(cache_path=cache_path):
        raise FileNotFoundError(
            f"CLIP embeddings for split {split!r} were not found or did not "
            "match the dataset. Run `python -m scripts.build_clip_embeddings`."
        )

    print(f"✓ Loaded {len(dataset)} examples, {dataset.num_classes} classes")
    print(f"✓ CLIP embeddings: {dataset.clip_embeddings.shape}")
    return dataset
