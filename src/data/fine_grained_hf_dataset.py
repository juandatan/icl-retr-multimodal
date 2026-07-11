"""
Generic fine-grained image classification dataset backed by one or more
HuggingFace dataset repos with a plain `image` / `label` (ClassLabel) schema.

Used for CUB-200-2011, FGVC Aircraft, Oxford-IIIT Pets, etc. — datasets that,
like Stanford Cars, are candidates for testing whether CLIP-similarity-based
1-shot retrieval actually helps ICL accuracy over 0-shot.

Supports two split modes:
- Class-level splits (default): disjoint classes across train/val/test.
- Image-level splits: all classes appear in every split, images are divided
  by a pre-computed JSON file (see scripts/create_image_split.py).
"""

import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

from PIL import Image
from datasets import concatenate_datasets, load_dataset, load_from_disk

from .base_dataset import BaseUtilityDataset, ClassificationExample


class FineGrainedHFDataset(BaseUtilityDataset):
    """
    Fine-grained classification dataset loaded from one or more HuggingFace
    dataset repos, merged into a single pool before splitting.

    Args:
        hf_repo_ids: HuggingFace dataset repo(s) to load and concatenate
            (each loaded with split="train" internally — most of these repos
            expose their own train/test as separate repos, e.g.
            Multimodal-Fatima/CUB_train + Multimodal-Fatima/CUB_test).
        cache_subdir: Subdirectory name under data_dir/hf_cache for the local
            on-disk cache (defaults to the dataset's own name).
    """

    def __init__(
        self,
        hf_repo_ids: List[str],
        split: str = 'train',
        data_dir: str = './data/fine_grained',
        cache_subdir: Optional[str] = None,
        class_split_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        image_split_path: Optional[str] = None,
    ):
        self.hf_repo_ids = hf_repo_ids
        self.cache_subdir = cache_subdir or Path(data_dir).name
        self.hf_dataset = None
        self.image_split_path = Path(image_split_path) if image_split_path else None
        self._image_split_hf_indices: Optional[Set[int]] = None

        super().__init__(
            split=split,
            data_dir=data_dir,
            class_split_seed=class_split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    # ------------------------------------------------------------------
    # BaseUtilityDataset hooks
    # ------------------------------------------------------------------

    def load_data(self):
        local_dataset_path = self.data_dir / "hf_cache" / f"{self.cache_subdir}_merged"

        if local_dataset_path.exists():
            print(f"Loading cached dataset from {local_dataset_path}...")
            self.hf_dataset = load_from_disk(str(local_dataset_path))
            print("✓ Loaded from cache")
        else:
            print(f"Downloading from HuggingFace: {self.hf_repo_ids} (this may take a few minutes)...")
            parts = []
            for repo_id in self.hf_repo_ids:
                repo_splits = load_dataset(repo_id, cache_dir=str(self.data_dir / "hf_cache"))
                # Each repo (e.g. CUB_train, CUB_test) exposes exactly one internal
                # split, under its own name ("train" or "test") -- concatenate
                # whatever split(s) each repo actually has rather than assuming "train".
                for hf_split in repo_splits.values():
                    parts.append(hf_split)
            self.hf_dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
            print(f"Saving merged dataset to {local_dataset_path}...")
            self.hf_dataset.save_to_disk(str(local_dataset_path))
            print("✓ Dataset cached locally")

        self.class_names = self._load_class_names()
        self.num_classes = len(self.class_names)

        if self.image_split_path is not None:
            if not self.image_split_path.exists():
                raise FileNotFoundError(
                    f"Image split file not found: {self.image_split_path}\n"
                    f"Generate it with: python scripts/create_image_split.py --dataset <name>"
                )
            with open(self.image_split_path) as f:
                split_data = json.load(f)
            if self.split not in split_data:
                raise ValueError(
                    f"Split '{self.split}' not found in {self.image_split_path}. "
                    f"Available: {list(split_data.keys())}"
                )
            self._image_split_hf_indices = set(split_data[self.split])
            print(f"Image split: {len(self._image_split_hf_indices)} images for '{self.split}'")

        self.examples = []
        for idx, item in enumerate(self.hf_dataset):
            label = item['label']
            example = ClassificationExample(
                index=len(self.examples),
                image_path=f"hf_dataset_idx_{idx}",
                label=label,
                label_name=self.class_names[label],
                split='',
                _hf_index=idx,
            )
            self.examples.append(example)

    def _filter_examples_by_split(self):
        """Override to use image-level filtering when image_split_path is set."""
        if self._image_split_hf_indices is not None:
            filtered = [ex for ex in self.examples if ex._hf_index in self._image_split_hf_indices]
            for i, ex in enumerate(filtered):
                ex.index = i
                ex.split = self.split
            self.examples = filtered
            print(f"Loaded {len(self.examples)} examples for '{self.split}' split (image-level)")
        else:
            super()._filter_examples_by_split()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_class_names(self):
        if self.hf_dataset is not None:
            if hasattr(self.hf_dataset, 'features') and 'label' in self.hf_dataset.features:
                feature = self.hf_dataset.features['label']
                if hasattr(feature, 'names'):
                    return feature.names
        raise ValueError("Could not determine class names from HF dataset features['label']")

    def __getitem__(self, idx: int) -> Tuple[ClassificationExample, Image.Image]:
        example = self.examples[idx]

        if self.hf_dataset is None or example._hf_index is None:
            raise ValueError("HuggingFace dataset not loaded")

        hf_item = self.hf_dataset[example._hf_index]
        image = hf_item['image']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return example, image
