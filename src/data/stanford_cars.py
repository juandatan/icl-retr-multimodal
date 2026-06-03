"""
Stanford Cars Dataset for ICL Utility Learning

Inherits from BaseUtilityDataset and implements dataset-specific loading logic.

Supports two split modes:
- Class-level splits (default): disjoint classes across train/val/test, used for
  cross-class generalisation experiments (Mini-ImageNet style).
- Image-level splits: all 196 classes appear in every split, images are divided
  by a pre-computed JSON file. Used for fine-grained within-distribution experiments
  (the main Stanford Cars evaluation).
"""

import json
from pathlib import Path
from typing import Any, Optional, Set, Tuple

from PIL import Image
from datasets import load_dataset, load_from_disk

from .base_dataset import BaseUtilityDataset, ClassificationExample


class StanfordCarsDataset(BaseUtilityDataset):
    """
    Stanford Cars dataset for ICL utility learning.

    Supports:
    - Disjoint class splits for train/val/test
    - Image-level splits (all classes present in every split)
    - CLIP embedding pre-computation
    - Semantic similarity-based candidate retrieval
    """

    def __init__(
        self,
        split: str = 'train',
        data_dir: str = './data/stanford_cars',
        class_split_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        build_embeddings: bool = False,
        clip_model=None,
        clip_preprocess=None,
        embedding_batch_size: int = 32,
        device: str = 'cpu',
        image_split_path: Optional[str] = None,
    ):
        """
        Args:
            split: One of 'train', 'val', 'test'
            data_dir: Directory to store dataset
            class_split_seed: Random seed for reproducible class splits
            train_ratio: Proportion of classes for training (class-split mode only)
            val_ratio: Proportion of classes for validation (class-split mode only)
            build_embeddings: Whether to build CLIP embeddings in constructor
            clip_model: CLIP model (required if build_embeddings=True)
            clip_preprocess: CLIP preprocessing (required if build_embeddings=True)
            embedding_batch_size: Batch size for embedding computation
            device: Device for embedding computation
            image_split_path: Path to a JSON file produced by
                create_stanford_cars_image_split.py.  When provided, the dataset
                uses image-level splits (all 196 classes in every split) instead
                of the default class-level splits.
        """
        self.hf_dataset: Any = None
        self.image_split_path = Path(image_split_path) if image_split_path else None
        # Populated before _filter_examples_by_split is called (see load_data)
        self._image_split_hf_indices: Optional[Set[int]] = None

        super().__init__(
            split=split,
            data_dir=data_dir,
            class_split_seed=class_split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        if build_embeddings:
            if clip_model is None or clip_preprocess is None:
                raise ValueError("clip_model and clip_preprocess required when build_embeddings=True")

            self.build_clip_embeddings(
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                batch_size=embedding_batch_size,
                device=device,
            )

    # ------------------------------------------------------------------
    # BaseUtilityDataset hooks
    # ------------------------------------------------------------------

    def load_data(self):
        """
        Load Stanford Cars dataset from HuggingFace.

        Populates self.hf_dataset, self.examples, self.num_classes,
        self.class_names.  When image_split_path is set, also loads the
        set of HF indices for the requested split so that
        _filter_examples_by_split (called by the parent) can delegate to
        _filter_examples_by_image_split instead.
        """
        print(f"Loading Stanford Cars dataset...")

        local_dataset_path = self.data_dir / "hf_cache" / "stanford_cars_train"

        if local_dataset_path.exists():
            print(f"Loading cached dataset from {local_dataset_path}...")
            self.hf_dataset = load_from_disk(str(local_dataset_path))
            print("✓ Loaded from cache")
        else:
            print("Downloading from HuggingFace (this may take a few minutes)...")
            self.hf_dataset = load_dataset(
                "tanganke/stanford_cars",
                split="train",
                cache_dir=str(self.data_dir / "hf_cache")
            )
            print(f"Saving dataset to {local_dataset_path}...")
            self.hf_dataset.save_to_disk(str(local_dataset_path))
            print("✓ Dataset cached locally")

        self.num_classes = 196
        self.class_names = self._load_class_names()

        # Load image-split indices before the parent calls _filter_examples_by_split
        if self.image_split_path is not None:
            if not self.image_split_path.exists():
                raise FileNotFoundError(
                    f"Image split file not found: {self.image_split_path}\n"
                    f"Generate it with: python scripts/create_stanford_cars_image_split.py"
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
        """Load human-readable class names from HF dataset."""
        if self.hf_dataset is not None:
            if hasattr(self.hf_dataset, 'features') and 'label' in self.hf_dataset.features:
                feature = self.hf_dataset.features['label']
                if hasattr(feature, 'names'):
                    return feature.names
        return [f"class_{i}" for i in range(self.num_classes)]

    def __getitem__(self, idx: int) -> Tuple[ClassificationExample, Image.Image]:
        example = self.examples[idx]

        if self.hf_dataset is None or example._hf_index is None:
            raise ValueError("HuggingFace dataset not loaded")

        hf_item = self.hf_dataset[example._hf_index]
        image = hf_item['image']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return example, image
