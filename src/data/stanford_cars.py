"""
Stanford Cars Dataset for ICL Utility Learning

Inherits from BaseUtilityDataset and implements dataset-specific loading logic.
"""

from typing import Any, Optional, Tuple

from PIL import Image
from datasets import load_dataset, load_from_disk

from .base_dataset import BaseUtilityDataset, ClassificationExample


class StanfordCarsDataset(BaseUtilityDataset):
    """
    Stanford Cars dataset for ICL utility learning.

    Supports:
    - Disjoint class splits for train/val/test
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
    ):
        """
        Args:
            split: One of 'train', 'val', 'test'
            data_dir: Directory to store dataset
            class_split_seed: Random seed for reproducible class splits
            train_ratio: Proportion of classes for training
            val_ratio: Proportion of classes for validation
            build_embeddings: Whether to build CLIP embeddings in constructor
            clip_model: CLIP model (required if build_embeddings=True)
            clip_preprocess: CLIP preprocessing (required if build_embeddings=True)
            embedding_batch_size: Batch size for embedding computation
            device: Device for embedding computation
        """
        # Store HF dataset reference - using Any to avoid type errors with dynamic HF API
        from typing import Any
        self.hf_dataset: Any = None

        # Call parent init (which calls load_data())
        super().__init__(
            split=split,
            data_dir=data_dir,
            class_split_seed=class_split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        # Optionally build embeddings in constructor
        if build_embeddings:
            if clip_model is None or clip_preprocess is None:
                raise ValueError("clip_model and clip_preprocess required when build_embeddings=True")

            self.build_clip_embeddings(
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                batch_size=embedding_batch_size,
                device=device,
            )

    def load_data(self):
        """
        Load Stanford Cars dataset from HuggingFace.

        Populates:
        - self.hf_dataset: HuggingFace dataset
        - self.examples: List of all examples (before split filtering)
        - self.num_classes: 196 classes
        - self.class_names: List of class names
        """
        print(f"Loading Stanford Cars dataset...")

        # Check if dataset is already saved locally
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

            # Save to disk for future use
            print(f"Saving dataset to {local_dataset_path}...")
            self.hf_dataset.save_to_disk(str(local_dataset_path))
            print("✓ Dataset cached locally")

        # Set class information
        self.num_classes = 196
        self.class_names = self._load_class_names()

        # Create examples from HF dataset (before split filtering)
        self.examples = []
        for idx, item in enumerate(self.hf_dataset):
            label = item['label']

            example = ClassificationExample(
                index=len(self.examples),
                image_path=f"hf_dataset_idx_{idx}",  # Placeholder
                label=label,
                label_name=self.class_names[label],
                split='',  # Will be set by parent class
                _hf_index=idx,  # Store original HF index for later retrieval
            )

            self.examples.append(example)

    def _load_class_names(self):
        """Load human-readable class names from HF dataset."""
        # Try to get class names from HuggingFace dataset features
        if self.hf_dataset is not None:
            if hasattr(self.hf_dataset, 'features') and 'label' in self.hf_dataset.features:
                feature = self.hf_dataset.features['label']
                if hasattr(feature, 'names'):
                    return feature.names

        # Fallback: create generic class names
        return [f"class_{i}" for i in range(self.num_classes)]

    def __getitem__(self, idx: int) -> Tuple[ClassificationExample, Image.Image]:
        """
        Get example and image.

        Returns:
            (ClassificationExample, PIL.Image)
        """
        example = self.examples[idx]

        # Load image from HuggingFace dataset using stored index
        if self.hf_dataset is None or example._hf_index is None:
            raise ValueError("HuggingFace dataset not loaded")

        hf_item = self.hf_dataset[example._hf_index]
        image = hf_item['image']

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return example, image
