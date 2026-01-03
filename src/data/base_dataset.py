"""
Base dataset class for ICL utility learning.

Provides common functionality for:
- CLIP embedding computation and caching
- Semantic similarity-based retrieval
- Class-based train/val/test splits
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


@dataclass
class ClassificationExample:
    """A single image classification example."""
    index: int
    image_path: str
    label: int
    label_name: str
    split: str  # 'train', 'val', or 'test'
    _hf_index: Optional[int] = None  # For HuggingFace dataset index

    def load_image(self) -> Image.Image:
        """Load the image from disk. May be overridden by subclasses."""
        return Image.open(self.image_path).convert('RGB')


class BaseUtilityDataset(Dataset, ABC):
    """
    Abstract base class for utility learning datasets.

    Provides common functionality:
    - CLIP embedding computation and caching
    - Semantic similarity retrieval
    - Disjoint class splits for train/val/test

    Subclasses must implement:
    - load_data(): Load raw data and return examples
    - __getitem__(): Return (example, image) tuple
    """

    def __init__(
        self,
        split: str = 'train',
        data_dir: str = './data',
        class_split_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        """
        Args:
            split: Dataset split ('train', 'val', or 'test')
            data_dir: Root directory for dataset storage
            class_split_seed: Random seed for reproducible class splits
            train_ratio: Proportion of classes for training
            val_ratio: Proportion of classes for validation
        """
        self.split = split
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.class_split_seed = class_split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Will be populated by subclass
        self.examples: List[ClassificationExample] = []
        self.num_classes: int = 0
        self.class_names: List[str] = []
        self.train_classes: np.ndarray = np.array([])
        self.val_classes: np.ndarray = np.array([])
        self.test_classes: np.ndarray = np.array([])

        # CLIP embeddings (computed on-demand)
        self.clip_embeddings: Optional[np.ndarray] = None
        self.clip_model_name: Optional[str] = None

        # Subclass calls load_data() to populate examples
        self.load_data()

        # Create class splits after loading
        self._create_class_splits()

        # Filter examples by split
        self._filter_examples_by_split()

    @abstractmethod
    def load_data(self):
        """
        Load dataset and populate:
        - self.examples: List of all examples (before split filtering)
        - self.num_classes: Total number of classes
        - self.class_names: List of class names

        This method should load the raw data but NOT filter by split.
        """
        pass

    def _create_class_splits(self):
        """Create disjoint class splits for train/val/test."""
        np.random.seed(self.class_split_seed)

        # Shuffle all class indices
        all_classes = np.arange(self.num_classes)
        np.random.shuffle(all_classes)

        # Calculate split sizes
        n_train = int(self.num_classes * self.train_ratio)
        n_val = int(self.num_classes * self.val_ratio)

        # Split classes
        self.train_classes = all_classes[:n_train]
        self.val_classes = all_classes[n_train:n_train + n_val]
        self.test_classes = all_classes[n_train + n_val:]

        # Verify disjoint
        assert len(set(self.train_classes) & set(self.val_classes)) == 0
        assert len(set(self.train_classes) & set(self.test_classes)) == 0
        assert len(set(self.val_classes) & set(self.test_classes)) == 0

        print(f"Class splits: {len(self.train_classes)} train, "
              f"{len(self.val_classes)} val, {len(self.test_classes)} test")

    def _filter_examples_by_split(self):
        """Filter examples to only include those from the current split's classes."""
        # Get relevant classes for this split
        if self.split == 'train':
            valid_classes = set(self.train_classes)
        elif self.split == 'val':
            valid_classes = set(self.val_classes)
        else:  # test
            valid_classes = set(self.test_classes)

        # Filter examples
        filtered_examples = []
        for example in self.examples:
            if example.label in valid_classes:
                # Re-index after filtering
                example.index = len(filtered_examples)
                example.split = self.split
                filtered_examples.append(example)

        self.examples = filtered_examples
        print(f"Loaded {len(self.examples)} examples for {self.split} split")

    def __len__(self) -> int:
        return len(self.examples)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[ClassificationExample, Image.Image]:
        """
        Get example and image.

        Returns:
            (ClassificationExample, PIL.Image)
        """
        pass

    def load_clip_embeddings(self, cache_path: Optional[str] = None) -> bool:
        """
        Load cached CLIP embeddings if available.

        Args:
            cache_path: Path to cached embeddings (default: data_dir/clip_embeddings_{split}.pkl)

        Returns:
            True if embeddings were loaded, False otherwise
        """
        if cache_path is None:
            cache_path = self.data_dir / f'clip_embeddings_{self.split}.pkl'

        if Path(cache_path).exists():
            print(f"Loading cached CLIP embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.clip_embeddings = cache_data['embeddings']
                self.clip_model_name = cache_data['model_name']
                return True
        return False

    def build_clip_embeddings(
        self,
        clip_model,
        clip_preprocess,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build CLIP embeddings for all images in this split.

        Args:
            clip_model: CLIP model
            clip_preprocess: CLIP preprocessing function
            batch_size: Batch size for embedding computation
            device: Device to run on
            cache_path: Path to save/load cached embeddings

        Returns:
            Array of shape (num_examples, embedding_dim)
        """
        # Check for cached embeddings
        if cache_path is None:
            cache_path = self.data_dir / f'clip_embeddings_{self.split}.pkl'

        if Path(cache_path).exists():
            print(f"Loading cached CLIP embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.clip_embeddings = cache_data['embeddings']
                self.clip_model_name = cache_data['model_name']
                return self.clip_embeddings

        print(f"Computing CLIP embeddings for {len(self)} images...")

        clip_model = clip_model.to(device)
        clip_model.eval()

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self), batch_size)):
                # Get batch of images
                batch_images = []
                for j in range(i, min(i + batch_size, len(self))):
                    _, image = self[j]
                    batch_images.append(clip_preprocess(image))

                # Stack and move to device
                batch_tensor = torch.stack(batch_images).to(device)

                # Compute embeddings
                batch_embeddings = clip_model.encode_image(batch_tensor)

                # Normalize (for cosine similarity)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

                embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate all embeddings
        self.clip_embeddings = np.vstack(embeddings)
        self.clip_model_name = 'ViT-B/32'  # TODO: Make configurable

        # Cache embeddings
        print(f"Caching embeddings to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.clip_embeddings,
                'model_name': self.clip_model_name,
            }, f)

        return self.clip_embeddings

    def get_semantic_similarity(
        self,
        query_idx: int,
        candidate_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute semantic similarity between query and candidates using CLIP.

        Args:
            query_idx: Index of query example
            candidate_indices: Indices of candidate examples (None = all)

        Returns:
            Array of similarity scores (cosine similarity)
        """
        if self.clip_embeddings is None:
            raise ValueError("CLIP embeddings not computed. Call build_clip_embeddings first.")

        query_embedding = self.clip_embeddings[query_idx]

        if candidate_indices is None:
            candidate_embeddings = self.clip_embeddings
        else:
            candidate_embeddings = self.clip_embeddings[candidate_indices]

        # Cosine similarity (embeddings are already normalized)
        similarities = candidate_embeddings @ query_embedding

        return similarities

    def get_top_k_similar(
        self,
        query_idx: int,
        k: int = 100,
        exclude_query: bool = True,
        exclude_same_class: bool = False,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Get top-k most similar examples to query based on CLIP embeddings.

        Args:
            query_idx: Index of query example
            k: Number of candidates to return
            exclude_query: Whether to exclude the query itself
            exclude_same_class: Whether to exclude examples from same class

        Returns:
            (candidate_indices, similarity_scores)
        """
        if self.clip_embeddings is None:
            raise ValueError("CLIP embeddings not computed. Call build_clip_embeddings first.")

        # Get all similarities
        similarities = self.get_semantic_similarity(query_idx)

        # Create mask for valid candidates
        valid_mask = np.ones(len(self), dtype=bool)

        if exclude_query:
            valid_mask[query_idx] = False

        if exclude_same_class:
            query_label = self.examples[query_idx].label
            for idx, example in enumerate(self.examples):
                if example.label == query_label:
                    valid_mask[idx] = False

        # Get valid indices and similarities
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_mask]

        # Get top-k
        if len(valid_indices) < k:
            print(f"Warning: Only {len(valid_indices)} valid candidates, requested {k}")
            k = len(valid_indices)

        top_k_in_valid = np.argsort(valid_similarities)[-k:][::-1]
        top_k_indices = valid_indices[top_k_in_valid]
        top_k_scores = valid_similarities[top_k_in_valid]

        return top_k_indices.tolist(), top_k_scores

    def save_split_info(self, save_path: Optional[str] = None):
        """Save class split information for reproducibility."""
        if save_path is None:
            save_path = self.data_dir / 'class_splits.json'

        import json

        split_info = {
            'num_classes': self.num_classes,
            'train_classes': self.train_classes.tolist(),
            'val_classes': self.val_classes.tolist(),
            'test_classes': self.test_classes.tolist(),
            'class_split_seed': self.class_split_seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
        }

        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"Saved split info to {save_path}")
