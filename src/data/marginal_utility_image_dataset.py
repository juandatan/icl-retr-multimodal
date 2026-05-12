"""
Image-based dataset for training reranker on marginal utility predictions.

Unlike marginal_utility_dataset.py which loads pre-computed CLIP embeddings,
this dataset loads raw images for patch-level cross-attention models.
"""

import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Import MarginalUtilityResult for pickle deserialization
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataclasses import MarginalUtilityResult


class MarginalUtilityImageDataset(Dataset):
    """
    Image-based dataset for training reranker to predict marginal utilities.

    Loads raw images instead of pre-computed embeddings, enabling patch-level
    cross-attention models to operate on spatial features.
    """

    def __init__(
        self,
        results_path: str,
        base_dataset: Optional['BaseUtilityDataset'] = None,
        split: str = 'train',
        seed: int = 42,
        image_size: int = 224,
        normalize_utilities: bool = False
    ):
        """
        Initialize dataset.

        Args:
            results_path: Path to marginal_utilities_train.pkl file
            base_dataset: BaseUtilityDataset object (e.g., StanfordCarsDataset) that provides images
            split: One of 'train', 'val', 'test'
            seed: Random seed for reproducible splitting
            image_size: Size to resize images to (default: 224 for CLIP)
            normalize_utilities: If True, normalize utilities to [0, 1] range
        """
        self.results_path = results_path
        self.base_dataset = base_dataset
        self.split = split
        self.seed = seed
        self.image_size = image_size
        self.normalize_utilities = normalize_utilities

        # Load marginal utility data
        print(f"Loading marginal utility results from {results_path}...")
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        all_results = data['results']
        print(f"✓ Loaded {len(all_results)} result pairs")

        # Check if we have a base dataset to load images from
        if base_dataset is None:
            raise ValueError(
                "base_dataset is required. Pass a BaseUtilityDataset object "
                "(e.g., StanfordCarsDataset) that provides images by index."
            )

        # Compute normalization statistics from ALL data (before splitting)
        if self.normalize_utilities:
            all_utilities = np.array([self._get_attr(r, 'marginal_utility') for r in all_results])
            self.utility_min = float(np.min(all_utilities))
            self.utility_max = float(np.max(all_utilities))
            self.utility_range = self.utility_max - self.utility_min
            print(f"✓ Utility normalization enabled: min={self.utility_min:.4f}, max={self.utility_max:.4f}")
        else:
            self.utility_min = None
            self.utility_max = None
            self.utility_range = None

        # Split by query
        print(f"Splitting data by query (80/10/10)...")
        train_results, val_results, test_results = self.split_by_query(
            all_results, seed=seed
        )

        # Select split
        if split == 'train':
            self.results = train_results
        elif split == 'val':
            self.results = val_results
        elif split == 'test':
            self.results = test_results
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        query_indices = [self._get_attr(r, 'query_idx') for r in self.results]
        print(f"✓ {split.upper()} split: {len(self.results)} pairs from "
              f"{len(set(query_indices))} queries")

        # Setup image transforms (CLIP preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP mean
                std=[0.26862954, 0.26130258, 0.27577711]   # CLIP std
            )
        ])

    @staticmethod
    def _get_attr(obj, key):
        """Get attribute from object or dictionary."""
        if isinstance(obj, dict):
            return obj[key]
        else:
            return getattr(obj, key)

    def _normalize_utility(self, utility: float) -> float:
        """Normalize utility to [0, 1] range using min-max scaling."""
        if not self.normalize_utilities:
            return utility

        if self.utility_range == 0:
            return 0.5

        return (utility - self.utility_min) / self.utility_range

    @staticmethod
    def split_by_query(
        results: List,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List, List, List]:
        """Split results by query (not by pairs) for proper generalization testing."""
        query_groups = defaultdict(list)
        for result in results:
            query_idx = result['query_idx'] if isinstance(result, dict) else result.query_idx
            query_groups[query_idx].append(result)

        query_ids = sorted(query_groups.keys())
        random.Random(seed).shuffle(query_ids)

        n_queries = len(query_ids)
        n_train = int(train_ratio * n_queries)
        n_val = int(val_ratio * n_queries)

        train_queries = query_ids[:n_train]
        val_queries = query_ids[n_train:n_train + n_val]
        test_queries = query_ids[n_train + n_val:]

        train_results = []
        for qid in train_queries:
            train_results.extend(query_groups[qid])

        val_results = []
        for qid in val_queries:
            val_results.extend(query_groups[qid])

        test_results = []
        for qid in test_queries:
            test_results.extend(query_groups[qid])

        print(f"  Train: {len(train_results)} pairs from {len(train_queries)} queries")
        print(f"  Val:   {len(val_results)} pairs from {len(val_queries)} queries")
        print(f"  Test:  {len(test_results)} pairs from {len(test_queries)} queries")

        return train_results, val_results, test_results

    def _load_image(self, image_idx: int) -> Image.Image:
        """Load and return PIL Image from base dataset."""
        try:
            # Get example from base dataset
            example = self.base_dataset.examples[image_idx]
            return example.image
        except Exception as e:
            print(f"Error loading image at index {image_idx}: {e}")
            # Return blank image as fallback
            return Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))

    def __len__(self) -> int:
        """Return number of training pairs."""
        return len(self.results)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Args:
            idx: Index of result pair

        Returns:
            (query_image, example_image, similarity, utility)
            - query_image: Preprocessed query image, shape (3, H, W)
            - example_image: Preprocessed example image, shape (3, H, W)
            - similarity: CLIP similarity score, shape (1,)
            - utility: Marginal utility score, shape (1,)
        """
        result = self.results[idx]

        # Get attributes
        query_idx = self._get_attr(result, 'query_idx')
        example_idx = self._get_attr(result, 'example_idx')
        similarity_score = self._get_attr(result, 'similarity_score')
        marginal_utility = self._get_attr(result, 'marginal_utility')

        # Load and preprocess images
        query_image = self._load_image(query_idx)
        example_image = self._load_image(example_idx)

        query_tensor = self.transform(query_image)
        example_tensor = self.transform(example_image)

        # Normalize utility if enabled
        normalized_utility = self._normalize_utility(marginal_utility)

        # Convert to tensors
        similarity = torch.tensor([similarity_score], dtype=torch.float32)
        utility = torch.tensor([normalized_utility], dtype=torch.float32)

        return query_tensor, example_tensor, similarity, utility

    def get_query_groups(self) -> Dict[int, List[int]]:
        """Group result indices by query_idx."""
        query_groups = defaultdict(list)
        for idx, result in enumerate(self.results):
            query_idx = self._get_attr(result, 'query_idx')
            query_groups[query_idx].append(idx)
        return dict(query_groups)

    def compute_baseline_mse(self) -> float:
        """Compute baseline MSE using CLIP similarity as predictor."""
        utilities = np.array([self._get_attr(r, 'marginal_utility') for r in self.results])
        similarities = np.array([self._get_attr(r, 'similarity_score') for r in self.results])

        predictions = similarities
        mse = np.mean((utilities - predictions) ** 2)
        return float(mse)

    def compute_baseline_spearman(self) -> float:
        """Compute baseline Spearman correlation using CLIP similarity as predictor."""
        from scipy.stats import spearmanr

        utilities = np.array([self._get_attr(r, 'marginal_utility') for r in self.results])
        similarities = np.array([self._get_attr(r, 'similarity_score') for r in self.results])

        spearman_corr, _ = spearmanr(similarities, utilities)
        return float(spearman_corr)


if __name__ == "__main__":
    # Quick test
    from stanford_cars import StanfordCarsDataset

    results_path = "outputs/marginal_utilities/marginal_utility_stanford_cars/marginal_utilities_train.pkl"

    print("Testing image dataset loading...")
    print("=" * 70)

    # Load base dataset (contains images)
    print("\nLoading Stanford Cars dataset...")
    base_dataset = StanfordCarsDataset(split='train')

    # Test train split
    train_ds = MarginalUtilityImageDataset(
        results_path=results_path,
        base_dataset=base_dataset,
        split='train'
    )
    print(f"\nTrain dataset size: {len(train_ds)}")

    # Test sample
    query_img, example_img, sim, util = train_ds[0]
    print(f"\nSample:")
    print(f"  Query image shape: {query_img.shape}")
    print(f"  Example image shape: {example_img.shape}")
    print(f"  Similarity: {sim.item():.4f}")
    print(f"  Utility: {util.item():.4f}")

    print("\n✓ Dataset test passed!")
