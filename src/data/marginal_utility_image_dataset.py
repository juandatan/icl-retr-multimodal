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
from data.marginal_utility_dataset import QuerySplitConfig


def clip_transform(image_size: int = 224) -> transforms.Compose:
    """Standard CLIP image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


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
        normalize_utilities: bool = False,
        top_k: Optional[int] = None,
        query_split: Optional[QuerySplitConfig] = None,
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
            top_k: If set, retain only the top-k candidates per query by similarity rank.
            query_split: Train/val/test split fractions. Defaults to 90/10/0.
        """
        self.results_path = results_path
        self.base_dataset = base_dataset
        self.split = split
        self.seed = seed
        self.image_size = image_size
        self.normalize_utilities = normalize_utilities
        self.top_k = top_k
        self.query_split = query_split or QuerySplitConfig()

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
        qs = self.query_split
        split_desc = f"{qs.train_ratio:.0%}/{qs.val_ratio:.0%}/{qs.test_ratio:.0%}"
        print(f"Splitting data by query ({split_desc})...")
        train_results, val_results, test_results = self.split_by_query(
            all_results, train_ratio=qs.train_ratio, val_ratio=qs.val_ratio, seed=seed
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

        if self.top_k is not None:
            self.results = self._apply_top_k(self.results, self.top_k)

        query_indices = [self._get_attr(r, 'query_idx') for r in self.results]
        print(f"✓ {split.upper()} split: {len(self.results)} pairs from "
              f"{len(set(query_indices))} queries")

        self.transform = clip_transform(image_size)

    @staticmethod
    def _get_attr(obj, key):
        """Get attribute from object or dictionary."""
        if isinstance(obj, dict):
            return obj[key]
        else:
            return getattr(obj, key)

    @staticmethod
    def prepare_splits(
        all_results: List,
        query_split: QuerySplitConfig,
        top_k: Optional[int],
        seed: int,
    ) -> Tuple[List, List, List]:
        """Split results and apply top_k filtering. Returns (train, val, test)."""
        qs = query_split
        train_results, val_results, test_results = MarginalUtilityImageDataset.split_by_query(
            all_results, train_ratio=qs.train_ratio, val_ratio=qs.val_ratio, seed=seed
        )
        if top_k is not None:
            train_results = MarginalUtilityImageDataset._apply_top_k(train_results, top_k)
            val_results = MarginalUtilityImageDataset._apply_top_k(val_results, top_k)
            test_results = MarginalUtilityImageDataset._apply_top_k(test_results, top_k)
        return train_results, val_results, test_results

    @staticmethod
    def _apply_top_k(results: List, top_k: int) -> List:
        """Retain only the top-k candidates per query ranked by similarity score."""
        get_attr = MarginalUtilityImageDataset._get_attr
        query_groups: Dict[int, List] = defaultdict(list)
        for r in results:
            query_groups[get_attr(r, 'query_idx')].append(r)

        filtered = []
        for group in query_groups.values():
            sorted_group = sorted(group, key=lambda r: get_attr(r, 'similarity_score'), reverse=True)
            filtered.extend(sorted_group[:top_k])

        print(f"  top_k={top_k}: {len(filtered)} pairs retained from {len(query_groups)} queries "
              f"(was {len(results)})")
        return filtered

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
        train_ratio: float = 0.9,
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
            _, image = self.base_dataset[image_idx]
            return image.convert('RGB')
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


class CachedPatchFeatureDataset(Dataset):
    """
    Dataset that serves pre-extracted CLIP patch features.

    Extracts patch features for all images once at initialization, then serves
    them as tensors during training. This avoids repeated CLIP forward passes
    and HuggingFace dataset access during training, enabling safe multi-GPU use.
    """

    def __init__(
        self,
        results: List,
        split: str,
        base_dataset,
        patch_model,
        normalize_utilities: bool = False,
        utility_min: Optional[float] = None,
        utility_max: Optional[float] = None,
        cache_path: Optional[str] = None,
        extraction_batch_size: int = 64,
        device: str = 'cuda',
        feature_cache: Optional[Dict] = None,
    ):
        """
        Args:
            results: Pre-split list of result dicts/objects for this split.
            split: Split name (for logging only).
            base_dataset: Dataset providing images by index.
            patch_model: Model with extract_patch_features() method.
            normalize_utilities: Whether to normalize utilities to [0, 1].
            utility_min: Pre-computed min utility (required if normalize_utilities).
            utility_max: Pre-computed max utility (required if normalize_utilities).
            cache_path: Path to save/load cached features.
            extraction_batch_size: Batch size for feature extraction.
            device: Device for feature extraction.
            feature_cache: Pre-computed feature cache dict (skips extraction if provided).
        """
        self.split = split
        self.results = results
        self.normalize_utilities = normalize_utilities
        self.utility_min = utility_min
        self.utility_max = utility_max
        self.utility_range = (utility_max - utility_min) if utility_min is not None and utility_max is not None else None

        query_indices = [self._get_attr(r, 'query_idx') for r in self.results]
        print(f"✓ {split.upper()} split: {len(self.results)} pairs from "
              f"{len(set(query_indices))} queries")

        # Use provided cache or extract features
        if feature_cache is not None:
            self.feature_cache = feature_cache
        else:
            self._extract_and_cache_features(
                base_dataset, patch_model, cache_path, extraction_batch_size, device
            )

    def _extract_and_cache_features(self, base_dataset, patch_model, cache_path, batch_size, device):
        """Extract CLIP patch features for all unique images used in this split."""
        import os
        from tqdm import tqdm

        # Find all unique image indices
        all_indices = set()
        for r in self.results:
            all_indices.add(self._get_attr(r, 'query_idx'))
            all_indices.add(self._get_attr(r, 'example_idx'))
        unique_indices = sorted(all_indices)
        print(f"  Extracting patch features for {len(unique_indices)} unique images...")

        # Check for cache
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached features from {cache_path}")
            self.feature_cache = torch.load(cache_path, map_location='cpu')
            print(f"  ✓ Loaded {len(self.feature_cache)} cached features")
            return

        transform = clip_transform(224)

        # Extract features in batches
        patch_model.eval()
        self.feature_cache = {}

        with torch.no_grad():
            for batch_start in tqdm(range(0, len(unique_indices), batch_size),
                                     desc=f"  Extracting patches ({self.split})"):
                batch_indices = unique_indices[batch_start:batch_start + batch_size]
                images = []

                for idx in batch_indices:
                    try:
                        _, img = base_dataset[idx]
                        images.append(transform(img.convert('RGB')))
                    except Exception as e:
                        # Fallback grey image
                        images.append(torch.zeros(3, 224, 224))

                image_batch = torch.stack(images).to(device)
                features = patch_model.extract_patch_features(image_batch)
                features = features.cpu().half()  # Store as float16 to save memory

                for i, idx in enumerate(batch_indices):
                    self.feature_cache[idx] = features[i]

        print(f"  ✓ Cached {len(self.feature_cache)} patch feature tensors")

        # Save cache if path provided
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.feature_cache, cache_path)
            print(f"  ✓ Saved cache to {cache_path}")

    @staticmethod
    def _get_attr(obj, key):
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        result = self.results[idx]
        query_idx = self._get_attr(result, 'query_idx')
        example_idx = self._get_attr(result, 'example_idx')
        similarity_score = self._get_attr(result, 'similarity_score')
        marginal_utility = self._get_attr(result, 'marginal_utility')

        query_features = self.feature_cache[query_idx].float()
        example_features = self.feature_cache[example_idx].float()
        similarity = torch.tensor([similarity_score], dtype=torch.float32)

        if self.normalize_utilities and self.utility_range and self.utility_range > 0:
            marginal_utility = (marginal_utility - self.utility_min) / self.utility_range
        utility = torch.tensor([marginal_utility], dtype=torch.float32)

        return query_features, example_features, similarity, utility

    def compute_baseline_mse(self) -> float:
        utilities = np.array([self._get_attr(r, 'marginal_utility') for r in self.results])
        similarities = np.array([self._get_attr(r, 'similarity_score') for r in self.results])
        return float(np.mean((utilities - similarities) ** 2))

    def compute_baseline_spearman(self) -> float:
        from scipy.stats import spearmanr
        utilities = np.array([self._get_attr(r, 'marginal_utility') for r in self.results])
        similarities = np.array([self._get_attr(r, 'similarity_score') for r in self.results])
        corr, _ = spearmanr(similarities, utilities)
        return float(corr)

    def get_query_groups(self) -> Dict[int, List[int]]:
        """Return {query_idx: [result_indices]} for building pairs."""
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, r in enumerate(self.results):
            groups[self._get_attr(r, 'query_idx')].append(i)
        return dict(groups)


class PairwiseCachedPatchFeatureDataset(CachedPatchFeatureDataset):
    """
    Pairwise variant of CachedPatchFeatureDataset for ranking loss training.

    Returns (query, better_example, worse_example, similarity_better, similarity_worse)
    pairs sampled from the same query so MarginRankingLoss can directly compare them.
    """

    def __init__(self, *args, pairs_per_query: int = 10, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs_per_query = pairs_per_query
        self.rng = random.Random(seed)
        self.pairs = self._build_pairs()
        print(f"  ✓ Built {len(self.pairs)} pairwise examples ({pairs_per_query} pairs/query)")

    def _build_pairs(self) -> List[Tuple[int, int]]:
        groups = self.get_query_groups()
        pairs = []
        for result_indices in groups.values():
            if len(result_indices) < 2:
                continue
            for _ in range(self.pairs_per_query):
                idx1, idx2 = self.rng.sample(result_indices, 2)
                u1 = self._get_attr(self.results[idx1], 'marginal_utility')
                u2 = self._get_attr(self.results[idx2], 'marginal_utility')
                if u1 > u2:
                    pairs.append((idx1, idx2))
                elif u2 > u1:
                    pairs.append((idx2, idx1))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        better_idx, worse_idx = self.pairs[idx]

        def _get_tensors(result_idx):
            r = self.results[result_idx]
            q_feat = self.feature_cache[self._get_attr(r, 'query_idx')].float()
            e_feat = self.feature_cache[self._get_attr(r, 'example_idx')].float()
            sim = torch.tensor([self._get_attr(r, 'similarity_score')], dtype=torch.float32)
            return q_feat, e_feat, sim

        q_feat, better_feat, better_sim = _get_tensors(better_idx)
        _, worse_feat, worse_sim = _get_tensors(worse_idx)
        return q_feat, better_feat, better_sim, worse_feat, worse_sim


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
