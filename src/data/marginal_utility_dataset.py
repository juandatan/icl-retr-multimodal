"""
PyTorch Dataset for training reranker on marginal utility predictions.

This dataset loads pre-computed marginal utilities and CLIP embeddings,
creating training pairs of (query_embedding, example_embedding, similarity, utility).
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
from torch.utils.data import Dataset

# Import MarginalUtilityResult for pickle deserialization
from .dataclasses import MarginalUtilityResult


@dataclass
class InteractionFeaturesConfig:
    """Configuration for which interaction features to compute."""
    use_product: bool = False  # Element-wise product of embeddings
    use_difference: bool = False  # Element-wise difference of embeddings
    use_l2_distance: bool = False  # L2 distance between embeddings

    @property
    def num_features(self) -> int:
        """Return total number of additional features."""
        count = 0
        if self.use_product:
            count += 512  # embedding_dim
        if self.use_difference:
            count += 512  # embedding_dim
        if self.use_l2_distance:
            count += 1  # scalar
        return count

    @property
    def enabled(self) -> bool:
        """Return True if any interaction features are enabled."""
        return self.use_product or self.use_difference or self.use_l2_distance

    def __repr__(self) -> str:
        """Pretty string representation."""
        features = []
        if self.use_product:
            features.append("product")
        if self.use_difference:
            features.append("difference")
        if self.use_l2_distance:
            features.append("l2_dist")
        return f"InteractionFeaturesConfig({', '.join(features) if features else 'none'})"


class MarginalUtilityDataset(Dataset):
    """
    Dataset for training reranker to predict marginal utilities.

    Loads pre-computed MarginalUtilityResult objects and CLIP embeddings,
    returning (query_emb, example_emb, similarity, utility) tuples.

    Key design: Query-based splitting to test generalization to new queries.
    """

    def __init__(
        self,
        results_path: str,
        embeddings_path: str,
        split: str = 'train',
        seed: int = 42,
        interaction_features: Optional[InteractionFeaturesConfig] = None
    ):
        """
        Initialize dataset.

        Args:
            results_path: Path to marginal_utilities_train.pkl file
            embeddings_path: Path to clip_embeddings_train.pkl file
            split: One of 'train', 'val', 'test'
            seed: Random seed for reproducible splitting
            interaction_features: Configuration for interaction features to compute
        """
        self.results_path = results_path
        self.embeddings_path = embeddings_path
        self.split = split
        self.seed = seed
        self.interaction_features = interaction_features or InteractionFeaturesConfig()

        # Load data
        print(f"Loading marginal utility results from {results_path}...")
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        all_results = data['results']
        print(f"✓ Loaded {len(all_results)} result pairs")

        # Load CLIP embeddings
        print(f"Loading CLIP embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            emb_data = pickle.load(f)
        self.embeddings = emb_data['embeddings']  # shape: (num_examples, 512)
        print(f"✓ Loaded embeddings: shape {self.embeddings.shape}")

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

    @staticmethod
    def _get_attr(obj, key):
        """Get attribute from object or dictionary."""
        if isinstance(obj, dict):
            return obj[key]
        else:
            return getattr(obj, key)

    @staticmethod
    def split_by_query(
        results: List,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List, List, List]:
        """
        Split results by query (not by pairs) for proper generalization testing.

        Args:
            results: List of MarginalUtilityResult objects or dictionaries
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            seed: Random seed

        Returns:
            (train_results, val_results, test_results)
        """
        # Group by query (handle both dataclass and dict)
        query_groups = defaultdict(list)
        for result in results:
            query_idx = result['query_idx'] if isinstance(result, dict) else result.query_idx
            query_groups[query_idx].append(result)

        # Split query IDs
        query_ids = sorted(query_groups.keys())
        random.Random(seed).shuffle(query_ids)

        n_queries = len(query_ids)
        n_train = int(train_ratio * n_queries)
        n_val = int(val_ratio * n_queries)

        train_queries = query_ids[:n_train]
        val_queries = query_ids[n_train:n_train + n_val]
        test_queries = query_ids[n_train + n_val:]

        # Collect results for each split
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

    def __len__(self) -> int:
        """Return number of training pairs."""
        return len(self.results)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a training example.

        Args:
            idx: Index of result pair

        Returns:
            Base: (query_embedding, example_embedding, similarity, utility)
            With interaction features, additional tensors are appended in order:
                - product (if use_product=True)
                - difference (if use_difference=True)
                - l2_distance (if use_l2_distance=True)
            Then utility is always last.

            All tensors have dtype=float32
        """
        result = self.results[idx]

        # Get attributes (handle both dataclass and dict)
        query_idx = self._get_attr(result, 'query_idx')
        example_idx = self._get_attr(result, 'example_idx')
        similarity_score = self._get_attr(result, 'similarity_score')
        marginal_utility = self._get_attr(result, 'marginal_utility')

        # Get embeddings
        query_emb = self.embeddings[query_idx]  # shape: (512,)
        example_emb = self.embeddings[example_idx]  # shape: (512,)

        # Convert to tensors
        query_emb_tensor = torch.from_numpy(query_emb).float()
        example_emb_tensor = torch.from_numpy(example_emb).float()
        similarity = torch.tensor([similarity_score], dtype=torch.float32)
        utility = torch.tensor([marginal_utility], dtype=torch.float32)

        # Build output tuple dynamically based on enabled features
        output = [query_emb_tensor, example_emb_tensor, similarity]

        if self.interaction_features.enabled:
            # Compute interaction features as needed
            if self.interaction_features.use_product:
                # Element-wise product (captures feature co-activation)
                product = query_emb_tensor * example_emb_tensor  # (512,)
                output.append(product)

            if self.interaction_features.use_difference:
                # Element-wise difference (captures feature differences)
                difference = query_emb_tensor - example_emb_tensor  # (512,)
                output.append(difference)

            if self.interaction_features.use_l2_distance:
                # L2 distance (single scalar)
                l2_distance = torch.norm(query_emb_tensor - example_emb_tensor, p=2).unsqueeze(0)  # (1,)
                output.append(l2_distance)

        # Utility is always last
        output.append(utility)

        return tuple(output)

    def get_query_groups(self) -> Dict[int, List[int]]:
        """
        Group result indices by query_idx.

        Useful for evaluation: compute metrics per-query then average.

        Returns:
            Dict mapping query_idx -> list of result indices in this dataset
        """
        query_groups = defaultdict(list)
        for idx, result in enumerate(self.results):
            query_idx = self._get_attr(result, 'query_idx')
            query_groups[query_idx].append(idx)
        return dict(query_groups)

    def compute_baseline_mse(self) -> float:
        """
        Compute baseline MSE using CLIP similarity as predictor.

        Naive baseline: predict utility = similarity (scaled to utility range).

        Returns:
            Baseline MSE
        """
        utilities = np.array([self._get_attr(r, 'marginal_utility') for r in self.results])
        similarities = np.array([self._get_attr(r, 'similarity_score') for r in self.results])

        # Naive prediction: utility = 2 * similarity - 1 (map [0,1] to [-1,1])
        # Or simpler: directly use similarity as prediction
        predictions = similarities

        mse = np.mean((utilities - predictions) ** 2)
        return float(mse)


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path

    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    results_path = "outputs/marginal_utilities/marginal_utility_stanford_cars/marginal_utilities_train.pkl"
    embeddings_path = "data/stanford_cars/clip_embeddings_train.pkl"

    print("Testing dataset loading...")
    print("=" * 70)

    # Test train split
    train_ds = MarginalUtilityDataset(results_path, embeddings_path, split='train')
    print(f"\nTrain dataset size: {len(train_ds)}")

    # Test sample
    query_emb, example_emb, sim, util = train_ds[0]
    print(f"\nSample:")
    print(f"  Query embedding shape: {query_emb.shape}")
    print(f"  Example embedding shape: {example_emb.shape}")
    print(f"  Similarity: {sim.item():.4f}")
    print(f"  Utility: {util.item():.4f}")

    # Test baseline
    baseline_mse = train_ds.compute_baseline_mse()
    print(f"\nBaseline MSE (using CLIP similarity): {baseline_mse:.4f}")

    # Test val split
    val_ds = MarginalUtilityDataset(results_path, embeddings_path, split='val')
    print(f"\nValidation dataset size: {len(val_ds)}")

    print("\n✓ Dataset test passed!")
