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

    Data volume ablation: Use `top_k` to simulate training on n=10/20/etc from a
    larger collected dataset (e.g. n=50), without re-running LLaVA inference.
    Filtering is by per-query similarity rank, so the top-k most similar candidates
    are retained for each query independently.

    Contrastiveness: A query is "contrastive" if its candidate set contains both
    helpful (utility > 0) and harmful (utility <= 0) examples. Use `contrastive_mode`
    to guarantee contrastiveness within the top-k by swapping in the nearest
    opposite-sign candidate from outside the top-k when needed.
    """

    def __init__(
        self,
        results_path: str,
        embeddings_path: str,
        split: str = 'train',
        seed: int = 42,
        interaction_features: Optional[InteractionFeaturesConfig] = None,
        normalize_utilities: bool = False,
        top_k: Optional[int] = None,
        contrastive_mode: str = 'none',
    ):
        """
        Initialize dataset.

        Args:
            results_path: Path to marginal_utilities_train.pkl file
            embeddings_path: Path to clip_embeddings_train.pkl file
            split: One of 'train', 'val', 'test'
            seed: Random seed for reproducible splitting
            interaction_features: Configuration for interaction features to compute
            normalize_utilities: If True, normalize utilities to [0, 1] range using min-max scaling
            top_k: If set, retain only the top-k candidates per query by similarity rank.
                Used for data volume ablations: generate data at n=50, then train at
                n=10/20/etc without re-running inference.
            contrastive_mode: How to select candidates per query.
                'none'         — top-k by similarity, no contrastiveness constraint (default)
                'contrastive'  — top-k by similarity, but guarantee both positive and negative
                                 utility examples are present by swapping in the nearest
                                 opposite-sign candidate from outside the top-k if needed.
                                 Requires top_k to be set, and that the data was generated
                                 with retrieval.contrastive_max_pool set in the data generation
                                 config, so every query has both signs available in the full pool.
        """
        if contrastive_mode not in ('none', 'contrastive'):
            raise ValueError(f"contrastive_mode must be 'none' or 'contrastive', got {contrastive_mode!r}")

        self.results_path = results_path
        self.embeddings_path = embeddings_path
        self.split = split
        self.seed = seed
        self.interaction_features = interaction_features or InteractionFeaturesConfig()
        self.normalize_utilities = normalize_utilities
        self.top_k = top_k
        self.contrastive_mode = contrastive_mode

        # Load data
        print(f"Loading marginal utility results from {results_path}...")
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        all_results = data['results']
        print(f"✓ Loaded {len(all_results)} result pairs")

        # Compute normalization statistics from ALL data (before any filtering) so
        # that stats are consistent across different top_k / contrastive_mode settings.
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

        # Apply top-k or contrastive top-k filtering.
        # 'contrastive' mode subsumes top-k filtering (it also selects top_k per query).
        if self.contrastive_mode == 'contrastive':
            if self.top_k is None:
                raise ValueError("top_k must be set when contrastive_mode='contrastive'")
            self.results = self._apply_contrastive_mode(self.results, self.top_k)
        elif self.top_k is not None:
            self.results = self._apply_top_k(self.results, self.top_k)

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

    def _normalize_utility(self, utility: float) -> float:
        """
        Normalize utility to [0, 1] range using min-max scaling.

        Args:
            utility: Raw utility value

        Returns:
            Normalized utility in [0, 1]
        """
        if not self.normalize_utilities:
            return utility

        if self.utility_range == 0:
            return 0.5  # All utilities are the same

        return (utility - self.utility_min) / self.utility_range

    @staticmethod
    def _group_by_query(results: List) -> Dict[int, List]:
        """Group results by query_idx. Handles both dataclass and dict result objects."""
        groups: Dict[int, List] = defaultdict(list)
        for r in results:
            groups[MarginalUtilityDataset._get_attr(r, 'query_idx')].append(r)
        return groups

    @staticmethod
    def _apply_top_k(results: List, top_k: int) -> List:
        """
        Retain only the top-k candidates per query ranked by similarity score.

        Enables data volume ablations: generate once at n=50, then instantiate
        datasets at n=10, n=20, etc. without re-running inference.
        """
        query_groups = MarginalUtilityDataset._group_by_query(results)

        filtered = []
        for group in query_groups.values():
            sorted_group = sorted(
                group,
                key=lambda r: MarginalUtilityDataset._get_attr(r, 'similarity_score'),
                reverse=True
            )
            filtered.extend(sorted_group[:top_k])

        print(f"  top_k={top_k}: {len(filtered)} pairs retained from {len(query_groups)} queries "
              f"(was {len(results)})")
        return filtered

    @staticmethod
    def _apply_contrastive_mode(results: List, top_k: int) -> List:
        """
        Build a contrastive top-k candidate set for each query.

        Requires that data was generated with retrieval.contrastive_max_pool set,
        so every query has both positive and negative utility examples available
        in the full stored pool.

        For each query:
          1. Sort all candidates by similarity (descending).
          2. Take the top-k by similarity.
          3. If top-k is already contrastive, keep it as-is.
          4. Otherwise, swap the lowest-similarity majority-sign example in top-k
             with the highest-similarity opposite-sign candidate from the remainder.
        """
        get_attr = MarginalUtilityDataset._get_attr
        query_groups = MarginalUtilityDataset._group_by_query(results)

        output = []
        n_swapped = 0

        for group in query_groups.values():
            sorted_group = sorted(
                group,
                key=lambda r: get_attr(r, 'similarity_score'),
                reverse=True
            )
            selected = sorted_group[:top_k]
            pool = sorted_group[top_k:]

            has_positive = any(get_attr(r, 'marginal_utility') > 0 for r in selected)
            has_negative = any(get_attr(r, 'marginal_utility') <= 0 for r in selected)

            if not (has_positive and has_negative) and pool:
                need_positive = not has_positive
                inject = None
                for pool_r in pool:  # pool is similarity-sorted descending
                    u = get_attr(pool_r, 'marginal_utility')
                    if (need_positive and u > 0) or (not need_positive and u <= 0):
                        inject = pool_r
                        break

                if inject is not None:
                    # Replace the lowest-similarity majority-sign element in selected.
                    # selected is similarity-sorted descending, so scan from the end.
                    for i in range(len(selected) - 1, -1, -1):
                        u = get_attr(selected[i], 'marginal_utility')
                        is_majority = (not need_positive and u > 0) or (need_positive and u <= 0)
                        if is_majority:
                            selected[i] = inject
                            n_swapped += 1
                            break

            output.extend(selected)

        print(f"  contrastive top-k={top_k}: {n_swapped}/{len(query_groups)} queries needed a swap, "
              f"{len(output)} pairs total")
        return output

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

        # Normalize utility if enabled
        normalized_utility = self._normalize_utility(marginal_utility)

        # Convert to tensors
        query_emb_tensor = torch.from_numpy(query_emb).float()
        example_emb_tensor = torch.from_numpy(example_emb).float()
        similarity = torch.tensor([similarity_score], dtype=torch.float32)
        utility = torch.tensor([normalized_utility], dtype=torch.float32)

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

    def compute_baseline_spearman(self) -> float:
        """
        Compute baseline Spearman correlation using CLIP similarity as predictor.

        This measures how well CLIP similarity alone ranks examples by utility.

        Returns:
            Baseline Spearman correlation
        """
        from scipy.stats import spearmanr

        utilities = np.array([self._get_attr(r, 'marginal_utility') for r in self.results])
        similarities = np.array([self._get_attr(r, 'similarity_score') for r in self.results])

        spearman_corr, _ = spearmanr(similarities, utilities)
        return float(spearman_corr)


class PairwiseMarginalUtilityDataset(MarginalUtilityDataset):
    """
    Dataset for pairwise ranking loss training.

    Returns pairs of examples (better, worse) for the same query,
    enabling margin ranking loss optimization.
    """

    def __init__(
        self,
        results_path: str,
        embeddings_path: str,
        split: str = 'train',
        seed: int = 42,
        interaction_features: Optional[InteractionFeaturesConfig] = None,
        pairs_per_query: int = 10,
        normalize_utilities: bool = False,
        top_k: Optional[int] = None,
        contrastive_mode: str = 'none',
    ):
        """
        Initialize pairwise dataset.

        Args:
            results_path: Path to marginal_utilities_train.pkl file
            embeddings_path: Path to clip_embeddings_train.pkl file
            split: One of 'train', 'val', 'test'
            seed: Random seed for reproducible splitting
            interaction_features: Configuration for interaction features to compute
            pairs_per_query: Number of pairs to sample per query
            normalize_utilities: If True, normalize utilities to [0, 1] range
            top_k: If set, retain only top-k candidates per query by similarity rank
            contrastive_mode: 'none' or 'contrastive'
        """
        super().__init__(results_path, embeddings_path, split, seed, interaction_features,
                         normalize_utilities, top_k, contrastive_mode)

        self.pairs_per_query = pairs_per_query
        self.rng = random.Random(seed)

        # Build pairs: for each query, sample pairs of (better, worse) examples
        print(f"Building pairwise examples (pairs_per_query={pairs_per_query})...")
        self.pairs = self._build_pairs()
        print(f"✓ Created {len(self.pairs)} pairs")

    def _build_pairs(self) -> List[Tuple[int, int]]:
        """
        Build pairs of (better_idx, worse_idx) for training.

        For each query, sample pairs_per_query pairs where utility_better > utility_worse.

        Returns:
            List of (better_result_idx, worse_result_idx) tuples
        """
        query_groups = self.get_query_groups()
        pairs = []

        for result_indices in query_groups.values():
            # Sample pairs where utility_better > utility_worse
            for _ in range(self.pairs_per_query):
                # Randomly sample two different examples
                if len(result_indices) < 2:
                    continue

                idx1, idx2 = self.rng.sample(result_indices, 2)
                util1 = self._get_attr(self.results[idx1], 'marginal_utility')
                util2 = self._get_attr(self.results[idx2], 'marginal_utility')

                # Order by utility (higher utility = better)
                if util1 > util2:
                    pairs.append((idx1, idx2))  # (better, worse)
                elif util2 > util1:
                    pairs.append((idx2, idx1))  # (better, worse)
                # Skip if utilities are equal

        return pairs

    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a pair of examples (better, worse).

        Returns:
            Tuple containing:
            - query_emb_better: Query embedding for better example
            - example_emb_better: Better example embedding
            - similarity_better: Similarity for better example
            - [interaction features for better example]
            - query_emb_worse: Query embedding for worse example
            - example_emb_worse: Worse example embedding
            - similarity_worse: Similarity for worse example
            - [interaction features for worse example]
        """
        better_idx, worse_idx = self.pairs[idx]

        # Get better example data
        better_result = self.results[better_idx]
        better_query_idx = self._get_attr(better_result, 'query_idx')
        better_example_idx = self._get_attr(better_result, 'example_idx')
        better_similarity = self._get_attr(better_result, 'similarity_score')

        # Get worse example data
        worse_result = self.results[worse_idx]
        worse_example_idx = self._get_attr(worse_result, 'example_idx')
        worse_similarity = self._get_attr(worse_result, 'similarity_score')

        # Get embeddings
        query_emb = self.embeddings[better_query_idx]  # Same query for both
        better_emb = self.embeddings[better_example_idx]
        worse_emb = self.embeddings[worse_example_idx]

        # Convert to tensors
        query_emb_tensor = torch.from_numpy(query_emb).float()
        better_emb_tensor = torch.from_numpy(better_emb).float()
        worse_emb_tensor = torch.from_numpy(worse_emb).float()
        better_sim = torch.tensor([better_similarity], dtype=torch.float32)
        worse_sim = torch.tensor([worse_similarity], dtype=torch.float32)

        # Build output tuple
        output = [query_emb_tensor, better_emb_tensor, better_sim]

        # Add interaction features for better example
        if self.interaction_features.enabled:
            if self.interaction_features.use_product:
                product = query_emb_tensor * better_emb_tensor
                output.append(product)
            if self.interaction_features.use_difference:
                difference = query_emb_tensor - better_emb_tensor
                output.append(difference)
            if self.interaction_features.use_l2_distance:
                l2_distance = torch.norm(query_emb_tensor - better_emb_tensor, p=2).unsqueeze(0)
                output.append(l2_distance)

        # Add worse example features
        output.extend([query_emb_tensor, worse_emb_tensor, worse_sim])

        # Add interaction features for worse example
        if self.interaction_features.enabled:
            if self.interaction_features.use_product:
                product = query_emb_tensor * worse_emb_tensor
                output.append(product)
            if self.interaction_features.use_difference:
                difference = query_emb_tensor - worse_emb_tensor
                output.append(difference)
            if self.interaction_features.use_l2_distance:
                l2_distance = torch.norm(query_emb_tensor - worse_emb_tensor, p=2).unsqueeze(0)
                output.append(l2_distance)

        return tuple(output)


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
