"""
Pytest tests for distractor-set construction logic (pure numpy, no model/GPU
dependency).

Run with:
    pytest src/tests/test_distractor_sets.py -v
"""

import numpy as np
import pytest

from src.data.distractor_sets import (
    build_distractor_ranking,
    materialize_distractor_set,
    restrict_pool_to_distractor_classes,
)


def make_class_text_embeddings(num_classes: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.normal(size=(num_classes, dim))
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    return embs


class TestBuildDistractorRanking:
    def test_excludes_true_class(self):
        embs = make_class_text_embeddings(20)
        query_emb = embs[3]  # aligned with true class, should rank first if included
        ranking = build_distractor_ranking(query_emb, embs, true_class_idx=3, query_idx=0)
        assert 3 not in ranking.ranked_class_indices
        assert len(ranking.ranked_class_indices) == 19

    def test_sorted_descending_by_similarity(self):
        embs = make_class_text_embeddings(10)
        query_emb = embs[0]
        ranking = build_distractor_ranking(query_emb, embs, true_class_idx=0, query_idx=0)
        sims = embs[ranking.ranked_class_indices] @ query_emb
        assert np.all(np.diff(sims) <= 1e-9)  # descending (non-increasing)


class TestMaterializeDistractorSet:
    def test_true_label_always_included(self):
        embs = make_class_text_embeddings(20)
        ranking = build_distractor_ranking(embs[5], embs, true_class_idx=5, query_idx=0)
        for k in [4, 8, 16]:
            dset = materialize_distractor_set(ranking, k=k)
            assert 5 in dset.letter_to_class_idx.values()
            assert dset.class_idx_to_letter[5] == dset.true_letter

    def test_set_size_exactly_k(self):
        embs = make_class_text_embeddings(20)
        ranking = build_distractor_ranking(embs[0], embs, true_class_idx=0, query_idx=0)
        for k in [4, 8, 12, 16]:
            dset = materialize_distractor_set(ranking, k=k)
            assert dset.k == k
            assert len(dset.letter_to_class_idx) == k
            assert len(dset.class_idx_to_letter) == k

    def test_k_clamped_when_exceeding_available_classes(self):
        embs = make_class_text_embeddings(5)
        ranking = build_distractor_ranking(embs[0], embs, true_class_idx=0, query_idx=0)
        dset = materialize_distractor_set(ranking, k=16)
        assert dset.k == 5
        assert len(dset.letter_to_class_idx) == 5

    def test_deterministic_given_same_query_and_k(self):
        embs = make_class_text_embeddings(20)
        ranking = build_distractor_ranking(embs[2], embs, true_class_idx=2, query_idx=7)
        dset_a = materialize_distractor_set(ranking, k=8, base_seed=42)
        dset_b = materialize_distractor_set(ranking, k=8, base_seed=42)
        assert dset_a.letter_to_class_idx == dset_b.letter_to_class_idx

    def test_different_query_idx_gives_different_shuffle(self):
        embs = make_class_text_embeddings(20)
        ranking_a = build_distractor_ranking(embs[2], embs, true_class_idx=2, query_idx=1)
        ranking_b = build_distractor_ranking(embs[2], embs, true_class_idx=2, query_idx=2)
        dset_a = materialize_distractor_set(ranking_a, k=8, base_seed=42)
        dset_b = materialize_distractor_set(ranking_b, k=8, base_seed=42)
        # Same candidate classes (same true class + ranking), but letters should
        # very likely differ between distinct query_idx seeds.
        assert dset_a.letter_to_class_idx != dset_b.letter_to_class_idx

    def test_different_k_gives_independent_shuffle(self):
        embs = make_class_text_embeddings(20)
        ranking = build_distractor_ranking(embs[2], embs, true_class_idx=2, query_idx=1)
        dset_8 = materialize_distractor_set(ranking, k=8, base_seed=42)
        dset_16 = materialize_distractor_set(ranking, k=16, base_seed=42)
        # k=8's classes are a subset of k=16's classes (prefix property)...
        assert set(dset_8.letter_to_class_idx.values()) <= set(dset_16.letter_to_class_idx.values())
        # ...but letter assignment for the shared classes need not match, and the
        # shuffles are independently seeded (not simply truncated).
        shared_classes = set(dset_8.letter_to_class_idx.values())
        letters_8 = {cls: dset_8.class_idx_to_letter[cls] for cls in shared_classes}
        letters_16 = {cls: dset_16.class_idx_to_letter[cls] for cls in shared_classes}
        assert letters_8 != letters_16

    def test_topk_prefix_property(self):
        """Smaller K's distractor classes are a prefix of the ranking used for larger K."""
        embs = make_class_text_embeddings(20)
        ranking = build_distractor_ranking(embs[0], embs, true_class_idx=0, query_idx=3)
        dset_4 = materialize_distractor_set(ranking, k=4, base_seed=42)
        dset_16 = materialize_distractor_set(ranking, k=16, base_seed=42)
        classes_4 = set(dset_4.letter_to_class_idx.values())
        classes_16 = set(dset_16.letter_to_class_idx.values())
        assert classes_4 <= classes_16
        assert classes_4 == set(ranking.ranked_class_indices[:3]) | {0}


class TestRestrictPoolToDistractorClasses:
    class _FakeExample:
        def __init__(self, index, label):
            self.index = index
            self.label = label

    class _FakeDataset:
        def __init__(self, labels):
            self.examples = [
                TestRestrictPoolToDistractorClasses._FakeExample(i, label)
                for i, label in enumerate(labels)
            ]

    def test_restricts_to_allowed_classes(self):
        dataset = self._FakeDataset(labels=[0, 1, 2, 3, 1, 2])
        embs = make_class_text_embeddings(4)
        ranking = build_distractor_ranking(embs[0], embs, true_class_idx=0, query_idx=0)
        dset = materialize_distractor_set(ranking, k=2)  # true class 0 + 1 distractor
        allowed_classes = set(dset.class_idx_to_letter.keys())

        allowed_indices = restrict_pool_to_distractor_classes(dataset, dset)
        expected = {ex.index for ex in dataset.examples if ex.label in allowed_classes}
        assert allowed_indices == expected
        assert all(dataset.examples[i].label in allowed_classes for i in allowed_indices)
