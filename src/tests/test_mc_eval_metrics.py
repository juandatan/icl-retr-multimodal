"""Tests for MC aggregation and cross-split support ranking."""

from types import SimpleNamespace

import numpy as np

from scripts.evaluate_mc_baselines import (
    _condition_summary,
    _paired_vs_zero_summary,
    deterministic_random_candidate,
    rank_support_candidates,
)


def _record(seed, class_idx, true_letter, zero, clip, zero_pred, clip_pred):
    return SimpleNamespace(
        query_idx=class_idx,
        letter_seed=seed,
        true_class_idx=class_idx,
        true_letter=true_letter,
        zero_shot_correct=zero,
        clip_correct=clip,
        zero_shot_pred_letter=zero_pred,
        clip_pred_letter=clip_pred,
    )


def test_condition_summary_keeps_seed_macro_and_letter_diagnostics():
    records = [
        _record(1, 0, "A", True, False, "A", "B"),
        _record(1, 1, "B", False, True, "A", "B"),
        _record(2, 0, "B", True, True, "B", "B"),
        _record(2, 1, "A", True, False, "A", "B"),
    ]

    summary = _condition_summary(records, "clip_correct", "clip_pred_letter")

    assert summary["accuracy"] == 0.5
    assert summary["mean_per_class_accuracy"] == 0.5
    assert summary["accuracy_by_letter_seed"] == {1: 0.5, 2: 0.5}
    assert summary["accuracy_by_true_letter"] == {"A": 0.0, "B": 1.0}
    assert summary["predicted_letter_counts"] == {"B": 4}


def test_paired_summary_uses_within_trial_transitions():
    records = [
        _record(1, 0, "A", True, False, "A", "B"),
        _record(1, 1, "B", False, True, "A", "B"),
        _record(2, 0, "B", True, True, "B", "B"),
        _record(2, 1, "A", True, False, "A", "B"),
    ]

    summary = _paired_vs_zero_summary(records, "clip_correct")

    assert summary["accuracy_difference"] == -0.25
    assert summary["zero_only_correct"] == 2
    assert summary["condition_only_correct"] == 1
    assert summary["num_paired_trials"] == 4
    assert summary["num_query_clusters"] == 2


def test_cross_split_ranking_and_allowed_mask():
    retrieval = SimpleNamespace(
        clip_embeddings=np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    )
    retrieval.__len__ = lambda: 3
    # Special methods are resolved on the class, not the instance.
    retrieval = type("Dataset", (), {
        "__len__": lambda self: 3,
        "clip_embeddings": retrieval.clip_embeddings,
    })()

    indices, scores = rank_support_candidates(
        np.array([1.0, 0.0]), retrieval, k=2, allowed_indices={1, 2}
    )

    assert indices == [1, 2]
    assert np.allclose(scores, [0.8, 0.0])


def test_random_control_is_reproducible():
    pool = [3, 7, 11, 15]
    assert deterministic_random_candidate(pool, 123) == deterministic_random_candidate(pool, 123)
