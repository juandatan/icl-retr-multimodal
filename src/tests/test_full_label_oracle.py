from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_full_label_oracle import (
    build_oracle_tasks,
    oracle_target_k_values,
    summarize_oracle_results,
)


def test_oracle_scope_selects_restricted_or_unrestricted_targets():
    assert oracle_target_k_values("restricted", [16, 4, 8, 8], 200) == [4, 8, 16]
    assert oracle_target_k_values("unrestricted", [4, 8], 200) == [200]
    assert oracle_target_k_values("all", [8, 4], 200) == [4, 8, 200]
    with pytest.raises(ValueError):
        oracle_target_k_values("invalid", [4], 200)


def test_oracle_pool_rebuild_reuses_baseline_clip_top1():
    baseline = [SimpleNamespace(query_idx=0, clip_example_idx=1)]
    eval_dataset = SimpleNamespace(clip_embeddings=np.array([[1.0, 0.0]]))
    retrieval_dataset = SimpleNamespace(clip_embeddings=np.array([
        [0.1, 0.0],
        [0.9, 0.0],
        [0.5, 0.0],
    ]))

    tasks = build_oracle_tasks(
        baseline, eval_dataset, retrieval_dataset, candidate_pool_size=2
    )

    assert tasks[0]["candidate_indices"] == [1, 2]
    assert tasks[0]["candidate_similarities"] == [0.9, 0.5]


def test_oracle_summary_reports_candidate_existence_ceiling_by_k():
    records = [
        SimpleNamespace(
            query_idx=0,
            true_class_idx=1,
            candidate_indices=[10, 11],
            candidate_class_indices=[2, 1],
            candidate_correct_by_k={4: [False, True], 8: [False, False]},
            candidate_margin_by_k={4: [-1.0, 0.5], 8: [-2.0, -0.2]},
            candidate_true_rank_by_k={4: [3, 1], 8: [5, 2]},
        ),
        SimpleNamespace(
            query_idx=1,
            true_class_idx=3,
            candidate_indices=[20, 21],
            candidate_class_indices=[3, 4],
            candidate_correct_by_k={4: [True, False], 8: [True, False]},
            candidate_margin_by_k={4: [0.7, -1.0], 8: [0.2, -1.5]},
            candidate_true_rank_by_k={4: [1, 4], 8: [1, 6]},
        ),
    ]

    results = summarize_oracle_results(records, [4, 8], class_count=200)

    assert results["by_k"][4]["oracle_accuracy"] == 1.0
    assert results["by_k"][8]["oracle_accuracy"] == 0.5
    assert results["by_k"][4]["clip_top1_accuracy"] == 0.5
    assert results["by_k"][4]["pool_has_true_class_rate"] == 1.0
    assert results["accuracy"] == 0.5  # largest restricted K is the primary metric
