from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.reranker_teacher_data import (
    derive_candidate_metrics,
    score_metrics,
    summarize_teacher_records,
    validate_teacher_record,
)


def _record(split="train", query_idx=5):
    zero_scores = np.asarray([2.0, 2.5, 0.0], dtype=np.float32)
    candidate_scores = np.asarray([
        [3.0, 2.0, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 1.0, 0.0],
    ], dtype=np.float32)
    zero, candidates = derive_candidate_metrics(
        zero_scores, candidate_scores, true_local_idx=0
    )
    return SimpleNamespace(
        query_split=split,
        query_idx=query_idx,
        true_class_idx=0,
        label_class_indices=np.asarray([0, 1, 2], dtype=np.int16),
        label_siglip_similarities=np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
        ranked_distractor_class_indices=np.asarray([1, 2, 3], dtype=np.int16),
        zero_shot_scores=zero_scores,
        zero_shot_metrics=zero,
        candidate_indices=np.asarray([10, 11, 12], dtype=np.int32),
        candidate_class_indices=np.asarray([0, 1, 2], dtype=np.int16),
        candidate_similarities=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        candidate_scores=candidate_scores,
        candidate_metrics=candidates,
    )


def test_score_metrics_and_incremental_targets():
    scores = np.asarray([[3.0, 2.0, 0.0], [2.0, 3.0, 0.0]])
    metrics = score_metrics(scores, true_local_idx=0)
    assert metrics["correct"].tolist() == [True, False]
    assert metrics["true_rank"].tolist() == [1, 2]
    np.testing.assert_allclose(metrics["margin"], [1.0, -1.0])

    zero, candidates = derive_candidate_metrics(
        [2.0, 2.5, 0.0], scores, true_local_idx=0
    )
    assert zero["margin"] == -0.5
    np.testing.assert_allclose(candidates["incremental_margin"], [1.5, -0.5])
    np.testing.assert_allclose(
        candidates["incremental_true_log_probability"],
        candidates["true_log_probability"] - zero["true_log_probability"],
    )


def test_teacher_record_validation_recomputes_derived_targets():
    record = _record()
    validate_teacher_record(record, k=3, pool_size=3)
    record.candidate_metrics["margin"][0] += 0.1
    with pytest.raises(ValueError, match="does not match raw scores"):
        validate_teacher_record(record, k=3, pool_size=3)


def test_teacher_summary_is_partitioned_by_official_query_split():
    train = _record("train", 5)
    val = _record("val", 0)
    train.scoring_mode = "full_sequence_batch"
    val.scoring_mode = "prefix_kv_cache"
    summary = summarize_teacher_records([train, val], k=3, pool_size=3)
    assert summary["num_queries"] == 2
    assert summary["num_pairs"] == 6
    assert summary["scoring_batch_sizes"] == {"8": 2}
    assert summary["scoring_modes"] == {
        "full_sequence_batch": 1,
        "prefix_kv_cache": 1,
    }
    assert set(summary["by_query_split"]) == {"train", "val"}
    assert summary["by_query_split"]["train"]["zero_shot_accuracy"] == 0.0
    assert summary["by_query_split"]["val"]["pool_oracle_accuracy"] == 1.0
    assert summary["by_query_split"]["train"]["contrastive_query_rate"] == 1.0
