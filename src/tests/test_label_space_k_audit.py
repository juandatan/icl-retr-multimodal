from types import SimpleNamespace

import numpy as np

from src.utils.label_space_audit import summarize_label_space_audit


def _record(scores, ranking=(1, 2, 3)):
    return SimpleNamespace(
        query_idx=0,
        true_class_idx=0,
        candidate_indices=[10, 11, 12],
        candidate_scores=scores,
        ranked_distractor_class_indices=list(ranking),
    )


def test_k_audit_detects_exemplar_ranking_change_from_omitted_class():
    # At K=2 (classes 0 and 1), exemplar 10 has the best margin. Class 2 makes
    # exemplar 10 fail in the full space, while exemplar 11 is the full oracle.
    record = _record([
        [3.0, 1.0, 4.0, 0.0],
        [3.0, 2.0, 1.0, 0.0],
        [2.0, 1.5, 1.0, 0.0],
    ])

    results = summarize_label_space_audit([record], [2, 4], class_count=4)
    restricted = results["by_k"][2]
    full = results["by_k"][4]

    assert restricted["strongest_wrong_coverage"] == 2 / 3
    assert restricted["top_exemplar_agreement"] == 0.0
    assert restricted["selected_full_accuracy"] == 0.0
    assert restricted["full_oracle_accuracy"] == 1.0
    assert restricted["full_accuracy_gap_to_oracle"] == 1.0
    assert restricted["full_margin_regret"]["mean"] == 2.0
    assert full["strongest_wrong_coverage"] == 1.0
    assert full["top_exemplar_agreement"] == 1.0
    assert full["full_margin_regret"]["mean"] == 0.0


def test_k_audit_full_space_is_identity_reference():
    records = [
        _record([
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 3.0, 1.0, 0.0],
            [4.0, 1.0, 2.0, 0.0],
        ])
    ]

    metrics = summarize_label_space_audit(records, [4], class_count=4)["by_k"][4]

    assert metrics["rank_correlation"]["mean_spearman"] == 1.0
    assert metrics["rank_correlation"]["mean_kendall"] == 1.0
    assert metrics["top_exemplar_agreement"] == 1.0
    assert metrics["selected_full_accuracy"] == metrics["full_oracle_accuracy"]
