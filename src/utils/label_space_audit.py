"""Pure metric computation for the full-label-space K audit."""

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _safe_correlation(function, first: np.ndarray, second: np.ndarray) -> float:
    """Return a finite rank correlation, including for constant tied arrays."""
    value = function(first, second).statistic
    if np.isfinite(value):
        return float(value)
    return 1.0 if np.array_equal(first, second) else 0.0


def _true_margins(scores: np.ndarray, true_class_idx: int, class_indices: list[int]) -> np.ndarray:
    """True score minus strongest wrong score for every exemplar."""
    wrong = [idx for idx in class_indices if idx != true_class_idx]
    if not wrong:
        raise ValueError("A margin requires at least one wrong class")
    return scores[:, true_class_idx] - np.max(scores[:, wrong], axis=1)


def _full_correct(scores: np.ndarray, true_class_idx: int) -> bool:
    """Match the evaluator's stable class-index tie breaking."""
    return int(np.argmax(scores)) == true_class_idx


def summarize_label_space_audit(
    records: list, k_values: list[int], class_count: int
) -> dict:
    """Compute the five target-fidelity diagnostics for every K."""
    if not records:
        raise ValueError("Cannot summarize an empty label-space audit")
    k_values = sorted(set(int(k) for k in k_values) | {class_count})
    if any(k < 2 or k > class_count for k in k_values):
        raise ValueError(f"K values must be between 2 and {class_count}")

    per_k = {
        k: {
            "coverage": [],
            "spearman": [],
            "kendall": [],
            "top_agreement": [],
            "selected_correct": [],
            "oracle_correct": [],
            "regret": [],
        }
        for k in k_values
    }

    for record in records:
        scores = np.asarray(record.candidate_scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != class_count:
            raise ValueError(f"Query {record.query_idx} has invalid candidate score shape {scores.shape}")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Query {record.query_idx} contains non-finite scores")
        if scores.shape[0] != len(record.candidate_indices):
            raise ValueError(f"Query {record.query_idx} candidate metadata does not match scores")

        true_idx = int(record.true_class_idx)
        full_margins = _true_margins(scores, true_idx, list(range(class_count)))
        full_best = int(np.argmax(full_margins))
        full_wrong_scores = scores.copy()
        full_wrong_scores[:, true_idx] = -np.inf
        strongest_wrong = np.argmax(full_wrong_scores, axis=1)

        ranking = list(record.ranked_distractor_class_indices)
        if len(ranking) != class_count - 1 or true_idx in ranking or len(set(ranking)) != len(ranking):
            raise ValueError(f"Query {record.query_idx} has an invalid distractor ranking")

        for k in k_values:
            class_indices = sorted(ranking[:k - 1] + [true_idx])
            restricted_margins = _true_margins(scores, true_idx, class_indices)
            selected = int(np.argmax(restricted_margins))
            values = per_k[k]
            values["coverage"].extend(int(idx in class_indices) for idx in strongest_wrong)
            values["spearman"].append(_safe_correlation(spearmanr, restricted_margins, full_margins))
            values["kendall"].append(_safe_correlation(kendalltau, restricted_margins, full_margins))
            values["top_agreement"].append(selected == full_best)
            values["selected_correct"].append(_full_correct(scores[selected], true_idx))
            values["oracle_correct"].append(_full_correct(scores[full_best], true_idx))
            values["regret"].append(float(full_margins[full_best] - full_margins[selected]))

    by_k = {}
    for k, values in per_k.items():
        selected_accuracy = float(np.mean(values["selected_correct"]))
        oracle_accuracy = float(np.mean(values["oracle_correct"]))
        regrets = np.asarray(values["regret"], dtype=np.float64)
        by_k[k] = {
            "strongest_wrong_coverage": float(np.mean(values["coverage"])),
            "rank_correlation": {
                "mean_spearman": float(np.mean(values["spearman"])),
                "mean_kendall": float(np.mean(values["kendall"])),
            },
            "top_exemplar_agreement": float(np.mean(values["top_agreement"])),
            "selected_full_accuracy": selected_accuracy,
            "full_oracle_accuracy": oracle_accuracy,
            "full_accuracy_gap_to_oracle": oracle_accuracy - selected_accuracy,
            "full_margin_regret": {
                "mean": float(np.mean(regrets)),
                "median": float(np.median(regrets)),
                "p95": float(np.quantile(regrets, 0.95)),
            },
        }

    return {
        "num_queries": len(records),
        "candidate_pool_size": len(records[0].candidate_indices),
        "class_count": class_count,
        "k_values": k_values,
        "by_k": by_k,
    }
