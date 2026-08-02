"""Selection-centric validation metrics for exemplar rerankers."""

import numpy as np
from scipy.stats import spearmanr


def reranker_selection_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    margins: np.ndarray,
    correct: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Summarize candidate rankings, with one row per independent query."""
    arrays = [np.asarray(value) for value in (scores, targets, margins, correct, mask)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("All reranker metric inputs must be rank-2")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("All reranker metric inputs must have equal shapes")
    scores, targets, margins, correct, mask = arrays
    mask = mask.astype(bool)
    if not mask.any(axis=1).all():
        raise ValueError("Every query must have a valid candidate")

    predicted = np.where(mask, scores, -np.inf).argmax(axis=1)
    target_best = np.where(mask, targets, -np.inf).argmax(axis=1)
    margin_best = np.where(mask, margins, -np.inf).argmax(axis=1)
    rows = np.arange(len(scores))
    valid_scores = scores[mask]
    if not np.isfinite(valid_scores).all():
        raise ValueError("Valid predictions contain non-finite values")

    correlations = []
    for row_scores, row_margins, row_mask in zip(scores, margins, mask):
        result = spearmanr(row_scores[row_mask], row_margins[row_mask]).statistic
        correlations.append(0.0 if not np.isfinite(result) else float(result))

    selected_margin = margins[rows, predicted]
    oracle_margin = margins[rows, margin_best]
    selected_target = targets[rows, predicted]
    oracle_target = targets[rows, target_best]
    return {
        "restricted_selected_accuracy": float(np.mean(correct[rows, predicted])),
        "restricted_clip_top1_accuracy": float(np.mean(correct[:, 0])),
        "restricted_pool_oracle_accuracy": float(
            np.mean(correct[rows, margin_best])
        ),
        "mean_selected_margin": float(np.mean(selected_margin)),
        "mean_margin_regret": float(np.mean(oracle_margin - selected_margin)),
        "margin_oracle_agreement": float(np.mean(predicted == margin_best)),
        "target_oracle_agreement": float(np.mean(predicted == target_best)),
        "mean_target_regret": float(np.mean(oracle_target - selected_target)),
        "mean_margin_spearman": float(np.mean(correlations)),
    }
