"""Validation and summaries for dense reranker teacher targets."""

from collections import Counter, defaultdict

import numpy as np


def score_metrics(scores, true_local_idx: int, temperature: float = 1.0) -> dict:
    """Derive dense classification targets from one or many score vectors."""
    values = np.asarray(scores, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must have shape [N, K] with K >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain non-finite values")
    if not 0 <= true_local_idx < values.shape[1]:
        raise ValueError("true_local_idx is outside the score vectors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    order = np.argsort(-values, axis=1, kind="stable")
    true_ranks = np.argmax(order == true_local_idx, axis=1) + 1
    wrong = values.copy()
    wrong[:, true_local_idx] = -np.inf
    strongest_wrong_local = np.argmax(wrong, axis=1)
    strongest_wrong_scores = wrong[np.arange(len(values)), strongest_wrong_local]
    margins = values[:, true_local_idx] - strongest_wrong_scores

    scaled = values / temperature
    maxima = np.max(scaled, axis=1, keepdims=True)
    log_normalizers = maxima[:, 0] + np.log(np.exp(scaled - maxima).sum(axis=1))
    true_log_probabilities = scaled[:, true_local_idx] - log_normalizers
    true_probabilities = np.exp(true_log_probabilities)

    result = {
        "correct": order[:, 0] == true_local_idx,
        "true_rank": true_ranks.astype(np.int16),
        "true_score": values[:, true_local_idx].astype(np.float32),
        "strongest_wrong_local_idx": strongest_wrong_local.astype(np.int16),
        "strongest_wrong_score": strongest_wrong_scores.astype(np.float32),
        "margin": margins.astype(np.float32),
        "true_log_probability": true_log_probabilities.astype(np.float32),
        "true_probability": true_probabilities.astype(np.float32),
    }
    if was_vector:
        return {
            key: value[0].item() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
    return result


def derive_candidate_metrics(
    zero_shot_scores,
    candidate_scores,
    true_local_idx: int,
    temperature: float = 1.0,
) -> tuple[dict, dict]:
    """Return zero-shot metrics and direct/incremental one-shot targets."""
    zero = score_metrics(zero_shot_scores, true_local_idx, temperature)
    candidates = score_metrics(candidate_scores, true_local_idx, temperature)
    candidates["incremental_margin"] = (
        candidates["margin"] - float(zero["margin"])
    ).astype(np.float32)
    candidates["incremental_true_log_probability"] = (
        candidates["true_log_probability"] - float(zero["true_log_probability"])
    ).astype(np.float32)
    candidates["incremental_true_probability"] = (
        candidates["true_probability"] - float(zero["true_probability"])
    ).astype(np.float32)
    return zero, candidates


def validate_teacher_record(
    record, k: int, pool_size: int, temperature: float = 1.0
) -> None:
    """Raise if a teacher record is incomplete or internally inconsistent."""
    labels = np.asarray(record.label_class_indices)
    label_similarities = np.asarray(record.label_siglip_similarities)
    ranking = np.asarray(record.ranked_distractor_class_indices)
    zero = np.asarray(record.zero_shot_scores)
    scores = np.asarray(record.candidate_scores)
    candidates = np.asarray(record.candidate_indices)
    candidate_classes = np.asarray(record.candidate_class_indices)
    similarities = np.asarray(record.candidate_similarities)

    if labels.shape != (k,) or len(np.unique(labels)) != k:
        raise ValueError(f"Query {record.query_split}:{record.query_idx} has invalid labels")
    class_count = len(ranking) + 1
    if (
        label_similarities.shape != (k,)
        or ranking.shape != (class_count - 1,)
        or len(np.unique(ranking)) != class_count - 1
        or record.true_class_idx in ranking
        or not 0 <= record.true_class_idx < class_count
        or np.any(labels < 0)
        or np.any(labels >= class_count)
        or np.any(labels[:-1] >= labels[1:])
        or np.any(ranking < 0)
        or np.any(ranking >= class_count)
        or not np.all(np.isfinite(label_similarities))
    ):
        raise ValueError(f"Query {record.query_split}:{record.query_idx} has invalid label metadata")
    if record.true_class_idx not in labels:
        raise ValueError(f"Query {record.query_split}:{record.query_idx} omits the true class")
    if zero.shape != (k,) or scores.shape != (pool_size, k):
        raise ValueError(f"Query {record.query_split}:{record.query_idx} has invalid score shapes")
    if not np.all(np.isfinite(zero)) or not np.all(np.isfinite(scores)):
        raise ValueError(f"Query {record.query_split}:{record.query_idx} has non-finite scores")
    if candidates.shape != (pool_size,) or len(np.unique(candidates)) != pool_size:
        raise ValueError(f"Query {record.query_split}:{record.query_idx} has invalid candidates")
    if candidate_classes.shape != (pool_size,) or similarities.shape != (pool_size,):
        raise ValueError(f"Query {record.query_split}:{record.query_idx} has invalid candidate metadata")
    if (
        np.any(candidate_classes < 0)
        or np.any(candidate_classes >= class_count)
        or not np.all(np.isfinite(similarities))
        or np.any(similarities[:-1] < similarities[1:])
    ):
        raise ValueError(f"Query {record.query_split}:{record.query_idx} candidates are not ranked")
    if record.query_split == "train" and record.query_idx in candidates:
        raise ValueError(f"Training query {record.query_idx} retrieves itself")

    true_local_idx = int(np.flatnonzero(labels == record.true_class_idx)[0])
    expected_zero, expected_candidates = derive_candidate_metrics(
        zero, scores, true_local_idx, temperature
    )
    for key, expected in expected_zero.items():
        if not np.allclose(record.zero_shot_metrics[key], expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Zero-shot metric {key} does not match raw scores")
    for key, expected in expected_candidates.items():
        if not np.allclose(record.candidate_metrics[key], expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Candidate metric {key} does not match raw scores")


def _distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def summarize_teacher_records(
    records: list, k: int, pool_size: int, temperature: float = 1.0
) -> dict:
    """Summarize target quality and selection ceilings for saved checkpoints."""
    if not records:
        raise ValueError("Cannot summarize empty teacher records")
    by_split = defaultdict(list)
    for record in records:
        validate_teacher_record(record, k, pool_size, temperature)
        by_split[record.query_split].append(record)

    def summarize(group):
        candidate_margin = np.concatenate([r.candidate_metrics["margin"] for r in group])
        incremental_margin = np.concatenate([
            r.candidate_metrics["incremental_margin"] for r in group
        ])
        candidate_probability = np.concatenate([
            r.candidate_metrics["true_probability"] for r in group
        ])
        incremental_probability = np.concatenate([
            r.candidate_metrics["incremental_true_probability"] for r in group
        ])
        incremental_log_probability = np.concatenate([
            r.candidate_metrics["incremental_true_log_probability"] for r in group
        ])
        candidate_correct = np.concatenate([r.candidate_metrics["correct"] for r in group])
        zero_correct = np.asarray([r.zero_shot_metrics["correct"] for r in group])
        clip_correct = np.asarray([r.candidate_metrics["correct"][0] for r in group])
        oracle_correct = np.asarray([
            np.any(r.candidate_metrics["correct"]) for r in group
        ])
        oracle_positions = np.asarray([
            int(np.argmax(r.candidate_metrics["margin"])) for r in group
        ])
        contrastive = np.asarray([
            np.any(r.candidate_metrics["incremental_margin"] > 0)
            and np.any(r.candidate_metrics["incremental_margin"] <= 0)
            for r in group
        ])
        return {
            "num_queries": len(group),
            "num_pairs": len(group) * pool_size,
            "zero_shot_accuracy": float(np.mean(zero_correct)),
            "clip_top1_accuracy": float(np.mean(clip_correct)),
            "candidate_pair_accuracy": float(np.mean(candidate_correct)),
            "pool_oracle_accuracy": float(np.mean(oracle_correct)),
            "oracle_top1_agreement": float(np.mean(oracle_positions == 0)),
            "contrastive_query_rate": float(np.mean(contrastive)),
            "positive_incremental_margin_rate": float(np.mean(incremental_margin > 0)),
            "same_class_candidate_rate": float(np.mean(np.concatenate([
                np.asarray(r.candidate_class_indices) == r.true_class_idx for r in group
            ]))),
            "candidate_margin": _distribution(candidate_margin),
            "incremental_margin": _distribution(incremental_margin),
            "true_probability": _distribution(candidate_probability),
            "incremental_true_probability": _distribution(incremental_probability),
            "incremental_true_log_probability": _distribution(incremental_log_probability),
        }

    return {
        "num_queries": len(records),
        "num_pairs": len(records) * pool_size,
        "k": k,
        "candidate_pool_size": pool_size,
        "scoring_batch_sizes": {
            str(batch_size): count
            for batch_size, count in sorted(Counter(
                int(getattr(record, "scoring_batch_size", 8)) for record in records
            ).items())
        },
        "scoring_modes": dict(sorted(Counter(
            str(getattr(record, "scoring_mode", "full_sequence_batch"))
            for record in records
        ).items())),
        "by_query_split": {
            split: summarize(group) for split, group in sorted(by_split.items())
        },
    }
