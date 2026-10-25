"""Held-out test evaluation of trained rerankers under full-200 Idefics2 scoring.

For every test query the parent process rebuilds the CLIP top-M training-image
pool exactly as teacher generation did, then runs each trained reranker
checkpoint on CPU to pick one exemplar. GPU workers score all 200 class names
after zero exemplars, the CLIP top-1 exemplar, and each selected exemplar
(selectors that agree share one Idefics2 pass). Restricted K-way diagnostics
mask the same fixed scores to nested SigLIP distractor sets, so K=32 test
accuracy is comparable to the validation selection accuracy. (The teacher
scored only the 32 labels; batching 200 labels can shift scores by
floating-point noise.)

Teacher pools ranked fp16 CLIP embeddings, whose similarities are heavily
tied, so exact pool reproduction requires the same stored embedding dtype.
Set ``reranker.teacher_artifact`` to verify feature tables and rebuilt
validation pools against the checkpoint's training artifact.

No test query is used for training, checkpoint selection, or choosing the
primary selector: ``primary_selector`` must be fixed from validation results.

Usage:
    python -m scripts.evaluate_reranker_test \
        selectors.reranker_m30.checkpoint=/path/to/m30/best.pt \
        selectors.reranker_m50.checkpoint=/path/to/m50/best.pt \
        primary_selector=reranker_m50
    python -m scripts.evaluate_reranker_test scoring.num_queries=20 ...
"""

import csv
import math
import pickle
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.evaluate_full_label_baselines import (
    _merge_records,
    _record_metrics,
    _validate_resume_records,
    build_query_tasks,
    run_multi_gpu_workers,
    summarize_condition,
)
from scripts.export_reranker_predictions import _load_checkpoint, _pooled_model
from src.data.dataclasses import RerankerTestEvalResult
from src.data.loading import load_dataset, load_siglip_inputs
from src.models.idefics2_wrapper import Idefics2Wrapper
from src.models.reranker import LabelAwareReranker
from src.utils.eval_utils import save_eval_results
from src.utils.kaggle_utils import kaggle_upload_eval_results
from src.utils.runtime import (
    atomic_pickle_dump as _atomic_pickle_dump,
    file_sha256 as _file_sha256,
    git_revision as _git_revision,
    setup_device as _setup_device,
    stratified_query_indices,
)


SCHEMA_VERSION = 1
METHOD = "reranker_test_eval"
ZERO_SHOT = "zero_shot"
CLIP_SELECTOR = "clip_top1"
# Inputs that must be identical for resumed records to be poolable.
RESUME_INVARIANT_ARGS = (
    "candidate_count", "k_values", "clip_model", "clip_embedding_dtype",
    "image_split_sha256", "siglip_text_embeddings_sha256",
    "siglip_retrieval_image_embeddings_sha256", "siglip_eval_image_embeddings_sha256",
    "retrieval_split", "idefics2_model", "load_in_8bit",
)
RESUME_INVARIANT_CONFIG = (
    "dataset.eval_split", "scoring.mode", "scoring.candidate_batch_size",
)


@dataclass(frozen=True)
class RerankerFeatureTables:
    """Model-input tables, matching the teacher artifact's bundled features."""

    query_clip: torch.Tensor
    query_siglip: torch.Tensor
    retrieval_clip: torch.Tensor
    retrieval_siglip: torch.Tensor
    label_siglip: torch.Tensor
    retrieval_classes: torch.Tensor


def clip_candidate_pool(
    retrieval_clip: np.ndarray, query_clip: np.ndarray, pool_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rank retrieval images exactly as ``build_teacher_tasks`` does.

    Test queries are not in the retrieval split, so no self-exclusion applies.
    """
    similarities = retrieval_clip @ query_clip
    order = np.argsort(-similarities, kind="stable")[:pool_size]
    return (
        np.asarray(order, dtype=np.int64),
        np.asarray(similarities[order], dtype=np.float32),
    )


def reranker_inputs(
    tables: RerankerFeatureTables,
    query_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_similarities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Batch pooled-reranker inputs as ``RerankerTeacherDataset`` + collation."""
    batch_size, count = candidate_indices.shape
    ranks = torch.arange(count, dtype=torch.float32) / max(count - 1, 1)
    return {
        "query_clip": tables.query_clip[query_indices],
        "candidate_clip": tables.retrieval_clip[candidate_indices],
        "query_siglip": tables.query_siglip[query_indices],
        "candidate_siglip": tables.retrieval_siglip[candidate_indices],
        "candidate_label_siglip": tables.label_siglip[
            tables.retrieval_classes[candidate_indices]
        ],
        "clip_similarities": candidate_similarities.to(torch.float32),
        "retrieval_ranks": ranks.expand(batch_size, count),
        "candidate_mask": torch.ones((batch_size, count), dtype=torch.bool),
    }


@torch.inference_mode()
def rerank_pools(
    model,
    tables: RerankerFeatureTables,
    query_indices: np.ndarray,
    pool_indices: np.ndarray,
    pool_similarities: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return each query's selected position within its candidate pool."""
    positions = []
    for start in range(0, len(query_indices), batch_size):
        rows = slice(start, start + batch_size)
        batch = reranker_inputs(
            tables,
            torch.as_tensor(query_indices[rows], dtype=torch.long),
            torch.as_tensor(pool_indices[rows], dtype=torch.long),
            torch.as_tensor(pool_similarities[rows]),
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        scores = model(**batch)
        positions.append(
            LabelAwareReranker.select(scores, batch["candidate_mask"]).cpu().numpy()
        )
    return np.concatenate(positions)


def checkpoint_provenance_mismatches(checkpoint: dict, expected: dict) -> list[str]:
    """Compare a checkpoint's teacher-artifact inputs with the evaluation inputs."""
    immutable = (checkpoint.get("provenance") or {}).get("artifact_immutable_args")
    if not immutable:
        return ["checkpoint records no teacher-artifact provenance"]
    retrieval_split = immutable.get("retrieval_split")
    observed = {
        "clip_model": immutable.get("clip_model"),
        "image_split_sha256": immutable.get("image_split_sha256"),
        "siglip_text_embeddings_sha256": immutable.get("siglip_text_embeddings_sha256"),
        "siglip_retrieval_image_embeddings_sha256": (
            immutable.get("siglip_image_embeddings_sha256_by_split") or {}
        ).get(retrieval_split),
        "retrieval_split": retrieval_split,
        "idefics2_model": immutable.get("idefics2_model"),
        "load_in_8bit": immutable.get("load_in_8bit"),
    }
    return [
        f"{key}: checkpoint={observed.get(key)!r}, evaluation={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]


def checkpoint_pool_size(checkpoint: dict) -> int:
    """Return the candidate-pool width a checkpoint was trained and validated on."""
    max_candidates = (
        (checkpoint.get("resolved_config") or {}).get("data", {}).get("max_candidates")
    )
    if max_candidates is not None:
        return int(max_candidates)
    immutable = (checkpoint.get("provenance") or {}).get("artifact_immutable_args") or {}
    if immutable.get("candidate_pool_size") is None:
        raise ValueError("Checkpoint records neither data.max_candidates nor the artifact pool size")
    return int(immutable["candidate_pool_size"])


def load_selectors(cfg: DictConfig, expected_provenance: dict, k_values: list[int], device):
    """Load enabled checkpoints; return JSON-safe specs and their eval-mode models."""
    specs, models = [], {}
    require_match = bool(cfg.reranker.get("require_matching_provenance", True))
    for name, entry in cfg.selectors.items():
        name = str(name)
        if entry.get("checkpoint") is None:
            continue
        if name in {ZERO_SHOT, CLIP_SELECTOR}:
            raise ValueError(f"Selector name {name!r} is reserved for a baseline")
        path = Path(str(entry.checkpoint))
        checkpoint = _load_checkpoint(path, allow_non_best=bool(cfg.reranker.allow_non_best))
        mismatches = checkpoint_provenance_mismatches(checkpoint, expected_provenance)
        if mismatches:
            message = f"Selector {name} inputs differ from its training data:\n  " + "\n  ".join(mismatches)
            if require_match:
                raise ValueError(message)
            print(f"WARNING: {message}")
        immutable = (checkpoint.get("provenance") or {}).get("artifact_immutable_args") or {}
        if str(cfg.dataset.eval_split) in [str(split) for split in immutable.get("query_splits", [])]:
            raise ValueError(
                f"Selector {name} was trained or selected on split {cfg.dataset.eval_split!r}"
            )
        training_k = immutable.get("k")
        if training_k is not None and int(training_k) not in k_values:
            raise ValueError(
                f"distractor_set.k_values must include the teacher K={training_k} of {name}"
            )
        training_pool_size = checkpoint_pool_size(checkpoint)
        pool_size = int(entry.pool_size) if entry.get("pool_size") is not None else training_pool_size
        if pool_size < 1:
            raise ValueError(f"Selector {name} pool_size must be positive")
        if pool_size != training_pool_size:
            print(
                f"WARNING: {name} evaluates M={pool_size} but was trained on "
                f"M={training_pool_size}; its rank feature scale changes."
            )
        models[name] = _pooled_model(checkpoint, device)
        specs.append({
            "name": name,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": _file_sha256(str(path)),
            "artifact_sha256": (checkpoint.get("provenance") or {}).get("artifact_sha256"),
            "experiment_name": checkpoint.get("experiment_name"),
            "epoch": int(checkpoint["epoch"]),
            "training_k": None if training_k is None else int(training_k),
            "training_pool_size": training_pool_size,
            "pool_size": pool_size,
            "validation_metrics": checkpoint.get("metrics"),
            "training_git_revision": (checkpoint.get("provenance") or {}).get("git_revision"),
        })
    if not specs:
        raise ValueError("Enable at least one selector by setting selectors.<name>.checkpoint")
    return specs, models


def verify_teacher_artifact(
    path, selector_specs: list[dict], retrieval_clip: np.ndarray,
    tables: RerankerFeatureTables, num_pools: int,
) -> dict:
    """Check evaluation features and pool construction against the training artifact.

    Validation pools are rebuilt from the artifact's query embeddings cast back
    to the stored retrieval dtype, then compared with the teacher's pools.
    """
    sha256 = _file_sha256(str(path))
    for spec in selector_specs:
        if spec["artifact_sha256"] not in (None, sha256):
            raise ValueError(f"{spec['name']} was trained on a different teacher artifact")
    with open(path, "rb") as file:
        payload = pickle.load(file)
    feature_tables = payload["feature_tables"]
    split = str(payload["immutable_args"]["retrieval_split"])
    for label, expected, actual in (
        ("retrieval CLIP", feature_tables["clip_image_embeddings_by_split"][split], tables.retrieval_clip),
        ("retrieval SigLIP", feature_tables["siglip_image_embeddings_by_split"][split], tables.retrieval_siglip),
        ("SigLIP class text", feature_tables["siglip_class_text_embeddings"], tables.label_siglip),
    ):
        if not np.array_equal(np.asarray(expected, dtype=np.float32), actual.numpy()):
            raise ValueError(f"{label} features differ from the teacher artifact")
    records = [r for r in payload["records"] if str(r.query_split) != split][:num_pools]
    for record in records:
        query = np.asarray(
            feature_tables["clip_image_embeddings_by_split"][str(record.query_split)][int(record.query_idx)]
        ).astype(retrieval_clip.dtype)
        order, similarities = clip_candidate_pool(retrieval_clip, query, len(record.candidate_indices))
        if not (
            np.array_equal(order, record.candidate_indices)
            and np.array_equal(similarities, record.candidate_similarities)
        ):
            raise ValueError(
                f"Rebuilt CLIP pool differs from the teacher for "
                f"{record.query_split}:{record.query_idx}"
            )
    print(f"Verified features and {len(records)} rebuilt pools against {path}")
    return {"teacher_artifact_sha256": sha256, "teacher_verified_pools": len(records)}


def resolve_primary_selector(primary: Optional[str], selector_names: list[str]) -> str:
    if primary is None:
        if len(selector_names) != 1:
            raise ValueError(
                "Set primary_selector (chosen on validation) when evaluating several selectors"
            )
        return selector_names[0]
    if primary not in selector_names:
        raise ValueError(f"primary_selector {primary!r} is not an enabled selector: {selector_names}")
    return primary


def build_selection_tasks(
    base_tasks: list[dict],
    true_classes: np.ndarray,
    pool_indices: np.ndarray,
    retrieval_classes: np.ndarray,
    selected_positions: dict[str, np.ndarray],
    pool_sizes: dict[str, int],
) -> list[dict]:
    """Attach per-selector exemplar choices to the distractor-set query tasks.

    Row ``i`` of every array describes ``base_tasks[i]``. CLIP top-1 is pool
    position 0, i.e. the stable argmax used by the baseline evaluator.
    """
    tasks = []
    for row, base in enumerate(base_tasks):
        true_class = int(true_classes[row])
        selected = {CLIP_SELECTOR: int(pool_indices[row, 0])}
        positions = {CLIP_SELECTOR: 0}
        pool_has_true = {}
        for name, row_positions in selected_positions.items():
            position = int(row_positions[row])
            if not 0 <= position < pool_sizes[name]:
                raise ValueError(f"{name} selected position {position} outside its pool")
            selected[name] = int(pool_indices[row, position])
            positions[name] = position
            pool_has_true[name] = bool(
                np.any(retrieval_classes[pool_indices[row, :pool_sizes[name]]] == true_class)
            )
        tasks.append({
            **base,
            "true_class_idx": true_class,
            "selected_example_indices": selected,
            "selected_positions": positions,
            "pool_contains_true_class": pool_has_true,
            "example_indices": sorted(set(selected.values())),
        })
    return tasks


def score_selector_tasks(
    cfg: DictConfig,
    tasks: list[dict],
    worker_id: int,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> list[RerankerTestEvalResult]:
    """Score one worker's task shard on its locally visible cuda:0 device."""
    if not tasks:
        if checkpoint_path:
            _atomic_pickle_dump([], checkpoint_path)
        return []
    eval_dataset = load_dataset(
        cfg.dataset.name, split=cfg.dataset.eval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    retrieval_dataset = load_dataset(
        cfg.dataset.name, split=cfg.dataset.retrieval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    candidate_labels = list(eval_dataset.class_names)
    device, _, _ = _setup_device(1)
    if device != "cuda":
        raise RuntimeError("Reranker test GPU worker requires CUDA")
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device="cuda:0",
        load_in_8bit=bool(cfg.model.load_in_8bit),
        scoring_mode=str(cfg.scoring.mode),
        cache_dir=cfg.model.get("cache_dir", None),
    )
    batch_size = int(cfg.scoring.candidate_batch_size)

    records = []
    for task in tqdm(tasks, desc=f"GPU {worker_id} reranker test scoring", position=worker_id):
        query_idx = int(task["query_idx"])
        query_example, query_image = eval_dataset[query_idx]
        if int(query_example.label) != int(task["true_class_idx"]):
            raise ValueError(f"Query {query_idx} label differs from the parent task")
        query_features = model.encode_full_label_scoring_images([query_image])
        zero_by_label = model.score_candidate_labels_with_image_features(
            query_features, [], candidate_labels, batch_size=batch_size,
        )
        example_scores, example_classes = {}, {}
        for example_idx in task["example_indices"]:
            example, example_image = retrieval_dataset[int(example_idx)]
            combined = model.combine_full_label_scoring_image_features(
                model.encode_full_label_scoring_images([example_image]), query_features
            )
            by_label = model.score_candidate_labels_with_image_features(
                combined, [candidate_labels[example.label]], candidate_labels,
                batch_size=batch_size,
            )
            example_scores[int(example_idx)] = [by_label[label] for label in candidate_labels]
            example_classes[int(example_idx)] = int(example.label)
        records.append(RerankerTestEvalResult(
            query_idx=query_idx,
            true_class_idx=int(query_example.label),
            zero_shot_scores=[zero_by_label[label] for label in candidate_labels],
            selected_example_indices=dict(task["selected_example_indices"]),
            selected_positions=dict(task["selected_positions"]),
            pool_contains_true_class=dict(task["pool_contains_true_class"]),
            example_class_indices=example_classes,
            example_scores=example_scores,
            distractor_class_indices={
                int(k): list(indices) for k, indices in task["distractor_class_indices"].items()
            },
        ))
        if checkpoint_path and checkpoint_every > 0 and len(records) % checkpoint_every == 0:
            _atomic_pickle_dump(records, checkpoint_path)

    if checkpoint_path:
        _atomic_pickle_dump(records, checkpoint_path)
    return records


def condition_views(records: list[RerankerTestEvalResult], condition: str) -> list:
    """Adapt records to the baseline summarizer's single-score-vector interface."""
    views = []
    for record in records:
        scores = (
            record.zero_shot_scores if condition == ZERO_SHOT
            else record.example_scores[record.selected_example_indices[condition]]
        )
        views.append(SimpleNamespace(
            query_idx=record.query_idx,
            true_class_idx=record.true_class_idx,
            scores=scores,
            distractor_class_indices=record.distractor_class_indices,
        ))
    return views


def mcnemar_exact_p(treatment_only: int, control_only: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs."""
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(treatment_only, control_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def paired_accuracy_difference(
    treatment_correct,
    control_correct,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict:
    """Paired accuracy difference with a percentile bootstrap interval.

    Per-query differences take only the values -1, 0, and +1, so resampling
    queries with replacement is exactly a multinomial draw over those counts.
    """
    treatment = np.asarray(treatment_correct, dtype=bool)
    control = np.asarray(control_correct, dtype=bool)
    if treatment.shape != control.shape or treatment.ndim != 1 or not len(treatment):
        raise ValueError("Paired correctness vectors must be equal-length and non-empty")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    n = len(treatment)
    treatment_only = int(np.sum(treatment & ~control))
    control_only = int(np.sum(control & ~treatment))
    counts = np.random.default_rng(seed).multinomial(
        n, np.asarray([control_only, n - treatment_only - control_only, treatment_only]) / n,
        size=int(bootstrap_samples),
    )
    differences = (counts[:, 2] - counts[:, 0]) / n
    alpha = (1 - confidence_level) / 2
    return {
        "accuracy_difference": (treatment_only - control_only) / n,
        "ci_low": float(np.quantile(differences, alpha)),
        "ci_high": float(np.quantile(differences, 1 - alpha)),
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(bootstrap_samples),
        "treatment_only_correct": treatment_only,
        "control_only_correct": control_only,
        "both_correct": int(np.sum(treatment & control)),
        "neither_correct": int(np.sum(~treatment & ~control)),
        "mcnemar_exact_p": mcnemar_exact_p(treatment_only, control_only),
    }


def _correctness(records, condition: str, temperature: float, k: Optional[int] = None) -> np.ndarray:
    return np.asarray([
        _record_metrics(view, "scores", temperature, k)["correct"]
        for view in condition_views(records, condition)
    ])


def summarize_test_results(
    records: list[RerankerTestEvalResult],
    selector_specs: list[dict],
    primary_selector: str,
    temperature: float,
    k_values: list[int],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty record list")
    rerankers = [spec["name"] for spec in selector_specs]
    conditions = [ZERO_SHOT, CLIP_SELECTOR, *rerankers]

    def paired(treatment, control) -> dict:
        return paired_accuracy_difference(
            treatment, control, bootstrap_samples, confidence_level, seed
        )

    def evaluate(k: Optional[int]) -> dict:
        correct = {name: _correctness(records, name, temperature, k) for name in conditions}
        return {
            "conditions": {
                name: summarize_condition(condition_views(records, name), "scores", temperature, k)
                for name in conditions
            },
            "paired_vs_clip_top1": {
                name: paired(correct[name], correct[CLIP_SELECTOR]) for name in rerankers
            },
            "paired_clip_top1_vs_zero_shot": paired(correct[CLIP_SELECTOR], correct[ZERO_SHOT]),
            "correct_by_condition": correct,
        }

    full = evaluate(None)
    correct = full.pop("correct_by_condition")
    restricted_by_k = {}
    nested_correct = []
    for k in sorted(set(k_values)):
        restricted_by_k[k] = evaluate(k)
        nested_correct.append(restricted_by_k[k].pop("correct_by_condition"))
    # Fixed scores over nested label sets: adding labels can never turn an
    # incorrect prediction correct. Full-200 is the largest set.
    nested_correct.append(correct)
    for name in conditions:
        for smaller, larger in zip(nested_correct, nested_correct[1:]):
            if np.any(~smaller[name] & larger[name]):
                raise ValueError(f"Non-monotonic K accuracy for condition {name}")

    true_classes = np.asarray([record.true_class_idx for record in records])
    clip_examples = np.asarray([record.selected_example_indices[CLIP_SELECTOR] for record in records])
    selection = {}
    for name in [CLIP_SELECTOR, *rerankers]:
        examples = np.asarray([record.selected_example_indices[name] for record in records])
        example_classes = np.asarray([
            record.example_class_indices[record.selected_example_indices[name]] for record in records
        ])
        stats = {
            "same_class_rate": float(np.mean(example_classes == true_classes)),
            "agreement_with_clip_top1": float(np.mean(examples == clip_examples)),
            "mean_selected_position": float(np.mean([
                record.selected_positions[name] for record in records
            ])),
        }
        if name != CLIP_SELECTOR:
            spec = next(spec for spec in selector_specs if spec["name"] == name)
            has_true = np.asarray([record.pool_contains_true_class[name] for record in records])
            stats.update({
                "pool_size": spec["pool_size"],
                "pool_contains_true_class_rate": float(np.mean(has_true)),
                "queries_with_true_class_in_pool": int(np.sum(has_true)),
            })
            for label, rows in (("pool_contains_true", has_true), ("pool_lacks_true", ~has_true)):
                stats[f"accuracy_when_{label}"] = (
                    float(np.mean(correct[name][rows])) if rows.any() else None
                )
                stats[f"clip_top1_accuracy_when_{label}"] = (
                    float(np.mean(correct[CLIP_SELECTOR][rows])) if rows.any() else None
                )
        selection[name] = stats

    primary = full["conditions"][primary_selector]
    return {
        "accuracy": primary["accuracy"],
        "mean_per_class_accuracy": primary["mean_per_class_accuracy"],
        "correct": primary["correct"],
        "total": primary["total"],
        "primary_selector": primary_selector,
        "candidate_count": len(records[0].zero_shot_scores),
        "k_values": sorted(set(k_values)),
        "score_definition": "mean_token_log_likelihood",
        "candidate_policy": "score_all_then_mask_to_nested_siglip_distractor_sets",
        "temperature": temperature,
        **full,
        "restricted_by_k": restricted_by_k,
        "selection": selection,
        "mean_distinct_exemplars_per_query": float(np.mean([
            len(record.example_scores) for record in records
        ])),
    }


def write_query_csv(path: Path, records, rerankers: list[str], k_values: list[int], temperature: float):
    """Write one row per query with each condition's choice and correctness."""
    rows = []
    for record in records:
        row = {"query_idx": record.query_idx, "true_class_idx": record.true_class_idx}
        for name in [ZERO_SHOT, CLIP_SELECTOR, *rerankers]:
            view = condition_views([record], name)[0]
            metrics = _record_metrics(view, "scores", temperature)
            if name != ZERO_SHOT:
                example_idx = record.selected_example_indices[name]
                row[f"{name}_example_idx"] = example_idx
                row[f"{name}_example_class_idx"] = record.example_class_indices[example_idx]
                row[f"{name}_selected_position"] = record.selected_positions[name]
            if name in record.pool_contains_true_class:
                row[f"{name}_pool_contains_true_class"] = record.pool_contains_true_class[name]
            row[f"{name}_predicted_class_idx"] = metrics["predicted_class_idx"]
            row[f"{name}_correct"] = metrics["correct"]
            row[f"{name}_true_rank"] = metrics["true_rank"]
            for k in k_values:
                row[f"{name}_k{k}_correct"] = _record_metrics(view, "scores", temperature, k)["correct"]
        rows.append(row)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(records, query_indices, candidate_labels, cfg, run_args, upload: bool):
    k_values = run_args["k_values"]
    rerankers = [spec["name"] for spec in run_args["selectors"]]
    temperature = float(cfg.scoring.temperature)
    results = summarize_test_results(
        records,
        run_args["selectors"],
        run_args["primary_selector"],
        temperature,
        k_values,
        int(cfg.statistics.bootstrap_samples),
        float(cfg.statistics.confidence_level),
        int(cfg.experiment.seed),
    )
    complete = len(records) == len(query_indices)
    # Partial results cover a class-biased prefix of the class-sorted query
    # order, so they are flagged and never written under the final name.
    results.update({
        "complete": complete,
        "completed_queries": len(records),
        "target_queries": len(query_indices),
    })
    run_id_parts = {
        "dataset": cfg.dataset.name,
        "eval": cfg.dataset.eval_split,
        # Checkpoint hashes keep re-trained selectors from overwriting results.
        "selectors": "+".join(
            f"{spec['name']}-{spec['checkpoint_sha256'][:8]}" for spec in run_args["selectors"]
        ),
        "n": len(query_indices),
    }
    if not complete:
        run_id_parts["status"] = "partial"
    out_dir = save_eval_results(
        method=METHOD,
        results=results,
        run_id_parts=run_id_parts,
        args=run_args,
        extra={
            "records": records,
            "query_indices": query_indices,
            "candidate_labels": candidate_labels,
        },
        output_root=cfg.output.save_dir,
    )
    run_id = "_".join(str(value) for value in run_id_parts.values())
    write_query_csv(out_dir / f"{run_id}_queries.csv", records, rerankers, k_values, temperature)
    dataset_name = cfg.output.get("results_kaggle_dataset", None)
    if dataset_name and upload:
        kaggle_upload_eval_results(out_dir, dataset_name)
    return out_dir, results


def _load_resume_records(path, cfg, run_args, candidate_labels, query_indices, task_by_query):
    with open(path, "rb") as file:
        payload = pickle.load(file)
    saved_args = payload["args"]
    if payload.get("method") != METHOD or saved_args.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Resume requires a schema-v{SCHEMA_VERSION} {METHOD} checkpoint")
    if payload["candidate_labels"] != candidate_labels:
        raise ValueError("Checkpoint candidate labels differ from the current dataset")
    if not set(payload["query_indices"]).issubset(query_indices):
        raise ValueError("Checkpoint contains queries outside the current query sample")

    def selector_key(specs):
        return [(spec["name"], spec["checkpoint_sha256"], spec["pool_size"]) for spec in specs]

    if selector_key(saved_args["selectors"]) != selector_key(run_args["selectors"]):
        raise ValueError("Checkpoint was produced by different selectors or pool sizes")
    for key in RESUME_INVARIANT_ARGS:
        if saved_args.get(key) != run_args[key]:
            raise ValueError(f"Checkpoint differs in {key}")
    for key in RESUME_INVARIANT_CONFIG:
        if OmegaConf.select(OmegaConf.create(saved_args["resolved_config"]), key) != OmegaConf.select(cfg, key):
            raise ValueError(f"Checkpoint uses a different {key}")
    records = payload["records"]
    _validate_resume_records(records, payload["query_indices"])
    for record in records:
        task = task_by_query[record.query_idx]
        for field in ("selected_example_indices", "distractor_class_indices"):
            if getattr(record, field) != task[field]:
                raise ValueError(f"Checkpoint {field} differ for query {record.query_idx}")
    return records


def _package_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@hydra.main(version_base=None, config_path="../configs", config_name="eval_reranker_test")
def main(cfg: DictConfig):
    eval_split = str(cfg.dataset.eval_split)
    retrieval_split = str(cfg.dataset.retrieval_split)
    if eval_split == retrieval_split:
        raise ValueError("The evaluation split must be disjoint from the retrieval split")
    eval_dataset, retrieval_dataset = (
        load_dataset(
            cfg.dataset.name, split=split,
            image_split_path=cfg.dataset.image_split_path,
            embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
        )
        for split in (eval_split, retrieval_split)
    )
    if eval_dataset.class_names != retrieval_dataset.class_names:
        raise ValueError("Evaluation and retrieval class mappings differ")
    if eval_dataset.clip_model_name != retrieval_dataset.clip_model_name:
        raise ValueError("Evaluation and retrieval embeddings use different CLIP models")
    candidate_labels = list(eval_dataset.class_names)
    k_values = sorted({int(k) for k in cfg.distractor_set.k_values})
    if not k_values or any(k < 2 or k > len(candidate_labels) for k in k_values):
        raise ValueError(f"distractor_set.k_values must be between 2 and {len(candidate_labels)}")

    siglip = {}
    for split, dataset in ((eval_split, eval_dataset), (retrieval_split, retrieval_dataset)):
        siglip[split] = load_siglip_inputs(
            str(cfg.dataset.name), split, cfg.dataset.get("siglip_kaggle_dataset", None),
            expected_class_names=candidate_labels,
            expected_example_ids=[example.image_path for example in dataset.examples],
        )
        if siglip[split][2].shape[0] != len(dataset):
            raise ValueError(f"SigLIP image embeddings do not align with split {split}")
    siglip_text, _, eval_siglip, text_path, eval_siglip_path = siglip[eval_split]
    _, _, retrieval_siglip, _, retrieval_siglip_path = siglip[retrieval_split]
    # The teacher cast SigLIP tables to float32 before ranking distractors.
    siglip_text, eval_siglip, retrieval_siglip = (
        np.asarray(values, dtype=np.float32) for values in (siglip_text, eval_siglip, retrieval_siglip)
    )
    # CLIP pools are ranked in the stored dtype (fp16 for the teacher); mixed
    # dtypes would promote the dot product and reorder heavily tied pools.
    clip_dtype = np.asarray(retrieval_dataset.clip_embeddings).dtype
    if np.asarray(eval_dataset.clip_embeddings).dtype != clip_dtype:
        raise ValueError("Evaluation and retrieval CLIP embeddings have different dtypes")
    if siglip_text.shape[0] != len(candidate_labels):
        raise ValueError("SigLIP text embeddings do not cover every candidate class")

    expected_provenance = {
        "clip_model": eval_dataset.clip_model_name,
        "image_split_sha256": _file_sha256(cfg.dataset.image_split_path),
        "siglip_text_embeddings_sha256": _file_sha256(str(text_path)),
        "siglip_retrieval_image_embeddings_sha256": _file_sha256(str(retrieval_siglip_path)),
        "retrieval_split": retrieval_split,
        "idefics2_model": str(cfg.model.idefics2_model),
        "load_in_8bit": bool(cfg.model.load_in_8bit),
    }
    torch.manual_seed(int(cfg.experiment.seed))
    reranker_device = torch.device(str(cfg.reranker.device))
    selector_specs, models = load_selectors(cfg, expected_provenance, k_values, reranker_device)
    rerankers = [spec["name"] for spec in selector_specs]
    primary_selector = resolve_primary_selector(cfg.get("primary_selector", None), rerankers)
    pool_sizes = {spec["name"]: spec["pool_size"] for spec in selector_specs}
    max_pool = max(pool_sizes.values())
    if max_pool > len(retrieval_dataset):
        raise ValueError("A selector pool exceeds the retrieval split")

    query_indices = stratified_query_indices(
        eval_dataset.examples, cfg.scoring.get("num_queries", None), int(cfg.experiment.seed)
    )
    query_array = np.asarray(query_indices, dtype=np.int64)
    pools = [
        clip_candidate_pool(retrieval_dataset.clip_embeddings, eval_dataset.clip_embeddings[q], max_pool)
        for q in query_indices
    ]
    pool_indices = np.stack([order for order, _ in pools])
    pool_similarities = np.stack([similarities for _, similarities in pools])
    retrieval_classes = np.asarray([example.label for example in retrieval_dataset.examples])
    tables = RerankerFeatureTables(
        query_clip=torch.as_tensor(np.asarray(eval_dataset.clip_embeddings, dtype=np.float32)),
        query_siglip=torch.as_tensor(eval_siglip),
        retrieval_clip=torch.as_tensor(np.asarray(retrieval_dataset.clip_embeddings, dtype=np.float32)),
        retrieval_siglip=torch.as_tensor(retrieval_siglip),
        label_siglip=torch.as_tensor(siglip_text),
        retrieval_classes=torch.as_tensor(retrieval_classes, dtype=torch.long),
    )
    teacher_verification = None
    if cfg.reranker.get("teacher_artifact", None):
        teacher_verification = verify_teacher_artifact(
            cfg.reranker.teacher_artifact, selector_specs, retrieval_dataset.clip_embeddings,
            tables, int(cfg.reranker.teacher_artifact_check_pools),
        )
    selected_positions = {
        name: rerank_pools(
            models[name], tables, query_array,
            pool_indices[:, :pool_sizes[name]], pool_similarities[:, :pool_sizes[name]],
            int(cfg.reranker.batch_size), reranker_device,
        )
        for name in rerankers
    }
    query_tasks = build_selection_tasks(
        build_query_tasks(eval_dataset, query_indices, eval_siglip, siglip_text, k_values),
        np.asarray([eval_dataset.examples[q].label for q in query_indices]),
        pool_indices, retrieval_classes, selected_positions, pool_sizes,
    )
    task_by_query = {task["query_idx"]: task for task in query_tasks}

    print(f"Reranker test evaluation: {len(query_indices)} stratified {eval_split} queries")
    clip_choice = pool_indices[:, 0]
    true_classes = np.asarray([task["true_class_idx"] for task in query_tasks])
    for name in rerankers:
        chosen = np.asarray([task["selected_example_indices"][name] for task in query_tasks])
        print(
            f"  {name} (M={pool_sizes[name]}): same-class pick "
            f"{np.mean(retrieval_classes[chosen] == true_classes):.1%} vs CLIP "
            f"{np.mean(retrieval_classes[clip_choice] == true_classes):.1%}; "
            f"agrees with CLIP on {np.mean(chosen == clip_choice):.1%}"
        )

    run_args = {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "selectors": selector_specs,
        "primary_selector": primary_selector,
        "query_indices": query_indices,
        "candidate_count": len(candidate_labels),
        "k_values": k_values,
        "eval_num_examples": len(eval_dataset),
        "retrieval_num_examples": len(retrieval_dataset),
        **expected_provenance,
        "clip_embedding_dtype": str(clip_dtype),
        "teacher_verification": teacher_verification,
        "siglip_eval_image_embeddings_sha256": _file_sha256(str(eval_siglip_path)),
        "environment": _package_versions(),
        "git_revision": _git_revision(),
    }

    records = []
    resume_from = cfg.limits.get("resume_from", None)
    if resume_from:
        records = _load_resume_records(
            resume_from, cfg, run_args, candidate_labels, query_indices, task_by_query
        )
    completed = {record.query_idx for record in records}
    pending_tasks = [task for task in query_tasks if task["query_idx"] not in completed]
    print(f"  {len(completed)} queries already completed; {len(pending_tasks)} pending")

    requested_gpus = int(cfg.hardware.num_gpus)
    device, num_gpus, _ = _setup_device(requested_gpus)
    if pending_tasks and device != "cuda":
        raise RuntimeError("The reranker test evaluation requires CUDA GPUs")
    if pending_tasks and bool(cfg.hardware.get("require_all", True)) and num_gpus != requested_gpus:
        raise RuntimeError(f"Requested {requested_gpus} GPUs but only {num_gpus} are available")

    last_saved_count = len(records)
    checkpoint_every = int(cfg.output.checkpoint_every)

    def save_progress(merged_records, force=False):
        nonlocal last_saved_count
        if len(merged_records) < len(query_indices) and (
            force or (
                checkpoint_every > 0
                and len(merged_records) - last_saved_count >= checkpoint_every
            )
        ):
            save_checkpoint(
                merged_records, query_indices, candidate_labels, cfg, run_args,
                upload=bool(cfg.output.get("upload_every_checkpoint", True)),
            )
            last_saved_count = len(merged_records)

    if pending_tasks:
        records = run_multi_gpu_workers(
            cfg=cfg,
            pending_tasks=pending_tasks,
            num_gpus=num_gpus,
            existing_records=records,
            query_indices=query_indices,
            progress_callback=save_progress,
            worker_module="scripts.evaluate_reranker_test_worker",
        )
    records = _merge_records(records, [], query_indices)
    if len(records) != len(query_indices):
        raise RuntimeError(f"Only completed {len(records)}/{len(query_indices)} queries")
    out_dir, results = save_checkpoint(
        records, query_indices, candidate_labels, cfg, run_args, upload=True
    )

    print(f"Full-{len(candidate_labels)} top-1 accuracy (primary: {primary_selector}):")
    for name in [ZERO_SHOT, CLIP_SELECTOR, *rerankers]:
        condition = results["conditions"][name]
        print(f"  {name:>16}: top-1 {condition['accuracy']:.2%}, top-5 {condition['top5_accuracy']:.2%}, "
              f"MRR {condition['mean_reciprocal_rank']:.3f}")
    for name in rerankers:
        diff = results["paired_vs_clip_top1"][name]
        print(f"  {name} - CLIP top-1: {diff['accuracy_difference']:+.2%} "
              f"[{diff['ci_low']:+.2%}, {diff['ci_high']:+.2%}], McNemar p={diff['mcnemar_exact_p']:.2g}")
    print("Restricted fixed-score accuracy:")
    for k, restricted in results["restricted_by_k"].items():
        conditions = restricted["conditions"]
        print(f"  K={k:3d}: " + ", ".join(
            f"{name}={conditions[name]['accuracy']:.2%}" for name in [ZERO_SHOT, CLIP_SELECTOR, *rerankers]
        ))
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
