"""Generate dense K-way Idefics2 supervision for an exemplar reranker.

The parent fixes query-specific label sets and CLIP candidate pools, shards
queries across isolated GPU workers, merges atomic worker checkpoints, and is
the only process that publishes to Kaggle.

Usage:
    python -m scripts.generate_reranker_teacher_data
    python -m scripts.generate_reranker_teacher_data \
        limits.max_queries_per_split.train=10 limits.max_queries_per_split.val=10
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.dataclasses import RerankerTeacherQueryRecord
from src.data.distractor_sets import build_distractor_ranking
from src.data.loading import load_dataset, resolve_siglip_cache_path
from src.models.idefics2_wrapper import (
    FULL_SEQUENCE_SCORING,
    PREFIX_KV_CACHE_SCORING,
    SUPPORTED_SCORING_MODES,
    Idefics2Wrapper,
)
from src.utils.eval_utils import _json_safe
from src.utils.kaggle_utils import kaggle_upload_eval_results
from src.utils.reranker_teacher_data import (
    derive_candidate_metrics,
    summarize_teacher_records,
)
from src.utils.runtime import (
    atomic_pickle_dump as _atomic_pickle_dump,
    file_sha256 as _file_sha256,
    git_revision as _git_revision,
    setup_device as _setup_device,
    stratified_query_indices,
)


def _task_key(value) -> tuple[str, int]:
    if isinstance(value, dict):
        return str(value["query_split"]), int(value["query_idx"])
    return str(value.query_split), int(value.query_idx)


def _assert_score_parity(
    first: dict[str, float],
    second: dict[str, float],
    labels: list[str],
    *,
    rtol: float,
    atol: float,
    context: str,
    allow_near_tie_label_flip: bool = False,
    true_label: Optional[str] = None,
) -> tuple[float, float, bool]:
    """Validate score geometry and return raw and shift-invariant deltas.

    All downstream teacher targets depend on score differences or softmax
    probabilities, so adding one constant to every label score is immaterial.
    Cached and uncached kernels may introduce such a common offset while still
    defining the same distribution over labels.
    """
    first_values = np.asarray([first[label] for label in labels], dtype=np.float64)
    second_values = np.asarray([second[label] for label in labels], dtype=np.float64)
    raw_maximum_delta = float(np.max(np.abs(first_values - second_values)))
    first_centered = first_values - np.max(first_values)
    second_centered = second_values - np.max(second_values)
    centered_deltas = np.abs(first_centered - second_centered)
    centered_maximum_delta = float(np.max(centered_deltas))
    worst_index = int(np.argmax(centered_deltas))
    if not np.allclose(first_centered, second_centered, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"Scoring parity failed for {context}: "
            f"raw_max_abs_delta={raw_maximum_delta:.6g}, "
            f"centered_max_abs_delta={centered_maximum_delta:.6g}, "
            f"worst_label={labels[worst_index]!r}, "
            f"scores=({first_values[worst_index]:.6g}, "
            f"{second_values[worst_index]:.6g}), rtol={rtol}, atol={atol}"
        )
    first_winner = int(np.argmax(first_values))
    second_winner = int(np.argmax(second_values))
    label_flip = first_winner != second_winner
    if label_flip:
        first_preference = float(
            first_values[first_winner] - first_values[second_winner]
        )
        second_preference = float(
            second_values[second_winner] - second_values[first_winner]
        )
        true_label_involved = true_label in {
            labels[first_winner], labels[second_winner]
        }
        detail = (
            f"{labels[first_winner]!r} vs {labels[second_winner]!r}; "
            f"preference_gaps=({first_preference:.6g}, "
            f"{second_preference:.6g}); "
            f"true_label_involved={true_label_involved}; "
            f"raw_max_abs_delta={raw_maximum_delta:.6g}, "
            f"centered_max_abs_delta={centered_maximum_delta:.6g}"
        )
        if not allow_near_tie_label_flip or true_label_involved:
            raise RuntimeError(
                f"Scoring parity changed the predicted label for {context}: {detail}"
            )
        print(
            f"Scoring parity accepted a within-tolerance label flip for "
            f"{context}: {detail}",
            flush=True,
        )
    return raw_maximum_delta, centered_maximum_delta, label_flip


def load_siglip_split(cfg: DictConfig, split: str):
    dataset_name = str(cfg.dataset.name)
    kaggle_dataset = cfg.dataset.get("siglip_kaggle_dataset", None)
    text_path = resolve_siglip_cache_path(
        dataset_name, "siglip_text_embeddings.pkl", kaggle_dataset
    )
    image_path = resolve_siglip_cache_path(
        dataset_name, f"siglip_image_embeddings_{split}.pkl", kaggle_dataset
    )
    for path in (text_path, image_path):
        if not path.exists():
            raise FileNotFoundError(f"Required SigLIP embedding file not found: {path}")
    with open(text_path, "rb") as file:
        text_data = pickle.load(file)
    with open(image_path, "rb") as file:
        image_data = pickle.load(file)
    text_model = text_data.get("model_name")
    image_model = image_data.get("model_name")
    if text_model and image_model and text_model != image_model:
        raise ValueError(f"SigLIP text/image model mismatch for split {split}")
    return (
        np.asarray(text_data["embeddings"], dtype=np.float32),
        list(text_data["class_names"]),
        np.asarray(image_data["embeddings"], dtype=np.float32),
        text_path,
        image_path,
        {
            "model_name": text_model or image_model,
            "example_ids": image_data.get("example_ids"),
        },
    )


def build_teacher_tasks(
    cfg: DictConfig,
    query_datasets: dict,
    retrieval_dataset,
    siglip_images: dict,
    siglip_text: np.ndarray,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Fix every query, label set, and retrieval pool before GPU inference."""
    k = int(cfg.targets.k)
    pool_size = int(cfg.retrieval.candidate_pool_size)
    tasks = []
    query_indices_by_split = {}
    for split in cfg.dataset.query_splits:
        split = str(split)
        dataset = query_datasets[split]
        maximum = cfg.limits.max_queries_per_split.get(split, None)
        query_indices = stratified_query_indices(
            dataset.examples,
            None if maximum is None else int(maximum),
            int(cfg.experiment.seed),
        )
        query_indices_by_split[split] = query_indices
        for query_idx in query_indices:
            query = dataset.examples[query_idx]
            similarities = retrieval_dataset.clip_embeddings @ dataset.clip_embeddings[query_idx]
            if split == str(cfg.dataset.retrieval_split):
                similarities = similarities.copy()
                similarities[query_idx] = -np.inf
            if bool(cfg.retrieval.exclude_same_class):
                if not similarities.flags.writeable:
                    similarities = similarities.copy()
                retrieval_classes = np.asarray([
                    example.label for example in retrieval_dataset.examples
                ])
                similarities[retrieval_classes == query.label] = -np.inf
            order = np.argsort(-similarities, kind="stable")[:pool_size]
            ranking = build_distractor_ranking(
                query_siglip_emb=siglip_images[split][query_idx],
                class_text_embeddings=siglip_text,
                true_class_idx=query.label,
                query_idx=query_idx,
            )
            label_indices = sorted(ranking.ranked_class_indices[:k - 1] + [query.label])
            label_similarities = siglip_images[split][query_idx] @ siglip_text[label_indices].T
            tasks.append({
                "query_split": split,
                "query_idx": int(query_idx),
                "true_class_idx": int(query.label),
                "candidate_indices": np.asarray(order, dtype=np.int32),
                "candidate_similarities": np.asarray(similarities[order], dtype=np.float32),
                "label_class_indices": np.asarray(label_indices, dtype=np.int16),
                "label_siglip_similarities": np.asarray(label_similarities, dtype=np.float32),
                "ranked_distractor_class_indices": np.asarray(
                    ranking.ranked_class_indices, dtype=np.int16
                ),
            })
    return tasks, query_indices_by_split


def score_teacher_tasks(
    cfg: DictConfig,
    tasks: list[dict],
    worker_id: int,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> list[RerankerTeacherQueryRecord]:
    """Score one task shard on the worker's locally visible cuda:0."""
    if not tasks:
        if checkpoint_path:
            _atomic_pickle_dump([], checkpoint_path)
        return []

    needed_splits = sorted({str(task["query_split"]) for task in tasks})
    query_datasets = {
        split: load_dataset(
            cfg.dataset.name,
            split=split,
            image_split_path=cfg.dataset.image_split_path,
            embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
        )
        for split in needed_splits
    }
    retrieval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.retrieval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    class_names = retrieval_dataset.class_names
    candidate_labels = list(class_names)
    scoring_mode = str(
        cfg.scoring.get("mode", FULL_SEQUENCE_SCORING)
    )
    if scoring_mode not in SUPPORTED_SCORING_MODES:
        raise ValueError(
            f"Unsupported scoring mode {scoring_mode!r}; expected one of "
            f"{sorted(SUPPORTED_SCORING_MODES)}"
        )
    device, _, _ = _setup_device(1)
    if device != "cuda":
        raise RuntimeError("Reranker teacher generation requires a CUDA GPU")
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device="cuda:0",
        load_in_8bit=bool(cfg.model.load_in_8bit),
        scoring_mode=scoring_mode,
    )

    records = []
    log_cfg = cfg.get("logging", {})
    heartbeat_query_every = int(log_cfg.get("heartbeat_every_queries", 25))
    heartbeat_candidate_every = int(log_cfg.get("heartbeat_every_candidates", 10))
    feature_cache_size = int(cfg.scoring.get("candidate_feature_cache_size", 0))
    empty_cache_every = int(cfg.scoring.get("empty_cache_every_queries", 50))
    parity_queries = int(cfg.scoring.get("parity_check_queries", 0))
    parity_rtol = float(cfg.scoring.get("parity_rtol", 1e-3))
    parity_atol = float(cfg.scoring.get("parity_atol", 1e-3))
    allow_near_tie_label_flips = bool(
        cfg.scoring.get("parity_allow_near_tie_label_flips", False)
    )
    parity_mode = (
        PREFIX_KV_CACHE_SCORING
        if scoring_mode == FULL_SEQUENCE_SCORING
        else FULL_SEQUENCE_SCORING
    )
    candidate_feature_cache = OrderedDict()
    feature_cache_hits = 0
    feature_cache_misses = 0
    progress = tqdm(tasks, desc=f"GPU {worker_id} K={cfg.targets.k} teacher", position=worker_id)
    for local_position, task in enumerate(progress, start=1):
        split = str(task["query_split"])
        query_idx = int(task["query_idx"])
        query_started = time.monotonic()
        emit_heartbeat = (
            local_position == 1
            or (
                heartbeat_query_every > 0
                and local_position % heartbeat_query_every == 0
            )
        )
        if emit_heartbeat:
            print(
                f"[GPU {worker_id}] scoring {split}:{query_idx} "
                f"(worker query {local_position}/{len(tasks)})",
                flush=True,
            )
        query, query_image = query_datasets[split][query_idx]
        label_indices = np.asarray(task["label_class_indices"], dtype=np.int16)
        labels = [candidate_labels[int(idx)] for idx in label_indices]
        true_local_idx = int(np.flatnonzero(label_indices == query.label)[0])
        true_label = candidate_labels[int(query.label)]

        query_features = model.encode_full_label_scoring_images([query_image])
        zero_by_label = model.score_candidate_labels_with_image_features(
            query_features,
            [],
            labels,
            batch_size=int(cfg.scoring.candidate_batch_size),
        )
        parity_raw_max_delta = 0.0
        parity_centered_max_delta = 0.0
        parity_label_flips = 0
        check_parity = local_position <= parity_queries
        if check_parity:
            parity_zero = model.score_candidate_labels_with_image_features(
                query_features,
                [],
                labels,
                batch_size=int(cfg.scoring.candidate_batch_size),
                scoring_mode=parity_mode,
            )
            raw_delta, centered_delta, label_flip = _assert_score_parity(
                zero_by_label,
                parity_zero,
                labels,
                rtol=parity_rtol,
                atol=parity_atol,
                context=f"{split}:{query_idx} zero-shot",
                allow_near_tie_label_flip=allow_near_tie_label_flips,
                true_label=true_label,
            )
            parity_raw_max_delta = max(parity_raw_max_delta, raw_delta)
            parity_centered_max_delta = max(
                parity_centered_max_delta, centered_delta
            )
            parity_label_flips += int(label_flip)
        zero_scores = np.asarray([zero_by_label[label] for label in labels], dtype=np.float32)

        candidate_scores = []
        candidate_classes = []
        for candidate_position, candidate_idx in enumerate(
            task["candidate_indices"], start=1
        ):
            candidate, candidate_image = retrieval_dataset[int(candidate_idx)]
            cache_key = int(candidate_idx)
            candidate_features = candidate_feature_cache.pop(cache_key, None)
            if candidate_features is None:
                feature_cache_misses += 1
                candidate_features = model.encode_full_label_scoring_images([candidate_image])
                if feature_cache_size > 0:
                    candidate_feature_cache[cache_key] = candidate_features
                    if len(candidate_feature_cache) > feature_cache_size:
                        candidate_feature_cache.popitem(last=False)
            else:
                feature_cache_hits += 1
                candidate_feature_cache[cache_key] = candidate_features
            combined = model.combine_full_label_scoring_image_features(
                candidate_features, query_features
            )
            scores_by_label = model.score_candidate_labels_with_image_features(
                combined,
                [candidate_labels[candidate.label]],
                labels,
                batch_size=int(cfg.scoring.candidate_batch_size),
            )
            if check_parity:
                parity_scores = model.score_candidate_labels_with_image_features(
                    combined,
                    [candidate_labels[candidate.label]],
                    labels,
                    batch_size=int(cfg.scoring.candidate_batch_size),
                    scoring_mode=parity_mode,
                )
                raw_delta, centered_delta, label_flip = _assert_score_parity(
                    scores_by_label,
                    parity_scores,
                    labels,
                    rtol=parity_rtol,
                    atol=parity_atol,
                    context=(
                        f"{split}:{query_idx} candidate {int(candidate_idx)}"
                    ),
                    allow_near_tie_label_flip=allow_near_tie_label_flips,
                    true_label=true_label,
                )
                parity_raw_max_delta = max(parity_raw_max_delta, raw_delta)
                parity_centered_max_delta = max(
                    parity_centered_max_delta, centered_delta
                )
                parity_label_flips += int(label_flip)
            candidate_classes.append(int(candidate.label))
            candidate_scores.append([scores_by_label[label] for label in labels])
            if (
                emit_heartbeat
                and heartbeat_candidate_every > 0
                and (
                    candidate_position % heartbeat_candidate_every == 0
                    or candidate_position == len(task["candidate_indices"])
                )
            ):
                elapsed = time.monotonic() - query_started
                print(
                    f"[GPU {worker_id}] {split}:{query_idx} exemplars "
                    f"{candidate_position}/{len(task['candidate_indices'])}; "
                    f"elapsed={elapsed:.1f}s; vision_cache="
                    f"{feature_cache_hits}/{feature_cache_hits + feature_cache_misses}",
                    flush=True,
                )
        if check_parity:
            print(
                f"[GPU {worker_id}] scoring parity passed for {split}:{query_idx}; "
                f"{scoring_mode} vs {parity_mode}; "
                f"raw_max_abs_delta={parity_raw_max_delta:.6g}; "
                f"centered_max_abs_delta={parity_centered_max_delta:.6g}; "
                f"within_tolerance_label_flips={parity_label_flips}",
                flush=True,
            )
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        zero_metrics, candidate_metrics = derive_candidate_metrics(
            zero_scores,
            candidate_scores,
            true_local_idx,
            temperature=float(cfg.targets.temperature),
        )
        records.append(RerankerTeacherQueryRecord(
            query_split=split,
            query_idx=query_idx,
            true_class_idx=int(query.label),
            label_class_indices=label_indices,
            label_siglip_similarities=np.asarray(
                task["label_siglip_similarities"], dtype=np.float32
            ),
            ranked_distractor_class_indices=np.asarray(
                task["ranked_distractor_class_indices"], dtype=np.int16
            ),
            zero_shot_scores=zero_scores,
            zero_shot_metrics=zero_metrics,
            candidate_indices=np.asarray(task["candidate_indices"], dtype=np.int32),
            candidate_class_indices=np.asarray(candidate_classes, dtype=np.int16),
            candidate_similarities=np.asarray(
                task["candidate_similarities"], dtype=np.float32
            ),
            candidate_scores=candidate_scores,
            candidate_metrics=candidate_metrics,
            scoring_batch_size=int(cfg.scoring.candidate_batch_size),
            scoring_mode=scoring_mode,
        ))
        if checkpoint_path and checkpoint_every > 0 and len(records) % checkpoint_every == 0:
            _atomic_pickle_dump(records, checkpoint_path)
        if (
            torch.cuda.is_available()
            and empty_cache_every > 0
            and local_position % empty_cache_every == 0
        ):
            torch.cuda.empty_cache()
    if checkpoint_path:
        _atomic_pickle_dump(records, checkpoint_path)
    return records


def _merge_records(existing_records, shard_records, task_keys):
    by_key = {_task_key(record): record for record in existing_records}
    for shard in shard_records:
        for record in shard:
            by_key[_task_key(record)] = record
    return [by_key[key] for key in task_keys if key in by_key]


def _resume_argument_differences(previous: dict, current: dict) -> dict:
    """Return substantive resume incompatibilities, excluding provenance only."""
    ignored_keys = {"git_revision"}
    differences = {}
    for key in sorted((set(previous) | set(current)) - ignored_keys):
        if previous.get(key) != current.get(key):
            differences[key] = {
                "checkpoint": previous.get(key),
                "current": current.get(key),
            }
    return differences


def run_workers(
    cfg: DictConfig,
    pending_tasks: list[dict],
    num_gpus: int,
    existing_records: list[RerankerTeacherQueryRecord],
    task_keys: list[tuple[str, int]],
    progress_callback: Optional[Callable[[list, bool], None]] = None,
) -> list[RerankerTeacherQueryRecord]:
    """Run isolated workers and merge their atomic progress files."""
    shards = [pending_tasks[gpu_id::num_gpus] for gpu_id in range(num_gpus)]
    print(f"Splitting {len(pending_tasks)} pending queries across {num_gpus} GPUs:")
    for gpu_id, shard in enumerate(shards):
        print(f"  GPU {gpu_id}: {len(shard)} queries")

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="reranker_teacher_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.pkl"
        _atomic_pickle_dump(OmegaConf.to_container(cfg, resolve=True), config_path)
        processes = []
        output_paths = []
        cadence = max(1, int(np.ceil(
            max(1, int(cfg.output.checkpoint_every_queries)) / num_gpus
        )))
        for gpu_id, shard in enumerate(shards):
            tasks_path = temp_path / f"tasks_{gpu_id}.pkl"
            output_path = temp_path / f"records_{gpu_id}.pkl"
            _atomic_pickle_dump(shard, tasks_path)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            command = [
                sys.executable,
                "-m",
                "scripts.generate_reranker_teacher_data_worker",
                "--worker-id", str(gpu_id),
                "--config-path", str(config_path),
                "--tasks-path", str(tasks_path),
                "--output-path", str(output_path),
                "--checkpoint-every", str(cadence),
            ]
            processes.append((
                gpu_id,
                subprocess.Popen(command, env=env, cwd=project_root),
            ))
            output_paths.append(output_path)

        last_count = len(existing_records)
        try:
            while True:
                failed = [
                    (gpu_id, process.returncode)
                    for gpu_id, process in processes
                    if process.poll() not in (None, 0)
                ]
                progress_shards = []
                for output_path in output_paths:
                    if output_path.exists():
                        with open(output_path, "rb") as file:
                            progress_shards.append(pickle.load(file))
                    else:
                        progress_shards.append([])
                merged = _merge_records(existing_records, progress_shards, task_keys)
                if len(merged) > last_count:
                    print(f"Merged teacher progress: {len(merged)}/{len(task_keys)} queries")
                    if progress_callback:
                        progress_callback(merged, bool(failed))
                    last_count = len(merged)
                elif failed and progress_callback and merged:
                    # Preserve the latest known merged state even when it has not
                    # reached the normal publication cadence.
                    progress_callback(merged, True)
                if failed:
                    raise RuntimeError(f"Teacher-data worker failure(s): {failed}")
                if all(process.poll() == 0 for _, process in processes):
                    break
                time.sleep(2)
        finally:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()

        final_shards = []
        for output_path in output_paths:
            with open(output_path, "rb") as file:
                final_shards.append(pickle.load(file))
        return _merge_records(existing_records, final_shards, task_keys)


def build_feature_tables(query_datasets, siglip_images, siglip_text, class_names):
    """Bundle compact, index-addressable inputs needed by the reranker."""
    return {
        "indexing": "Rows use the original dataset-local index for each named split.",
        "clip_image_embeddings_by_split": {
            split: np.asarray(dataset.clip_embeddings, dtype=np.float32)
            for split, dataset in query_datasets.items()
        },
        "siglip_image_embeddings_by_split": {
            split: np.asarray(values, dtype=np.float32)
            for split, values in siglip_images.items()
        },
        "siglip_class_text_embeddings": np.asarray(siglip_text, dtype=np.float32),
        "class_names": list(class_names),
    }


def _write_outputs(path: Path, payload: dict):
    _atomic_pickle_dump(payload, path)
    summary_path = path.with_suffix(".json")
    temporary_path = summary_path.with_suffix(".json.tmp")
    with open(temporary_path, "w") as file:
        json.dump(_json_safe({
            key: value for key, value in payload.items()
            if key not in ("records", "feature_tables")
        }), file, indent=2)
    os.replace(temporary_path, summary_path)


@hydra.main(version_base=None, config_path="../configs", config_name="generate_reranker_teacher_data")
def main(cfg: DictConfig):
    query_splits = [str(split) for split in cfg.dataset.query_splits]
    if "test" in query_splits:
        raise ValueError("The final test split must not be used for teacher generation")
    if len(query_splits) != len(set(query_splits)):
        raise ValueError("dataset.query_splits contains duplicates")
    k = int(cfg.targets.k)
    pool_size = int(cfg.retrieval.candidate_pool_size)
    if k < 2:
        raise ValueError("targets.k must be at least 2")

    query_datasets = {
        split: load_dataset(
            cfg.dataset.name,
            split=split,
            image_split_path=cfg.dataset.image_split_path,
            embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
        )
        for split in query_splits
    }
    retrieval_dataset = query_datasets.get(str(cfg.dataset.retrieval_split))
    if retrieval_dataset is None:
        retrieval_dataset = load_dataset(
            cfg.dataset.name,
            split=cfg.dataset.retrieval_split,
            image_split_path=cfg.dataset.image_split_path,
            embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
        )
    class_names = list(retrieval_dataset.class_names)
    class_count = len(class_names)
    if k > class_count or pool_size > len(retrieval_dataset):
        raise ValueError("K or candidate pool exceeds the available class/data space")
    for split, dataset in query_datasets.items():
        if list(dataset.class_names) != class_names:
            raise ValueError(f"Class mapping differs for query split {split}")
        if dataset.clip_model_name != retrieval_dataset.clip_model_name:
            raise ValueError(f"CLIP model differs for query split {split}")

    siglip_text = None
    siglip_class_names = None
    siglip_images = {}
    siglip_paths = {}
    siglip_model = None
    for split in query_splits:
        (
            split_text,
            split_names,
            split_images,
            text_path,
            image_path,
            split_metadata,
        ) = load_siglip_split(cfg, split)
        if siglip_text is None:
            siglip_text, siglip_class_names = split_text, split_names
        elif not np.array_equal(siglip_text, split_text) or siglip_class_names != split_names:
            raise ValueError("SigLIP class text embeddings differ across split caches")
        if siglip_model is None:
            siglip_model = split_metadata["model_name"]
        elif split_metadata["model_name"] != siglip_model:
            raise ValueError("SigLIP model differs across split caches")
        if split_names != class_names or split_images.shape[0] != len(query_datasets[split]):
            raise ValueError(f"SigLIP inputs do not align with query split {split}")
        example_ids = split_metadata["example_ids"]
        expected_ids = [example.image_path for example in query_datasets[split].examples]
        if example_ids is not None and list(example_ids) != expected_ids:
            raise ValueError(f"SigLIP row identifiers do not align with query split {split}")
        siglip_images[split] = split_images
        siglip_paths[split] = str(image_path)

    tasks, query_indices_by_split = build_teacher_tasks(
        cfg, query_datasets, retrieval_dataset, siglip_images, siglip_text
    )
    task_keys = [_task_key(task) for task in tasks]
    task_by_key = {_task_key(task): task for task in tasks}
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Teacher query keys are not unique")

    feature_tables = None
    if bool(cfg.output.get("include_feature_tables", True)):
        feature_tables = build_feature_tables(
            query_datasets, siglip_images, siglip_text, class_names
        )
    split_slug = "_".join(query_splits)
    run_id = f"{cfg.dataset.name}_{split_slug}_k_{k}_pool_{pool_size}"
    output_dir = Path(cfg.output.save_dir) / run_id
    output_path = output_dir / "reranker_teacher_data.pkl"
    immutable_args = {
        "schema_version": 2,
        "dataset": str(cfg.dataset.name),
        "query_splits": query_splits,
        "retrieval_split": str(cfg.dataset.retrieval_split),
        "query_indices_by_split": query_indices_by_split,
        "k": k,
        "candidate_pool_size": pool_size,
        "temperature": float(cfg.targets.temperature),
        "score_definition": str(cfg.targets.score_definition),
        "candidate_policy": str(cfg.targets.candidate_policy),
        "label_policy": str(cfg.targets.label_policy),
        "idefics2_model": str(cfg.model.idefics2_model),
        "load_in_8bit": bool(cfg.model.load_in_8bit),
        "clip_model": retrieval_dataset.clip_model_name,
        "siglip_model": siglip_model,
        "image_feature_representation": "post_connector_pooler_output",
        "image_split_sha256": _file_sha256(cfg.dataset.image_split_path),
        "siglip_text_embeddings_sha256": _file_sha256(str(text_path)),
        "siglip_image_embeddings_sha256_by_split": {
            split: _file_sha256(path) for split, path in siglip_paths.items()
        },
        "git_revision": _git_revision(),
    }

    records = []
    resume_from = cfg.limits.get("resume_from", None)
    if resume_from:
        with open(resume_from, "rb") as file:
            previous = pickle.load(file)
        previous_args = previous.get("immutable_args", {})
        argument_differences = _resume_argument_differences(previous_args, immutable_args)
        if argument_differences:
            raise ValueError(
                "Resume artifact has incompatible scoring/data arguments: "
                f"{json.dumps(_json_safe(argument_differences), indent=2)}"
            )
        if previous_args.get("git_revision") != immutable_args.get("git_revision"):
            print(
                "Resume note: Git revision changed "
                f"({previous_args.get('git_revision')} -> {immutable_args.get('git_revision')}); "
                "continuing because all scoring/data arguments match."
            )
        records = previous["records"]
        previous_batch_size = int(
            previous.get("resolved_config", {})
            .get("scoring", {})
            .get("candidate_batch_size", 8)
        )
        for record in records:
            if "scoring_batch_size" not in getattr(record, "__dict__", {}):
                record.scoring_batch_size = previous_batch_size
            if "scoring_mode" not in getattr(record, "__dict__", {}):
                record.scoring_mode = FULL_SEQUENCE_SCORING
            if record.scoring_mode not in SUPPORTED_SCORING_MODES:
                raise ValueError(
                    "Resume artifact contains unsupported scoring mode "
                    f"{record.scoring_mode!r} for query {_task_key(record)}"
                )
        if len({_task_key(record) for record in records}) != len(records):
            raise ValueError("Resume artifact contains duplicate query records")
        for record in records:
            key = _task_key(record)
            if key not in task_by_key:
                raise ValueError(f"Resume artifact contains unexpected query {key}")
            task = task_by_key[key]
            if not np.array_equal(record.candidate_indices, task["candidate_indices"]):
                raise ValueError(f"Candidate pool changed for query {key}")
            if not np.array_equal(record.label_class_indices, task["label_class_indices"]):
                raise ValueError(f"Label set changed for query {key}")

    def checkpoint(current_records, complete: bool):
        current_records = _merge_records([], [current_records], task_keys)
        summary = summarize_teacher_records(
            current_records, k, pool_size, float(cfg.targets.temperature)
        )
        payload = {
            "method": "reranker_teacher_data",
            "run_id": run_id,
            "complete": complete,
            "completed_queries": len(current_records),
            "requested_queries": len(task_keys),
            "results": summary,
            "immutable_args": immutable_args,
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "records": current_records,
            "feature_tables": feature_tables,
        }
        _write_outputs(output_path, payload)
        print(f"Checkpoint: {len(current_records)}/{len(task_keys)} queries -> {output_path}")
        dataset_name = cfg.output.get("kaggle_dataset", None)
        should_upload = complete or bool(cfg.output.get("upload_every_checkpoint", True))
        if dataset_name and should_upload:
            uploaded = kaggle_upload_eval_results(
                output_dir,
                str(dataset_name),
                title=f"CUB-200 Reranker Teacher Data (K={k})",
            )
            if not uploaded and bool(cfg.output.get("require_upload_success", True)):
                raise RuntimeError(
                    f"Checkpoint was saved locally but upload to {dataset_name} failed"
                )

    completed = {_task_key(record) for record in records}
    pending = [task for task in tasks if _task_key(task) not in completed]
    requested_gpus = int(cfg.hardware.num_gpus)
    device, num_gpus, _ = _setup_device(requested_gpus)
    if pending and device != "cuda":
        raise RuntimeError("Reranker teacher generation requires CUDA GPUs")
    if pending and bool(cfg.hardware.get("require_all", True)) and num_gpus != requested_gpus:
        raise RuntimeError(f"Requested {requested_gpus} GPUs but only {num_gpus} are available")

    print(
        f"Teacher generation: {len(task_keys)} train/val queries, {pool_size} exemplars/query, "
        f"K={k}, {len(completed)} completed"
    )
    checkpoint_every = int(cfg.output.checkpoint_every_queries)
    last_saved_count = len(records)

    def save_progress(merged, force: bool = False):
        nonlocal last_saved_count
        if (
            force
            or (
                checkpoint_every > 0
                and len(merged) < len(task_keys)
                and len(merged) - last_saved_count >= checkpoint_every
            )
        ):
            checkpoint(merged, complete=False)
            last_saved_count = len(merged)

    if pending:
        records = run_workers(
            cfg, pending, num_gpus, records, task_keys, progress_callback=save_progress
        )
    records = _merge_records([], [records], task_keys)
    if len(records) != len(task_keys):
        raise RuntimeError(f"Only completed {len(records)}/{len(task_keys)} teacher queries")
    checkpoint(records, complete=True)
    print(f"Teacher data complete: {output_path}")


if __name__ == "__main__":
    main()
