"""Option-free, all-class label-likelihood classification with Idefics2.

Every canonical class name is teacher-forced after one identical prompt. The
model always scores all classes; K-way results are derived afterward by masking
those fixed scores to nested SigLIP distractor sets. Thus K does not change the
prompt, exemplar, or any existing class score.

The parent process builds the query/distractor specification, then launches one
isolated subprocess per GPU. Each subprocess sees only one physical GPU through
CUDA_VISIBLE_DEVICES and loads its own Idefics2 copy.

Usage:
    python -m scripts.evaluate_full_label_baselines
    python -m scripts.evaluate_full_label_baselines scoring.num_queries=10
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.dataclasses import FullLabelEvalResult
from src.data.distractor_sets import build_distractor_ranking
from src.data.loading import load_dataset, load_siglip_inputs
from src.models.idefics2_wrapper import Idefics2Wrapper
from src.utils.eval_utils import save_eval_results
from src.utils.kaggle_utils import kaggle_upload_eval_results
from src.utils.runtime import (
    atomic_pickle_dump as _atomic_pickle_dump,
    file_sha256 as _file_sha256,
    git_revision as _git_revision,
    setup_device as _setup_device,
    stratified_query_indices,
)


def closed_set_metrics(scores, true_class_idx: int, temperature: float = 1.0) -> dict:
    """Calculate stable ranks, margins, and softmax diagnostics from class energies."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("scores must be a non-empty 1D sequence")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    if not 0 <= true_class_idx < len(scores):
        raise ValueError("true_class_idx is outside the score vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    order = np.argsort(-scores, kind="stable")
    predicted = int(order[0])
    true_rank = int(np.flatnonzero(order == true_class_idx)[0]) + 1
    wrong_scores = np.delete(scores, true_class_idx)
    true_margin = (
        float(scores[true_class_idx] - np.max(wrong_scores))
        if len(wrong_scores) else float("inf")
    )
    top_gap = float(scores[order[0]] - scores[order[1]]) if len(scores) > 1 else float("inf")

    scaled = scores / temperature
    max_scaled = np.max(scaled)
    shifted = scaled - max_scaled
    log_normalizer = float(max_scaled + np.log(np.exp(shifted).sum()))
    true_log_probability = float(scaled[true_class_idx] - log_normalizer)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))

    return {
        "predicted_class_idx": predicted,
        "correct": predicted == true_class_idx,
        "top5_correct": true_rank <= min(5, len(scores)),
        "true_rank": true_rank,
        "reciprocal_rank": 1.0 / true_rank,
        "true_score": float(scores[true_class_idx]),
        "true_log_probability": true_log_probability,
        "true_probability": float(probabilities[true_class_idx]),
        "true_margin": true_margin,
        "top1_top2_gap": top_gap,
        "entropy": entropy,
        "effective_num_classes": float(np.exp(entropy)),
    }


def _record_metrics(record, score_field: str, temperature: float, k: Optional[int] = None) -> dict:
    scores = getattr(record, score_field)
    if k is None:
        return closed_set_metrics(scores, record.true_class_idx, temperature)

    if k not in record.distractor_class_indices:
        raise ValueError(f"Record for query {record.query_idx} has no K={k} distractor set")
    # Class sets have no meaningful presentation order. Sorting makes score ties
    # resolve identically for a class as K grows.
    class_indices = sorted(record.distractor_class_indices[k])
    if record.true_class_idx not in class_indices:
        raise ValueError(f"K={k} set for query {record.query_idx} omits the true class")
    local_true_idx = class_indices.index(record.true_class_idx)
    metrics = closed_set_metrics(
        [scores[class_idx] for class_idx in class_indices], local_true_idx, temperature
    )
    metrics["predicted_class_idx"] = class_indices[metrics["predicted_class_idx"]]
    return metrics


def summarize_condition(
    records: list,
    score_field: str,
    temperature: float,
    k: Optional[int] = None,
) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty record list")
    per_query = [_record_metrics(record, score_field, temperature, k) for record in records]
    by_class = defaultdict(list)
    for record, metrics in zip(records, per_query):
        by_class[record.true_class_idx].append(metrics["correct"])
    ranks = np.array([metrics["true_rank"] for metrics in per_query])
    candidate_count = k if k is not None else len(getattr(records[0], score_field))
    cutoffs = sorted({cutoff for cutoff in (1, 5, 10, 20, 50, 100, 200) if cutoff <= candidate_count}
                     | {candidate_count})
    return {
        "accuracy": float(np.mean([metrics["correct"] for metrics in per_query])),
        "top5_accuracy": float(np.mean([metrics["top5_correct"] for metrics in per_query])),
        "mean_per_class_accuracy": float(np.mean([np.mean(values) for values in by_class.values()])),
        "mean_reciprocal_rank": float(np.mean([metrics["reciprocal_rank"] for metrics in per_query])),
        "mean_true_rank": float(np.mean(ranks)),
        "median_true_rank": float(np.median(ranks)),
        "mean_true_probability": float(np.mean([metrics["true_probability"] for metrics in per_query])),
        "mean_true_margin": float(np.mean([metrics["true_margin"] for metrics in per_query])),
        "mean_top1_top2_gap": float(np.mean([metrics["top1_top2_gap"] for metrics in per_query])),
        "mean_entropy": float(np.mean([metrics["entropy"] for metrics in per_query])),
        "mean_effective_num_classes": float(np.mean([
            metrics["effective_num_classes"] for metrics in per_query
        ])),
        "true_rank_cdf": {
            f"top_{cutoff}": float(np.mean(ranks <= cutoff)) for cutoff in cutoffs
        },
        "correct": int(sum(metrics["correct"] for metrics in per_query)),
        "total": len(per_query),
        "candidate_count": candidate_count,
    }


def _paired_summary(zero_metrics: list[dict], clip_metrics: list[dict]) -> dict:
    return {
        "accuracy_difference": float(np.mean([
            int(c["correct"]) - int(z["correct"])
            for z, c in zip(zero_metrics, clip_metrics)
        ])),
        "zero_only_correct": int(sum(
            z["correct"] and not c["correct"] for z, c in zip(zero_metrics, clip_metrics)
        )),
        "clip_only_correct": int(sum(
            c["correct"] and not z["correct"] for z, c in zip(zero_metrics, clip_metrics)
        )),
        "mean_true_log_probability_difference": float(np.mean([
            c["true_log_probability"] - z["true_log_probability"]
            for z, c in zip(zero_metrics, clip_metrics)
        ])),
        "mean_true_margin_difference": float(np.mean([
            c["true_margin"] - z["true_margin"]
            for z, c in zip(zero_metrics, clip_metrics)
        ])),
    }


def summarize_results(records: list, temperature: float, k_values: list[int]) -> dict:
    k_values = sorted(set(k_values))
    zero = summarize_condition(records, "zero_shot_scores", temperature)
    clip = summarize_condition(records, "clip_scores", temperature)
    zero_metrics = [_record_metrics(r, "zero_shot_scores", temperature) for r in records]
    clip_metrics = [_record_metrics(r, "clip_scores", temperature) for r in records]

    restricted_by_k = {}
    for k in k_values:
        restricted_zero = summarize_condition(records, "zero_shot_scores", temperature, k)
        restricted_clip = summarize_condition(records, "clip_scores", temperature, k)
        restricted_zero_metrics = [
            _record_metrics(r, "zero_shot_scores", temperature, k) for r in records
        ]
        restricted_clip_metrics = [
            _record_metrics(r, "clip_scores", temperature, k) for r in records
        ]
        restricted_by_k[k] = {
            "conditions": {
                "zero_shot": restricted_zero,
                "clip_top1": restricted_clip,
            },
            "paired": _paired_summary(restricted_zero_metrics, restricted_clip_metrics),
        }

    # With nested candidate sets and fixed scores, adding candidates cannot turn
    # an incorrect prediction into a correct one. Enforce this core baseline invariant.
    for score_field in ("zero_shot_scores", "clip_scores"):
        for record in records:
            correctness = [
                _record_metrics(record, score_field, temperature, k)["correct"]
                for k in k_values
            ]
            if any(not earlier and later for earlier, later in zip(correctness, correctness[1:])):
                raise ValueError(
                    f"Non-monotonic K accuracy for query {record.query_idx}, field {score_field}"
                )

    return {
        "accuracy": zero["accuracy"],
        "mean_per_class_accuracy": zero["mean_per_class_accuracy"],
        "correct": zero["correct"],
        "total": zero["total"],
        "candidate_count": len(records[0].zero_shot_scores),
        "k_values": k_values,
        "score_definition": "mean_token_log_likelihood",
        "candidate_policy": "score_all_then_mask_to_nested_siglip_distractor_sets",
        "temperature": temperature,
        "conditions": {"zero_shot": zero, "clip_top1": clip},
        "paired": _paired_summary(zero_metrics, clip_metrics),
        "restricted_by_k": restricted_by_k,
        "retrieval": {
            "same_class_rate": float(np.mean([
                r.clip_example_class_idx == r.true_class_idx for r in records
            ])),
            "mean_clip_similarity": float(np.mean([r.clip_similarity for r in records])),
        },
    }


def build_query_tasks(
    eval_dataset,
    query_indices: list[int],
    siglip_image_embeddings: np.ndarray,
    siglip_text_embeddings: np.ndarray,
    k_values: list[int],
) -> list[dict]:
    tasks = []
    for query_idx in query_indices:
        example = eval_dataset.examples[query_idx]
        ranking = build_distractor_ranking(
            query_siglip_emb=siglip_image_embeddings[query_idx],
            class_text_embeddings=siglip_text_embeddings,
            true_class_idx=example.label,
            query_idx=query_idx,
        )
        distractor_sets = {}
        for k in k_values:
            distractor_sets[k] = sorted(
                ranking.ranked_class_indices[:k - 1] + [ranking.true_class_idx]
            )
        tasks.append({
            "query_idx": query_idx,
            "distractor_class_indices": distractor_sets,
        })
    return tasks


def score_query_tasks(
    cfg: DictConfig,
    tasks: list[dict],
    worker_id: int,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> list[FullLabelEvalResult]:
    """Score one worker's task shard on its locally visible cuda:0 device."""
    if not tasks:
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
    if eval_dataset.class_names != retrieval_dataset.class_names:
        raise ValueError("Evaluation and retrieval class mappings differ")
    if eval_dataset.clip_model_name != retrieval_dataset.clip_model_name:
        raise ValueError("Evaluation and retrieval embeddings use different CLIP models")

    candidate_labels = list(eval_dataset.class_names)
    device, _, _ = _setup_device(1)
    if device != "cuda":
        raise RuntimeError("Full-label GPU worker requires CUDA")
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device="cuda:0",
        load_in_8bit=cfg.model.load_in_8bit,
    )

    records = []
    for task in tqdm(tasks, desc=f"GPU {worker_id} full-label scoring", position=worker_id):
        query_idx = int(task["query_idx"])
        query_example, query_image = eval_dataset[query_idx]
        similarities = retrieval_dataset.clip_embeddings @ eval_dataset.clip_embeddings[query_idx]
        clip_example_idx = int(np.argmax(similarities))
        clip_example, clip_image = retrieval_dataset[clip_example_idx]
        clip_label = candidate_labels[clip_example.label]

        query_features = model.encode_full_label_scoring_images([query_image])
        zero_by_label = model.score_candidate_labels_with_image_features(
            query_features, [], candidate_labels,
            batch_size=int(cfg.scoring.candidate_batch_size),
        )
        clip_features = model.encode_full_label_scoring_images([clip_image])
        clip_query_features = model.combine_full_label_scoring_image_features(
            clip_features, query_features
        )
        clip_by_label = model.score_candidate_labels_with_image_features(
            clip_query_features, [clip_label], candidate_labels,
            batch_size=int(cfg.scoring.candidate_batch_size),
        )
        records.append(FullLabelEvalResult(
            query_idx=query_idx,
            true_class_idx=query_example.label,
            clip_example_idx=clip_example_idx,
            clip_example_class_idx=clip_example.label,
            clip_similarity=float(similarities[clip_example_idx]),
            zero_shot_scores=[zero_by_label[label] for label in candidate_labels],
            clip_scores=[clip_by_label[label] for label in candidate_labels],
            distractor_class_indices={
                int(k): list(indices)
                for k, indices in task["distractor_class_indices"].items()
            },
        ))
        if checkpoint_path and checkpoint_every > 0 and len(records) % checkpoint_every == 0:
            _atomic_pickle_dump(records, checkpoint_path)

    if checkpoint_path:
        _atomic_pickle_dump(records, checkpoint_path)
    return records


def _merge_records(existing_records, shard_records, query_indices):
    by_query = {record.query_idx: record for record in existing_records}
    for records in shard_records:
        for record in records:
            by_query[record.query_idx] = record
    return [by_query[idx] for idx in query_indices if idx in by_query]


def _validate_resume_records(records, saved_query_indices):
    """Validate completed records from either a partial or final checkpoint."""
    record_query_indices = [record.query_idx for record in records]
    if len(record_query_indices) != len(set(record_query_indices)):
        raise ValueError("Checkpoint contains duplicate query records")
    if not set(record_query_indices).issubset(saved_query_indices):
        raise ValueError("Checkpoint records fall outside its declared query sample")


def run_multi_gpu_workers(
    cfg: DictConfig,
    pending_tasks: list[dict],
    num_gpus: int,
    existing_records: list,
    query_indices: list[int],
    progress_callback: Optional[Callable[[list], None]] = None,
) -> list[FullLabelEvalResult]:
    """Launch isolated GPU workers and merge their atomic shard checkpoints."""
    task_shards = [pending_tasks[gpu_id::num_gpus] for gpu_id in range(num_gpus)]
    print(f"Splitting {len(pending_tasks)} pending queries across {num_gpus} GPUs:")
    for gpu_id, shard in enumerate(task_shards):
        print(f"  GPU {gpu_id}: {len(shard)} queries")

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="full_label_baselines_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.pkl"
        _atomic_pickle_dump(OmegaConf.to_container(cfg, resolve=True), config_path)
        processes = []
        output_paths = []
        worker_checkpoint_every = max(
            1, int(np.ceil(max(1, int(cfg.output.checkpoint_every)) / num_gpus))
        )
        for gpu_id, task_shard in enumerate(task_shards):
            tasks_path = temp_path / f"tasks_{gpu_id}.pkl"
            output_path = temp_path / f"records_{gpu_id}.pkl"
            _atomic_pickle_dump(task_shard, tasks_path)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            command = [
                sys.executable,
                "-m",
                "scripts.evaluate_full_label_baselines_worker",
                "--worker-id", str(gpu_id),
                "--config-path", str(config_path),
                "--tasks-path", str(tasks_path),
                "--output-path", str(output_path),
                "--checkpoint-every", str(worker_checkpoint_every),
            ]
            processes.append((
                gpu_id,
                subprocess.Popen(command, env=env, cwd=project_root),
            ))
            output_paths.append(output_path)

        last_progress_count = len(existing_records)
        try:
            while True:
                failed = [(gpu_id, process.returncode) for gpu_id, process in processes
                          if process.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Full-label GPU worker failure(s): {failed}")

                shard_records = []
                for output_path in output_paths:
                    if output_path.exists():
                        with open(output_path, "rb") as file:
                            shard_records.append(pickle.load(file))
                    else:
                        shard_records.append([])
                merged = _merge_records(existing_records, shard_records, query_indices)
                if len(merged) > last_progress_count:
                    print(f"Merged worker progress: {len(merged)}/{len(query_indices)} queries")
                    if progress_callback:
                        progress_callback(merged)
                    last_progress_count = len(merged)

                if all(process.poll() == 0 for _, process in processes):
                    break
                time.sleep(2)
        finally:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
        return merged


def save_checkpoint(records, query_indices, candidate_labels, cfg, run_args, upload: bool):
    k_values = sorted({int(k) for k in cfg.distractor_set.k_values})
    results = summarize_results(records, float(cfg.scoring.temperature), k_values)
    out_dir = save_eval_results(
        method="full_label_baselines",
        results=results,
        run_id_parts={
            "dataset": cfg.dataset.name,
            "eval": cfg.dataset.eval_split,
            "retrieval": cfg.dataset.retrieval_split,
            "n": len(query_indices),
        },
        args=run_args,
        extra={
            "records": records,
            "query_indices": query_indices,
            "candidate_labels": candidate_labels,
        },
        output_root=cfg.output.save_dir,
    )
    dataset_name = cfg.output.get("results_kaggle_dataset", None)
    if dataset_name and upload:
        kaggle_upload_eval_results(out_dir, dataset_name)
    return out_dir


@hydra.main(version_base=None, config_path="../configs", config_name="eval_full_label_baselines")
def main(cfg: DictConfig):
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
    if eval_dataset.class_names != retrieval_dataset.class_names:
        raise ValueError("Evaluation and retrieval class mappings differ")
    if eval_dataset.clip_model_name != retrieval_dataset.clip_model_name:
        raise ValueError("Evaluation and retrieval embeddings use different CLIP models")

    candidate_labels = list(eval_dataset.class_names)
    if len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError("Canonical candidate labels are not unique")
    k_values = sorted({int(k) for k in cfg.distractor_set.k_values})
    if not k_values or any(k < 2 or k > len(candidate_labels) for k in k_values):
        raise ValueError(f"distractor_set.k_values must be between 2 and {len(candidate_labels)}")

    query_indices = stratified_query_indices(
        eval_dataset.examples, cfg.scoring.get("num_queries", None), int(cfg.experiment.seed)
    )
    siglip_text, siglip_class_names, siglip_images, text_path, image_path = load_siglip_inputs(
        str(cfg.dataset.name),
        str(cfg.dataset.eval_split),
        cfg.dataset.get("siglip_kaggle_dataset", None),
        expected_class_names=eval_dataset.class_names,
        expected_example_ids=[example.image_path for example in eval_dataset.examples],
    )
    if siglip_class_names != list(eval_dataset.class_names):
        raise ValueError("SigLIP text-embedding class mapping differs from the evaluation dataset")
    if siglip_text.shape[0] != len(candidate_labels):
        raise ValueError("SigLIP text embeddings do not cover every candidate class")
    if siglip_images.shape[0] != len(eval_dataset):
        raise ValueError("SigLIP image embeddings do not align with the evaluation split")
    query_tasks = build_query_tasks(
        eval_dataset, query_indices, siglip_images, siglip_text, k_values
    )
    task_by_query = {task["query_idx"]: task for task in query_tasks}

    records = []
    resume_from = cfg.limits.get("resume_from", None)
    if resume_from:
        with open(resume_from, "rb") as file:
            payload = pickle.load(file)
        if payload["args"].get("schema_version") != 2:
            raise ValueError("Resume requires a schema-v2 full-label baseline or legacy pilot checkpoint")
        if payload["candidate_labels"] != candidate_labels:
            raise ValueError("Checkpoint candidate labels differ from the current dataset")
        saved_query_indices = payload["query_indices"]
        if not set(saved_query_indices).issubset(query_indices):
            raise ValueError("Checkpoint contains queries outside the current query sample")
        saved_model_cfg = payload["args"]["resolved_config"]["model"]
        if saved_model_cfg["idefics2_model"] != cfg.model.idefics2_model:
            raise ValueError("Checkpoint uses a different Idefics2 model")
        if bool(saved_model_cfg["load_in_8bit"]) != bool(cfg.model.load_in_8bit):
            raise ValueError("Checkpoint uses a different quantization setting")
        records = payload["records"]
        # Progress checkpoints declare the complete target sample but contain
        # only the subset completed when they were uploaded. Legacy pilot and
        # final checkpoints contain one record for every declared query.
        _validate_resume_records(records, saved_query_indices)
        for record in records:
            expected = task_by_query[record.query_idx]["distractor_class_indices"]
            if record.distractor_class_indices != expected:
                raise ValueError(f"Checkpoint distractor sets differ for query {record.query_idx}")

    completed = {record.query_idx for record in records}
    pending_tasks = [task for task in query_tasks if task["query_idx"] not in completed]
    print(
        f"Full-label baselines: {len(query_indices)} stratified {cfg.dataset.eval_split} queries, "
        f"{len(candidate_labels)} labels, K={k_values}, {len(completed)} completed"
    )

    requested_gpus = int(cfg.hardware.num_gpus)
    device, num_gpus, _ = _setup_device(requested_gpus)
    if pending_tasks and device != "cuda":
        raise RuntimeError("The full-label baseline evaluation requires CUDA GPUs")
    if pending_tasks and bool(cfg.hardware.get("require_all", True)) and num_gpus != requested_gpus:
        raise RuntimeError(f"Requested {requested_gpus} GPUs but only {num_gpus} are available")

    run_args = {
        "schema_version": 2,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "query_indices": query_indices,
        "candidate_count": len(candidate_labels),
        "k_values": k_values,
        "eval_num_examples": len(eval_dataset),
        "retrieval_num_examples": len(retrieval_dataset),
        "clip_model": eval_dataset.clip_model_name,
        "siglip_text_embeddings_sha256": _file_sha256(str(text_path)),
        "siglip_image_embeddings_sha256": _file_sha256(str(image_path)),
        "image_split_sha256": _file_sha256(cfg.dataset.image_split_path),
        "git_revision": _git_revision(),
    }

    last_saved_count = len(records)
    checkpoint_every = int(cfg.output.checkpoint_every)

    def save_progress(merged_records):
        nonlocal last_saved_count
        if (
            checkpoint_every > 0
            and len(merged_records) < len(query_indices)
            and len(merged_records) - last_saved_count >= checkpoint_every
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
        )

    if len(records) != len(query_indices):
        raise RuntimeError(f"Only completed {len(records)}/{len(query_indices)} queries")
    out_dir = save_checkpoint(records, query_indices, candidate_labels, cfg, run_args, upload=True)
    results = summarize_results(records, float(cfg.scoring.temperature), k_values)
    print("Unrestricted (all classes):")
    print(f"  Zero-shot top-1/top-5: {results['conditions']['zero_shot']['accuracy']:.2%} / "
          f"{results['conditions']['zero_shot']['top5_accuracy']:.2%}")
    print(f"  CLIP 1-shot top-1/top-5: {results['conditions']['clip_top1']['accuracy']:.2%} / "
          f"{results['conditions']['clip_top1']['top5_accuracy']:.2%}")
    print("Restricted fixed-score accuracy:")
    for k in k_values:
        conditions = results["restricted_by_k"][k]["conditions"]
        print(f"  K={k:3d}: zero={conditions['zero_shot']['accuracy']:.2%}, "
              f"CLIP 1-shot={conditions['clip_top1']['accuracy']:.2%}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
