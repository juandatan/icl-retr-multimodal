"""Exhaustive candidate-pool oracle for option-free full-label evaluation.

Consumes a completed schema-v2 baseline produced by
evaluate_full_label_baselines.py. For each query it evaluates every exemplar in one
fixed unrestricted CLIP pool. Restricted mode scores only the union of the
nested K-way label sets; unrestricted mode scores all labels. The previously
scored CLIP top-1 exemplar is reused without another model call.
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

from scripts.evaluate_full_label_baselines import closed_set_metrics
from src.data.dataclasses import FullLabelOracleResult
from src.data.loading import load_dataset
from src.models.idefics2_wrapper import Idefics2Wrapper
from src.utils.eval_utils import save_eval_results
from src.utils.kaggle_utils import kaggle_upload_eval_results
from src.utils.runtime import (
    atomic_pickle_dump as _atomic_pickle_dump,
    file_sha256 as _file_sha256,
    git_revision as _git_revision,
    setup_device as _setup_device,
)


def load_baseline_results(path: str) -> dict:
    if not path:
        raise ValueError("input.baseline_results_path must point to a completed baseline pickle")
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if payload.get("method") not in {"full_label_baselines", "full_label_pilot"}:
        raise ValueError("Oracle input is not a full-label baseline result")
    if payload.get("args", {}).get("schema_version") != 2:
        raise ValueError("Oracle input must use the schema-v2 fixed-score baseline")
    if len(payload.get("records", [])) != len(payload.get("query_indices", [])):
        raise ValueError("Oracle input baseline is incomplete")
    return payload


def oracle_target_k_values(scope: str, restricted_k_values: list[int], class_count: int) -> list[int]:
    restricted = sorted(set(int(k) for k in restricted_k_values))
    if not restricted or any(k < 2 or k >= class_count for k in restricted):
        raise ValueError(
            f"Restricted oracle K values must be between 2 and {class_count - 1}"
        )
    if scope == "restricted":
        return restricted
    if scope == "unrestricted":
        return [class_count]
    if scope == "all":
        return sorted(set(restricted + [class_count]))
    raise ValueError("oracle.scope must be restricted, unrestricted, or all")


def build_oracle_tasks(
    baseline_records: list,
    eval_dataset,
    retrieval_dataset,
    candidate_pool_size: int,
) -> list[dict]:
    if candidate_pool_size <= 0:
        raise ValueError("oracle.candidate_pool_size must be positive")
    tasks = []
    for baseline_record in baseline_records:
        query_idx = baseline_record.query_idx
        similarities = retrieval_dataset.clip_embeddings @ eval_dataset.clip_embeddings[query_idx]
        order = np.argsort(-similarities, kind="stable")[:candidate_pool_size]
        candidate_indices = [int(idx) for idx in order]
        if not candidate_indices or candidate_indices[0] != baseline_record.clip_example_idx:
            raise ValueError(
                f"Rebuilt CLIP pool for query {query_idx} does not match its baseline top-1"
            )
        tasks.append({
            "query_idx": query_idx,
            "baseline_record": baseline_record,
            "candidate_indices": candidate_indices,
            "candidate_similarities": [float(similarities[idx]) for idx in candidate_indices],
        })
    return tasks


def _metrics_for_class_set(
    score_by_class: dict[int, float],
    true_class_idx: int,
    class_indices: list[int],
    temperature: float,
) -> dict:
    class_indices = sorted(class_indices)
    local_true_idx = class_indices.index(true_class_idx)
    metrics = closed_set_metrics(
        [score_by_class[class_idx] for class_idx in class_indices],
        local_true_idx,
        temperature,
    )
    metrics["predicted_class_idx"] = class_indices[metrics["predicted_class_idx"]]
    return metrics


def score_oracle_tasks(
    cfg: DictConfig,
    tasks: list[dict],
    worker_id: int,
    class_count: int,
    target_k_values: list[int],
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> list[FullLabelOracleResult]:
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
    if len(candidate_labels) != class_count:
        raise ValueError("Worker dataset class count differs from the baseline")

    device, _, _ = _setup_device(1)
    if device != "cuda":
        raise RuntimeError("Full-label oracle worker requires CUDA")
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device="cuda:0",
        load_in_8bit=cfg.model.load_in_8bit,
    )
    temperature = float(cfg.scoring.temperature)
    batch_size = int(cfg.scoring.candidate_batch_size)
    records = []

    for task in tqdm(tasks, desc=f"GPU {worker_id} oracle", position=worker_id):
        query_idx = int(task["query_idx"])
        baseline = task["baseline_record"]
        _, query_image = eval_dataset[query_idx]
        query_features = model.encode_full_label_scoring_images([query_image])

        class_sets = {}
        for k in target_k_values:
            class_sets[k] = (
                list(range(class_count))
                if k == class_count
                else list(baseline.distractor_class_indices[k])
            )
        classes_to_score = sorted({idx for values in class_sets.values() for idx in values})
        labels_to_score = [candidate_labels[idx] for idx in classes_to_score]

        correct_by_k = {k: [] for k in target_k_values}
        margin_by_k = {k: [] for k in target_k_values}
        rank_by_k = {k: [] for k in target_k_values}
        candidate_class_indices = []

        for pool_position, candidate_idx in enumerate(task["candidate_indices"]):
            candidate, candidate_image = retrieval_dataset[candidate_idx]
            candidate_class_indices.append(candidate.label)
            if pool_position == 0:
                score_by_class = {
                    class_idx: float(baseline.clip_scores[class_idx])
                    for class_idx in classes_to_score
                }
            else:
                candidate_features = model.encode_full_label_scoring_images([candidate_image])
                combined_features = model.combine_full_label_scoring_image_features(
                    candidate_features, query_features
                )
                scores_by_label = model.score_candidate_labels_with_image_features(
                    combined_features,
                    [candidate_labels[candidate.label]],
                    labels_to_score,
                    batch_size=batch_size,
                )
                score_by_class = {
                    class_idx: scores_by_label[candidate_labels[class_idx]]
                    for class_idx in classes_to_score
                }

            for k in target_k_values:
                metrics = _metrics_for_class_set(
                    score_by_class, baseline.true_class_idx, class_sets[k], temperature
                )
                correct_by_k[k].append(bool(metrics["correct"]))
                margin_by_k[k].append(float(metrics["true_margin"]))
                rank_by_k[k].append(int(metrics["true_rank"]))

        records.append(FullLabelOracleResult(
            query_idx=query_idx,
            true_class_idx=baseline.true_class_idx,
            candidate_indices=list(task["candidate_indices"]),
            candidate_class_indices=candidate_class_indices,
            candidate_similarities=list(task["candidate_similarities"]),
            candidate_correct_by_k=correct_by_k,
            candidate_margin_by_k=margin_by_k,
            candidate_true_rank_by_k=rank_by_k,
        ))
        if checkpoint_path and checkpoint_every > 0 and len(records) % checkpoint_every == 0:
            _atomic_pickle_dump(records, checkpoint_path)

    if checkpoint_path:
        _atomic_pickle_dump(records, checkpoint_path)
    return records


def summarize_oracle_results(records: list, target_k_values: list[int], class_count: int) -> dict:
    if not records:
        raise ValueError("Cannot summarize empty oracle records")
    by_k = {}
    oracle_correct_by_k = {}
    for k in target_k_values:
        oracle_correct = [any(record.candidate_correct_by_k[k]) for record in records]
        oracle_correct_by_k[k] = oracle_correct
        clip_correct = [record.candidate_correct_by_k[k][0] for record in records]
        best_positions = [
            int(np.argmax(record.candidate_margin_by_k[k])) for record in records
        ]
        by_class = defaultdict(list)
        for record, correct in zip(records, oracle_correct):
            by_class[record.true_class_idx].append(correct)
        by_k[k] = {
            "oracle_accuracy": float(np.mean(oracle_correct)),
            "oracle_correct": int(sum(oracle_correct)),
            "clip_top1_accuracy": float(np.mean(clip_correct)),
            "accuracy_gain_over_clip_top1": float(np.mean(oracle_correct) - np.mean(clip_correct)),
            "mean_per_class_accuracy": float(np.mean([
                np.mean(values) for values in by_class.values()
            ])),
            "mean_best_true_margin": float(np.mean([
                record.candidate_margin_by_k[k][position]
                for record, position in zip(records, best_positions)
            ])),
            "mean_best_true_rank": float(np.mean([
                min(record.candidate_true_rank_by_k[k]) for record in records
            ])),
            "oracle_selected_same_class_rate": float(np.mean([
                record.candidate_class_indices[position] == record.true_class_idx
                for record, position in zip(records, best_positions)
            ])),
            "pool_has_true_class_rate": float(np.mean([
                record.true_class_idx in record.candidate_class_indices for record in records
            ])),
            "total": len(records),
        }

    restricted = [k for k in target_k_values if k < class_count]
    for record_idx in range(len(records)):
        correctness = [oracle_correct_by_k[k][record_idx] for k in restricted]
        if any(not earlier and later for earlier, later in zip(correctness, correctness[1:])):
            raise ValueError(f"Non-monotonic restricted oracle result at record {record_idx}")

    primary_k = class_count if class_count in by_k else max(target_k_values)
    primary = by_k[primary_k]
    return {
        "accuracy": primary["oracle_accuracy"],
        "mean_per_class_accuracy": primary["mean_per_class_accuracy"],
        "correct": primary["oracle_correct"],
        "total": primary["total"],
        "candidate_pool_size": len(records[0].candidate_indices),
        "target_k_values": target_k_values,
        "oracle_definition": "candidate_existence_via_best_true_vs_best_wrong_margin",
        "by_k": by_k,
    }


def _merge_records(existing_records, shard_records, query_indices):
    by_query = {record.query_idx: record for record in existing_records}
    for shard in shard_records:
        for record in shard:
            by_query[record.query_idx] = record
    return [by_query[idx] for idx in query_indices if idx in by_query]


def run_oracle_workers(
    cfg: DictConfig,
    pending_tasks: list[dict],
    num_gpus: int,
    class_count: int,
    target_k_values: list[int],
    existing_records: list,
    query_indices: list[int],
    progress_callback: Optional[Callable[[list], None]] = None,
) -> list[FullLabelOracleResult]:
    task_shards = [pending_tasks[gpu_id::num_gpus] for gpu_id in range(num_gpus)]
    print(f"Splitting {len(pending_tasks)} oracle queries across {num_gpus} GPUs:")
    for gpu_id, shard in enumerate(task_shards):
        print(f"  GPU {gpu_id}: {len(shard)} queries")

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="full_label_oracle_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.pkl"
        _atomic_pickle_dump(OmegaConf.to_container(cfg, resolve=True), config_path)
        processes = []
        output_paths = []
        per_worker_checkpoint = max(
            1, int(np.ceil(max(1, int(cfg.output.checkpoint_every)) / num_gpus))
        )
        for gpu_id, shard in enumerate(task_shards):
            tasks_path = temp_path / f"tasks_{gpu_id}.pkl"
            output_path = temp_path / f"records_{gpu_id}.pkl"
            _atomic_pickle_dump(shard, tasks_path)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            command = [
                sys.executable, "-m", "scripts.evaluate_full_label_oracle_worker",
                "--worker-id", str(gpu_id),
                "--config-path", str(config_path),
                "--tasks-path", str(tasks_path),
                "--output-path", str(output_path),
                "--class-count", str(class_count),
                "--target-k-values", *[str(k) for k in target_k_values],
                "--checkpoint-every", str(per_worker_checkpoint),
            ]
            processes.append((
                gpu_id,
                subprocess.Popen(command, env=env, cwd=project_root),
            ))
            output_paths.append(output_path)

        last_count = len(existing_records)
        try:
            while True:
                failed = [(gpu_id, process.returncode) for gpu_id, process in processes
                          if process.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Oracle GPU worker failure(s): {failed}")
                shards = []
                for output_path in output_paths:
                    if output_path.exists():
                        with open(output_path, "rb") as file:
                            shards.append(pickle.load(file))
                    else:
                        shards.append([])
                merged = _merge_records(existing_records, shards, query_indices)
                if len(merged) > last_count:
                    print(f"Merged oracle progress: {len(merged)}/{len(query_indices)} queries")
                    if progress_callback:
                        progress_callback(merged)
                    last_count = len(merged)
                if all(process.poll() == 0 for _, process in processes):
                    break
                time.sleep(2)
        finally:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
        return merged


def save_oracle_checkpoint(
    records, query_indices, cfg, run_args, target_k_values, class_count, upload: bool
):
    results = summarize_oracle_results(records, target_k_values, class_count)
    scope = str(cfg.oracle.scope)
    method = f"full_label_oracle_{scope}"
    out_dir = save_eval_results(
        method=method,
        results=results,
        run_id_parts={
            "dataset": cfg.dataset.name,
            "eval": cfg.dataset.eval_split,
            "retrieval": cfg.dataset.retrieval_split,
            "scope": scope,
            "pool": int(cfg.oracle.candidate_pool_size),
            "n": len(query_indices),
        },
        args=run_args,
        extra={"records": records, "query_indices": query_indices},
        output_root=cfg.output.save_dir,
    )
    dataset_name = cfg.output.get("results_kaggle_dataset", None)
    if dataset_name and upload:
        kaggle_upload_eval_results(out_dir, dataset_name)
    return out_dir


@hydra.main(version_base=None, config_path="../configs", config_name="eval_full_label_oracle")
def main(cfg: DictConfig):
    baseline_path = cfg.input.get("baseline_results_path", None)
    baseline = load_baseline_results(baseline_path)
    baseline_records = baseline["records"]
    query_indices = baseline["query_indices"]
    candidate_labels = baseline["candidate_labels"]
    class_count = len(candidate_labels)
    target_k_values = oracle_target_k_values(
        str(cfg.oracle.scope), list(cfg.oracle.k_values), class_count
    )
    for record in baseline_records:
        for k in target_k_values:
            if k < class_count and k not in record.distractor_class_indices:
                raise ValueError(f"Baseline record {record.query_idx} lacks K={k}")

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
    if len(eval_dataset.class_names) != class_count:
        raise ValueError("Baseline class count differs from the current dataset")
    current_candidate_labels = list(eval_dataset.class_names)
    if current_candidate_labels != candidate_labels:
        raise ValueError("Baseline candidate labels differ from the current dataset")
    baseline_model_cfg = baseline["args"]["resolved_config"]["model"]
    if baseline_model_cfg["idefics2_model"] != cfg.model.idefics2_model:
        raise ValueError("Oracle and baseline must use the same Idefics2 model")
    if bool(baseline_model_cfg["load_in_8bit"]) != bool(cfg.model.load_in_8bit):
        raise ValueError("Oracle and baseline must use the same quantization setting")

    all_tasks = build_oracle_tasks(
        baseline_records, eval_dataset, retrieval_dataset, int(cfg.oracle.candidate_pool_size)
    )
    task_by_query = {task["query_idx"]: task for task in all_tasks}
    records = []
    resume_from = cfg.limits.get("resume_from", None)
    if resume_from:
        with open(resume_from, "rb") as file:
            payload = pickle.load(file)
        args = payload.get("args", {})
        if args.get("schema_version") != 1:
            raise ValueError("Unsupported oracle checkpoint schema")
        if args.get("baseline_results_sha256") != _file_sha256(str(baseline_path)):
            raise ValueError("Oracle checkpoint uses a different baseline result")
        if args.get("target_k_values") != target_k_values:
            raise ValueError("Oracle checkpoint target K values differ")
        if args.get("candidate_pool_size") != int(cfg.oracle.candidate_pool_size):
            raise ValueError("Oracle checkpoint candidate-pool size differs")
        if args.get("idefics2_model") != str(cfg.model.idefics2_model):
            raise ValueError("Oracle checkpoint model differs")
        if bool(args.get("load_in_8bit")) != bool(cfg.model.load_in_8bit):
            raise ValueError("Oracle checkpoint quantization setting differs")
        if payload.get("query_indices") != query_indices:
            raise ValueError("Oracle checkpoint query sample differs")
        records = payload["records"]
        for record in records:
            expected_indices = task_by_query[record.query_idx]["candidate_indices"]
            if record.candidate_indices != expected_indices:
                raise ValueError(
                    f"Oracle checkpoint candidate pool differs for query {record.query_idx}"
                )

    completed = {record.query_idx for record in records}
    pending_tasks = [task_by_query[idx] for idx in query_indices if idx not in completed]
    requested_gpus = int(cfg.hardware.num_gpus)
    device, num_gpus, _ = _setup_device(requested_gpus)
    if pending_tasks and device != "cuda":
        raise RuntimeError("Full-label oracle requires CUDA GPUs")
    if pending_tasks and bool(cfg.hardware.get("require_all", True)) and num_gpus != requested_gpus:
        raise RuntimeError(f"Requested {requested_gpus} GPUs but only {num_gpus} are available")

    run_args = {
        "schema_version": 1,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "baseline_results_sha256": _file_sha256(str(baseline_path)),
        "baseline_run_id": baseline["run_id"],
        "target_k_values": target_k_values,
        "candidate_pool_size": int(cfg.oracle.candidate_pool_size),
        "class_count": class_count,
        "idefics2_model": str(cfg.model.idefics2_model),
        "load_in_8bit": bool(cfg.model.load_in_8bit),
        "query_indices": query_indices,
        "git_revision": _git_revision(),
    }
    checkpoint_every = int(cfg.output.checkpoint_every)
    last_saved_count = len(records)

    def save_progress(merged):
        nonlocal last_saved_count
        if (
            checkpoint_every > 0
            and len(merged) < len(query_indices)
            and len(merged) - last_saved_count >= checkpoint_every
        ):
            save_oracle_checkpoint(
                merged, query_indices, cfg, run_args, target_k_values, class_count,
                upload=bool(cfg.output.get("upload_every_checkpoint", True)),
            )
            last_saved_count = len(merged)

    if pending_tasks:
        records = run_oracle_workers(
            cfg, pending_tasks, num_gpus, class_count, target_k_values,
            records, query_indices, save_progress,
        )
    if len(records) != len(query_indices):
        raise RuntimeError(f"Only completed {len(records)}/{len(query_indices)} oracle queries")

    out_dir = save_oracle_checkpoint(
        records, query_indices, cfg, run_args, target_k_values, class_count, upload=True
    )
    results = summarize_oracle_results(records, target_k_values, class_count)
    for k in target_k_values:
        summary = results["by_k"][k]
        print(
            f"K={k}: oracle={summary['oracle_accuracy']:.2%}, "
            f"CLIP top-1={summary['clip_top1_accuracy']:.2%}, "
            f"gain={summary['accuracy_gain_over_clip_top1']:+.2%}"
        )
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
