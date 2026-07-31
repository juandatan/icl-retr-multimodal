"""Audit how well restricted label sets preserve full-space exemplar utility.

Each validation query retrieves a fixed CLIP candidate pool from the training
split. Idefics2 scores every candidate exemplar against all classes exactly
once. Nested query-specific SigLIP hard-label sets are then applied to those
fixed scores to compare each K with the full-class exemplar ranking.

Usage:
    python -m scripts.audit_label_space_k
    python -m scripts.audit_label_space_k audit.num_queries=10
    python -m scripts.audit_label_space_k limits.resume_from=/path/to/audit.pkl
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.dataclasses import LabelSpaceAuditResult
from src.data.distractor_sets import build_distractor_ranking
from src.data.loading import load_dataset, load_siglip_inputs
from src.models.idefics2_wrapper import Idefics2Wrapper
from src.utils.eval_utils import _json_safe
from src.utils.kaggle_utils import kaggle_upload_eval_results
from src.utils.label_space_audit import summarize_label_space_audit
from src.utils.runtime import (
    atomic_pickle_dump as _atomic_pickle_dump,
    file_sha256 as _file_sha256,
    git_revision as _git_revision,
    setup_device as _setup_device,
    stratified_query_indices,
)


def build_audit_tasks(
    eval_dataset,
    retrieval_dataset,
    query_indices: list[int],
    siglip_image_embeddings: np.ndarray,
    siglip_text_embeddings: np.ndarray,
    candidate_pool_size: int,
) -> list[dict]:
    """Fix the retrieval pool and complete hard-label ranking for each query."""
    tasks = []
    for query_idx in query_indices:
        query = eval_dataset.examples[query_idx]
        similarities = retrieval_dataset.clip_embeddings @ eval_dataset.clip_embeddings[query_idx]
        order = np.argsort(-similarities, kind="stable")[:candidate_pool_size]
        ranking = build_distractor_ranking(
            query_siglip_emb=siglip_image_embeddings[query_idx],
            class_text_embeddings=siglip_text_embeddings,
            true_class_idx=query.label,
            query_idx=query_idx,
        )
        tasks.append({
            "query_idx": int(query_idx),
            "candidate_indices": [int(idx) for idx in order],
            "candidate_similarities": [float(similarities[idx]) for idx in order],
            "ranked_distractor_class_indices": ranking.ranked_class_indices,
        })
    return tasks


def score_audit_tasks(
    cfg: DictConfig,
    tasks: list[dict],
    worker_id: int,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> list[LabelSpaceAuditResult]:
    """Score one task shard over the complete class space on local cuda:0."""
    if not tasks:
        if checkpoint_path:
            _atomic_pickle_dump([], checkpoint_path)
        return []
    eval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.eval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    retrieval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.retrieval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    candidate_labels = list(eval_dataset.class_names)
    device, _, _ = _setup_device(1)
    if device != "cuda":
        raise RuntimeError("The label-space audit requires a CUDA GPU")
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device="cuda:0",
        load_in_8bit=bool(cfg.model.load_in_8bit),
    )

    records = []
    for task in tqdm(
        tasks,
        desc=f"GPU {worker_id} full-200 audit",
        position=worker_id,
    ):
        query_idx = int(task["query_idx"])
        query, query_image = eval_dataset[query_idx]
        query_features = model.encode_full_label_scoring_images([query_image])
        zero_by_label = model.score_candidate_labels_with_image_features(
            query_features,
            [],
            candidate_labels,
            batch_size=int(cfg.scoring.candidate_batch_size),
        )

        candidate_scores = []
        candidate_classes = []
        for candidate_idx in task["candidate_indices"]:
            candidate, candidate_image = retrieval_dataset[candidate_idx]
            candidate_features = model.encode_full_label_scoring_images([candidate_image])
            combined = model.combine_full_label_scoring_image_features(
                candidate_features, query_features
            )
            scores_by_label = model.score_candidate_labels_with_image_features(
                combined,
                [candidate_labels[candidate.label]],
                candidate_labels,
                batch_size=int(cfg.scoring.candidate_batch_size),
            )
            candidate_classes.append(int(candidate.label))
            candidate_scores.append([float(scores_by_label[label]) for label in candidate_labels])

        records.append(LabelSpaceAuditResult(
            query_idx=query_idx,
            true_class_idx=int(query.label),
            zero_shot_scores=[float(zero_by_label[label]) for label in candidate_labels],
            candidate_indices=list(task["candidate_indices"]),
            candidate_class_indices=candidate_classes,
            candidate_similarities=list(task["candidate_similarities"]),
            candidate_scores=candidate_scores,
            ranked_distractor_class_indices=list(task["ranked_distractor_class_indices"]),
        ))
        if checkpoint_path and checkpoint_every > 0 and len(records) % checkpoint_every == 0:
            _atomic_pickle_dump(records, checkpoint_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if checkpoint_path:
        _atomic_pickle_dump(records, checkpoint_path)
    return records


def _merge_records(
    existing_records: list[LabelSpaceAuditResult],
    shard_records: list[list[LabelSpaceAuditResult]],
    query_indices: list[int],
) -> list[LabelSpaceAuditResult]:
    """Merge worker progress deterministically in the declared query order."""
    by_query = {record.query_idx: record for record in existing_records}
    for shard in shard_records:
        for record in shard:
            by_query[record.query_idx] = record
    return [by_query[idx] for idx in query_indices if idx in by_query]


def run_multi_gpu_workers(
    cfg: DictConfig,
    pending_tasks: list[dict],
    num_gpus: int,
    existing_records: list[LabelSpaceAuditResult],
    query_indices: list[int],
    progress_callback: Optional[Callable[[list[LabelSpaceAuditResult]], None]] = None,
) -> list[LabelSpaceAuditResult]:
    """Run isolated audit workers and merge their atomic shard checkpoints."""
    task_shards = [pending_tasks[gpu_id::num_gpus] for gpu_id in range(num_gpus)]
    print(f"Splitting {len(pending_tasks)} pending queries across {num_gpus} GPUs:")
    for gpu_id, shard in enumerate(task_shards):
        print(f"  GPU {gpu_id}: {len(shard)} queries")

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="label_space_k_audit_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.pkl"
        _atomic_pickle_dump(OmegaConf.to_container(cfg, resolve=True), config_path)
        processes = []
        output_paths = []
        worker_checkpoint_every = max(
            1,
            int(np.ceil(max(1, int(cfg.output.checkpoint_every)) / num_gpus)),
        )
        for gpu_id, shard in enumerate(task_shards):
            tasks_path = temp_path / f"tasks_{gpu_id}.pkl"
            output_path = temp_path / f"records_{gpu_id}.pkl"
            _atomic_pickle_dump(shard, tasks_path)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            command = [
                sys.executable,
                "-m",
                "scripts.audit_label_space_k_worker",
                "--worker-id",
                str(gpu_id),
                "--config-path",
                str(config_path),
                "--tasks-path",
                str(tasks_path),
                "--output-path",
                str(output_path),
                "--checkpoint-every",
                str(worker_checkpoint_every),
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
                if failed:
                    raise RuntimeError(f"Label-space audit worker failure(s): {failed}")

                shards = []
                for output_path in output_paths:
                    if output_path.exists():
                        with open(output_path, "rb") as file:
                            shards.append(pickle.load(file))
                    else:
                        shards.append([])
                merged = _merge_records(existing_records, shards, query_indices)
                if len(merged) > last_count:
                    print(f"Merged audit progress: {len(merged)}/{len(query_indices)} queries")
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

        final_shards = []
        for output_path in output_paths:
            with open(output_path, "rb") as file:
                final_shards.append(pickle.load(file))
        return _merge_records(existing_records, final_shards, query_indices)


def _write_outputs(path: Path, payload: dict):
    _atomic_pickle_dump(payload, path)
    summary_path = path.with_suffix(".json")
    temporary_path = summary_path.with_suffix(".json.tmp")
    with open(temporary_path, "w") as file:
        summary = {key: value for key, value in payload.items() if key != "records"}
        json.dump(_json_safe(summary), file, indent=2)
    os.replace(temporary_path, summary_path)


def _print_summary(summary: dict):
    print("\nK target-fidelity audit (all accuracy values evaluated over 200 classes):")
    header = "K  cover  spear  kendall  top-agree  selected-acc  gap  mean-regret"
    print(header)
    for k in summary["k_values"]:
        metrics = summary["by_k"][k]
        print(
            f"{k:3d} "
            f"{metrics['strongest_wrong_coverage']:6.2%} "
            f"{metrics['rank_correlation']['mean_spearman']:6.3f} "
            f"{metrics['rank_correlation']['mean_kendall']:7.3f} "
            f"{metrics['top_exemplar_agreement']:9.2%} "
            f"{metrics['selected_full_accuracy']:12.2%} "
            f"{metrics['full_accuracy_gap_to_oracle']:5.2%} "
            f"{metrics['full_margin_regret']['mean']:11.4f}"
        )


@hydra.main(version_base=None, config_path="../configs", config_name="audit_label_space_k")
def main(cfg: DictConfig):
    eval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.eval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    retrieval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.retrieval_split,
        image_split_path=cfg.dataset.image_split_path,
        embeddings_kaggle_dataset=cfg.dataset.embeddings_kaggle_dataset,
    )
    if eval_dataset.class_names != retrieval_dataset.class_names:
        raise ValueError("Evaluation and retrieval class mappings differ")
    if eval_dataset.clip_model_name != retrieval_dataset.clip_model_name:
        raise ValueError("Evaluation and retrieval CLIP models differ")

    class_count = len(eval_dataset.class_names)
    k_values = sorted(set(int(k) for k in cfg.audit.k_values) | {class_count})
    if any(k < 2 or k > class_count for k in k_values):
        raise ValueError(f"audit.k_values must be between 2 and {class_count}")
    pool_size = int(cfg.audit.candidate_pool_size)
    if pool_size <= 0 or pool_size > len(retrieval_dataset):
        raise ValueError("audit.candidate_pool_size is outside the retrieval dataset")

    query_indices = stratified_query_indices(
        eval_dataset.examples, int(cfg.audit.num_queries), int(cfg.experiment.seed)
    )
    siglip_text, siglip_class_names, siglip_images, text_path, image_path = load_siglip_inputs(
        str(cfg.dataset.name),
        str(cfg.dataset.eval_split),
        cfg.dataset.get("siglip_kaggle_dataset", None),
        expected_class_names=eval_dataset.class_names,
        expected_example_ids=[example.image_path for example in eval_dataset.examples],
    )
    if siglip_class_names != list(eval_dataset.class_names):
        raise ValueError("SigLIP class mapping differs from the evaluation dataset")
    if siglip_text.shape[0] != class_count:
        raise ValueError("SigLIP text embeddings do not cover the full class space")
    if siglip_images.shape[0] != len(eval_dataset):
        raise ValueError("SigLIP image embeddings do not align with the evaluation split")

    tasks = build_audit_tasks(
        eval_dataset,
        retrieval_dataset,
        query_indices,
        siglip_images,
        siglip_text,
        pool_size,
    )
    task_by_query = {task["query_idx"]: task for task in tasks}
    candidate_labels = list(eval_dataset.class_names)
    records = []

    resume_from = cfg.limits.get("resume_from", None)
    if resume_from:
        with open(resume_from, "rb") as file:
            previous = pickle.load(file)
        if previous.get("query_indices") != query_indices:
            raise ValueError("Resume checkpoint uses a different query sample")
        if previous.get("candidate_labels") != candidate_labels:
            raise ValueError("Resume checkpoint uses a different class mapping")
        previous_args = previous.get("args", {})
        if previous_args.get("candidate_pool_size") != pool_size:
            raise ValueError("Resume checkpoint uses a different candidate pool size")
        if previous_args.get("idefics2_model") != str(cfg.model.idefics2_model):
            raise ValueError("Resume checkpoint uses a different model")
        if bool(previous_args.get("load_in_8bit")) != bool(cfg.model.load_in_8bit):
            raise ValueError("Resume checkpoint uses different quantization")
        records = previous["records"]
        if len({record.query_idx for record in records}) != len(records):
            raise ValueError("Resume checkpoint contains duplicate queries")
        for record in records:
            if record.query_idx not in task_by_query:
                raise ValueError("Resume checkpoint contains an unexpected query")
            expected = task_by_query[record.query_idx]
            if record.candidate_indices != expected["candidate_indices"]:
                raise ValueError(f"Candidate pool changed for query {record.query_idx}")
            if record.ranked_distractor_class_indices != expected["ranked_distractor_class_indices"]:
                raise ValueError(f"Distractor ranking changed for query {record.query_idx}")

    run_id = (
        f"{cfg.dataset.name}_{cfg.dataset.eval_split}_{cfg.dataset.retrieval_split}_"
        f"pool_{pool_size}_n_{len(query_indices)}"
    )
    output_path = Path(cfg.output.save_dir) / "label_space_k_audit" / f"{run_id}.pkl"
    args = {
        "schema_version": 1,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "query_indices": query_indices,
        "candidate_pool_size": pool_size,
        "k_values": k_values,
        "class_count": class_count,
        "idefics2_model": str(cfg.model.idefics2_model),
        "load_in_8bit": bool(cfg.model.load_in_8bit),
        "clip_model": eval_dataset.clip_model_name,
        "image_split_sha256": _file_sha256(cfg.dataset.image_split_path),
        "siglip_text_embeddings_sha256": _file_sha256(str(text_path)),
        "siglip_image_embeddings_sha256": _file_sha256(str(image_path)),
        "git_revision": _git_revision(),
    }

    def checkpoint(current_records, complete: bool):
        ordered = {record.query_idx: record for record in current_records}
        current_records = [ordered[idx] for idx in query_indices if idx in ordered]
        summary = summarize_label_space_audit(current_records, k_values, class_count)
        payload = {
            "method": "label_space_k_audit",
            "run_id": run_id,
            "complete": complete,
            "completed_queries": len(current_records),
            "requested_queries": len(query_indices),
            "results": summary,
            "args": args,
            "query_indices": query_indices,
            "candidate_labels": candidate_labels,
            "records": current_records,
        }
        _write_outputs(output_path, payload)
        print(f"Checkpoint: {len(current_records)}/{len(query_indices)} queries -> {output_path}")
        dataset_name = cfg.output.get("results_kaggle_dataset", None)
        should_upload = complete or bool(cfg.output.get("upload_every_checkpoint", True))
        if dataset_name and should_upload:
            kaggle_upload_eval_results(
                output_path.parent,
                str(dataset_name),
                title="CUB-200 Label-Space Audit",
            )

    completed = {record.query_idx for record in records}
    pending = [task for task in tasks if task["query_idx"] not in completed]
    print(
        f"Label-space audit: {len(query_indices)} stratified {cfg.dataset.eval_split} queries, "
        f"{pool_size} exemplars/query, {class_count} labels, {len(completed)} completed"
    )
    requested_gpus = int(cfg.hardware.num_gpus)
    device, num_gpus, _ = _setup_device(requested_gpus)
    if pending and device != "cuda":
        raise RuntimeError("The label-space audit requires CUDA GPUs")
    if pending and bool(cfg.hardware.get("require_all", True)) and num_gpus != requested_gpus:
        raise RuntimeError(f"Requested {requested_gpus} GPUs but only {num_gpus} are available")

    last_saved_count = len(records)
    checkpoint_every = int(cfg.output.checkpoint_every)

    def save_progress(merged_records):
        nonlocal last_saved_count
        if (
            checkpoint_every > 0
            and len(merged_records) < len(query_indices)
            and len(merged_records) - last_saved_count >= checkpoint_every
        ):
            checkpoint(merged_records, complete=False)
            last_saved_count = len(merged_records)

    if pending:
        records = run_multi_gpu_workers(
            cfg,
            pending,
            num_gpus,
            records,
            query_indices,
            progress_callback=save_progress,
        )
    ordered = {record.query_idx: record for record in records}
    records = [ordered[idx] for idx in query_indices if idx in ordered]
    if len(records) != len(query_indices):
        raise RuntimeError(f"Only completed {len(records)}/{len(query_indices)} audit queries")
    checkpoint(records, complete=True)
    summary = summarize_label_space_audit(records, k_values, class_count)
    _print_summary(summary)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
