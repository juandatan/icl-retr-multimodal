"""Export aligned validation predictions from pooled and frozen rerankers.

The training checkpoints contain weights and resolved configuration, not the
candidate scores themselves.  This script replays inference on the teacher
artifact and writes both a human-readable query summary and a lossless NPZ for
paired selector/complementarity analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.idefics2_probe_dataset import (  # noqa: E402
    FrozenIdefics2ProbeDataset,
    collate_frozen_idefics2_probe_queries,
)
from src.data.reranker_dataset import (  # noqa: E402
    RerankerTeacherDataset,
    collate_reranker_queries,
)
from src.models.idefics2_probe import FrozenIdefics2UtilityProbe  # noqa: E402
from src.models.reranker import LabelAwareReranker, RerankerConfig  # noqa: E402
from src.utils.reranker_metrics import reranker_selection_metrics  # noqa: E402
from src.utils.runtime import file_sha256  # noqa: E402


POOLED_INPUTS = (
    "query_clip",
    "candidate_clip",
    "query_siglip",
    "candidate_siglip",
    "candidate_label_siglip",
    "clip_similarities",
    "retrieval_ranks",
    "candidate_mask",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--pooled-checkpoint", required=True, type=Path)
    parser.add_argument("--frozen-checkpoint", required=True, type=Path)
    parser.add_argument("--probe-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--allow-non-best",
        action="store_true",
        help="Allow last.pt or another checkpoint whose epoch is not best_epoch.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(path: Path, *, allow_non_best: bool) -> dict[str, Any]:
    # Checkpoints are trusted local training outputs and contain config/history
    # dictionaries in addition to tensors, hence weights_only=False is required.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "model_config", "epoch"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing fields: {sorted(missing)}")
    best_epoch = checkpoint.get("best_epoch")
    if best_epoch is None:
        best_epoch = checkpoint.get("early_stopping", {}).get("best_epoch")
    if (
        best_epoch is not None
        and int(checkpoint["epoch"]) != int(best_epoch)
        and not allow_non_best
    ):
        raise ValueError(
            f"{path} stores epoch {checkpoint['epoch']}, but its selected epoch is "
            f"{best_epoch}. Pass best.pt, or explicitly use --allow-non-best."
        )
    return checkpoint


def _verify_artifact(
    checkpoint: dict[str, Any], artifact_path: Path, actual_hash: str | None
) -> None:
    expected = checkpoint.get("provenance", {}).get("artifact_sha256")
    if expected is not None and actual_hash != expected:
        raise ValueError(
            "Checkpoint teacher-artifact hash does not match --artifact: "
            f"expected {expected}, got {actual_hash}"
        )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _pooled_model(checkpoint: dict[str, Any], device: torch.device):
    valid_names = {field.name for field in fields(RerankerConfig)}
    config_values = {
        name: value
        for name, value in checkpoint["model_config"].items()
        if name in valid_names
    }
    config = RerankerConfig(**config_values)
    if config.architecture == "visual_token_cross_encoder":
        raise ValueError(
            "This paired exporter expects a pooled reranker checkpoint, not "
            "visual_token_cross_encoder"
        )
    model = LabelAwareReranker(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _frozen_model(checkpoint: dict[str, Any], device: torch.device):
    config = checkpoint["model_config"]
    model = FrozenIdefics2UtilityProbe(
        input_dim=int(config["input_dim"]),
        architecture=str(config.get("architecture", "linear")),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def _infer_pooled(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "query_idx",
            "candidate_indices",
            "candidate_class_indices",
            "candidate_mask",
            "clip_similarities",
            "teacher_margins",
            "teacher_probabilities",
            "teacher_correct",
            "scores",
        )
    }
    for batch in loader:
        batch = _move(batch, device)
        scores = model(**{name: batch[name] for name in POOLED_INPUTS})
        values = {name: batch[name] for name in collected if name != "scores"}
        values["scores"] = scores
        for name, value in values.items():
            collected[name].append(value.detach().cpu().numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}


@torch.inference_mode()
def _infer_frozen(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "query_idx",
            "candidate_indices",
            "candidate_mask",
            "teacher_margins",
            "teacher_correct",
            "scores",
        )
    }
    for batch in loader:
        batch = _move(batch, device)
        scores = model(batch["pair_representations"], batch["candidate_mask"])
        values = {name: batch[name] for name in collected if name != "scores"}
        values["scores"] = scores
        for name, value in values.items():
            collected[name].append(value.detach().cpu().numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}


def _validate_alignment(
    pooled: dict[str, np.ndarray], frozen: dict[str, np.ndarray]
) -> None:
    for name in ("query_idx", "candidate_indices", "candidate_mask"):
        if not np.array_equal(pooled[name], frozen[name]):
            raise ValueError(f"Pooled and frozen inference are misaligned at {name}")
    for name in ("teacher_margins", "teacher_correct"):
        if not np.allclose(pooled[name], frozen[name]):
            raise ValueError(f"Pooled and frozen teacher values differ at {name}")


def _candidate_ranks(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a zero-based rank at each candidate position; padding is -1."""
    ranks = np.full(scores.shape, -1, dtype=np.int16)
    for row, valid in enumerate(mask.astype(bool)):
        valid_positions = np.flatnonzero(valid)
        order = valid_positions[
            np.argsort(-scores[row, valid_positions], kind="stable")
        ]
        ranks[row, order] = np.arange(len(order), dtype=np.int16)
    return ranks


def _teacher_output_classes(dataset: RerankerTeacherDataset) -> np.ndarray:
    max_candidates = max(len(record.candidate_indices) for record in dataset.records)
    outputs = np.full((len(dataset), max_candidates), -1, dtype=np.int16)
    for row, record in enumerate(dataset.records):
        scores = np.asarray(record.candidate_scores)
        labels = np.asarray(record.label_class_indices)
        outputs[row, : len(scores)] = labels[np.argmax(scores, axis=1)]
    return outputs


def _selection_fields(
    prefix: str,
    scores: np.ndarray,
    arrays: dict[str, np.ndarray],
    output_classes: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = arrays["candidate_mask"].astype(bool)
    selected = np.where(mask, scores, -np.inf).argmax(axis=1)
    rows = np.arange(len(selected))
    return {
        f"{prefix}_selected_position": selected.astype(np.int16),
        f"{prefix}_selected_candidate_idx": arrays["candidate_indices"][rows, selected],
        f"{prefix}_selected_candidate_class_idx": arrays[
            "candidate_class_indices"
        ][rows, selected],
        f"{prefix}_selected_score": scores[rows, selected],
        f"{prefix}_selected_teacher_output_class_idx": output_classes[rows, selected],
        f"{prefix}_selected_teacher_correct": arrays["teacher_correct"][rows, selected],
        f"{prefix}_selected_teacher_margin": arrays["teacher_margins"][rows, selected],
        f"{prefix}_selected_teacher_probability": arrays[
            "teacher_probabilities"
        ][rows, selected],
    }


def _metrics(scores: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return reranker_selection_metrics(
        scores,
        arrays["teacher_margins"],
        arrays["teacher_margins"],
        arrays["teacher_correct"],
        arrays["candidate_mask"],
    )


def _write_query_csv(path: Path, exports: dict[str, np.ndarray]) -> None:
    selected_keys = [
        key for key, value in exports.items()
        if "_selected_" in key and np.asarray(value).ndim == 1
    ]
    fieldnames = [
        "query_split",
        "query_idx",
        "true_class_idx",
        "candidate_count",
        *selected_keys,
        "selectors_agree",
        "either_selector_correct",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        count = len(exports["query_idx"])
        for row in range(count):
            values: dict[str, Any] = {
                "query_split": str(exports["query_split"]),
                "query_idx": int(exports["query_idx"][row]),
                "true_class_idx": int(exports["true_class_idx"][row]),
                "candidate_count": int(exports["candidate_mask"][row].sum()),
            }
            for key in selected_keys:
                value = exports[key][row]
                values[key] = value.item() if hasattr(value, "item") else value
            values["selectors_agree"] = bool(
                exports["pooled_selected_position"][row]
                == exports["frozen_selected_position"][row]
            )
            values["either_selector_correct"] = bool(
                exports["pooled_selected_teacher_correct"][row]
                or exports["frozen_selected_teacher_correct"][row]
            )
            writer.writerow(values)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    device = _device(args.device)
    pooled_checkpoint = _load_checkpoint(
        args.pooled_checkpoint, allow_non_best=args.allow_non_best
    )
    frozen_checkpoint = _load_checkpoint(
        args.frozen_checkpoint, allow_non_best=args.allow_non_best
    )
    artifact_hash = file_sha256(args.artifact)
    _verify_artifact(pooled_checkpoint, args.artifact, artifact_hash)
    _verify_artifact(frozen_checkpoint, args.artifact, artifact_hash)

    with args.artifact.open("rb") as file:
        artifact = pickle.load(file)
    pooled_data = RerankerTeacherDataset(
        artifact,
        split=args.split,
        target="margin",
    )
    frozen_data = FrozenIdefics2ProbeDataset(
        artifact,
        args.probe_cache,
        split=args.split,
        target="margin",
    )
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    pooled_loader = DataLoader(
        pooled_data, collate_fn=collate_reranker_queries, **loader_options
    )
    frozen_loader = DataLoader(
        frozen_data,
        collate_fn=collate_frozen_idefics2_probe_queries,
        **loader_options,
    )
    print(
        f"Exporting {len(pooled_data):,} aligned {args.split} queries on {device}",
        flush=True,
    )
    pooled = _infer_pooled(_pooled_model(pooled_checkpoint, device), pooled_loader, device)
    frozen = _infer_frozen(_frozen_model(frozen_checkpoint, device), frozen_loader, device)
    _validate_alignment(pooled, frozen)

    output_classes = _teacher_output_classes(pooled_data)
    true_classes = np.asarray(
        [int(record.true_class_idx) for record in pooled_data.records], dtype=np.int16
    )
    exports: dict[str, Any] = {
        "query_split": args.split,
        "query_idx": pooled["query_idx"],
        "true_class_idx": true_classes,
        "candidate_indices": pooled["candidate_indices"],
        "candidate_class_indices": pooled["candidate_class_indices"],
        "candidate_mask": pooled["candidate_mask"],
        "clip_similarities": pooled["clip_similarities"],
        "teacher_output_class_indices": output_classes,
        "teacher_margins": pooled["teacher_margins"],
        "teacher_probabilities": pooled["teacher_probabilities"],
        "teacher_correct": pooled["teacher_correct"],
        "clip_ranks": _candidate_ranks(
            pooled["clip_similarities"], pooled["candidate_mask"]
        ),
        "pooled_scores": pooled["scores"],
        "pooled_ranks": _candidate_ranks(pooled["scores"], pooled["candidate_mask"]),
        "frozen_scores": frozen["scores"],
        "frozen_ranks": _candidate_ranks(frozen["scores"], frozen["candidate_mask"]),
    }
    exports.update(
        _selection_fields(
            "clip", pooled["clip_similarities"], pooled, output_classes
        )
    )
    exports.update(
        _selection_fields("pooled", pooled["scores"], pooled, output_classes)
    )
    exports.update(
        _selection_fields("frozen", frozen["scores"], pooled, output_classes)
    )

    pooled_metrics = _metrics(pooled["scores"], pooled)
    frozen_metrics = _metrics(frozen["scores"], pooled)
    clip_metrics = _metrics(pooled["clip_similarities"], pooled)
    agree = float(np.mean(
        exports["pooled_selected_position"] == exports["frozen_selected_position"]
    ))
    two_selector_oracle = float(np.mean(
        exports["pooled_selected_teacher_correct"]
        | exports["frozen_selected_teacher_correct"]
    ))
    metadata = {
        "schema_version": 1,
        "method": "aligned_reranker_prediction_export",
        "split": args.split,
        "query_count": len(pooled_data),
        "candidate_width": int(pooled["candidate_mask"].shape[1]),
        "device": str(device),
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": artifact_hash,
        "pooled_checkpoint": str(args.pooled_checkpoint.resolve()),
        "pooled_checkpoint_sha256": file_sha256(args.pooled_checkpoint),
        "pooled_epoch": int(pooled_checkpoint["epoch"]),
        "frozen_checkpoint": str(args.frozen_checkpoint.resolve()),
        "frozen_checkpoint_sha256": file_sha256(args.frozen_checkpoint),
        "frozen_epoch": int(frozen_checkpoint["epoch"]),
        "pooled_experiment_name": pooled_checkpoint.get("experiment_name"),
        "frozen_experiment_name": frozen_checkpoint.get("experiment_name"),
        "pooled_metrics": pooled_metrics,
        "frozen_metrics": frozen_metrics,
        "clip_metrics": clip_metrics,
        "selector_agreement": agree,
        "two_selector_correctness_oracle": two_selector_oracle,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "candidate_predictions.npz",
        **{key: value for key, value in exports.items() if key != "query_split"},
    )
    _write_query_csv(args.output_dir / "query_predictions.csv", exports)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        "Export complete\n"
        f"  pooled accuracy {pooled_metrics['restricted_selected_accuracy']:.3%}\n"
        f"  frozen accuracy {frozen_metrics['restricted_selected_accuracy']:.3%}\n"
        f"  same exemplar {agree:.3%}\n"
        f"  either selected exemplar correct {two_selector_oracle:.3%}\n"
        f"  output {args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
