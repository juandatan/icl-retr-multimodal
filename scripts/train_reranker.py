"""Train and validate a frozen-feature exemplar reranker."""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reranker_dataset import (  # noqa: E402
    RerankerTeacherDataset,
    collate_reranker_queries,
)
from src.losses.listwise import MultiplePositiveListwiseLoss  # noqa: E402
from src.losses.pairwise_ranking import PairwiseRankingLoss  # noqa: E402
from src.losses.pointwise import MaskedHuberLoss, MaskedSoftLabelBCELoss  # noqa: E402
from src.models.reranker import LabelAwareReranker, RerankerConfig  # noqa: E402
from src.utils.reranker_metrics import reranker_selection_metrics  # noqa: E402
from src.utils.runtime import file_sha256, git_revision  # noqa: E402


MODEL_INPUTS = (
    "query_clip",
    "candidate_clip",
    "query_siglip",
    "candidate_siglip",
    "candidate_label_siglip",
    "clip_similarities",
    "retrieval_ranks",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _limit(dataset, count: int | None):
    if count is None or int(count) >= len(dataset):
        return dataset
    if int(count) <= 0:
        raise ValueError("Query limits must be positive or null")
    return Subset(dataset, range(int(count)))


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _build_objective(cfg: DictConfig):
    name = str(cfg.objective.name)
    if name == "pointwise_bce":
        return MaskedSoftLabelBCELoss()
    if name == "pairwise":
        return PairwiseRankingLoss(
            min_target_gap=float(cfg.objective.pairwise_min_target_gap),
            score_temperature=float(cfg.objective.pairwise_score_temperature),
        )
    if name == "listwise_correctness":
        return MultiplePositiveListwiseLoss()
    if name == "huber":
        return MaskedHuberLoss(delta=float(cfg.objective.huber_delta))
    raise ValueError(f"Unsupported objective: {name}")


def _validate_target_objective(target: str, objective: str) -> None:
    bounded = {
        "true_probability",
        "bounded_margin",
        "bounded_incremental_margin",
        "normalized_incremental_probability",
    }
    if objective == "pointwise_bce" and target not in bounded:
        raise ValueError(
            f"pointwise_bce requires a [0,1] target; got {target!r}"
        )


def _model_scores(model, batch):
    return model(**{name: batch[name] for name in MODEL_INPUTS})


def _objective_targets(batch: dict[str, Any], objective_name: str) -> torch.Tensor:
    if objective_name == "listwise_correctness":
        return batch["teacher_correct"]
    return batch["targets"]


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _print_epoch_metrics(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    metrics: dict[str, float],
    improved: bool,
    best_epoch: int,
    epochs_without_improvement: int,
) -> None:
    """Print changing validation metrics in a compact human-readable layout."""
    marker = "  new best" if improved else ""
    print(f"\nEpoch {epoch}/{total_epochs}{marker}")
    print(
        f"  Loss       train {train_loss:.6f}   "
        f"validation {metrics['loss']:.6f}"
    )
    print(
        "  Selection  "
        f"accuracy {metrics['restricted_selected_accuracy']:.3%}   "
        f"margin {metrics['mean_selected_margin']:.6f}   "
        f"regret {metrics['mean_margin_regret']:.6f}"
    )
    print(
        "  Ranking    "
        f"Spearman {metrics['mean_margin_spearman']:.6f}   "
        f"margin-oracle {metrics['margin_oracle_agreement']:.3%}   "
        f"target-oracle {metrics['target_oracle_agreement']:.3%}"
    )
    print(f"  Target     regret {metrics['mean_target_regret']:.6f}")
    print(
        f"  Checkpoint best epoch {best_epoch}   "
        f"epochs without improvement {epochs_without_improvement}"
    )


def _print_final_summary(
    *,
    stop_reason: str,
    stop_epoch: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    early_stopping_patience: int,
    learning_rate: float,
) -> None:
    """Print constants and the complete selected-checkpoint result once."""
    print(f"\nTraining complete: {stop_reason}")
    print(
        f"  Stopped at epoch {stop_epoch}; selected epoch {best_epoch}; "
        f"patience {early_stopping_patience}; learning rate {learning_rate:g}"
    )
    print(
        "  Fixed K=32 baselines  "
        f"CLIP top-1 {best_metrics['restricted_clip_top1_accuracy']:.3%}   "
        f"pool oracle {best_metrics['restricted_pool_oracle_accuracy']:.3%}"
    )
    print("  Selected-checkpoint metrics")
    print(
        f"    loss {best_metrics['loss']:.6f}   "
        f"accuracy {best_metrics['restricted_selected_accuracy']:.3%}"
    )
    print(
        f"    selected margin {best_metrics['mean_selected_margin']:.6f}   "
        f"margin regret {best_metrics['mean_margin_regret']:.6f}   "
        f"Spearman {best_metrics['mean_margin_spearman']:.6f}"
    )
    print(
        f"    margin-oracle {best_metrics['margin_oracle_agreement']:.3%}   "
        f"target-oracle {best_metrics['target_oracle_agreement']:.3%}   "
        f"target regret {best_metrics['mean_target_regret']:.6f}"
    )


@torch.no_grad()
def evaluate(
    model, loader, objective, objective_name: str, device, use_amp: bool
) -> dict[str, float]:
    model.eval()
    losses = []
    collected = {name: [] for name in (
        "scores", "targets", "teacher_margins", "teacher_correct", "candidate_mask"
    )}
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            scores = _model_scores(model, batch)
            loss = objective(
                scores,
                _objective_targets(batch, objective_name),
                batch["candidate_mask"],
            )
        losses.append(float(loss.item()))
        values = {
            "scores": scores,
            "targets": batch["targets"],
            "teacher_margins": batch["teacher_margins"],
            "teacher_correct": batch["teacher_correct"],
            "candidate_mask": batch["candidate_mask"],
        }
        for name, value in values.items():
            collected[name].append(value.detach().cpu().numpy())
    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    metrics = reranker_selection_metrics(
        arrays["scores"],
        arrays["targets"],
        arrays["teacher_margins"],
        arrays["teacher_correct"],
        arrays["candidate_mask"],
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(
    model,
    loader,
    objective,
    objective_name: str,
    optimizer,
    device,
    gradient_clip_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            scores = _model_scores(model, batch)
            loss = objective(
                scores,
                _objective_targets(batch, objective_name),
                batch["candidate_mask"],
            )
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@hydra.main(version_base=None, config_path="../configs", config_name="train_reranker")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.experiment.seed)
    _seed_everything(seed)
    target = str(cfg.data.target)
    objective_name = str(cfg.objective.name)
    _validate_target_objective(target, objective_name)
    artifact_path = Path(cfg.data.artifact_path)
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)

    dataset_kwargs = dict(
        artifact=artifact,
        target=target,
        target_temperature=float(cfg.data.target_temperature),
        incremental_lambda=float(cfg.data.incremental_lambda),
    )
    train_data = RerankerTeacherDataset(
        split=str(cfg.data.train_split), **dataset_kwargs
    )
    val_data = RerankerTeacherDataset(split=str(cfg.data.val_split), **dataset_kwargs)
    train_data = _limit(train_data, cfg.data.max_train_queries)
    val_data = _limit(val_data, cfg.data.max_val_queries)

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        collate_fn=collate_reranker_queries,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_data, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    model_config = RerankerConfig(
        clip_dim=train_data.dataset.clip_dim if isinstance(train_data, Subset) else train_data.clip_dim,
        siglip_dim=train_data.dataset.siglip_dim if isinstance(train_data, Subset) else train_data.siglip_dim,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )
    device = _device()
    model = LabelAwareReranker(model_config).to(device)
    objective = _build_objective(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.optimization.learning_rate),
        weight_decay=float(cfg.optimization.weight_decay),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training {model_config.architecture} on {device}: "
        f"{len(train_data)} train / {len(val_data)} val queries; "
        f"{parameter_count:,} parameters"
    )

    run_dir = Path(cfg.output.dir) / str(cfg.experiment.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor = str(cfg.optimization.monitor)
    monitor_mode = str(cfg.optimization.monitor_mode)
    if monitor_mode not in {"min", "max"}:
        raise ValueError("optimization.monitor_mode must be 'min' or 'max'")
    secondary_monitor = str(cfg.optimization.secondary_monitor)
    secondary_mode = str(cfg.optimization.secondary_monitor_mode)
    if secondary_mode not in {"min", "max"}:
        raise ValueError(
            "optimization.secondary_monitor_mode must be 'min' or 'max'"
        )
    best_value = math.inf if monitor_mode == "min" else -math.inf
    best_secondary = math.inf if secondary_mode == "min" else -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    patience = 0
    history = []
    provenance = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_immutable_args": artifact.get("immutable_args"),
        "git_revision": git_revision(),
    }
    total_epochs = int(cfg.optimization.epochs)
    early_stopping_patience = int(cfg.optimization.early_stopping_patience)
    stop_epoch = 0
    stop_reason = f"reached the {total_epochs}-epoch limit"

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            objective,
            objective_name,
            optimizer,
            device,
            float(cfg.optimization.gradient_clip_norm),
            bool(cfg.optimization.amp),
        )
        metrics = evaluate(
            model,
            val_loader,
            objective,
            objective_name,
            device,
            bool(cfg.optimization.amp),
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            **metrics,
        }
        history.append(row)
        if monitor not in metrics:
            raise ValueError(f"Unknown validation monitor: {monitor}")
        if secondary_monitor not in metrics:
            raise ValueError(
                f"Unknown secondary validation monitor: {secondary_monitor}"
            )
        value = float(metrics[monitor])
        secondary_value = float(metrics[secondary_monitor])
        primary_improved = (
            value < best_value if monitor_mode == "min" else value > best_value
        )
        primary_tied = math.isclose(value, best_value, rel_tol=0, abs_tol=1e-12)
        secondary_improved = (
            secondary_value < best_secondary
            if secondary_mode == "min"
            else secondary_value > best_secondary
        )
        improved = primary_improved or (primary_tied and secondary_improved)
        if improved:
            best_value = value
            best_secondary = secondary_value
            best_epoch = epoch
            best_metrics = {
                name: float(metric_value)
                for name, metric_value in metrics.items()
            }
            patience = 0
        else:
            patience += 1
        row.update({
            "best_epoch": best_epoch,
            f"best_{monitor}": best_value,
            f"best_{secondary_monitor}": best_secondary,
            "epochs_without_improvement": patience,
        })
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            "metrics": metrics,
            "history": history,
            "early_stopping": {
                "monitor": monitor,
                "best_value": best_value,
                "secondary_monitor": secondary_monitor,
                "best_secondary_value": best_secondary,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "epochs_without_improvement": patience,
            },
            "provenance": provenance,
        }
        _atomic_torch_save(checkpoint, run_dir / "last.pt")
        if improved:
            _atomic_torch_save(checkpoint, run_dir / "best.pt")
        _print_epoch_metrics(
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=train_loss,
            metrics=metrics,
            improved=improved,
            best_epoch=best_epoch,
            epochs_without_improvement=patience,
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        stop_epoch = epoch
        if patience >= early_stopping_patience:
            stop_reason = "early stopping"
            break

    _print_final_summary(
        stop_reason=stop_reason,
        stop_epoch=stop_epoch,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        early_stopping_patience=early_stopping_patience,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
    )


if __name__ == "__main__":
    main()
