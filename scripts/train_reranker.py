"""Train and validate a frozen-feature exemplar reranker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
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
from src.losses.listwise import (  # noqa: E402
    HybridListwisePairwiseLoss,
    MultiplePositiveListwiseLoss,
)
from src.losses.pairwise_ranking import PairwiseRankingLoss  # noqa: E402
from src.losses.pointwise import MaskedHuberLoss, MaskedSoftLabelBCELoss  # noqa: E402
from src.models.reranker import LabelAwareReranker, RerankerConfig  # noqa: E402
from src.utils.reranker_metrics import reranker_selection_metrics  # noqa: E402
from src.utils.runtime import file_sha256, git_revision  # noqa: E402


POOLED_MODEL_INPUTS = (
    "query_clip",
    "candidate_clip",
    "query_siglip",
    "candidate_siglip",
    "candidate_label_siglip",
    "clip_similarities",
    "retrieval_ranks",
    "candidate_mask",
)

VISUAL_TOKEN_MODEL_INPUTS = (
    "query_visual_tokens",
    "candidate_visual_tokens",
    "candidate_label_tokens",
    "candidate_label_token_mask",
    "candidate_mask",
)


def _safe_name_component(value: Any) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return component or "unset"


def _resolve_experiment_name(cfg: DictConfig) -> str:
    """Return an explicit name or a stable summary plus config fingerprint."""
    configured = cfg.experiment.get("name")
    if configured is not None and str(configured).strip().lower() not in {"", "auto"}:
        return str(configured)

    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    # Paths vary across machines without changing the experiment itself.
    data_config.pop("artifact_path", None)
    data_config.pop("visual_token_cache_path", None)
    signature = {
        "seed": int(cfg.experiment.seed),
        "data": data_config,
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "objective": OmegaConf.to_container(cfg.objective, resolve=True),
        "optimization": OmegaConf.to_container(cfg.optimization, resolve=True),
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()[:8]
    parts = (
        cfg.model.architecture,
        cfg.data.target,
        cfg.objective.name,
        f"seed{int(cfg.experiment.seed)}",
        fingerprint,
    )
    return "-".join(_safe_name_component(part) for part in parts)


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
    if name == "hybrid_listwise_pairwise":
        return HybridListwisePairwiseLoss(
            listwise_weight=float(cfg.objective.hybrid_listwise_weight),
            min_target_gap=float(cfg.objective.pairwise_min_target_gap),
            score_temperature=float(cfg.objective.pairwise_score_temperature),
        )
    if name == "huber":
        return MaskedHuberLoss(delta=float(cfg.objective.huber_delta))
    raise ValueError(f"Unsupported objective: {name}")


def _validate_target_objective(target: str, objective: str) -> None:
    bounded = {
        "true_probability",
        "bounded_margin",
        "bounded_incremental_margin",
        "mean_token_probability",
        "normalized_incremental_mean_token_probability",
        "normalized_incremental_probability",
    }
    if objective == "pointwise_bce" and target not in bounded:
        raise ValueError(
            f"pointwise_bce requires a [0,1] target; got {target!r}"
        )
    if objective == "hybrid_listwise_pairwise" and target != "margin":
        raise ValueError(
            "hybrid_listwise_pairwise requires data.target=margin"
        )


def _model_scores(model, batch):
    names = (
        VISUAL_TOKEN_MODEL_INPUTS
        if model.config.architecture == "visual_token_cross_encoder"
        else POOLED_MODEL_INPUTS
    )
    return model(**{name: batch[name] for name in names})


def _objective_targets(batch: dict[str, Any], objective_name: str) -> torch.Tensor:
    if objective_name == "listwise_correctness":
        return batch["teacher_correct"]
    return batch["targets"]


def _objective_loss(objective, objective_name: str, scores, batch):
    if objective_name == "hybrid_listwise_pairwise":
        return objective(
            scores,
            batch["targets"],
            batch["teacher_correct"],
            batch["candidate_mask"],
        )
    return objective(
        scores,
        _objective_targets(batch, objective_name),
        batch["candidate_mask"],
    )


def _hybrid_objective_components(objective, scores, batch) -> dict[str, float]:
    return {
        "pairwise_loss_component": float(objective.pairwise(
            scores, batch["targets"], batch["candidate_mask"]
        ).item()),
        "listwise_loss_component": float(objective.listwise(
            scores, batch["teacher_correct"], batch["candidate_mask"]
        ).item()),
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _print_phase_progress(
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
    batch_number: int,
    total_batches: int,
    running_loss: float,
    elapsed_seconds: float,
) -> None:
    """Emit a flushed heartbeat during long train and validation passes."""
    percent = 100.0 * batch_number / max(total_batches, 1)
    print(
        f"Epoch {epoch}/{total_epochs}  {phase} "
        f"{batch_number}/{total_batches} batches ({percent:.1f}%)  "
        f"running loss {running_loss:.6f}  elapsed {elapsed_seconds / 60:.1f}m",
        flush=True,
    )


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
    if "pairwise_loss_component" in metrics:
        print(
            "  Components "
            f"pairwise {metrics['pairwise_loss_component']:.6f}   "
            f"listwise {metrics['listwise_loss_component']:.6f}"
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
    # Notebook shell commands commonly attach Python to a non-interactive pipe,
    # where stdout is block-buffered. Flush the complete epoch block explicitly
    # so long-running jobs expose progress as soon as validation finishes.
    sys.stdout.flush()


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
    if "pairwise_loss_component" in best_metrics:
        print(
            "    loss components  "
            f"pairwise {best_metrics['pairwise_loss_component']:.6f}   "
            f"listwise {best_metrics['listwise_loss_component']:.6f}"
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
    sys.stdout.flush()


@torch.no_grad()
def evaluate(
    model,
    loader,
    objective,
    objective_name: str,
    device,
    use_amp: bool,
    *,
    epoch: int = 0,
    total_epochs: int = 0,
    progress_every_seconds: float = 0.0,
) -> dict[str, float]:
    model.eval()
    losses = []
    component_losses: dict[str, list[float]] = {}
    collected = {name: [] for name in (
        "scores", "targets", "teacher_margins", "teacher_correct", "candidate_mask"
    )}
    phase_started = time.monotonic()
    last_progress = phase_started
    total_batches = len(loader)
    for batch_number, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            scores = _model_scores(model, batch)
            loss = _objective_loss(objective, objective_name, scores, batch)
            if objective_name == "hybrid_listwise_pairwise":
                for name, value in _hybrid_objective_components(
                    objective, scores, batch
                ).items():
                    component_losses.setdefault(name, []).append(value)
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
        now = time.monotonic()
        if progress_every_seconds > 0 and now - last_progress >= progress_every_seconds:
            _print_phase_progress(
                epoch=epoch,
                total_epochs=total_epochs,
                phase="validation",
                batch_number=batch_number,
                total_batches=total_batches,
                running_loss=float(np.mean(losses)),
                elapsed_seconds=now - phase_started,
            )
            last_progress = now
    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    metrics = reranker_selection_metrics(
        arrays["scores"],
        arrays["targets"],
        arrays["teacher_margins"],
        arrays["teacher_correct"],
        arrays["candidate_mask"],
    )
    metrics["loss"] = float(np.mean(losses))
    metrics.update({
        name: float(np.mean(values))
        for name, values in component_losses.items()
    })
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
    scaler=None,
    *,
    epoch: int = 0,
    total_epochs: int = 0,
    progress_every_seconds: float = 0.0,
) -> float:
    model.train()
    losses = []
    phase_started = time.monotonic()
    last_progress = phase_started
    total_batches = len(loader)
    for batch_number, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            scores = _model_scores(model, batch)
            loss = _objective_loss(objective, objective_name, scores, batch)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            optimizer.step()
        losses.append(float(loss.item()))
        now = time.monotonic()
        if progress_every_seconds > 0 and now - last_progress >= progress_every_seconds:
            _print_phase_progress(
                epoch=epoch,
                total_epochs=total_epochs,
                phase="train",
                batch_number=batch_number,
                total_batches=total_batches,
                running_loss=float(np.mean(losses)),
                elapsed_seconds=now - phase_started,
            )
            last_progress = now
    return float(np.mean(losses))


@hydra.main(version_base=None, config_path="../configs", config_name="train_reranker")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.experiment.seed)
    experiment_name = _resolve_experiment_name(cfg)
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
        visual_token_cache_path=cfg.data.get("visual_token_cache_path", None),
    )
    train_data = RerankerTeacherDataset(
        split=str(cfg.data.train_split), **dataset_kwargs
    )
    val_data = RerankerTeacherDataset(
        split=str(cfg.data.val_split), **dataset_kwargs
    )
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

    underlying_train_data = (
        train_data.dataset if isinstance(train_data, Subset) else train_data
    )
    architecture = str(cfg.model.architecture)
    if (
        architecture == "visual_token_cross_encoder"
        and underlying_train_data.visual_tokens is None
    ):
        raise ValueError(
            "visual_token_cross_encoder requires data.visual_token_cache_path"
        )
    model_config = RerankerConfig(
        clip_dim=underlying_train_data.clip_dim,
        siglip_dim=underlying_train_data.siglip_dim,
        visual_token_dim=underlying_train_data.visual_token_dim,
        visual_token_count=underlying_train_data.visual_token_count,
        visual_label_token_count=underlying_train_data.label_token_count,
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
    scaler_enabled = bool(cfg.optimization.amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # Compatibility with older PyTorch.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training {model_config.architecture} on {device}: "
        f"{len(train_data)} train / {len(val_data)} val queries; "
        f"{parameter_count:,} parameters"
    )
    run_description = (
        f"Run {experiment_name}: seed={seed}, target={target}, "
        f"objective={objective_name}"
    )
    if objective_name == "hybrid_listwise_pairwise":
        run_description += (
            f", listwise_weight={float(cfg.objective.hybrid_listwise_weight):g}"
        )
    print(run_description)
    optional_inputs = [
        name
        for name, enabled in (
            ("clip_embeddings", model_config.use_clip_embeddings),
            ("clip_similarity", model_config.use_clip_similarity),
            ("retrieval_rank", model_config.use_retrieval_rank),
            (
                "derived_siglip_similarities",
                model_config.use_derived_siglip_similarities,
            ),
        )
        if enabled
    ]
    if architecture == "visual_token_cross_encoder":
        optional_inputs.append(
            "idefics2_visual_tokens"
            f"[{model_config.visual_token_count}x{model_config.visual_token_dim}]"
        )
    print(
        "Optional inputs: "
        + (", ".join(optional_inputs) if optional_inputs else "none")
    )
    progress_every_seconds = float(cfg.logging.progress_every_seconds)
    if progress_every_seconds > 0:
        print(
            "Progress: train/validation heartbeat every "
            f"{progress_every_seconds:g}s; metrics after each complete epoch"
        )
    sys.stdout.flush()

    run_dir = Path(cfg.output.dir) / experiment_name
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
        "experiment_name": experiment_name,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_immutable_args": artifact.get("immutable_args"),
        "visual_token_cache": underlying_train_data.visual_token_cache_metadata,
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
            scaler,
            epoch=epoch,
            total_epochs=total_epochs,
            progress_every_seconds=progress_every_seconds,
        )
        metrics = evaluate(
            model,
            val_loader,
            objective,
            objective_name,
            device,
            bool(cfg.optimization.amp),
            epoch=epoch,
            total_epochs=total_epochs,
            progress_every_seconds=progress_every_seconds,
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
            "experiment_name": experiment_name,
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
