"""Train a lightweight utility probe on frozen pair-conditioned Idefics2 states."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.idefics2_probe_dataset import (  # noqa: E402
    FrozenIdefics2ProbeDataset,
    collate_frozen_idefics2_probe_queries,
)
from src.models.idefics2_probe import FrozenIdefics2UtilityProbe  # noqa: E402
from src.utils.reranker_metrics import reranker_selection_metrics  # noqa: E402
from src.utils.runtime import file_sha256, git_revision  # noqa: E402


SUPPORTED_TARGETS = {
    "mean_token_probability",
    "normalized_incremental_mean_token_probability",
    # Retained as K=32 closed-set ablations, not as paper-faithful defaults.
    "true_probability",
    "normalized_incremental_probability",
}


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


def _experiment_name(cfg: DictConfig) -> str:
    configured = cfg.experiment.get("name")
    if configured is not None and str(configured).strip().lower() not in {"", "auto"}:
        return str(configured)
    signature = OmegaConf.to_container(cfg, resolve=True)
    signature["data"].pop("artifact_path", None)
    signature["data"].pop("probe_cache_path", None)
    signature["output"] = {}
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"frozen_idefics2_probe-{cfg.data.target}-"
        f"seed{int(cfg.experiment.seed)}-{digest}"
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _pointwise_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(scores[mask], targets[mask])


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(batch["pair_representations"], batch["candidate_mask"])
        loss = _pointwise_loss(scores, batch["targets"], batch["candidate_mask"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    losses = []
    collected = {
        name: []
        for name in (
            "scores",
            "targets",
            "teacher_margins",
            "teacher_correct",
            "candidate_mask",
        )
    }
    for batch in loader:
        batch = _move(batch, device)
        scores = model(batch["pair_representations"], batch["candidate_mask"])
        loss = _pointwise_loss(scores, batch["targets"], batch["candidate_mask"])
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


def _atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _print_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    metrics: dict[str, float],
    *,
    improved: bool,
    best_epoch: int,
    patience: int,
) -> None:
    marker = "  new best" if improved else ""
    print(f"\nEpoch {epoch}/{total_epochs}{marker}")
    print(f"  Loss       train {train_loss:.6f}   validation {metrics['loss']:.6f}")
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
    print(
        f"  Target     regret {metrics['mean_target_regret']:.6f}\n"
        f"  Checkpoint best epoch {best_epoch}   epochs without improvement {patience}"
    )
    sys.stdout.flush()


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_frozen_idefics2_probe",
)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.experiment.seed)
    _seed_everything(seed)
    target = str(cfg.data.target)
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Frozen probe target must be one of {sorted(SUPPORTED_TARGETS)}"
        )
    artifact_path = Path(cfg.data.artifact_path)
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    dataset_kwargs = {
        "artifact": artifact,
        "cache_path": cfg.data.probe_cache_path,
        "target": target,
        "target_temperature": float(cfg.data.target_temperature),
        "incremental_lambda": float(cfg.data.incremental_lambda),
    }
    train_data = FrozenIdefics2ProbeDataset(split="train", **dataset_kwargs)
    val_data = FrozenIdefics2ProbeDataset(split="val", **dataset_kwargs)
    if train_data.input_dim != val_data.input_dim:
        raise ValueError("Train and validation probe representations differ in width")
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(cfg.data.batch_size),
        "num_workers": int(cfg.data.num_workers),
        "collate_fn": collate_frozen_idefics2_probe_queries,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_data, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    device = _device()
    model = FrozenIdefics2UtilityProbe(
        input_dim=train_data.input_dim,
        dropout=float(cfg.model.dropout),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.optimization.learning_rate),
        weight_decay=float(cfg.optimization.weight_decay),
    )
    run_name = _experiment_name(cfg)
    run_dir = Path(cfg.output.dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training frozen Idefics2 linear probe on {device}: "
        f"{len(train_data)} train / {len(val_data)} val queries; "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters"
    )
    print(
        f"Run {run_name}: target={target}, seed={seed}, "
        "inputs=[exemplar image, exemplar label, query image]"
    )
    print("K=32 labels and query ground truth are supervision only, never model inputs")
    sys.stdout.flush()

    total_epochs = int(cfg.optimization.epochs)
    early_stopping_patience = int(cfg.optimization.early_stopping_patience)
    best_accuracy = -math.inf
    best_regret = math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    patience = 0
    history = []
    stop_reason = f"reached the {total_epochs}-epoch limit"
    stop_epoch = 0
    provenance = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "probe_cache_path": str(Path(cfg.data.probe_cache_path).resolve()),
        "probe_cache_metadata": train_data.metadata,
        "git_revision": git_revision(),
    }
    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        accuracy = float(metrics["restricted_selected_accuracy"])
        regret = float(metrics["mean_margin_regret"])
        improved = accuracy > best_accuracy or (
            math.isclose(accuracy, best_accuracy, rel_tol=0, abs_tol=1e-12)
            and regret < best_regret
        )
        if improved:
            best_accuracy = accuracy
            best_regret = regret
            best_epoch = epoch
            best_metrics = {name: float(value) for name, value in metrics.items()}
            patience = 0
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "best_epoch": best_epoch,
            "epochs_without_improvement": patience,
        }
        history.append(row)
        checkpoint = {
            "experiment_name": run_name,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": train_data.input_dim,
                "dropout": float(cfg.model.dropout),
            },
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            "metrics": metrics,
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "provenance": provenance,
        }
        _atomic_save(checkpoint, run_dir / "last.pt")
        if improved:
            _atomic_save(checkpoint, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        _print_epoch(
            epoch,
            total_epochs,
            train_loss,
            metrics,
            improved=improved,
            best_epoch=best_epoch,
            patience=patience,
        )
        stop_epoch = epoch
        if patience >= early_stopping_patience:
            stop_reason = "early stopping"
            break

    print(f"\nTraining complete: {stop_reason}")
    print(
        f"  Stopped at epoch {stop_epoch}; selected epoch {best_epoch}; "
        f"target {target}"
    )
    print(
        "  Fixed K=32 baselines  "
        f"CLIP top-1 {best_metrics['restricted_clip_top1_accuracy']:.3%}   "
        f"pool oracle {best_metrics['restricted_pool_oracle_accuracy']:.3%}"
    )
    print(
        "  Selected-checkpoint metrics\n"
        f"    loss {best_metrics['loss']:.6f}   "
        f"accuracy {best_metrics['restricted_selected_accuracy']:.3%}\n"
        f"    selected margin {best_metrics['mean_selected_margin']:.6f}   "
        f"margin regret {best_metrics['mean_margin_regret']:.6f}   "
        f"Spearman {best_metrics['mean_margin_spearman']:.6f}\n"
        f"    margin-oracle {best_metrics['margin_oracle_agreement']:.3%}   "
        f"target-oracle {best_metrics['target_oracle_agreement']:.3%}   "
        f"target regret {best_metrics['mean_target_regret']:.6f}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
