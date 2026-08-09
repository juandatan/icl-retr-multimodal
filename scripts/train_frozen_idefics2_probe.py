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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.idefics2_probe_dataset import (  # noqa: E402
    FrozenIdefics2ProbeDataset,
    collate_frozen_idefics2_probe_queries,
)
from src.losses.listwise import HybridListwisePairwiseLoss  # noqa: E402
from src.losses.pairwise_ranking import (  # noqa: E402
    CorrectnessCrossingPairwiseLoss,
    PairwiseRankingLoss,
)
from src.losses.pointwise import (  # noqa: E402
    HybridPointwisePairwiseLoss,
    MaskedSoftLabelBCELoss,
)
from src.models.idefics2_probe import FrozenIdefics2UtilityProbe  # noqa: E402
from src.utils.reranker_metrics import reranker_selection_metrics  # noqa: E402
from src.utils.runtime import (  # noqa: E402
    file_sha256,
    git_revision,
    stratified_sample_indices,
)


SUPPORTED_TARGETS = {
    "margin",
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
        f"frozen_idefics2_probe-{cfg.model.architecture}-{cfg.data.target}-"
        f"{cfg.objective.name}-seed{int(cfg.experiment.seed)}-{digest}"
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _stratified_training_subset(
    dataset: FrozenIdefics2ProbeDataset,
    *,
    max_queries: int | None,
    fraction: float | None,
    seed: int,
):
    if max_queries is not None and fraction is not None:
        raise ValueError("Set only one of max_train_queries and train_fraction")
    if fraction is not None:
        fraction = float(fraction)
        if not 0 < fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        count = max(1, int(round(len(dataset) * fraction)))
    elif max_queries is not None:
        count = int(max_queries)
        if count <= 0:
            raise ValueError("max_train_queries must be positive or null")
        count = min(count, len(dataset))
    else:
        count = len(dataset)
    if count == len(dataset):
        return dataset
    labels = [int(record.true_class_idx) for record in dataset.teacher.records]
    return Subset(dataset, stratified_sample_indices(labels, count, seed))


def _build_objective(cfg: DictConfig):
    name = str(cfg.objective.name)
    configured_teacher_temperature = cfg.objective.get(
        "pairwise_teacher_weight_temperature", None
    )
    teacher_temperature = (
        None
        if configured_teacher_temperature is None
        else float(configured_teacher_temperature)
    )
    if name == "pointwise_bce":
        return MaskedSoftLabelBCELoss()
    if name == "pairwise":
        return PairwiseRankingLoss(
            min_target_gap=float(cfg.objective.pairwise_min_target_gap),
            score_temperature=float(cfg.objective.pairwise_score_temperature),
            teacher_weight_temperature=teacher_temperature,
        )
    if name == "pointwise_pairwise":
        return HybridPointwisePairwiseLoss(
            pairwise_weight=float(cfg.objective.hybrid_pairwise_weight),
            min_target_gap=float(cfg.objective.pairwise_min_target_gap),
            score_temperature=float(cfg.objective.pairwise_score_temperature),
            teacher_weight_temperature=teacher_temperature,
        )
    if name == "hybrid_listwise_pairwise":
        if teacher_temperature is not None:
            raise ValueError(
                "pairwise_teacher_weight_temperature is not supported by "
                "hybrid_listwise_pairwise"
            )
        return HybridListwisePairwiseLoss(
            listwise_weight=float(cfg.objective.hybrid_listwise_weight),
            min_target_gap=float(cfg.objective.pairwise_min_target_gap),
            score_temperature=float(cfg.objective.pairwise_score_temperature),
        )
    if name == "correctness_crossing_pairwise":
        if teacher_temperature is not None:
            raise ValueError(
                "pairwise_teacher_weight_temperature is not supported by "
                "correctness_crossing_pairwise"
            )
        return CorrectnessCrossingPairwiseLoss(
            score_temperature=float(cfg.objective.pairwise_score_temperature),
            margin_aux_weight=float(cfg.objective.correctness_margin_aux_weight),
            margin_min_target_gap=float(cfg.objective.pairwise_min_target_gap),
        )
    raise ValueError(
        "objective.name must be pointwise_bce, pairwise, pointwise_pairwise, "
        "hybrid_listwise_pairwise, or correctness_crossing_pairwise"
    )


def _objective_loss(objective, scores, batch) -> torch.Tensor:
    if isinstance(objective, HybridListwisePairwiseLoss):
        return objective(
            scores,
            batch["targets"],
            batch["teacher_correct"],
            batch["candidate_mask"],
        )
    if isinstance(objective, CorrectnessCrossingPairwiseLoss):
        return objective(
            scores,
            batch["teacher_margins"],
            batch["teacher_correct"],
            batch["candidate_mask"],
        )
    return objective(scores, batch["targets"], batch["candidate_mask"])


def train_one_epoch(model, loader, optimizer, objective, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(batch["pair_representations"], batch["candidate_mask"])
        loss = _objective_loss(objective, scores, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, objective, device) -> dict[str, float]:
    model.eval()
    losses = []
    component_losses: dict[str, list[float]] = {}
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
        loss = _objective_loss(objective, scores, batch)
        losses.append(float(loss.item()))
        if isinstance(objective, HybridPointwisePairwiseLoss):
            components = {
                "pointwise_loss_component": objective.pointwise(
                    scores, batch["targets"], batch["candidate_mask"]
                ),
                "pairwise_loss_component": objective.pairwise(
                    scores, batch["targets"], batch["candidate_mask"]
                ),
            }
            for name, value in components.items():
                component_losses.setdefault(name, []).append(float(value.item()))
        elif isinstance(objective, HybridListwisePairwiseLoss):
            components = {
                "pairwise_loss_component": objective.pairwise(
                    scores, batch["targets"], batch["candidate_mask"]
                ),
                "listwise_loss_component": objective.listwise(
                    scores, batch["teacher_correct"], batch["candidate_mask"]
                ),
            }
            for name, value in components.items():
                component_losses.setdefault(name, []).append(float(value.item()))
        elif isinstance(objective, CorrectnessCrossingPairwiseLoss):
            components = {
                "correctness_loss_component": objective.correctness(
                    scores,
                    batch["teacher_correct"],
                    batch["candidate_mask"],
                ),
                "margin_aux_loss_component": objective.margin(
                    scores,
                    batch["teacher_margins"],
                    batch["candidate_mask"],
                ),
            }
            for name, value in components.items():
                component_losses.setdefault(name, []).append(float(value.item()))
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
    metrics.update({
        name: float(np.mean(values))
        for name, values in component_losses.items()
    })
    return metrics


def _atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _print_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_metrics: dict[str, float] | None,
    metrics: dict[str, float],
    *,
    improved: bool,
    best_epoch: int,
    patience: int,
) -> None:
    marker = "  new best" if improved else ""
    print(f"\nEpoch {epoch}/{total_epochs}{marker}")
    eval_train_loss = (
        f"   train-eval {train_metrics['loss']:.6f}"
        if train_metrics is not None
        else ""
    )
    print(
        f"  Loss       optimization {train_loss:.6f}{eval_train_loss}   "
        f"validation {metrics['loss']:.6f}"
    )
    if "pointwise_loss_component" in metrics:
        print(
            "  Components "
            f"pointwise {metrics['pointwise_loss_component']:.6f}   "
            f"pairwise {metrics['pairwise_loss_component']:.6f}"
        )
    elif "listwise_loss_component" in metrics:
        print(
            "  Components "
            f"pairwise {metrics['pairwise_loss_component']:.6f}   "
            f"listwise {metrics['listwise_loss_component']:.6f}"
        )
    elif "correctness_loss_component" in metrics:
        print(
            "  Components "
            f"correctness {metrics['correctness_loss_component']:.6f}   "
            f"margin-aux {metrics['margin_aux_loss_component']:.6f}"
        )
    train_accuracy = (
        f"train {train_metrics['restricted_selected_accuracy']:.3%}   "
        if train_metrics is not None
        else ""
    )
    print(
        "  Selection  "
        f"accuracy {train_accuracy}validation "
        f"{metrics['restricted_selected_accuracy']:.3%}   "
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
    objective_name = str(cfg.objective.name)
    if objective_name == "pointwise_bce" and target == "margin":
        raise ValueError("pointwise_bce requires a probability target, not margin")
    if objective_name == "pointwise_pairwise" and target == "margin":
        raise ValueError(
            "pointwise_pairwise includes BCE and requires a probability target"
        )
    if objective_name == "hybrid_listwise_pairwise" and target != "margin":
        raise ValueError(
            "hybrid_listwise_pairwise requires data.target=margin"
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
    input_dim = train_data.input_dim
    cache_metadata = train_data.metadata
    train_data = _stratified_training_subset(
        train_data,
        max_queries=cfg.data.get("max_train_queries", None),
        fraction=cfg.data.get("train_fraction", None),
        seed=int(cfg.data.get("stratified_subset_seed", seed)),
    )
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
    train_eval_loader = DataLoader(train_data, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    device = _device()
    architecture = str(cfg.model.architecture)
    model = FrozenIdefics2UtilityProbe(
        input_dim=input_dim,
        dropout=float(cfg.model.dropout),
        architecture=architecture,
        hidden_dim=int(cfg.model.hidden_dim),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.optimization.learning_rate),
        weight_decay=float(cfg.optimization.weight_decay),
    )
    objective = _build_objective(cfg)
    run_name = _experiment_name(cfg)
    run_dir = Path(cfg.output.dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training frozen Idefics2 {architecture} probe on {device}: "
        f"{len(train_data)} train / {len(val_data)} val queries; "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters"
    )
    print(
        f"Run {run_name}: architecture={architecture}, target={target}, seed={seed}, "
        f"objective={objective_name}, "
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
    best_train_metrics: dict[str, float] | None = None
    patience = 0
    history = []
    stop_reason = f"reached the {total_epochs}-epoch limit"
    stop_epoch = 0
    provenance = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "probe_cache_path": str(Path(cfg.data.probe_cache_path).resolve()),
        "probe_cache_metadata": cache_metadata,
        "git_revision": git_revision(),
    }
    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, objective, device
        )
        train_metrics = (
            evaluate(model, train_eval_loader, objective, device)
            if bool(cfg.logging.get("evaluate_train_metrics", True))
            else None
        )
        metrics = evaluate(model, val_loader, objective, device)
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
            best_train_metrics = (
                {name: float(value) for name, value in train_metrics.items()}
                if train_metrics is not None
                else None
            )
            patience = 0
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **(
                {
                    f"train_eval_{name}": float(value)
                    for name, value in train_metrics.items()
                }
                if train_metrics is not None
                else {}
            ),
            **metrics,
            "best_epoch": best_epoch,
            "epochs_without_improvement": patience,
        }
        history.append(row)
        checkpoint = {
            "experiment_name": run_name,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": input_dim,
                "architecture": architecture,
                "hidden_dim": int(cfg.model.hidden_dim),
                "dropout": float(cfg.model.dropout),
            },
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            "metrics": metrics,
            "train_metrics": train_metrics,
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_train_metrics": best_train_metrics,
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
            train_metrics,
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
        f"architecture {architecture}; target {target}; objective {objective_name}"
    )
    print(
        "  Fixed K=32 baselines  "
        f"CLIP top-1 {best_metrics['restricted_clip_top1_accuracy']:.3%}   "
        f"pool oracle {best_metrics['restricted_pool_oracle_accuracy']:.3%}"
    )
    print(
        "  Selected-checkpoint metrics\n"
        f"    loss {best_metrics['loss']:.6f}   "
        + (
            f"train accuracy {best_train_metrics['restricted_selected_accuracy']:.3%}   "
            if best_train_metrics is not None
            else ""
        )
        + f"validation accuracy {best_metrics['restricted_selected_accuracy']:.3%}\n"
        f"    selected margin {best_metrics['mean_selected_margin']:.6f}   "
        f"margin regret {best_metrics['mean_margin_regret']:.6f}   "
        f"Spearman {best_metrics['mean_margin_spearman']:.6f}\n"
        f"    margin-oracle {best_metrics['margin_oracle_agreement']:.3%}   "
        f"target-oracle {best_metrics['target_oracle_agreement']:.3%}   "
        f"target regret {best_metrics['mean_target_regret']:.6f}"
    )
    if "pointwise_loss_component" in best_metrics:
        print(
            "    loss components  "
            f"pointwise {best_metrics['pointwise_loss_component']:.6f}   "
            f"pairwise {best_metrics['pairwise_loss_component']:.6f}"
        )
    elif "listwise_loss_component" in best_metrics:
        print(
            "    loss components  "
            f"pairwise {best_metrics['pairwise_loss_component']:.6f}   "
            f"listwise {best_metrics['listwise_loss_component']:.6f}"
        )
    elif "correctness_loss_component" in best_metrics:
        print(
            "    loss components  "
            f"correctness {best_metrics['correctness_loss_component']:.6f}   "
            f"margin-aux {best_metrics['margin_aux_loss_component']:.6f}"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
