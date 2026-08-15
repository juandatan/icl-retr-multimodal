"""Grid-search wrapper around scripts/train_reranker.py.

Forwards all arguments to `python -m scripts.train_reranker --multirun`, so
comma-separated override values define the grid (Hydra's basic sweeper takes
the Cartesian product), then summarizes every completed run's best checkpoint
into one ranked CSV/console table.

Usage:
    python -m scripts.sweep_reranker \
        data.artifact_path=/path/to/reranker_teacher_data.pkl \
        data.max_candidates=30 \
        data.target=margin \
        model.architecture=interaction_mlp \
        model.dropout=0.1,0.3 \
        model.hidden_dim=128,256 \
        objective.name=hybrid_listwise_pairwise \
        objective.hybrid_listwise_weight=0.1 \
        optimization.learning_rate=0.0003,0.001 \
        optimization.weight_decay=0.0001,0.001 \
        optimization.epochs=100
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import torch

DEFAULT_OUTPUT_DIR = "outputs/reranker_training"
DEFAULT_MONITOR = "restricted_selected_accuracy"
DEFAULT_MONITOR_MODE = "max"
DEFAULT_SECONDARY_MONITOR = "mean_margin_regret"
DEFAULT_SECONDARY_MONITOR_MODE = "min"
SUMMARY_METRICS = (
    "restricted_selected_accuracy",
    "restricted_pool_oracle_accuracy",
    "mean_margin_regret",
    "mean_margin_spearman",
    "margin_oracle_agreement",
    "loss",
)


def _split_wrapper_args(argv: list[str]) -> tuple[list[str], Path | None]:
    """Pull --summary-csv out of the argument list; forward everything else."""
    overrides = []
    summary_csv = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--summary-csv":
            summary_csv = Path(argv[index + 1])
            index += 2
            continue
        overrides.append(arg)
        index += 1
    return overrides, summary_csv


def _output_dir_from_overrides(overrides: list[str]) -> Path:
    for override in overrides:
        if override.startswith("output.dir="):
            return Path(override.split("=", 1)[1])
    return Path(DEFAULT_OUTPUT_DIR)


def _swept_keys(overrides: list[str]) -> list[str]:
    """Return override keys using Hydra's discrete-sweep syntax `key=a,b,c`.

    A bracketed/braced value (`key=[1,2,3]` or `key={a:1}`) is a single list
    or dict override, not a sweep, even though it contains a comma.
    """
    keys = []
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if "," in value and not value.startswith(("[", "{")):
            keys.append(key)
    return keys


def _monitor_settings(overrides: list[str]) -> tuple[str, str, str, str]:
    """Read the accuracy-selection monitors, honoring any explicit override."""
    resolved = {
        "optimization.monitor": DEFAULT_MONITOR,
        "optimization.monitor_mode": DEFAULT_MONITOR_MODE,
        "optimization.secondary_monitor": DEFAULT_SECONDARY_MONITOR,
        "optimization.secondary_monitor_mode": DEFAULT_SECONDARY_MONITOR_MODE,
    }
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if key in resolved:
            resolved[key] = value
    return (
        resolved["optimization.monitor"],
        resolved["optimization.monitor_mode"],
        resolved["optimization.secondary_monitor"],
        resolved["optimization.secondary_monitor_mode"],
    )


def _get_nested(container: dict, dotted_key: str):
    node = container
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _load_run_summary(
    run_dir: Path, swept_keys: list[str], summary_metrics: tuple[str, ...]
) -> dict | None:
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {"experiment_name", "epoch", "metrics", "resolved_config"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise ValueError(f"{checkpoint_path} is missing checkpoint keys: {missing_keys}")
    metrics = checkpoint["metrics"]
    missing_metrics = set(summary_metrics) - metrics.keys()
    if missing_metrics:
        raise ValueError(f"{checkpoint_path} is missing metrics: {missing_metrics}")
    resolved_config = checkpoint["resolved_config"]
    row = {
        "experiment_name": checkpoint["experiment_name"],
        "best_epoch": checkpoint["epoch"],
    }
    for key in swept_keys:
        row[key] = _get_nested(resolved_config, key)
    row.update({name: metrics[name] for name in summary_metrics})
    return row


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    overrides, summary_csv = _split_wrapper_args(argv)
    swept_keys = _swept_keys(overrides)
    if not swept_keys:
        raise SystemExit(
            "At least one override must use a comma-separated grid, "
            "e.g. model.dropout=0.1,0.3"
        )
    output_dir = _output_dir_from_overrides(overrides)
    if summary_csv is None:
        summary_csv = output_dir / "sweep_summary.csv"
    monitor, monitor_mode, secondary_monitor, secondary_mode = _monitor_settings(
        overrides
    )
    summary_metrics = tuple(dict.fromkeys((*SUMMARY_METRICS, monitor, secondary_monitor)))

    sweep_started_at = time.time()
    command = [sys.executable, "-m", "scripts.train_reranker", "--multirun", *overrides]
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(
            f"One or more sweep jobs failed (exit code {result.returncode}); "
            "summarizing whichever runs completed."
        )

    rows = []
    if output_dir.is_dir():
        for run_dir in sorted(output_dir.iterdir()):
            # Only summarize runs this invocation produced; output.dir may
            # already contain unrelated runs from earlier sweeps or manual
            # train_reranker.py calls.
            if not run_dir.is_dir() or run_dir.stat().st_mtime < sweep_started_at:
                continue
            row = _load_run_summary(run_dir, swept_keys, summary_metrics)
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit(f"No completed runs with best.pt found under {output_dir}")

    monitor_sign = -1 if monitor_mode == "max" else 1
    secondary_sign = -1 if secondary_mode == "max" else 1
    rows.sort(
        key=lambda row: (
            monitor_sign * row[monitor],
            secondary_sign * row[secondary_monitor],
        )
    )

    fieldnames = ["experiment_name", "best_epoch", *swept_keys, *summary_metrics]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} run(s) to {summary_csv}\n")
    display_fields = ["best_epoch", *swept_keys, *summary_metrics]
    header = "  ".join(f"{name:>26}" for name in display_fields)
    print(header)
    for row in rows:
        print("  ".join(f"{row[name]!s:>26}" for name in display_fields))


if __name__ == "__main__":
    main()
