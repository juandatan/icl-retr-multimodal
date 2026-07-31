"""Isolated single-GPU worker for evaluate_full_label_baselines.py."""

import argparse
import os
import pickle
from pathlib import Path

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be set before starting a baseline worker")

from omegaconf import OmegaConf

from scripts.evaluate_full_label_baselines import score_query_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--tasks-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    with open(args.config_path, "rb") as file:
        cfg = OmegaConf.create(pickle.load(file))
    with open(args.tasks_path, "rb") as file:
        tasks = pickle.load(file)

    print(
        f"[Worker {args.worker_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}; "
        f"{len(tasks)} queries on local cuda:0",
        flush=True,
    )
    score_query_tasks(
        cfg=cfg,
        tasks=tasks,
        worker_id=args.worker_id,
        checkpoint_path=Path(args.output_path),
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
