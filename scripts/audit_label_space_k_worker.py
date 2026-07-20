"""Isolated single-GPU worker for audit_label_space_k.py."""

import argparse
import os
import pickle
import sys
from pathlib import Path

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be set before starting an audit worker")

sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf

from audit_label_space_k import score_audit_tasks


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
        f"[Audit worker {args.worker_id}] "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}; "
        f"{len(tasks)} queries on local cuda:0",
        flush=True,
    )
    score_audit_tasks(
        cfg=cfg,
        tasks=tasks,
        worker_id=args.worker_id,
        checkpoint_path=Path(args.output_path),
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
