"""
Single-GPU worker for ICL evaluation. Called by evaluate_icl_performance.py
as a subprocess with CUDA_VISIBLE_DEVICES set in the environment.
"""

import sys
import os
from pathlib import Path
import pickle
import argparse

# CUDA_VISIBLE_DEVICES must be set by the parent before launching this script.
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be set by the parent process")

print(f"[Worker] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate_icl_performance import (
    evaluate_icl_worker,
    determine_retrieval_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--config-path", type=str, required=True, help="Pickled worker kwargs")
    parser.add_argument("--queries-path", type=str, required=True, help="Pickled list of query indices")
    parser.add_argument("--output-path", type=str, required=True, help="Path to write pickled results")
    args = parser.parse_args()

    with open(args.config_path, 'rb') as f:
        worker_kwargs = pickle.load(f)

    with open(args.queries_path, 'rb') as f:
        query_indices = pickle.load(f)

    worker_id = args.worker_id
    print(f"[Worker {worker_id}] Processing {len(query_indices)} queries on cuda:0", flush=True)

    result = evaluate_icl_worker(
        gpu_id=0,
        query_indices=query_indices,
        worker_id=worker_id,
        **worker_kwargs
    )

    with open(args.output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"[Worker {worker_id}] Done. Results saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
