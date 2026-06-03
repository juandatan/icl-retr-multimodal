"""
Shared evaluation result persistence utilities.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def save_eval_results(
    method: str,
    results: Dict,
    run_id_parts: Dict[str, Any],
    args: Dict,
    extra: Optional[Dict] = None,
    output_root: str = "outputs/evals",
) -> Path:
    """
    Persist evaluation results for a single method as both .pkl and .json.

    Args:
        method: Method name used as sub-directory (e.g. 'reranker_best_model').
        results: Dict containing at minimum 'accuracy', 'mean_per_class_accuracy',
                 'correct', 'total'.
        run_id_parts: Ordered key-value pairs used to build a unique run identifier.
                      Values are joined as "k1_v1_k2_v2_...".
        args: Full argument namespace (vars(args)) stored for reproducibility.
        extra: Optional additional keys merged into the .pkl payload.
        output_root: Root directory for all eval outputs.

    Returns:
        Path to the output directory.
    """
    run_id = "_".join(str(v) for v in run_id_parts.values())

    out_dir = Path(output_root) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"run_id": run_id, "method": method, "results": results, "args": args}
    if extra:
        payload.update(extra)

    with open(out_dir / f"{run_id}.pkl", "wb") as f:
        pickle.dump(payload, f)

    summary = {
        "run_id": run_id,
        "method": method,
        "accuracy": results["accuracy"],
        "mean_per_class_accuracy": results["mean_per_class_accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "args": args,
    }
    with open(out_dir / f"{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ {method} results saved to {out_dir}/{run_id}{{.pkl,.json}}")
    return out_dir
