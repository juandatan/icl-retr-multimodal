import json

import numpy as np

from src.utils.eval_utils import save_eval_results


def test_json_contains_complete_nested_results_and_numpy_values(tmp_path):
    results = {
        "accuracy": np.float64(0.5),
        "mean_per_class_accuracy": np.float64(0.4),
        "correct": np.int64(1),
        "total": 2,
        "accuracy_by_k": {4: {"conditions": {"zero_shot": {"accuracy": 0.5}}}},
    }

    out_dir = save_eval_results(
        method="mc", results=results, run_id_parts={"dataset": "fake"},
        args={"seeds": [1, 2]}, output_root=str(tmp_path),
    )
    payload = json.loads((out_dir / "fake.json").read_text())

    assert payload["results"]["accuracy_by_k"]["4"]["conditions"]["zero_shot"]["accuracy"] == 0.5
    assert payload["mean_per_class_accuracy"] == 0.4
