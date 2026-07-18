"""Deprecated compatibility entry point for full-label baseline evaluation."""

import sys
import runpy
from pathlib import Path

if __name__ == "__main__":
    print(
        "evaluate_full_label_pilot.py was renamed to "
        "evaluate_full_label_baselines.py; forwarding this run.",
        file=sys.stderr,
    )
    runpy.run_path(
        str(Path(__file__).with_name("evaluate_full_label_baselines.py")),
        run_name="__main__",
    )
