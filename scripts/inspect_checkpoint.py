"""
Inspect checkpoint files.

Usage:
    python scripts/inspect_checkpoint.py
    python scripts/inspect_checkpoint.py path/to/checkpoint.pkl
"""

import sys
from pathlib import Path
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataclasses import MarginalUtilityResult


def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint file."""
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"=" * 70)
    print(f"Last query idx: {data['last_query_idx']}")
    print(f"Total results: {len(data['results'])} pairs")
    print(f"Queries completed: {data['num_queries']}")
    print(f"Total pairs: {data['num_pairs']}")
    print(f"=" * 70)


if __name__ == "__main__":
    # Default path - latest checkpoint
    default_dir = Path("outputs/marginal_utilities/marginal_utility_stanford_cars/checkpoints")

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Find latest checkpoint
        if default_dir.exists():
            checkpoints = sorted(default_dir.glob("checkpoint_*.pkl"))
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])
            else:
                print("No checkpoints found")
                sys.exit(1)
        else:
            print("Checkpoint directory not found")
            sys.exit(1)

    inspect_checkpoint(checkpoint_path)
