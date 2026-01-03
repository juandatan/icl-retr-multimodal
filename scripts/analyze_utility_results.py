"""
Quick analysis of marginal utility results.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py outputs/marginal_utilities/my_experiment/marginal_utilities_train.pkl
"""

import sys
import pickle
from pathlib import Path
import numpy as np

# Add parent directory to path to enable importing from scripts module
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.compute_marginal_utilities import MarginalUtilityResult


def analyze_results(result_path: str):
    """Analyze marginal utility results."""
    # Load results
    with open(result_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    config = data['config']

    print(f"{'='*70}")
    print(f"Results: {result_path}")
    print(f"{'='*70}\n")

    print(f"Total results: {len(results)} pairs from {data['num_queries']} queries")
    print(f"Top-k: {config['retrieval']['top_k']}")
    print(f"Dataset: {config['dataset']['name']} ({config['dataset']['split']} split)\n")

    # Sample result
    print(f"Sample result:")
    r = results[0]
    print(f"  Query: {r.query_label} (idx={r.query_idx})")
    print(f"  Example: {r.example_label} (idx={r.example_idx})")
    print(f"  Baseline log prob: {r.baseline_log_prob:.4f}")
    print(f"  1-shot log prob: {r.oneshot_log_prob:.4f}")
    print(f"  Marginal utility: {r.marginal_utility:.4f}")
    print(f"  CLIP similarity: {r.similarity_score:.4f}")
    print(f"  Same class: {r.same_class}\n")

    # Overall statistics
    utilities = np.array([r.marginal_utility for r in results])
    similarities = np.array([r.similarity_score for r in results])

    print(f"Marginal Utility Statistics:")
    print(f"  Mean:   {utilities.mean():7.4f}")
    print(f"  Std:    {utilities.std():7.4f}")
    print(f"  Min:    {utilities.min():7.4f}")
    print(f"  Max:    {utilities.max():7.4f}")
    print(f"  Median: {np.median(utilities):7.4f}\n")

    print(f"CLIP Similarity Statistics:")
    print(f"  Mean:   {similarities.mean():7.4f}")
    print(f"  Std:    {similarities.std():7.4f}")
    print(f"  Min:    {similarities.min():7.4f}")
    print(f"  Max:    {similarities.max():7.4f}\n")

    # By class
    same_class = np.array([r.marginal_utility for r in results if r.same_class])
    diff_class = np.array([r.marginal_utility for r in results if not r.same_class])

    print(f"By Class:")
    print(f"  Same class: {same_class.mean():7.4f} ± {same_class.std():.4f} (n={len(same_class)})")
    print(f"  Diff class: {diff_class.mean():7.4f} ± {diff_class.std():.4f} (n={len(diff_class)})")

    if len(same_class) > 0 and len(diff_class) > 0:
        print(f"  Difference: {same_class.mean() - diff_class.mean():7.4f}")

    # Correlation
    print(f"\nCorrelation:")
    corr = np.corrcoef(similarities, utilities)[0, 1]
    print(f"  CLIP similarity vs Marginal utility: {corr:.4f}")

    # Top examples by utility
    print(f"\nTop 5 examples by marginal utility:")
    sorted_results = sorted(results, key=lambda r: r.marginal_utility, reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. Query: {r.query_label}, Example: {r.example_label}")
        print(f"     Utility: {r.marginal_utility:.4f}, Similarity: {r.similarity_score:.4f}, Same class: {r.same_class}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    # Default path
    default_path = "outputs/marginal_utilities/marginal_utility_stanford_cars/marginal_utilities_train.pkl"

    # Use command line arg if provided
    result_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    analyze_results(result_path)
