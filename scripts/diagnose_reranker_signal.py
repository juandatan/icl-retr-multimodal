"""
Diagnose whether a dataset has learnable reranker signal, before committing to
a reranker architecture.

Two diagnostics are computed on the same small query subset (see
`limits.max_queries` in configs/marginal_utility_*.yaml):

1. Utility variance: how much does marginal utility vary across the CLIP-retrieved
   candidate pool for each query? Near-zero variance means candidate choice barely
   affects the model's confidence in the true label -- there is little signal left
   for a reranker to learn to distinguish good from bad candidates.

2. Oracle-gap accuracy: compares 0-shot, CLIP-top-1, and "oracle" 1-shot
   classification accuracy, where the oracle example per query is the pool
   candidate with the highest marginal utility (from diagnostic 1). A large
   oracle-vs-CLIP-top-1 gap means there's real accuracy headroom that a learned
   reranker could capture; a small gap means CLIP similarity already picks
   near-optimal examples, so a reranker likely won't help much on this dataset.

Usage:
    # Step 1 (once per dataset): compute marginal utilities for a query subset
    python scripts/compute_marginal_utilities.py --config-name marginal_utility_cub_200_idefics2

    # Step 2: run this diagnostic against those results
    python scripts/diagnose_reranker_signal.py \
        --dataset cub_200 \
        --marginal-utility-results outputs/marginal_utilities/marginal_utility_cub_200_idefics2/marginal_utilities_train.pkl \
        --image-split-path data/cub_200/image_split.json

    # Skip the (expensive) oracle-gap classification pass, just report variance stats
    python scripts/diagnose_reranker_signal.py \
        --dataset cub_200 \
        --marginal-utility-results outputs/marginal_utilities/marginal_utility_cub_200_idefics2/marginal_utilities_train.pkl \
        --skip-classification
"""

import sys
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataclasses import MarginalUtilityResult
from data.dataset_registry import FINE_GRAINED_DATASETS
from models.idefics2_wrapper import Idefics2Wrapper
from utils.imagenet_names import get_readable_name

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_icl_performance import load_dataset, _setup_device

ALL_DATASETS = ["stanford_cars", "mini_imagenet"] + list(FINE_GRAINED_DATASETS.keys())


def load_marginal_utility_results(path: Path) -> List[MarginalUtilityResult]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['results']


def compute_utility_variance_stats(results: List[MarginalUtilityResult]) -> Dict:
    """Group marginal utilities by query and summarize spread within each pool."""
    by_query: Dict[int, List[float]] = defaultdict(list)
    for r in results:
        by_query[r.query_idx].append(r.marginal_utility)

    per_query_std = [float(np.std(u)) for u in by_query.values()]
    per_query_range = [float(np.max(u) - np.min(u)) for u in by_query.values()]
    per_query_max = [float(np.max(u)) for u in by_query.values()]

    all_utils = np.array([r.marginal_utility for r in results])

    return {
        'num_queries': len(by_query),
        'candidates_per_query_mean': float(np.mean([len(v) for v in by_query.values()])),
        'overall_utility_mean': float(all_utils.mean()),
        'overall_utility_std': float(all_utils.std()),
        'mean_per_query_std': float(np.mean(per_query_std)),
        'mean_per_query_range': float(np.mean(per_query_range)),
        'mean_per_query_max_utility': float(np.mean(per_query_max)),
        'frac_queries_with_positive_best_utility': float(np.mean(np.array(per_query_max) > 0)),
    }


def build_oracle_lookup(results: List[MarginalUtilityResult]) -> Dict[int, int]:
    """Map query_idx -> example_idx of the pool candidate with max marginal utility."""
    best: Dict[int, Tuple[int, float]] = {}
    for r in results:
        cur = best.get(r.query_idx)
        if cur is None or r.marginal_utility > cur[1]:
            best[r.query_idx] = (r.example_idx, r.marginal_utility)
    return {q: idx for q, (idx, _) in best.items()}


def build_clip_top1_lookup(results: List[MarginalUtilityResult]) -> Dict[int, int]:
    """Map query_idx -> example_idx of the highest-CLIP-similarity pool candidate.

    Reuses the similarity_score already stored per pair, so this reflects the
    exact same retrieved pool used for utility computation (no re-retrieval).
    """
    best: Dict[int, Tuple[int, float]] = {}
    for r in results:
        cur = best.get(r.query_idx)
        if cur is None or r.similarity_score > cur[1]:
            best[r.query_idx] = (r.example_idx, r.similarity_score)
    return {q: idx for q, (idx, _) in best.items()}


def run_condition(
    model: Idefics2Wrapper,
    dataset,
    query_indices: List[int],
    example_idx_lookup: Optional[Dict[int, int]],
    candidate_labels: List[str],
    candidate_batch_size: int,
    desc: str,
) -> Dict:
    """Run 0-shot (lookup=None) or 1-shot classification and report accuracy."""
    correct = 0
    total = 0

    for query_idx in tqdm(query_indices, desc=desc):
        query_example, query_image = dataset[query_idx]
        true_label_readable = get_readable_name(query_example.label_name)

        context_examples = []
        if example_idx_lookup is not None:
            example_idx = example_idx_lookup.get(query_idx)
            if example_idx is None:
                continue
            ex_example, ex_image = dataset[example_idx]
            context_examples.append((ex_image, get_readable_name(ex_example.label_name)))

        predicted = model.classify_with_context(
            query_image=query_image,
            context_examples=context_examples,
            candidate_labels=candidate_labels,
            batch_size=candidate_batch_size,
        )

        correct += int(predicted == true_label_readable)
        total += 1

        if model.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {'correct': correct, 'total': total, 'accuracy': correct / total if total else 0.0}


def plot_diagnostics(mu_results: List[MarginalUtilityResult], results_payload: Dict,
                      output_dir: Path, dataset_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_query: Dict[int, List[float]] = defaultdict(list)
    for r in mu_results:
        by_query[r.query_idx].append(r.marginal_utility)
    per_query_range = [max(v) - min(v) for v in by_query.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(per_query_range, bins=30, color='tab:blue', edgecolor='black')
    axes[0].set_xlabel("Marginal utility range within candidate pool")
    axes[0].set_ylabel("Number of queries")
    axes[0].set_title(f"{dataset_name}: utility spread per query")

    if 'oracle' in results_payload:
        methods = ['0-shot', 'CLIP-top-1', 'Oracle-best']
        accs = [
            results_payload['zero_shot']['accuracy'],
            results_payload['clip_top1']['accuracy'],
            results_payload['oracle']['accuracy'],
        ]
        axes[1].bar(methods, accs, color=['tab:gray', 'tab:orange', 'tab:green'])
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"{dataset_name}: oracle-gap accuracy")
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plot_path = output_dir / "diagnostic_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {plot_path}")


def save_results(mu_results: List[MarginalUtilityResult], results_payload: Dict,
                  output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "diagnostic_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    with open(output_dir / "diagnostic_results.pkl", "wb") as f:
        pickle.dump({**results_payload, 'marginal_utility_results': mu_results}, f)

    print(f"✓ Saved diagnostic results to {output_dir}/diagnostic_results.{{json,pkl}}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether a dataset has learnable reranker signal"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=ALL_DATASETS,
                        help="Dataset to diagnose")
    parser.add_argument("--marginal-utility-results", type=str, required=True,
                        help="Path to marginal_utilities_{split}.pkl from compute_marginal_utilities.py")
    parser.add_argument("--split", type=str, default="train",
                        help="Split the marginal utility results were computed on "
                             "(default: train, matching use_train_as_pool)")
    parser.add_argument("--image-split-path", type=str, default=None,
                        help="Path to image-level split JSON, if used for this dataset")
    parser.add_argument("--model", type=str, default="HuggingFaceM4/idefics2-8b",
                        help="Idefics2 model to use for the oracle-gap accuracy comparison")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load Idefics2 in 8-bit mode")
    parser.add_argument("--candidate-batch-size", type=int, default=8,
                        help="Candidate labels processed in parallel (default: 8)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Cap the number of queries evaluated in the oracle-gap "
                             "comparison (default: all queries in the utility results)")
    parser.add_argument("--output-dir", type=str, default="outputs/diagnostics",
                        help="Where to save diagnostic results and plots")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Only compute utility-variance stats; skip the (expensive) "
                             "oracle-gap accuracy comparison")

    args = parser.parse_args()

    print(f"Loading marginal utility results from {args.marginal_utility_results}...")
    mu_results = load_marginal_utility_results(Path(args.marginal_utility_results))
    print(f"✓ Loaded {len(mu_results)} (query, candidate) pairs\n")

    # --- Diagnostic 1: utility variance ---
    variance_stats = compute_utility_variance_stats(mu_results)
    print("=" * 70)
    print("DIAGNOSTIC 1: Marginal utility variance within candidate pools")
    print("=" * 70)
    for key, value in variance_stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print()
    if variance_stats['mean_per_query_std'] < 0.02:
        print("  ⚠️  Utility barely varies across candidates -- CLIP similarity choice")
        print("      likely doesn't matter much; weak signal for a reranker to learn.")
    else:
        print("  ✓ Utility varies meaningfully across candidates -- there is headroom")
        print("    for a reranker to distinguish good from bad candidates.")
    print()

    output_dir = Path(args.output_dir) / args.dataset
    results_payload = {'dataset': args.dataset, 'variance_stats': variance_stats}

    if args.skip_classification:
        save_results(mu_results, results_payload, output_dir)
        plot_diagnostics(mu_results, results_payload, output_dir, args.dataset)
        return

    # --- Diagnostic 2: oracle-gap accuracy ---
    oracle_lookup = build_oracle_lookup(mu_results)
    clip_top1_lookup = build_clip_top1_lookup(mu_results)
    query_indices = sorted(oracle_lookup.keys())
    if args.max_queries:
        query_indices = query_indices[:args.max_queries]
    print(f"Running oracle-gap accuracy comparison on {len(query_indices)} queries...\n")

    dataset = load_dataset(args.dataset, split=args.split, image_split_path=args.image_split_path)
    device, _, _ = _setup_device(1)
    model = Idefics2Wrapper(model_name=args.model, device=device, load_in_8bit=args.load_in_8bit)

    candidate_labels = sorted({get_readable_name(ex.label_name) for ex in dataset.examples})

    zero_shot = run_condition(model, dataset, query_indices, None,
                               candidate_labels, args.candidate_batch_size, "0-shot")
    clip_top1 = run_condition(model, dataset, query_indices, clip_top1_lookup,
                               candidate_labels, args.candidate_batch_size, "CLIP-top-1")
    oracle = run_condition(model, dataset, query_indices, oracle_lookup,
                            candidate_labels, args.candidate_batch_size, "Oracle")

    gap = oracle['accuracy'] - clip_top1['accuracy']

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Oracle-gap accuracy")
    print("=" * 70)
    print(f"  0-shot accuracy:      {zero_shot['accuracy']:.2%} ({zero_shot['correct']}/{zero_shot['total']})")
    print(f"  CLIP-top-1 accuracy:  {clip_top1['accuracy']:.2%} ({clip_top1['correct']}/{clip_top1['total']})")
    print(f"  Oracle-best accuracy: {oracle['accuracy']:.2%} ({oracle['correct']}/{oracle['total']})")
    print(f"\n  Oracle - CLIP gap: {gap:+.2%}")
    if gap < 0.03:
        print("  ⚠️  Small gap -- CLIP similarity already picks near-optimal examples;")
        print("      a learned reranker has little headroom to improve over CLIP retrieval.")
    else:
        print("  ✓ Meaningful gap -- CLIP similarity leaves accuracy on the table that")
        print("    a learned reranker could potentially recover.")
    print()

    results_payload.update({
        'zero_shot': zero_shot,
        'clip_top1': clip_top1,
        'oracle': oracle,
        'oracle_clip_gap': gap,
    })

    save_results(mu_results, results_payload, output_dir)
    plot_diagnostics(mu_results, results_payload, output_dir, args.dataset)


if __name__ == "__main__":
    main()
