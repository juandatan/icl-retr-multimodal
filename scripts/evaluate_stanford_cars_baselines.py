"""
Evaluate Stanford Cars baseline retrieval methods: 0-shot and 1-shot CLIP similarity.

These are static benchmarks — run once and store as reference points for reranker comparison.
Results are saved to outputs/evals/ via save_eval_results.

Usage:
    # Image-level split (within-distribution, recommended)
    python scripts/evaluate_stanford_cars_baselines.py \
        --image-split-path data/stanford_cars/image_split.json \
        --eval-split test

    # Class-level split (cross-class generalisation)
    python scripts/evaluate_stanford_cars_baselines.py \
        --eval-split val+test
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.eval_utils import save_eval_results

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_icl_performance import (
    _setup_device,
    evaluate_icl_multigpu,
    load_dataset,
    CombinedDataset,
    determine_retrieval_split,
    get_cache_path,
    load_cached_results,
    save_cached_results,
    evaluate_icl,
    retrieve_by_clip,
    Idefics2Wrapper,
)


def _run_baseline(
    *,
    method: str,
    retrieval_fn,
    k: int,
    candidate_pool_size: int,
    test_dataset,
    retrieval_dataset,
    vlm,
    test_dataset_size: int,
    eval_split: str,
    retrieval_split: str,
    args,
    device: str,
    num_gpus: int,
    use_multi_gpu: bool,
    num_queries: int,
    eval_mode: str,
    run_id_parts: dict,
    use_all_classes: bool = False,
):
    """Run one baseline evaluation (single-GPU or multi-GPU) and save results."""
    cache = get_cache_path(
        dataset_name="stanford_cars",
        method=method,
        k=k,
        num_queries=num_queries,
        seed=args.seed,
        use_generative=args.use_generative,
        use_all_classes=use_all_classes,
    )

    results = None
    if args.use_cache and not args.force_recompute:
        results = load_cached_results(cache)

    if results is None:
        print("\n" + "=" * 70)
        print(f"EVALUATING: {method.replace('_', ' ').upper()}")
        print("=" * 70)

        if use_multi_gpu:
            results = evaluate_icl_multigpu(
                dataset_name="stanford_cars",
                test_dataset_size=test_dataset_size,
                eval_split=eval_split,
                retrieval_split=retrieval_split,
                reranker_checkpoint=None,
                kaggle_dataset=None,
                llava_model_name=args.model,
                load_in_8bit=args.load_in_8bit,
                k=k,
                candidate_pool_size=candidate_pool_size,
                num_queries=num_queries,
                seed=args.seed,
                return_predictions=True,
                use_reranker=False,
                num_gpus=num_gpus,
                use_generative=args.use_generative,
                candidate_batch_size=args.candidate_batch_size,
                cache_path=cache,
                image_split_path=args.image_split_path,
                use_all_classes=use_all_classes,
                force_recompute=args.force_recompute,
            )
        else:
            results = evaluate_icl(
                test_dataset=test_dataset,
                retrieval_dataset=retrieval_dataset,
                llava_model=vlm,
                retrieval_fn=retrieval_fn,
                k=k,
                num_queries=args.num_queries,
                seed=args.seed,
                return_predictions=True,
                use_generative=args.use_generative,
                device=device,
                candidate_batch_size=args.candidate_batch_size,
                candidate_pool_size=candidate_pool_size,
            )

        if args.use_cache:
            save_cached_results(cache, results)

    print(f"\n{method} results:")
    print(f"  Accuracy:           {results['accuracy']:.2%}")
    print(f"  Mean per-class acc: {results['mean_per_class_accuracy']:.2%}")
    print(f"  Correct:            {results['correct']}/{results['total']}")

    save_eval_results(
        method=method,
        results=results,
        run_id_parts={**run_id_parts, "k": k, "pool": candidate_pool_size},
        args=vars(args),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 0-shot and 1-shot CLIP-retrieval baselines on Stanford Cars"
    )
    parser.add_argument("--eval-split", type=str, default="test",
                        choices=["train", "val", "test", "val+test"],
                        help="Which split(s) to evaluate on (default: test)")
    parser.add_argument("--retrieval-split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Which split to retrieve ICL examples from (default: auto)")
    parser.add_argument("--image-split-path", type=str, default=None,
                        help="Path to image-level split JSON for within-distribution eval")
    parser.add_argument("--model", type=str, default="HuggingFaceM4/idefics2-8b",
                        help="Vision-language model to use (default: idefics2-8b)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit mode")
    parser.add_argument("--num-queries", type=int, default=None,
                        help="Number of test queries to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load cached predictions if available")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recompute even if cache exists")
    parser.add_argument("--use-generative", action="store_true",
                        help="Use generative evaluation instead of discriminative")
    parser.add_argument("--candidate-batch-size", type=int, default=8,
                        help="Candidate labels processed in parallel (default: 8)")
    parser.add_argument("--skip-zero-shot", action="store_true",
                        help="Skip 0-shot evaluation")
    parser.add_argument("--skip-clip-retrieval", action="store_true",
                        help="Skip 1-shot CLIP-retrieval evaluation")
    parser.add_argument("--use-all-classes", action="store_true",
                        help="Score all 196 classes as candidates (discriminative only)")

    args = parser.parse_args()

    device, num_gpus, use_multi_gpu = _setup_device(args.num_gpus)

    retrieval_split = determine_retrieval_split(args.eval_split, args.retrieval_split)
    eval_mode = "generative" if args.use_generative else "discriminative"
    print(f"\nEvaluation setup:")
    print(f"  Dataset:      stanford_cars")
    print(f"  Queries from: {args.eval_split}")
    print(f"  ICL from:     {retrieval_split}")
    print(f"  Mode:         {eval_mode}")
    if args.image_split_path:
        print(f"  Image split:  {args.image_split_path}")

    print(f"\nLoading {args.eval_split} split(s)...")
    if args.eval_split == "val+test":
        temp_datasets = [
            load_dataset("stanford_cars", split="val", image_split_path=args.image_split_path),
            load_dataset("stanford_cars", split="test", image_split_path=args.image_split_path),
        ]
        test_dataset_size = sum(len(ds) for ds in temp_datasets)
        print(f"Combined val+test: {test_dataset_size} examples")
    else:
        temp_datasets = [load_dataset("stanford_cars", split=args.eval_split,
                                      image_split_path=args.image_split_path)]
        test_dataset_size = len(temp_datasets[0])
        print(f"{args.eval_split}: {test_dataset_size} examples")

    if use_multi_gpu:
        del temp_datasets
        test_dataset = None
        retrieval_dataset = None
    else:
        test_dataset = CombinedDataset(temp_datasets) if len(temp_datasets) > 1 else temp_datasets[0]
        retrieval_dataset = test_dataset

    vlm = None  # loaded once on first evaluation block that needs it
    num_queries = args.num_queries or test_dataset_size

    run_id_parts = {
        "dataset": "stanford_cars",
        "n": num_queries,
        "seed": args.seed,
        "mode": eval_mode,
    }

    shared = dict(
        test_dataset=test_dataset,
        retrieval_dataset=retrieval_dataset,
        test_dataset_size=test_dataset_size,
        eval_split=args.eval_split,
        retrieval_split=retrieval_split,
        args=args,
        device=device,
        num_gpus=num_gpus,
        use_multi_gpu=use_multi_gpu,
        num_queries=num_queries,
        eval_mode=eval_mode,
        run_id_parts=run_id_parts,
    )

    if not args.skip_zero_shot:
        if vlm is None and not use_multi_gpu:
            print(f"\nInitializing vision-language model: {args.model}")
            vlm = Idefics2Wrapper(model_name=args.model, device=device, load_in_8bit=args.load_in_8bit)

        _run_baseline(
            method="zero_shot",
            retrieval_fn=lambda *a, **kw: [],
            k=0,
            candidate_pool_size=0,
            vlm=vlm,
            use_all_classes=args.use_all_classes,
            **shared,
        )

    if not args.skip_clip_retrieval:
        if vlm is None and not use_multi_gpu:
            print(f"\nInitializing vision-language model: {args.model}")
            vlm = Idefics2Wrapper(model_name=args.model, device=device, load_in_8bit=args.load_in_8bit)

        def clip_fn(query_emb, retr_ds, k, exclude_indices=None, query_image=None):
            return retrieve_by_clip(query_emb, retr_ds, k, exclude_indices)

        _run_baseline(
            method="clip_retrieval_1shot",
            retrieval_fn=clip_fn,
            k=1,
            candidate_pool_size=50,
            vlm=vlm,
            use_all_classes=args.use_all_classes,
            **shared,
        )


if __name__ == "__main__":
    main()
