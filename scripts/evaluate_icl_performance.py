"""
Evaluate In-Context Learning (ICL) performance using different retrieval methods.

This script compares:
1. CLIP similarity baseline: retrieve top-K examples by cosine similarity
2. Learned reranker: retrieve top-K examples by predicted marginal utility

For each method, we:
- Select K in-context examples for each test query
- Query LLaVA for classification
- Measure accuracy

Usage:
    # Mini-ImageNet with k=1
    python scripts/evaluate_icl_performance.py \
        --dataset mini_imagenet \
        --reranker-checkpoint outputs/reranker_checkpoints/reranker_mini_imagenet_v2/best_model.pt \
        --k 1 \
        --num-queries 100

    # Stanford Cars with k=1
    python scripts/evaluate_icl_performance.py \
        --dataset stanford_cars \
        --reranker-checkpoint outputs/reranker_checkpoints/reranker_stanford_cars/best_model.pt \
        --k 1 \
        --num-queries 100
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.stanford_cars import StanfordCarsDataset
from data.mini_imagenet import MiniImageNetDataset
from data.marginal_utility_dataset import InteractionFeaturesConfig
from models.reranker import CLIPReranker
from models.llava_wrapper import LLaVAWrapper


def load_dataset(dataset_name: str, split: str = "test"):
    """Load dataset for evaluation."""
    print(f"\nLoading {dataset_name} dataset ({split} split)...")

    if dataset_name == "stanford_cars":
        dataset = StanfordCarsDataset(
            split=split,
            data_dir="data/stanford_cars",
            class_split_seed=42
        )
    elif dataset_name == "mini_imagenet":
        dataset = MiniImageNetDataset(
            split=split,
            data_dir="data/mini_imagenet",
            class_split_seed=42
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load CLIP embeddings
    success = dataset.load_clip_embeddings()
    if not success:
        raise FileNotFoundError(
            f"CLIP embeddings not found. Please run: "
            f"python scripts/build_clip_embeddings.py --dataset {dataset_name} --splits {split}"
        )

    print(f"✓ Loaded {len(dataset)} examples")
    print(f"✓ Embeddings shape: {dataset.clip_embeddings.shape}")
    print(f"✓ Num classes: {dataset.num_classes}")

    return dataset


def load_reranker(checkpoint_path: str, device: str) -> CLIPReranker:
    """Load trained reranker model from checkpoint."""
    print(f"\nLoading reranker from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    config = checkpoint['config']
    model_config = config['model']

    # Create interaction features config
    interaction_features = InteractionFeaturesConfig(
        use_product=model_config.get('use_product', False),
        use_difference=model_config.get('use_difference', False),
        use_l2_distance=model_config.get('use_l2_distance', False)
    )

    # Initialize model
    model = CLIPReranker(
        embedding_dim=model_config['embedding_dim'],
        hidden_dims=model_config['hidden_dims'],
        dropout=model_config.get('dropout', 0.1),
        interaction_features=interaction_features,
        use_sigmoid=model_config.get('use_sigmoid', False)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model with {model.get_num_parameters():,} parameters")
    print(f"✓ Interaction features: {interaction_features}")

    return model, interaction_features


def compute_similarity(query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and candidates."""
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    candidate_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)

    # Cosine similarity
    similarities = candidate_norms @ query_norm
    return similarities


def retrieve_by_clip(
    query_idx: int,
    train_dataset,
    k: int
) -> List[int]:
    """Retrieve top-K examples using CLIP similarity."""
    query_emb = train_dataset.clip_embeddings[query_idx]
    candidate_embs = train_dataset.clip_embeddings

    # Compute similarities
    similarities = compute_similarity(query_emb, candidate_embs)

    # Exclude self
    similarities[query_idx] = -np.inf

    # Get top-K
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return top_k_indices.tolist()


def retrieve_by_reranker(
    query_idx: int,
    train_dataset,
    reranker: CLIPReranker,
    interaction_features: InteractionFeaturesConfig,
    device: str,
    k: int
) -> List[int]:
    """Retrieve top-K examples using learned reranker."""
    query_emb = train_dataset.clip_embeddings[query_idx]
    candidate_embs = train_dataset.clip_embeddings

    # Compute CLIP similarities first (needed as input to reranker)
    similarities = compute_similarity(query_emb, candidate_embs)

    # Prepare batch inputs for reranker
    query_emb_tensor = torch.from_numpy(query_emb).float().to(device)
    candidate_embs_tensor = torch.from_numpy(candidate_embs).float().to(device)
    similarities_tensor = torch.from_numpy(similarities).float().to(device)

    # Expand query embedding to match batch size
    query_emb_batch = query_emb_tensor.unsqueeze(0).expand(len(candidate_embs), -1)

    # Compute interaction features if needed
    product = None
    difference = None
    l2_distance = None

    if interaction_features.use_product:
        product = query_emb_batch * candidate_embs_tensor
    if interaction_features.use_difference:
        difference = query_emb_batch - candidate_embs_tensor
    if interaction_features.use_l2_distance:
        l2_distance = torch.norm(query_emb_batch - candidate_embs_tensor, dim=1, keepdim=True)

    # Get predictions
    with torch.no_grad():
        utilities = reranker(
            query_emb_batch,
            candidate_embs_tensor,
            similarities_tensor.unsqueeze(1),
            product=product,
            difference=difference,
            l2_distance=l2_distance
        ).squeeze().cpu().numpy()

    # Exclude self
    utilities[query_idx] = -np.inf

    # Get top-K
    top_k_indices = np.argsort(utilities)[-k:][::-1]

    return top_k_indices.tolist()


def evaluate_icl(
    test_dataset,
    train_dataset,
    llava_model: LLaVAWrapper,
    retrieval_fn,
    k: int,
    num_queries: int = None,
    seed: int = 42,
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate ICL performance using a given retrieval method.

    Args:
        test_dataset: Test examples to query on
        train_dataset: Training examples to retrieve from
        llava_model: LLaVA model for classification
        retrieval_fn: Function(query_idx, train_dataset, k) -> List[example_indices]
        k: Number of in-context examples
        num_queries: Number of test queries to evaluate (None = all)
        seed: Random seed for reproducibility
        return_predictions: If True, return detailed predictions for each query

    Returns:
        Dictionary with accuracy and per-class results
    """
    random.seed(seed)

    # Sample queries if specified
    query_indices = list(range(len(test_dataset)))
    if num_queries is not None:
        query_indices = random.sample(query_indices, min(num_queries, len(test_dataset)))

    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    predictions = []

    print(f"\nEvaluating on {len(query_indices)} queries with k={k}...")

    for query_idx in tqdm(query_indices, desc="Querying LLaVA"):
        # Get query example
        query_image = test_dataset.get_image(query_idx)
        true_label = test_dataset.labels[query_idx]

        # Retrieve k examples
        example_indices = retrieval_fn(query_idx, train_dataset, k)

        # Build ICL prompt
        context_examples = []
        for ex_idx in example_indices:
            ex_image = train_dataset.get_image(ex_idx)
            ex_label = train_dataset.labels[ex_idx]
            ex_label_text = train_dataset.label_names[ex_label]
            context_examples.append((ex_image, ex_label_text))

        # Query LLaVA (get probabilities if possible)
        predicted_label_text = llava_model.classify_with_context(
            query_image=query_image,
            context_examples=context_examples,
            candidate_labels=train_dataset.label_names
        )

        # Convert prediction to label index
        try:
            predicted_label = train_dataset.label_names.index(predicted_label_text)
        except ValueError:
            # If prediction not in label list, mark as incorrect
            predicted_label = -1

        # Track accuracy
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
        total += 1

        # Track per-class accuracy
        if true_label not in per_class_correct:
            per_class_correct[true_label] = 0
            per_class_total[true_label] = 0

        if is_correct:
            per_class_correct[true_label] += 1
        per_class_total[true_label] += 1

        # Store detailed prediction info
        if return_predictions:
            predictions.append({
                'query_idx': query_idx,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'is_correct': is_correct,
                'example_indices': example_indices,
                'predicted_label_text': predicted_label_text
            })

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0

    per_class_accuracy = {}
    for label in per_class_total:
        per_class_accuracy[label] = (
            per_class_correct[label] / per_class_total[label]
            if per_class_total[label] > 0 else 0.0
        )

    mean_per_class_accuracy = np.mean(list(per_class_accuracy.values()))

    results = {
        'accuracy': accuracy,
        'mean_per_class_accuracy': mean_per_class_accuracy,
        'correct': correct,
        'total': total,
        'per_class_accuracy': per_class_accuracy
    }

    if return_predictions:
        results['predictions'] = predictions

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL performance with different retrieval methods")
    parser.add_argument("--dataset", type=str, required=True, choices=["stanford_cars", "mini_imagenet"],
                        help="Dataset to evaluate on")
    parser.add_argument("--reranker-checkpoint", type=str, default=None,
                        help="Path to trained reranker checkpoint (required for reranker evaluation)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of in-context examples")
    parser.add_argument("--num-queries", type=int, default=None,
                        help="Number of test queries to evaluate (default: all)")
    parser.add_argument("--llava-model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model to use")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load LLaVA in 8-bit mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path to save results")

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load datasets
    test_dataset = load_dataset(args.dataset, split="test")
    train_dataset = load_dataset(args.dataset, split="train")

    # Load reranker if provided
    reranker = None
    interaction_features = None
    if args.reranker_checkpoint:
        reranker, interaction_features = load_reranker(args.reranker_checkpoint, device)

    # Initialize LLaVA
    print(f"\nInitializing LLaVA model: {args.llava_model}")
    llava_model = LLaVAWrapper(
        model_name=args.llava_model,
        device=device,
        load_in_8bit=args.load_in_8bit
    )
    print("✓ LLaVA model loaded")

    # Evaluate CLIP similarity baseline
    print("\n" + "="*70)
    print("EVALUATING: CLIP Similarity Baseline")
    print("="*70)

    def clip_retrieval_fn(query_idx, train_ds, k):
        return retrieve_by_clip(query_idx, train_ds, k)

    clip_results = evaluate_icl(
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        llava_model=llava_model,
        retrieval_fn=clip_retrieval_fn,
        k=args.k,
        num_queries=args.num_queries,
        seed=args.seed,
        return_predictions=True
    )

    print(f"\nCLIP Baseline Results:")
    print(f"  Accuracy: {clip_results['accuracy']:.2%}")
    print(f"  Mean per-class accuracy: {clip_results['mean_per_class_accuracy']:.2%}")
    print(f"  Correct: {clip_results['correct']}/{clip_results['total']}")

    # Evaluate reranker if provided
    reranker_results = None
    if reranker:
        print("\n" + "="*70)
        print("EVALUATING: Learned Reranker")
        print("="*70)

        def reranker_retrieval_fn(query_idx, train_ds, k):
            return retrieve_by_reranker(
                query_idx, train_ds, reranker, interaction_features, device, k
            )

        reranker_results = evaluate_icl(
            test_dataset=test_dataset,
            train_dataset=train_dataset,
            llava_model=llava_model,
            retrieval_fn=reranker_retrieval_fn,
            k=args.k,
            num_queries=args.num_queries,
            seed=args.seed,
            return_predictions=True
        )

        print(f"\nReranker Results:")
        print(f"  Accuracy: {reranker_results['accuracy']:.2%}")
        print(f"  Mean per-class accuracy: {reranker_results['mean_per_class_accuracy']:.2%}")
        print(f"  Correct: {reranker_results['correct']}/{reranker_results['total']}")

        # Compute improvement
        improvement = reranker_results['accuracy'] - clip_results['accuracy']
        relative_improvement = (improvement / clip_results['accuracy']) * 100 if clip_results['accuracy'] > 0 else 0

        print(f"\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Absolute improvement: {improvement:+.2%}")
        print(f"Relative improvement: {relative_improvement:+.1f}%")

        # Detailed comparison analysis
        print(f"\n" + "="*70)
        print("DETAILED COMPARISON")
        print("="*70)

        clip_preds = clip_results['predictions']
        reranker_preds = reranker_results['predictions']

        # Track comparison categories
        reranker_wins = 0  # Reranker correct, CLIP wrong
        clip_wins = 0      # CLIP correct, reranker wrong
        both_correct = 0   # Both correct
        both_wrong = 0     # Both wrong

        # Track when both correct but selected different examples
        both_correct_diff_examples = []

        for clip_pred, reranker_pred in zip(clip_preds, reranker_preds):
            assert clip_pred['query_idx'] == reranker_pred['query_idx']

            clip_correct = clip_pred['is_correct']
            reranker_correct = reranker_pred['is_correct']

            if reranker_correct and not clip_correct:
                reranker_wins += 1
            elif clip_correct and not reranker_correct:
                clip_wins += 1
            elif clip_correct and reranker_correct:
                both_correct += 1
                # Check if they selected different examples
                if clip_pred['example_indices'] != reranker_pred['example_indices']:
                    both_correct_diff_examples.append({
                        'query_idx': clip_pred['query_idx'],
                        'clip_examples': clip_pred['example_indices'],
                        'reranker_examples': reranker_pred['example_indices']
                    })
            else:
                both_wrong += 1

        total_queries = len(clip_preds)

        print(f"\nOutcome Categories:")
        print(f"  Reranker wins (reranker ✓, CLIP ✗):     {reranker_wins:4d} ({reranker_wins/total_queries:6.2%})")
        print(f"  CLIP wins (CLIP ✓, reranker ✗):         {clip_wins:4d} ({clip_wins/total_queries:6.2%})")
        print(f"  Both correct:                            {both_correct:4d} ({both_correct/total_queries:6.2%})")
        print(f"  Both wrong:                              {both_wrong:4d} ({both_wrong/total_queries:6.2%})")
        print(f"  Total:                                   {total_queries:4d}")

        print(f"\nNet gain from reranker: {reranker_wins - clip_wins:+d} queries")

        print(f"\nWhen both methods are correct:")
        print(f"  Same examples selected:    {both_correct - len(both_correct_diff_examples):4d}")
        print(f"  Different examples:        {len(both_correct_diff_examples):4d}")

        if len(both_correct_diff_examples) > 0:
            print(f"\n  Note: Even when both are correct, reranker selects different")
            print(f"        examples in {len(both_correct_diff_examples)/both_correct:.1%} of cases")

        # Analyze example selection overlap
        if args.k == 1:
            same_examples = sum(
                1 for cp, rp in zip(clip_preds, reranker_preds)
                if cp['example_indices'] == rp['example_indices']
            )
            print(f"\nExample Selection Overlap:")
            print(f"  Same top-1 example selected: {same_examples}/{total_queries} ({same_examples/total_queries:.2%})")
            print(f"  Different top-1 selected:    {total_queries - same_examples}/{total_queries} ({(total_queries - same_examples)/total_queries:.2%})")

        # Store comparison details in results
        reranker_results['comparison'] = {
            'reranker_wins': reranker_wins,
            'clip_wins': clip_wins,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'both_correct_diff_examples': both_correct_diff_examples
        }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'dataset': args.dataset,
            'k': args.k,
            'num_queries': args.num_queries or len(test_dataset),
            'clip_results': clip_results,
            'reranker_results': reranker_results,
            'args': vars(args)
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
