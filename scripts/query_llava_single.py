"""
Query LLaVA with a single test image and optional in-context examples.

This script allows you to:
1. Select a query image from Mini-ImageNet test set
2. Select examples via CLIP similarity, reranker, or manual indices
3. Customize the prompt text
4. Query LLaVA and see the response

Usage:
    # Query with no examples (k=0)
    python scripts/query_llava_single.py \
        --query-idx 0

    # Query with manually specified examples
    python scripts/query_llava_single.py \
        --query-idx 0 \
        --example-indices 10 25 42

    # Query with CLIP-selected examples
    python scripts/query_llava_single.py \
        --query-idx 0 \
        --use-clip-retrieval \
        --k 3

    # Query with reranker-selected examples
    python scripts/query_llava_single.py \
        --query-idx 0 \
        --use-reranker \
        --reranker-checkpoint outputs/reranker_checkpoints/reranker_mini_imagenet_v2/best_model.pt \
        --k 3

    # Use custom prompt text
    python scripts/query_llava_single.py \
        --query-idx 0 \
        --k 1 \
        --use-clip-retrieval \
        --prompt-text "What breed of dog is in this image?"

    # Use generative mode (free-form generation)
    python scripts/query_llava_single.py \
        --query-idx 0 \
        --k 1 \
        --use-clip-retrieval \
        --use-generative
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.mini_imagenet import MiniImageNetDataset
from data.marginal_utility_dataset import InteractionFeaturesConfig
from models.mlp_reranker import MLPReranker
from models.llava_wrapper import LLaVAWrapper
from utils.imagenet_names import get_readable_name, get_synset_id, IMAGENET_SYNSET_TO_NAME


def compute_similarity(query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and candidates."""
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    candidate_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)

    # Cosine similarity
    similarities = candidate_norms @ query_norm
    return similarities


def retrieve_by_clip(
    query_emb: np.ndarray,
    train_dataset,
    k: int
) -> List[int]:
    """Retrieve top-K examples using CLIP similarity."""
    candidate_embs = train_dataset.clip_embeddings

    # Compute similarities
    similarities = compute_similarity(query_emb, candidate_embs)

    # Get top-K
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return top_k_indices.tolist()


def retrieve_by_reranker(
    query_emb: np.ndarray,
    train_dataset,
    reranker: MLPReranker,
    interaction_features: InteractionFeaturesConfig,
    device: str,
    k: int
) -> List[int]:
    """Retrieve top-K examples using learned reranker."""
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

    # Get top-K
    top_k_indices = np.argsort(utilities)[-k:][::-1]

    return top_k_indices.tolist()


def load_reranker(checkpoint_path: str, device: str):
    """Load trained reranker model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    print(f"\nLoading reranker from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    model = MLPReranker(
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


def main():
    parser = argparse.ArgumentParser(description="Query LLaVA with a single image and examples")
    parser.add_argument("--dataset", type=str, default="mini_imagenet",
                        help="Dataset to use (currently only mini_imagenet supported)")
    parser.add_argument("--query-idx", type=int, required=True,
                        help="Index of query image in test set")

    # Example selection methods (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--example-indices", type=int, nargs="*", default=None,
                        help="Manually specify indices of example images in train set")
    selection_group.add_argument("--use-clip-retrieval", action="store_true",
                        help="Use CLIP similarity to select examples")
    selection_group.add_argument("--use-reranker", action="store_true",
                        help="Use learned reranker to select examples")

    parser.add_argument("--k", type=int, default=1,
                        help="Number of examples to retrieve (for CLIP/reranker)")
    parser.add_argument("--reranker-checkpoint", type=str, default=None,
                        help="Path to reranker checkpoint (required if --use-reranker)")

    # Prompt customization
    parser.add_argument("--prompt-text", type=str, default=None,
                        help="Custom prompt text (overrides default classification prompt)")

    # Model settings
    parser.add_argument("--llava-model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model to use")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load LLaVA in 8-bit mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--use-generative", action="store_true",
                        help="Use generative evaluation (free-form generation) instead of discriminative")

    args = parser.parse_args()

    # Validate arguments
    if args.use_reranker and not args.reranker_checkpoint:
        parser.error("--reranker-checkpoint is required when using --use-reranker")

    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load datasets
    print(f"\nLoading {args.dataset} dataset...")

    if args.dataset == "mini_imagenet":
        test_dataset = MiniImageNetDataset(
            split="test",
            data_dir="data/mini_imagenet",
            class_split_seed=42
        )
        # Only load train dataset if we need it for example selection
        train_dataset = None
        need_train_dataset = (
            (args.example_indices is not None and len(args.example_indices) > 0) or
            args.use_clip_retrieval or
            args.use_reranker
        )
        if need_train_dataset:
            train_dataset = MiniImageNetDataset(
                split="train",
                data_dir="data/mini_imagenet",
                class_split_seed=42
            )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load CLIP embeddings (only what we need)
    test_dataset.load_clip_embeddings()
    print(f"✓ Test set: {len(test_dataset)} examples")

    if train_dataset is not None:
        train_dataset.load_clip_embeddings()
        print(f"✓ Train set: {len(train_dataset)} examples")
    else:
        print(f"✓ Train set: not loaded (no examples needed)")

    # Get query image and label
    query_example, query_image = test_dataset[args.query_idx]
    query_emb = test_dataset.clip_embeddings[args.query_idx]
    true_label = query_example.label
    true_label_name = query_example.label_name
    true_label_readable = get_readable_name(true_label_name)

    print(f"\n{'='*70}")
    print(f"QUERY IMAGE")
    print(f"{'='*70}")
    print(f"Index: {args.query_idx}")
    print(f"True label: {true_label_name} ({true_label_readable})")
    print(f"Image size: {query_image.size}")

    # Determine example indices
    example_indices = []
    if args.example_indices is not None:
        example_indices = args.example_indices
        print(f"\nUsing manually specified examples: {example_indices}")
    elif args.use_clip_retrieval:
        print(f"\nRetrieving top-{args.k} examples using CLIP similarity...")
        example_indices = retrieve_by_clip(query_emb, train_dataset, args.k)
        print(f"Selected indices: {example_indices}")
    elif args.use_reranker:
        # Load reranker
        reranker, interaction_features = load_reranker(args.reranker_checkpoint, device)
        print(f"\nRetrieving top-{args.k} examples using reranker...")
        example_indices = retrieve_by_reranker(
            query_emb, train_dataset, reranker, interaction_features, device, args.k
        )
        print(f"Selected indices: {example_indices}")

    # Build context examples
    context_examples = []
    if len(example_indices) > 0:
        print(f"\n{'='*70}")
        print(f"EXAMPLE IMAGES (k={len(example_indices)})")
        print(f"{'='*70}")

        for i, ex_idx in enumerate(example_indices):
            ex_example, ex_image = train_dataset[ex_idx]
            ex_label_text = ex_example.label_name
            ex_label_readable = get_readable_name(ex_label_text)

            # Convert to readable name for generative evaluation
            if args.use_generative:
                ex_label_text = ex_label_readable

            context_examples.append((ex_image, ex_label_text))

            print(f"Example {i+1}:")
            print(f"  Index: {ex_idx}")
            print(f"  Label: {ex_example.label_name} ({ex_label_readable})")
            print(f"  Image size: {ex_image.size}")

    # Initialize LLaVA
    print(f"\n{'='*70}")
    print(f"INITIALIZING LLAVA")
    print(f"{'='*70}")
    print(f"Model: {args.llava_model}")
    print(f"8-bit mode: {args.load_in_8bit}")

    llava_model = LLaVAWrapper(
        model_name=args.llava_model,
        device=device,
        load_in_8bit=args.load_in_8bit
    )
    print("✓ LLaVA model loaded")

    # Get candidate labels (for standard classification mode)
    if args.use_generative:
        # Use all 100 Mini-ImageNet classes
        candidate_label_names = sorted([name for name in IMAGENET_SYNSET_TO_NAME.values()])
    else:
        # For discriminative evaluation, use only test split classes
        candidate_label_names = [ex.label_name for ex in test_dataset.examples]
        # Deduplicate while preserving order
        seen = set()
        candidate_label_names = [x for x in candidate_label_names if not (x in seen or seen.add(x))]

    # Query LLaVA
    print(f"\n{'='*70}")
    print(f"QUERYING LLAVA")
    print(f"{'='*70}")
    print(f"Evaluation mode: {'Generative' if args.use_generative else 'Discriminative (probability-based)'}")
    print(f"Number of examples: {len(context_examples)}")
    print(f"Number of candidates: {len(candidate_label_names)}")

    if args.prompt_text:
        print(f"Custom prompt: {args.prompt_text}")

    if args.use_generative:
        if args.prompt_text:
            # Use custom prompt with generative mode
            predicted_label_text = llava_model.classify_with_context_generative(
                query_image=query_image,
                context_examples=context_examples,
                candidate_labels=candidate_label_names,
                custom_prompt=args.prompt_text
            )
        else:
            predicted_label_text = llava_model.classify_with_context_generative(
                query_image=query_image,
                context_examples=context_examples,
                candidate_labels=candidate_label_names
            )
        # Convert back from readable name to synset ID for comparison
        predicted_synset = get_synset_id(predicted_label_text)
    else:
        if args.prompt_text:
            # Use custom prompt with discriminative mode
            predicted_label_text = llava_model.classify_with_context(
                query_image=query_image,
                context_examples=context_examples,
                candidate_labels=candidate_label_names,
                custom_prompt=args.prompt_text
            )
        else:
            predicted_label_text = llava_model.classify_with_context(
                query_image=query_image,
                context_examples=context_examples,
                candidate_labels=candidate_label_names
            )
        predicted_synset = predicted_label_text

    # Convert prediction to label index
    predicted_label = -1
    for ex in test_dataset.examples:
        if ex.label_name == predicted_synset:
            predicted_label = ex.label
            break

    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    if args.use_generative:
        print(f"Raw prediction: {predicted_label_text}")
        print(f"Mapped to synset: {predicted_synset}")
    else:
        print(f"Predicted label: {predicted_label_text}")

    predicted_readable = get_readable_name(predicted_synset) if predicted_synset != predicted_label_text else predicted_label_text
    print(f"Predicted (readable): {predicted_readable}")
    print(f"True label: {true_label_name} ({true_label_readable})")

    is_correct = (predicted_label == true_label)
    print(f"\nResult: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    if not is_correct:
        print(f"\nPredicted index: {predicted_label}")
        print(f"True index: {true_label}")


if __name__ == "__main__":
    main()
