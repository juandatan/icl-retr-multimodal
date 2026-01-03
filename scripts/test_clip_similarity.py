"""
Test script to verify CLIP embeddings and semantic similarity retrieval.

This script loads pre-computed CLIP embeddings and tests the semantic
similarity retrieval functionality.

Usage:
    python scripts/test_clip_similarity.py
    python scripts/test_clip_similarity.py --query_idx 100 --top_k 10
"""

import argparse
import sys
from pathlib import Path

import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.stanford_cars import StanfordCarsDataset


def visualize_similar_images(dataset, query_idx, top_k_indices, top_k_scores, save_path=None):
    """Visualize query image and top-k similar images."""
    n_show = min(len(top_k_indices) + 1, 6)  # Show query + top 5
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 5))

    # Query image
    query_example, query_image = dataset[query_idx]
    axes[0].imshow(query_image)
    axes[0].set_title(f"Query (idx={query_idx})\n{query_example.label_name}", fontsize=10)
    axes[0].axis('off')

    # Top-k similar images
    for i, (idx, score) in enumerate(zip(top_k_indices[:n_show-1], top_k_scores[:n_show-1])):
        example, image = dataset[idx]
        axes[i+1].imshow(image)

        same_class = "✓" if example.label == query_example.label else "✗"
        axes[i+1].set_title(
            f"Rank {i+1} (idx={idx})\nSim: {score:.3f} {same_class}\n{example.label_name}",
            fontsize=9
        )
        axes[i+1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test CLIP semantic similarity retrieval"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/stanford_cars',
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Which split to test (default: train)'
    )
    parser.add_argument(
        '--query_idx',
        type=int,
        default=0,
        help='Index of query example (default: 0)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of similar examples to retrieve (default: 10)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize query and top-k similar images'
    )
    parser.add_argument(
        '--exclude_same_class',
        action='store_true',
        help='Exclude examples from the same class'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CLIP Semantic Similarity Test")
    print(f"{'='*70}\n")

    # Load dataset
    print(f"Loading {args.split} split...")
    dataset = StanfordCarsDataset(
        split=args.split,
        data_dir=args.data_dir,
    )
    print(f"✓ Loaded {len(dataset)} examples\n")

    # Load cached embeddings
    print("Loading CLIP embeddings...")
    if not dataset.load_clip_embeddings():
        print("ERROR: CLIP embeddings not found!")
        print("Please run: python scripts/build_clip_embeddings.py")
        return 1

    if dataset.clip_embeddings is not None:
        print(f"✓ CLIP embeddings loaded: {dataset.clip_embeddings.shape}\n")

    # Validate query index
    if args.query_idx >= len(dataset):
        print(f"ERROR: query_idx {args.query_idx} out of range [0, {len(dataset)-1}]")
        return 1

    # Get query example
    query_example, query_image = dataset[args.query_idx]
    print(f"Query Example:")
    print(f"  Index: {args.query_idx}")
    print(f"  Label: {query_example.label}")
    print(f"  Label Name: {query_example.label_name}")
    print(f"  Image Size: {query_image.size}\n")

    # Retrieve top-k similar examples
    print(f"Retrieving top-{args.top_k} similar examples...")
    top_k_indices, top_k_scores = dataset.get_top_k_similar(
        query_idx=args.query_idx,
        k=args.top_k,
        exclude_query=True,
        exclude_same_class=args.exclude_same_class,
    )

    print(f"\nTop-{len(top_k_indices)} similar examples:\n")
    print(f"{'Rank':<6} {'Index':<8} {'Similarity':<12} {'Same Class':<12} {'Label Name'}")
    print(f"{'-'*70}")

    for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        example = dataset.examples[idx]
        same_class = "✓" if example.label == query_example.label else "✗"
        print(f"{i+1:<6} {idx:<8} {score:<12.4f} {same_class:<12} {example.label_name}")

    # Compute statistics
    same_class_count = sum(
        1 for idx in top_k_indices
        if dataset.examples[idx].label == query_example.label
    )

    print(f"\n{'-'*70}")
    print(f"Statistics:")
    print(f"  Same class in top-{args.top_k}: {same_class_count}/{len(top_k_indices)} "
          f"({100*same_class_count/len(top_k_indices):.1f}%)")
    print(f"  Similarity range: [{min(top_k_scores):.4f}, {max(top_k_scores):.4f}]")
    print(f"  Mean similarity: {sum(top_k_scores)/len(top_k_scores):.4f}")

    # Visualize if requested
    if args.visualize:
        print(f"\nGenerating visualization...")
        save_path = f"clip_similarity_query_{args.query_idx}.png"
        visualize_similar_images(
            dataset, args.query_idx, top_k_indices, top_k_scores,
            save_path=save_path
        )

    print(f"\n✓ Test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
