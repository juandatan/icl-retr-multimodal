"""
Script to build and cache CLIP embeddings for Stanford Cars dataset.

This script pre-computes CLIP embeddings for all images in the dataset,
enabling fast semantic similarity-based candidate retrieval during utility
computation and training.

Usage:
    python scripts/build_clip_embeddings.py --splits train val test
    python scripts/build_clip_embeddings.py --splits train  # Just train split
    python scripts/build_clip_embeddings.py --model ViT-L/14  # Use larger CLIP model
"""

import argparse
import sys
from pathlib import Path

import torch
import clip

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.stanford_cars import StanfordCarsDataset


def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP embeddings for Stanford Cars dataset"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/stanford_cars',
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='Which splits to process (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
        help='CLIP model to use (default: ViT-B/32)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for embedding computation (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use. If None, automatically selects best available.'
    )
    parser.add_argument(
        '--class_split_seed',
        type=int,
        default=42,
        help='Random seed for class splits (default: 42)'
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"\n{'='*70}")
    print(f"CLIP Embedding Generation for Stanford Cars")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  CLIP Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Splits: {', '.join(args.splits)}")
    print(f"  Data Directory: {args.data_dir}")
    print(f"\n{'='*70}\n")

    # Load CLIP model
    print(f"Loading CLIP model: {args.model}...")
    clip_model, clip_preprocess = clip.load(args.model, device=device)
    print(f"✓ CLIP model loaded successfully\n")

    # Process each split
    results = {}
    for split in args.splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}\n")

        # Load dataset
        dataset = StanfordCarsDataset(
            split=split,
            data_dir=args.data_dir,
            class_split_seed=args.class_split_seed,
        )

        print(f"Loaded {len(dataset)} examples\n")

        # Build embeddings
        embeddings = dataset.build_clip_embeddings(
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            batch_size=args.batch_size,
            device=device,
        )

        print(f"\n✓ Completed {split} split")
        print(f"  Examples: {len(dataset)}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Embedding dim: {embeddings.shape[1]}")

        # Test semantic similarity
        if len(dataset) > 1:
            query_idx = 0
            top_k_indices, top_k_scores = dataset.get_top_k_similar(
                query_idx,
                k=min(5, len(dataset) - 1),
                exclude_query=True
            )

            print(f"\n  Example: Top-5 similar to index 0:")
            query_label = dataset.examples[query_idx].label
            for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
                cand_label = dataset.examples[idx].label
                same_class = "✓" if cand_label == query_label else "✗"
                print(f"    {i+1}. Index {idx:4d}, Similarity: {score:.4f}, Same class: {same_class}")

        results[split] = {
            'num_examples': len(dataset),
            'embedding_shape': embeddings.shape,
        }

    # Save split info (once)
    print(f"\n{'='*70}")
    print("Saving metadata")
    print(f"{'='*70}\n")

    # Use train split to save class split info
    train_dataset = StanfordCarsDataset(
        split='train',
        data_dir=args.data_dir,
        class_split_seed=args.class_split_seed,
    )
    train_dataset.save_split_info()

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"CLIP Model: {args.model}")
    print(f"Device: {device}\n")

    print(f"Embeddings generated for {len(results)} splits:\n")
    total_examples = 0
    for split, info in results.items():
        print(f"  {split:5s}: {info['num_examples']:5d} examples, "
              f"shape {info['embedding_shape']}")
        total_examples += info['num_examples']

    print(f"\n  Total: {total_examples:5d} examples")

    print(f"\nEmbeddings cached in: {args.data_dir}/")
    print(f"  Format: clip_embeddings_{{split}}.pkl")
    print(f"\n✓ All embeddings generated successfully!")


if __name__ == "__main__":
    main()
