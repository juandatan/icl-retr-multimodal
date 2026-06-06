"""
Script to build and cache CLIP embeddings for datasets.

This script pre-computes CLIP embeddings for all images in the dataset,
enabling fast semantic similarity-based candidate retrieval during utility
computation and training.

Usage:
    python scripts/build_clip_embeddings.py --dataset stanford_cars --splits train val test
    python scripts/build_clip_embeddings.py --dataset mini_imagenet --splits train
    python scripts/build_clip_embeddings.py --dataset stanford_cars --model ViT-L/14

    # With image-level split (Stanford Cars within-distribution eval)
    python scripts/build_clip_embeddings.py \
        --dataset stanford_cars \
        --image-split-path data/stanford_cars/image_split.json \
        --kaggle-dataset juandatan/stanford-cars-clip
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import clip

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.stanford_cars import StanfordCarsDataset
from src.data.mini_imagenet import MiniImageNetDataset


def _upload_embeddings_to_kaggle(data_dir: Path, splits: list, kaggle_dataset: str):
    """Upload clip_embeddings_{split}.pkl files to a Kaggle dataset."""
    username, slug = kaggle_dataset.split('/')
    title = slug.replace('-', ' ').title()

    embedding_files = [data_dir / f'clip_embeddings_{s}.pkl' for s in splits]
    missing = [f for f in embedding_files if not f.exists()]
    if missing:
        print(f"✗ Missing embedding files: {missing}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in embedding_files:
            shutil.copy(f, tmp / f.name)

        metadata = {
            "title": title,
            "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

        exists = subprocess.run(
            ["kaggle", "datasets", "status", kaggle_dataset],
            capture_output=True,
        ).returncode == 0

        if exists:
            cmd = ["kaggle", "datasets", "version", "-p", str(tmp),
                   "-m", f"Update embeddings: {', '.join(splits)}", "--dir-mode", "zip"]
        else:
            cmd = ["kaggle", "datasets", "create", "-p", str(tmp), "--dir-mode", "zip"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Uploaded embeddings to https://www.kaggle.com/datasets/{kaggle_dataset}")
        else:
            print(f"✗ Upload failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP embeddings for datasets"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='stanford_cars',
        choices=['stanford_cars', 'mini_imagenet'],
        help='Dataset to process (default: stanford_cars)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory containing the dataset (default: ./data/{dataset_name})'
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
    parser.add_argument(
        '--image-split-path',
        type=str,
        default=None,
        help='Path to image-level split JSON (for Stanford Cars within-distribution eval)'
    )
    parser.add_argument(
        '--kaggle-dataset',
        type=str,
        default=None,
        help='Kaggle dataset to upload embeddings to (username/dataset-slug)'
    )

    args = parser.parse_args()

    # Set default data_dir if not provided
    if args.data_dir is None:
        args.data_dir = f'./data/{args.dataset}'

    # Select dataset class
    if args.dataset == 'stanford_cars':
        DatasetClass = StanfordCarsDataset
        dataset_display_name = "Stanford Cars"
    elif args.dataset == 'mini_imagenet':
        DatasetClass = MiniImageNetDataset
        dataset_display_name = "Mini-ImageNet"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
    print(f"CLIP Embedding Generation for {dataset_display_name}")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  Dataset: {dataset_display_name}")
    print(f"  CLIP Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Splits: {', '.join(args.splits)}")
    print(f"  Data Directory: {args.data_dir}")
    if args.image_split_path:
        print(f"  Image Split: {args.image_split_path}")
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
        kwargs = dict(split=split, data_dir=args.data_dir, class_split_seed=args.class_split_seed)
        if args.dataset == 'stanford_cars' and args.image_split_path:
            kwargs['image_split_path'] = args.image_split_path
        dataset = DatasetClass(**kwargs)

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
    train_kwargs = dict(split='train', data_dir=args.data_dir, class_split_seed=args.class_split_seed)
    if args.dataset == 'stanford_cars' and args.image_split_path:
        train_kwargs['image_split_path'] = args.image_split_path
    train_dataset = DatasetClass(**train_kwargs)
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

    if args.kaggle_dataset:
        _upload_embeddings_to_kaggle(
            data_dir=Path(args.data_dir),
            splits=args.splits,
            kaggle_dataset=args.kaggle_dataset,
        )


if __name__ == "__main__":
    main()
