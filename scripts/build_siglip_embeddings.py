"""Build and cache SigLIP image/text embeddings for CUB-200.

Precomputes SigLIP image embeddings (per split, mirroring the CLIP embedding
cache convention) and SigLIP text embeddings for all class labels (once per
dataset, since text embeddings don't depend on split). These are used to build
image-to-text (I2T) distractor sets for multiple-choice ICL evaluation --
kept separate from the CLIP embeddings used for actual candidate retrieval.

Usage:
    python -m scripts.build_siglip_embeddings \
        --dataset cub_200 \
        --image-split-path data/cub_200/image_split.json
"""

import argparse
import pickle
from pathlib import Path

from src.data.fine_grained_hf_dataset import FineGrainedHFDataset
from src.data.dataset_registry import FINE_GRAINED_DATASETS, get_dataset_spec
from src.models.siglip_encoder import SiglipEncoder


def main():
    parser = argparse.ArgumentParser(
        description="Build SigLIP image and text embeddings for CUB-200"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cub_200',
        choices=list(FINE_GRAINED_DATASETS),
        help='Dataset to process (default: cub_200)'
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
        help='Which splits to build image embeddings for (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/siglip-so400m-patch14-384',
        help='SigLIP model to use (default: google/siglip-so400m-patch14-384)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for embedding computation (default: 16)'
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
        help='Path to the canonical CUB image-level split JSON'
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f'./data/{args.dataset}'
    data_dir = Path(args.data_dir)

    spec = get_dataset_spec(args.dataset)
    dataset_display_name = spec.display_name

    print(f"\n{'='*70}")
    print(f"SigLIP Embedding Generation for {dataset_display_name}")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  Dataset: {dataset_display_name}")
    print(f"  SigLIP Model: {args.model}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Splits (image embeddings): {', '.join(args.splits)}")
    print(f"  Data Directory: {args.data_dir}")
    if args.image_split_path:
        print(f"  Image Split: {args.image_split_path}")
    print(f"\n{'='*70}\n")

    print(f"Loading SigLIP model: {args.model}...")
    encoder = SiglipEncoder(model_name=args.model, device=args.device)
    print(f"✓ SigLIP model loaded on {encoder.device}\n")

    def _load_dataset(split: str):
        kwargs = dict(
            hf_repo_ids=list(spec.hf_repo_ids),
            split=split,
            data_dir=args.data_dir,
            class_split_seed=args.class_split_seed,
        )
        if args.image_split_path:
            kwargs['image_split_path'] = args.image_split_path
        return FineGrainedHFDataset(**kwargs)

    # --- Text embeddings: once per dataset, not per split ---
    text_cache_path = data_dir / 'siglip_text_embeddings.pkl'
    if text_cache_path.exists():
        print(f"Text embeddings already cached at {text_cache_path}, skipping.\n")
    else:
        print(f"\n{'='*70}")
        print("Building text embeddings for all class labels")
        print(f"{'='*70}\n")

        # class_names is identical across splits (it's set before split filtering),
        # so any split's dataset instance works here.
        reference_dataset = _load_dataset(args.splits[0])
        readable_class_names = list(reference_dataset.class_names)

        text_embeddings = encoder.encode_texts(readable_class_names, batch_size=args.batch_size)
        print(f"✓ Text embedding shape: {text_embeddings.shape}")

        with open(text_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': text_embeddings,
                'model_name': args.model,
                'class_names': readable_class_names,
            }, f)
        print(f"✓ Cached text embeddings to {text_cache_path}\n")

    # --- Image embeddings: per split ---
    for split in args.splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split (image embeddings)")
        print(f"{'='*70}\n")

        image_cache_path = data_dir / f'siglip_image_embeddings_{split}.pkl'
        if image_cache_path.exists():
            print(f"Image embeddings already cached at {image_cache_path}, skipping.")
            continue

        dataset = _load_dataset(split)
        print(f"Loaded {len(dataset)} examples")

        images = [dataset[i][1] for i in range(len(dataset))]
        embeddings = encoder.encode_images(images, batch_size=args.batch_size)
        print(f"✓ Image embedding shape: {embeddings.shape}")

        with open(image_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'model_name': args.model,
                'example_ids': [ex.image_path for ex in dataset.examples],
            }, f)
        print(f"✓ Cached image embeddings to {image_cache_path}")

    print(f"\n{'='*70}")
    print("✓ All SigLIP embeddings generated successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
