"""
Script to build and cache SigLIP embeddings for datasets.

Precomputes SigLIP image embeddings (per split, mirroring the CLIP embedding
cache convention) and SigLIP text embeddings for all class labels (once per
dataset, since text embeddings don't depend on split). These are used to build
image-to-text (I2T) distractor sets for multiple-choice ICL evaluation --
kept separate from the CLIP embeddings used for actual candidate retrieval.

Usage:
    python scripts/build_siglip_embeddings.py --dataset stanford_cars --splits train val test
    python scripts/build_siglip_embeddings.py --dataset mini_imagenet --splits test

    # With image-level split (Stanford Cars within-distribution eval)
    python scripts/build_siglip_embeddings.py \
        --dataset stanford_cars \
        --image-split-path data/stanford_cars/image_split.json
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.stanford_cars import StanfordCarsDataset
from src.data.mini_imagenet import MiniImageNetDataset
from src.data.fine_grained_hf_dataset import FineGrainedHFDataset
from src.data.dataset_registry import FINE_GRAINED_DATASETS, get_dataset_spec
from src.models.siglip_encoder import SiglipEncoder
from src.utils.imagenet_names import get_readable_name


def main():
    parser = argparse.ArgumentParser(
        description="Build SigLIP image and text embeddings for datasets"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='stanford_cars',
        choices=['stanford_cars', 'mini_imagenet'] + list(FINE_GRAINED_DATASETS.keys()),
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
        help='Path to image-level split JSON (for Stanford Cars within-distribution eval)'
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f'./data/{args.dataset}'
    data_dir = Path(args.data_dir)

    fine_grained_spec = None
    supports_image_split = args.dataset == 'stanford_cars'
    if args.dataset == 'stanford_cars':
        DatasetClass = StanfordCarsDataset
        dataset_display_name = "Stanford Cars"
    elif args.dataset == 'mini_imagenet':
        DatasetClass = MiniImageNetDataset
        dataset_display_name = "Mini-ImageNet"
    elif args.dataset in FINE_GRAINED_DATASETS:
        DatasetClass = FineGrainedHFDataset
        fine_grained_spec = get_dataset_spec(args.dataset)
        dataset_display_name = fine_grained_spec.display_name
        supports_image_split = True
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
        kwargs = dict(split=split, data_dir=args.data_dir, class_split_seed=args.class_split_seed)
        if supports_image_split and args.image_split_path:
            kwargs['image_split_path'] = args.image_split_path
        if fine_grained_spec is not None:
            kwargs['hf_repo_ids'] = fine_grained_spec.hf_repo_ids
        return DatasetClass(**kwargs)

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
        is_mini_imagenet = args.dataset == 'mini_imagenet'
        readable_class_names = (
            [get_readable_name(name) for name in reference_dataset.class_names]
            if is_mini_imagenet else list(reference_dataset.class_names)
        )

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
