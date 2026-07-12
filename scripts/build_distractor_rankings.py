"""
Compute the raw top-K SigLIP image-to-text (I2T) class ranking per query image,
from already-built SigLIP embeddings (see build_siglip_embeddings.py).

For each query image, ranks all classes by dot product between the image's
SigLIP embedding and every class's SigLIP text embedding, and keeps the top K
class indices/scores. This is the raw ranking (the true class is included only
if it naturally ranks in the top K) -- distractor-set materialization (force-
including the true class, shuffling letters) happens downstream in
scripts/evaluate_mc_baselines.py via src/data/distractor_sets.py.

Usage:
    python scripts/build_distractor_rankings.py --dataset cub_200 --splits test
    python scripts/build_distractor_rankings.py --dataset cub_200 --splits train val test --k 16
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compute top-K SigLIP I2T class rankings per query image"
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (must already have siglip_text_embeddings.pkl '
                             'and siglip_image_embeddings_{split}.pkl, see build_siglip_embeddings.py)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing the dataset (default: ./data/{dataset_name})')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Which splits to process (default: all)')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of top classes to keep per query image (default: 16)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(f'./data/{args.dataset}')

    text_cache_path = data_dir / 'siglip_text_embeddings.pkl'
    if not text_cache_path.exists():
        raise FileNotFoundError(
            f"SigLIP text embeddings not found at {text_cache_path}. Please run: "
            f"python scripts/build_siglip_embeddings.py --dataset {args.dataset}"
        )
    with open(text_cache_path, 'rb') as f:
        text_data = pickle.load(f)
    class_text_embeddings = text_data['embeddings']  # (C, D)
    print(f"Loaded {class_text_embeddings.shape[0]} class text embeddings "
          f"(dim={class_text_embeddings.shape[1]})")

    for split in args.splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}\n")

        image_cache_path = data_dir / f'siglip_image_embeddings_{split}.pkl'
        if not image_cache_path.exists():
            raise FileNotFoundError(
                f"SigLIP image embeddings not found at {image_cache_path}. Please run: "
                f"python scripts/build_siglip_embeddings.py --dataset {args.dataset} --splits {split}"
            )
        with open(image_cache_path, 'rb') as f:
            image_data = pickle.load(f)
        image_embeddings = image_data['embeddings']  # (N, D)
        print(f"Loaded {image_embeddings.shape[0]} image embeddings")

        k = min(args.k, class_text_embeddings.shape[0])

        # (N, C) dot products, then top-k indices per row (descending).
        dots = image_embeddings @ class_text_embeddings.T
        top_k_indices = np.argsort(dots, axis=1)[:, ::-1][:, :k]
        top_k_scores = np.take_along_axis(dots, top_k_indices, axis=1)

        output_path = data_dir / f'siglip_top{k}_rankings_{split}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump({
                'top_k_class_indices': top_k_indices,
                'top_k_scores': top_k_scores,
                'k': k,
                'model_name': text_data.get('model_name'),
                'example_ids': image_data.get('example_ids'),
            }, f)
        print(f"✓ Saved top-{k} rankings to {output_path} "
              f"(shape={top_k_indices.shape})")

    print(f"\n{'='*70}")
    print("✓ All distractor rankings computed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
