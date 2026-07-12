"""
Evaluate 0-shot and CLIP-top-1 accuracy on a K-way multiple-choice classification
task, swept across several values of K.

This measures actual classification accuracy under a well-defined, cheaply
scoreable task, in contrast to the marginal-utility signal (a log-prob delta)
used to train the reranker, which does not guarantee the true label becomes the
model's argmax choice.

Design:
1. Classification is constrained to a K-way multiple-choice task with a single
   letter answer token, so a full closed-set softmax over K options is obtained
   from a single forward pass (see Idefics2Wrapper.classify_with_context_mc).
2. The K-1 distractor options are built from SigLIP image-to-text similarity
   (SigLIP is architecturally what Idefics2's vision tower matches), with the
   true label force-included.
3. CLIP retrieval (the mechanism this whole project is testing) is restricted to
   only the images whose label falls within that distractor set, guaranteeing
   every retrieved candidate is a valid answer option.
4. 0-shot and CLIP-top-1-retrieved 1-shot accuracy are compared on the same
   K-way task, per query, using the same letter assignment for both conditions.

This is a baselines-only script (Phase 1). A follow-up phase adds a
reranker-top-1 condition using the same DistractorSet/pool machinery.

Usage:
    python scripts/evaluate_mc_baselines.py
    python scripts/evaluate_mc_baselines.py limits.max_queries=5
"""

import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig
import numpy as np
import pickle
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataclasses import MCEvalResult
from data.distractor_sets import (
    build_distractor_ranking,
    materialize_distractor_set,
    restrict_pool_to_distractor_classes,
)
from models.idefics2_wrapper import Idefics2Wrapper
from utils.imagenet_names import get_readable_name
from utils.eval_utils import save_eval_results

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_icl_performance import load_dataset, _setup_device


def resolve_siglip_cache_path(dataset_name: str, filename: str, siglip_kaggle_dataset: Optional[str]) -> Path:
    """
    Resolve a local path to a siglip_*.pkl file, preferring a Kaggle dataset
    mounted as a notebook input (via scripts/upload_siglip_distractor_set_to_kaggle.py)
    over the local data/{dataset_name}/ cache.

    Assumes the dataset has already been added as an input in the Kaggle
    notebook -- no download-via-CLI fallback, unlike
    resolve_embeddings_cache_path in evaluate_icl_performance.py.

    Tries both known Kaggle mount conventions: the older flat
    /kaggle/input/{slug}/ and the newer /kaggle/input/datasets/{owner}/{slug}/
    (seen when a dataset is referenced by full owner/slug).
    """
    if siglip_kaggle_dataset:
        owner, slug = siglip_kaggle_dataset.split('/')
        candidate_paths = [
            Path(f"/kaggle/input/datasets/{owner}/{slug}/{filename}"),
            Path(f"/kaggle/input/{slug}/{filename}"),
        ]
        for mounted_path in candidate_paths:
            if mounted_path.exists():
                print(f"✓ Using mounted Kaggle input: {mounted_path}")
                return mounted_path

    return Path(f"data/{dataset_name}/{filename}")


def load_siglip_text_embeddings(dataset_name: str, siglip_kaggle_dataset: Optional[str] = None):
    cache_path = resolve_siglip_cache_path(dataset_name, "siglip_text_embeddings.pkl", siglip_kaggle_dataset)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"SigLIP text embeddings not found at {cache_path}. Please run: "
            f"python scripts/build_siglip_embeddings.py --dataset {dataset_name}"
        )
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["class_names"]


def load_siglip_image_embeddings(dataset_name: str, split: str, siglip_kaggle_dataset: Optional[str] = None):
    filename = f"siglip_image_embeddings_{split}.pkl"
    cache_path = resolve_siglip_cache_path(dataset_name, filename, siglip_kaggle_dataset)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"SigLIP image embeddings not found at {cache_path}. Please run: "
            f"python scripts/build_siglip_embeddings.py --dataset {dataset_name} --splits {split}"
        )
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"]


def resolve_label_text(dataset_name: str, class_name: str) -> str:
    """Mini-ImageNet class_names are synset IDs; everything else is already readable."""
    if dataset_name == "mini_imagenet":
        return get_readable_name(class_name)
    return class_name


def label_for_class_idx(dataset, dataset_name: str, class_idx: int) -> str:
    return resolve_label_text(dataset_name, dataset.class_names[class_idx])


@hydra.main(version_base=None, config_path="../configs", config_name="eval_mc_baselines")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    split = cfg.dataset.split
    image_split_path = cfg.dataset.get("image_split_path", None)

    embeddings_kaggle_dataset = cfg.dataset.get("embeddings_kaggle_dataset", None)
    dataset = load_dataset(
        dataset_name, split=split, image_split_path=image_split_path,
        embeddings_kaggle_dataset=embeddings_kaggle_dataset,
    )

    siglip_kaggle_dataset = cfg.dataset.get("siglip_kaggle_dataset", None)
    print("Loading SigLIP embeddings...")
    siglip_text_embs, siglip_class_names = load_siglip_text_embeddings(dataset_name, siglip_kaggle_dataset)
    siglip_image_embs = load_siglip_image_embeddings(dataset_name, split, siglip_kaggle_dataset)
    assert len(siglip_class_names) == len(dataset.class_names), (
        f"SigLIP text embeddings were built for {len(siglip_class_names)} classes, "
        f"but dataset has {len(dataset.class_names)} classes. Rebuild with "
        f"scripts/build_siglip_embeddings.py."
    )
    assert siglip_image_embs.shape[0] == len(dataset), (
        f"SigLIP image embeddings ({siglip_image_embs.shape[0]}) don't match "
        f"dataset size ({len(dataset)}) for split '{split}'. Rebuild with "
        f"scripts/build_siglip_embeddings.py."
    )

    device, _, _ = _setup_device(1)
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device=device,
        load_in_8bit=cfg.model.load_in_8bit,
    )

    k_values = list(cfg.distractor_set.k_values)
    base_seed = cfg.distractor_set.letter_seed
    candidate_pool_k = cfg.retrieval.candidate_pool_k

    max_queries = cfg.limits.get("max_queries", None)
    query_indices = list(range(len(dataset)))
    if max_queries:
        query_indices = query_indices[:max_queries]

    results_by_k = {k: [] for k in k_values}

    for query_idx in tqdm(query_indices, desc="Evaluating queries"):
        query_example, query_image = dataset[query_idx]
        true_class_idx = query_example.label

        ranking = build_distractor_ranking(
            query_siglip_emb=siglip_image_embs[query_idx],
            class_text_embeddings=siglip_text_embs,
            true_class_idx=true_class_idx,
            query_idx=query_idx,
        )

        for k in k_values:
            dset = materialize_distractor_set(ranking, k=k, base_seed=base_seed)
            letter_to_label = {
                letter: label_for_class_idx(dataset, dataset_name, class_idx)
                for letter, class_idx in dset.letter_to_class_idx.items()
            }

            zero_shot_letter, zero_shot_probs = model.classify_with_context_mc(
                query_image, [], letter_to_label
            )
            zero_shot_correct = zero_shot_letter == dset.true_letter

            allowed_indices = restrict_pool_to_distractor_classes(dataset, dset)
            pool_indices, _ = dataset.get_top_k_similar(
                query_idx,
                k=candidate_pool_k,
                exclude_query=True,
                allowed_indices=allowed_indices,
            )

            clip_example_idx = None
            clip_pred_letter = None
            clip_probs = None
            clip_correct = None
            if pool_indices:
                clip_example_idx = pool_indices[0]
                ex_example, ex_image = dataset[clip_example_idx]
                ex_label_text = resolve_label_text(dataset_name, ex_example.label_name)
                clip_pred_letter, clip_probs = model.classify_with_context_mc(
                    query_image, [(ex_image, ex_label_text)], letter_to_label
                )
                clip_correct = clip_pred_letter == dset.true_letter

            results_by_k[k].append(MCEvalResult(
                query_idx=query_idx,
                true_class_idx=true_class_idx,
                true_letter=dset.true_letter,
                k=dset.k,
                letter_to_class_idx=dset.letter_to_class_idx,
                clip_example_idx=clip_example_idx,
                reranker_example_idx=None,
                zero_shot_pred_letter=zero_shot_letter,
                clip_pred_letter=clip_pred_letter,
                reranker_pred_letter=None,
                zero_shot_probs=zero_shot_probs,
                clip_probs=clip_probs,
                reranker_probs=None,
                zero_shot_correct=zero_shot_correct,
                clip_correct=clip_correct,
                reranker_correct=None,
                pool_size=len(pool_indices),
            ))

    print("\n" + "=" * 70)
    print("RESULTS: 0-shot vs. CLIP-top-1 accuracy, K-way multiple choice")
    print("=" * 70)

    accuracy_by_k = {}
    for k in k_values:
        records = results_by_k[k]
        zero_shot_acc = np.mean([r.zero_shot_correct for r in records])
        clip_records = [r for r in records if r.clip_correct is not None]
        clip_acc = np.mean([r.clip_correct for r in clip_records]) if clip_records else None

        accuracy_by_k[k] = {
            "zero_shot_accuracy": float(zero_shot_acc),
            "clip_top1_accuracy": float(clip_acc) if clip_acc is not None else None,
            "num_queries": len(records),
            "num_queries_with_clip_pool": len(clip_records),
        }

        clip_acc_str = f"{clip_acc:.2%}" if clip_acc is not None else "N/A"
        print(f"  K={k:2d}: 0-shot={zero_shot_acc:.2%}  CLIP-top-1={clip_acc_str}  "
              f"(n={len(records)}, n_with_pool={len(clip_records)})")

    results_payload = {
        "accuracy": accuracy_by_k[k_values[0]]["zero_shot_accuracy"],
        "mean_per_class_accuracy": accuracy_by_k[k_values[0]]["zero_shot_accuracy"],
        "correct": sum(r.zero_shot_correct for r in results_by_k[k_values[0]]),
        "total": len(results_by_k[k_values[0]]),
        "accuracy_by_k": accuracy_by_k,
    }

    save_eval_results(
        method="mc_baselines",
        results=results_payload,
        run_id_parts={"dataset": dataset_name, "split": split},
        args={"k_values": k_values, "candidate_pool_k": candidate_pool_k,
              "base_seed": base_seed, "max_queries": max_queries},
        extra={"results_by_k": results_by_k},
        output_root=cfg.output.save_dir,
    )


if __name__ == "__main__":
    main()
