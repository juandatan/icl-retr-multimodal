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
3. Labeled references come from a configurable support split (train by default),
   separate from the evaluation queries.
4. The primary CLIP condition is restricted to images whose labels fall within
   the option set; random-pool, unrestricted-CLIP, and same-class controls expose
   what that restriction and the retrieved label contribute.
5. Every condition is paired on the same query/options and repeated under
   configurable option-letter seeds to measure answer-position sensitivity.

This is a baselines-only script (Phase 1). A follow-up phase adds a
reranker-top-1 condition using the same DistractorSet/pool machinery.

Usage:
    python scripts/evaluate_mc_baselines.py
    python scripts/evaluate_mc_baselines.py limits.max_queries=5
    python scripts/evaluate_mc_baselines.py limits.compute_oracle=true

    # Resume oracle-only from a previously saved mc_baselines .pkl -- reuses the
    # saved 0-shot/CLIP-top-1 results as-is (no Idefics2 calls re-run for them)
    # and only pays the oracle pass's cost:
    python scripts/evaluate_mc_baselines.py \
        limits.resume_from=outputs/evals/mc_baselines/mc_baselines/cub_200_test_train.pkl
"""

import hashlib
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pickle
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataclasses import MCEvalResult
from data.distractor_sets import (
    build_distractor_ranking,
    materialize_distractor_set,
)
from models.idefics2_wrapper import Idefics2Wrapper
from utils.imagenet_names import get_readable_name
from utils.eval_utils import save_eval_results
from utils.kaggle_utils import kaggle_upload_eval_results

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


def rank_support_candidates(
    query_embedding: np.ndarray,
    retrieval_dataset,
    k: int,
    allowed_indices: Optional[set] = None,
    exclude_index: Optional[int] = None,
):
    """Rank a support dataset against an evaluation-query CLIP embedding."""
    similarities = retrieval_dataset.clip_embeddings @ query_embedding
    valid_mask = np.ones(len(retrieval_dataset), dtype=bool)
    if allowed_indices is not None:
        valid_mask[:] = False
        valid_mask[list(allowed_indices)] = True
    if exclude_index is not None:
        valid_mask[exclude_index] = False

    valid_indices = np.flatnonzero(valid_mask)
    if not len(valid_indices):
        return [], np.array([], dtype=similarities.dtype)
    k = min(k, len(valid_indices))
    valid_similarities = similarities[valid_indices]
    order = np.argsort(valid_similarities)[-k:][::-1]
    return valid_indices[order].tolist(), valid_similarities[order]


def deterministic_random_candidate(pool_indices: list, seed: int) -> Optional[int]:
    """Choose a reproducible random control from an already materialized pool."""
    if not pool_indices:
        return None
    return int(np.random.default_rng(seed).choice(pool_indices))


def evaluate_reference(
    model, query_image, retrieval_dataset, dataset_name: str, example_idx, letter_to_label
):
    """Evaluate one labeled reference, returning nullable prediction fields."""
    if example_idx is None:
        return None, None
    example, image = retrieval_dataset[example_idx]
    label_text = resolve_label_text(dataset_name, example.label_name)
    pred_letter, probs = model.classify_with_context_mc(
        query_image, [(image, label_text)], letter_to_label
    )
    return pred_letter, probs


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def load_saved_mc_results(resume_from: str):
    """Load a previously saved mc_baselines .pkl (see save_eval_results) for oracle-only resume."""
    with open(resume_from, "rb") as f:
        payload = pickle.load(f)
    results_by_k = payload["results_by_k"]
    args = payload["args"]
    if args.get("schema_version", 1) < 2:
        raise ValueError(
            "This is a legacy MC result produced before the prompt and retrieval-split fixes. "
            "It cannot be resumed as a schema-v2 oracle run; rerun the baselines first."
        )
    return results_by_k, args


def build_oracle_context_for_k(
    query_dataset, retrieval_dataset, dataset_name: str, records: list, candidate_pool_k: int
) -> list:
    """
    Deterministically rebuild each record's CLIP-restricted retrieval pool from its
    saved letter_to_class_idx (no SigLIP ranking or Idefics2 calls needed), for
    oracle-only resume from a previously saved results file.
    """
    context = []
    for record_idx, record in enumerate(records):
        if record.oracle_correct is not None:
            continue
        allowed_classes = set(record.letter_to_class_idx.values())
        allowed_indices = {
            ex.index for ex in retrieval_dataset.examples if ex.label in allowed_classes
        }
        exclude_index = record.query_idx if query_dataset is retrieval_dataset else None
        pool_indices, _ = rank_support_candidates(
            query_dataset.clip_embeddings[record.query_idx], retrieval_dataset,
            k=candidate_pool_k, allowed_indices=allowed_indices, exclude_index=exclude_index,
        )
        if not pool_indices:
            continue
        if record.clip_example_idx is not None and pool_indices[0] != record.clip_example_idx:
            raise ValueError(
                f"Rebuilt pool for query {record.query_idx} (K={record.k}) doesn't match the "
                f"saved clip_example_idx ({pool_indices[0]} != {record.clip_example_idx}). The "
                f"dataset's CLIP embeddings or candidate_pool_k likely differ from the original run."
            )
        letter_to_label = {
            letter: label_for_class_idx(query_dataset, dataset_name, class_idx)
            for letter, class_idx in record.letter_to_class_idx.items()
        }
        context.append({
            "record_idx": record_idx,
            "query_idx": record.query_idx,
            "pool_indices": pool_indices,
            "clip_example_idx": record.clip_example_idx,
            "clip_correct": record.clip_correct,
            "true_letter": record.true_letter,
            "letter_to_label": letter_to_label,
        })
    return context


def run_oracle_pass_for_k(
    model, query_dataset, retrieval_dataset, dataset_name: str, k: int, context: list, records: list
):
    """Test every pool candidate per query for one K, mutating oracle_correct/oracle_example_idx in place."""
    for ctx in tqdm(context, desc=f"Oracle K={k}"):
        oracle_correct = False
        oracle_example_idx = None
        _, query_image = query_dataset[ctx["query_idx"]]
        for cand_idx in ctx["pool_indices"]:
            if cand_idx == ctx["clip_example_idx"]:
                cand_pred_correct = ctx["clip_correct"]
            else:
                cand_example, cand_image = retrieval_dataset[cand_idx]
                cand_label_text = resolve_label_text(dataset_name, cand_example.label_name)
                cand_pred_letter, _ = model.classify_with_context_mc(
                    query_image, [(cand_image, cand_label_text)], ctx["letter_to_label"]
                )
                cand_pred_correct = cand_pred_letter == ctx["true_letter"]
            if cand_pred_correct:
                oracle_correct = True
                oracle_example_idx = cand_idx
                break

        record = records[ctx["record_idx"]]
        record.oracle_correct = oracle_correct
        record.oracle_example_idx = oracle_example_idx


def _condition_summary(
    records: list, correct_field: str, prediction_field: Optional[str] = None
) -> Optional[dict]:
    valid = [r for r in records if getattr(r, correct_field, None) is not None]
    if not valid:
        return None
    by_seed = defaultdict(list)
    by_class = defaultdict(list)
    by_true_letter = defaultdict(list)
    predicted_letter_counts = defaultdict(int)
    for record in valid:
        correct = bool(getattr(record, correct_field))
        by_seed[record.letter_seed].append(correct)
        by_class[record.true_class_idx].append(correct)
        by_true_letter[record.true_letter].append(correct)
        if prediction_field:
            prediction = getattr(record, prediction_field, None)
            if prediction is not None:
                predicted_letter_counts[prediction] += 1
    return {
        "accuracy": float(np.mean([getattr(r, correct_field) for r in valid])),
        "correct": int(sum(bool(getattr(r, correct_field)) for r in valid)),
        "num_trials": len(valid),
        "mean_per_class_accuracy": float(np.mean([np.mean(v) for v in by_class.values()])),
        "accuracy_by_letter_seed": {
            int(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())
        },
        "accuracy_by_true_letter": {
            letter: float(np.mean(values)) for letter, values in sorted(by_true_letter.items())
        },
        "predicted_letter_counts": dict(sorted(predicted_letter_counts.items())),
    }


def _paired_vs_zero_summary(records: list, correct_field: str) -> Optional[dict]:
    paired = [r for r in records if getattr(r, correct_field, None) is not None]
    if not paired:
        return None
    def transition_counts(seed_records):
        zero_only = sum(
            r.zero_shot_correct and not getattr(r, correct_field) for r in seed_records
        )
        condition_only = sum(
            not r.zero_shot_correct and getattr(r, correct_field) for r in seed_records
        )
        return zero_only, condition_only

    def exact_mcnemar(zero_only, condition_only):
        discordant = zero_only + condition_only
        if not discordant:
            return 1.0
        tail = sum(
            math.comb(discordant, i) for i in range(min(zero_only, condition_only) + 1)
        ) / (2 ** discordant)
        return min(1.0, 2.0 * tail)

    zero_only, condition_only = transition_counts(paired)
    by_seed = defaultdict(list)
    records_by_seed = defaultdict(list)
    by_query = defaultdict(list)
    for record in paired:
        difference = int(bool(getattr(record, correct_field))) - int(record.zero_shot_correct)
        by_seed[record.letter_seed].append(difference)
        records_by_seed[record.letter_seed].append(record)
        by_query[record.query_idx].append(difference)
    query_differences = np.array([np.mean(values) for values in by_query.values()])
    standard_error = (
        float(np.std(query_differences, ddof=1) / np.sqrt(len(query_differences)))
        if len(query_differences) > 1 else 0.0
    )
    mean_difference = float(np.mean(query_differences))
    return {
        "accuracy_difference": mean_difference,
        "query_clustered_standard_error": standard_error,
        "query_clustered_normal_95ci": [
            mean_difference - 1.95996398454 * standard_error,
            mean_difference + 1.95996398454 * standard_error,
        ],
        "accuracy_difference_by_letter_seed": {
            int(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())
        },
        # McNemar is calculated separately per seed so repeated permutations of
        # the same query are not incorrectly treated as independent observations.
        "exact_mcnemar_p_by_letter_seed": {
            int(seed): float(exact_mcnemar(*transition_counts(seed_records)))
            for seed, seed_records in sorted(records_by_seed.items())
        },
        "zero_only_correct": int(zero_only),
        "condition_only_correct": int(condition_only),
        "num_paired_trials": len(paired),
        "num_query_clusters": len(by_query),
    }


def summarize_and_save(
    results_by_k: dict, k_values: list, dataset_name: str, split: str,
    save_dir: str, results_kaggle_dataset: Optional[str], args: dict,
    benchmark_scope: dict,
) -> Path:
    """Aggregate per-condition/per-seed stats and persist the complete report."""
    print("\n" + "=" * 70)
    print("RESULTS: 0-shot vs. CLIP-top-1 accuracy, K-way multiple choice")
    print("=" * 70)

    accuracy_by_k = {}
    for k in k_values:
        records = results_by_k[k]
        unique_query_records = {}
        for record in records:
            unique_query_records.setdefault(record.query_idx, record)
        query_records = list(unique_query_records.values())

        conditions = {
            "zero_shot": _condition_summary(
                records, "zero_shot_correct", "zero_shot_pred_letter"
            ),
            "restricted_clip_top1": _condition_summary(
                records, "clip_correct", "clip_pred_letter"
            ),
            "random_pool": _condition_summary(
                records, "random_correct", "random_pred_letter"
            ),
            "unrestricted_clip_top1": _condition_summary(
                records, "unrestricted_clip_correct", "unrestricted_clip_pred_letter"
            ),
            "same_class_top1_oracle": _condition_summary(
                records, "same_class_correct", "same_class_pred_letter"
            ),
            "candidate_existence_oracle": _condition_summary(records, "oracle_correct"),
        }
        zero = conditions["zero_shot"]
        clip = conditions["restricted_clip_top1"]
        oracle = conditions["candidate_existence_oracle"]
        pool_true_class_rate = float(np.mean([r.pool_has_true_class for r in query_records]))
        unrestricted_in_options = [
            r.unrestricted_clip_example_in_options for r in query_records
            if r.unrestricted_clip_example_in_options is not None
        ]
        restricted_same_class = [
            r.clip_example_same_class for r in query_records
            if r.clip_example_same_class is not None
        ]
        random_same_class = [
            r.random_example_same_class for r in query_records
            if r.random_example_same_class is not None
        ]

        accuracy_by_k[k] = {
            "conditions": conditions,
            "paired_vs_zero_shot": {
                "restricted_clip_top1": _paired_vs_zero_summary(records, "clip_correct"),
                "random_pool": _paired_vs_zero_summary(records, "random_correct"),
                "unrestricted_clip_top1": _paired_vs_zero_summary(
                    records, "unrestricted_clip_correct"
                ),
                "same_class_top1_oracle": _paired_vs_zero_summary(
                    records, "same_class_correct"
                ),
            },
            # Backward-compatible headline aliases.
            "zero_shot_accuracy": zero["accuracy"],
            "clip_top1_accuracy": clip["accuracy"] if clip else None,
            "oracle_accuracy": oracle["accuracy"] if oracle else None,
            "pool_true_class_rate": pool_true_class_rate,
            "unrestricted_clip_in_option_set_rate": (
                float(np.mean(unrestricted_in_options)) if unrestricted_in_options else None
            ),
            "restricted_clip_same_class_rate": (
                float(np.mean(restricted_same_class)) if restricted_same_class else None
            ),
            "random_pool_same_class_rate": (
                float(np.mean(random_same_class)) if random_same_class else None
            ),
            "num_queries": len(query_records),
            "num_trials": len(records),
            "num_letter_seeds": len({r.letter_seed for r in records}),
        }

        def acc_str(summary):
            return f"{summary['accuracy']:.2%}" if summary else "N/A"

        print(
            f"  K={k:2d}: 0-shot={acc_str(zero)}  restricted-CLIP={acc_str(clip)}  "
            f"random={acc_str(conditions['random_pool'])}  "
            f"unrestricted-CLIP={acc_str(conditions['unrestricted_clip_top1'])}  "
            f"same-class={acc_str(conditions['same_class_top1_oracle'])}  "
            f"candidate-oracle={acc_str(oracle)}  pool_has_true_class={pool_true_class_rate:.2%} "
            f"(queries={len(query_records)}, trials={len(records)})"
        )

    results_payload = {
        "accuracy": accuracy_by_k[k_values[0]]["conditions"]["zero_shot"]["accuracy"],
        "mean_per_class_accuracy": accuracy_by_k[k_values[0]]["conditions"]["zero_shot"]["mean_per_class_accuracy"],
        "correct": accuracy_by_k[k_values[0]]["conditions"]["zero_shot"]["correct"],
        "total": accuracy_by_k[k_values[0]]["conditions"]["zero_shot"]["num_trials"],
        "accuracy_by_k": accuracy_by_k,
        "benchmark_scope": benchmark_scope,
    }

    out_dir = save_eval_results(
        method="mc_baselines",
        results=results_payload,
        run_id_parts={
            "dataset": dataset_name,
            "split": split,
            "retrieval_split": args["retrieval_split"],
        },
        args=args,
        extra={"results_by_k": results_by_k},
        output_root=save_dir,
    )

    if results_kaggle_dataset:
        kaggle_upload_eval_results(out_dir, results_kaggle_dataset)

    return out_dir


@hydra.main(version_base=None, config_path="../configs", config_name="eval_mc_baselines")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    split = cfg.dataset.split
    retrieval_split = cfg.retrieval.get("split", "train")
    image_split_path = cfg.dataset.get("image_split_path", None)
    results_kaggle_dataset = cfg.output.get("results_kaggle_dataset", None)
    resume_from = cfg.limits.get("resume_from", None)

    embeddings_kaggle_dataset = cfg.dataset.get("embeddings_kaggle_dataset", None)
    query_dataset = load_dataset(
        dataset_name, split=split, image_split_path=image_split_path,
        embeddings_kaggle_dataset=embeddings_kaggle_dataset,
    )
    if retrieval_split == split:
        retrieval_dataset = query_dataset
    else:
        retrieval_dataset = load_dataset(
            dataset_name, split=retrieval_split, image_split_path=image_split_path,
            embeddings_kaggle_dataset=embeddings_kaggle_dataset,
        )
    if query_dataset.class_names != retrieval_dataset.class_names:
        raise ValueError("Evaluation and retrieval datasets have different class-name mappings.")
    if query_dataset.clip_embeddings.shape[1] != retrieval_dataset.clip_embeddings.shape[1]:
        raise ValueError("Evaluation and retrieval CLIP embeddings have different dimensions.")
    if query_dataset.clip_model_name != retrieval_dataset.clip_model_name:
        raise ValueError(
            "Evaluation and retrieval embeddings were built by different CLIP models: "
            f"{query_dataset.clip_model_name!r} vs {retrieval_dataset.clip_model_name!r}."
        )

    device, _, _ = _setup_device(1)
    model = Idefics2Wrapper(
        model_name=cfg.model.idefics2_model,
        device=device,
        load_in_8bit=cfg.model.load_in_8bit,
    )

    if resume_from:
        print(f"Resuming oracle-only from {resume_from}...")
        results_by_k, saved_args = load_saved_mc_results(resume_from)
        if saved_args["eval_split"] != split or saved_args["retrieval_split"] != retrieval_split:
            raise ValueError(
                "Configured eval/retrieval splits do not match the saved run: "
                f"configured={split}/{retrieval_split}, "
                f"saved={saved_args['eval_split']}/{saved_args['retrieval_split']}"
            )
        k_values = list(saved_args["k_values"])
        candidate_pool_k = saved_args["candidate_pool_k"]
        benchmark_scope = saved_args["benchmark_scope"]
        for k in k_values:
            context = build_oracle_context_for_k(
                query_dataset, retrieval_dataset, dataset_name, results_by_k[k], candidate_pool_k
            )
            run_oracle_pass_for_k(
                model, query_dataset, retrieval_dataset, dataset_name, k, context, results_by_k[k]
            )
            summarize_and_save(
                results_by_k, k_values, dataset_name, split, cfg.output.save_dir,
                results_kaggle_dataset, saved_args, benchmark_scope,
            )
        return

    siglip_kaggle_dataset = cfg.dataset.get("siglip_kaggle_dataset", None)
    print("Loading SigLIP embeddings...")
    siglip_text_embs, siglip_class_names = load_siglip_text_embeddings(dataset_name, siglip_kaggle_dataset)
    siglip_image_embs = load_siglip_image_embeddings(dataset_name, split, siglip_kaggle_dataset)
    assert len(siglip_class_names) == len(query_dataset.class_names), (
        f"SigLIP text embeddings were built for {len(siglip_class_names)} classes, "
        f"but dataset has {len(query_dataset.class_names)} classes. Rebuild with "
        f"scripts/build_siglip_embeddings.py."
    )
    if list(siglip_class_names) != list(query_dataset.class_names):
        raise ValueError("SigLIP and dataset class names have different ordering.")
    assert siglip_image_embs.shape[0] == len(query_dataset), (
        f"SigLIP image embeddings ({siglip_image_embs.shape[0]}) don't match "
        f"dataset size ({len(query_dataset)}) for split '{split}'. Rebuild with "
        f"scripts/build_siglip_embeddings.py."
    )

    k_values = list(cfg.distractor_set.k_values)
    letter_seeds = list(cfg.distractor_set.get(
        "letter_seeds", [cfg.distractor_set.get("letter_seed", 42)]
    ))
    if not letter_seeds:
        raise ValueError("distractor_set.letter_seeds must contain at least one seed")
    candidate_pool_k = cfg.retrieval.candidate_pool_k
    compute_oracle = cfg.limits.get("compute_oracle", False)
    controls = cfg.get("controls", {})
    run_random_control = controls.get("random_pool", True)
    run_unrestricted_control = controls.get("unrestricted_clip", True)
    run_same_class_control = controls.get("same_class_oracle", True)

    max_queries = cfg.limits.get("max_queries", None)
    query_indices = list(range(len(query_dataset)))
    if max_queries:
        query_indices = query_indices[:max_queries]

    benchmark_scope = {
        "task": "query-specific K-way multiple-choice image classification",
        "distractors": "SigLIP image-to-text hard negatives with the true class force-included",
        "primary_retrieval": "CLIP top-1 from a support split, restricted to query option classes",
        "limitations": [
            "This is not unconstrained 200-way CUB classification accuracy.",
            "SigLIP hard-negative construction is coupled to Idefics2's vision architecture.",
            "Restricted retrieval assumes the query's answer-option set is known at retrieval time.",
            "Same-class and candidate-existence oracle conditions use ground-truth information and are ceilings, not deployable methods.",
        ],
    }
    run_args = {
        "schema_version": 2,
        "k_values": k_values,
        "candidate_pool_k": candidate_pool_k,
        "letter_seeds": letter_seeds,
        "max_queries": max_queries,
        "eval_split": split,
        "retrieval_split": retrieval_split,
        "eval_num_examples": len(query_dataset),
        "retrieval_num_examples": len(retrieval_dataset),
        "clip_embedding_model_eval": query_dataset.clip_model_name,
        "clip_embedding_model_retrieval": retrieval_dataset.clip_model_name,
        "idefics2_model": cfg.model.idefics2_model,
        "siglip_model": cfg.distractor_set.siglip_model,
        "load_in_8bit": bool(cfg.model.load_in_8bit),
        "image_split_path": image_split_path,
        "image_split_sha256": _file_sha256(image_split_path),
        "git_revision": _git_revision(),
        "controls": OmegaConf.to_container(controls, resolve=True),
        "benchmark_scope": benchmark_scope,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }

    results_by_k = {k: [] for k in k_values}
    oracle_context_by_k = {k: [] for k in k_values}

    for query_idx in tqdm(query_indices, desc="Evaluating queries"):
        query_example, query_image = query_dataset[query_idx]
        true_class_idx = query_example.label

        ranking = build_distractor_ranking(
            query_siglip_emb=siglip_image_embs[query_idx],
            class_text_embeddings=siglip_text_embs,
            true_class_idx=true_class_idx,
            query_idx=query_idx,
        )

        for k in k_values:
            # Candidate classes are seed-independent; only their letter assignment changes.
            class_set = materialize_distractor_set(ranking, k=k, base_seed=letter_seeds[0])
            allowed_classes = set(class_set.class_idx_to_letter)
            allowed_indices = {
                ex.index for ex in retrieval_dataset.examples if ex.label in allowed_classes
            }
            exclude_index = query_idx if query_dataset is retrieval_dataset else None
            query_clip_embedding = query_dataset.clip_embeddings[query_idx]
            pool_indices, _ = rank_support_candidates(
                query_clip_embedding, retrieval_dataset, candidate_pool_k,
                allowed_indices=allowed_indices, exclude_index=exclude_index,
            )
            unrestricted_indices, _ = rank_support_candidates(
                query_clip_embedding, retrieval_dataset, 1, exclude_index=exclude_index,
            )
            same_class_indices = {
                ex.index for ex in retrieval_dataset.examples if ex.label == true_class_idx
            }
            same_class_ranked, _ = rank_support_candidates(
                query_clip_embedding, retrieval_dataset, 1,
                allowed_indices=same_class_indices, exclude_index=exclude_index,
            )
            random_example_idx = deterministic_random_candidate(
                pool_indices, int(cfg.experiment.seed) + query_idx + k * 100003
            )
            clip_example_idx = pool_indices[0] if pool_indices else None
            unrestricted_example_idx = unrestricted_indices[0] if unrestricted_indices else None
            same_class_example_idx = same_class_ranked[0] if same_class_ranked else None
            pool_has_true_class = any(
                retrieval_dataset.examples[idx].label == true_class_idx for idx in pool_indices
            )
            clip_example_same_class = (
                retrieval_dataset.examples[clip_example_idx].label == true_class_idx
                if clip_example_idx is not None else None
            )
            random_example_same_class = (
                retrieval_dataset.examples[random_example_idx].label == true_class_idx
                if random_example_idx is not None else None
            )
            unrestricted_in_options = (
                retrieval_dataset.examples[unrestricted_example_idx].label in allowed_classes
                if unrestricted_example_idx is not None else None
            )

            for letter_seed in letter_seeds:
                dset = materialize_distractor_set(ranking, k=k, base_seed=letter_seed)
                letter_to_label = {
                    letter: label_for_class_idx(query_dataset, dataset_name, class_idx)
                    for letter, class_idx in dset.letter_to_class_idx.items()
                }
                zero_shot_letter, zero_shot_probs = model.classify_with_context_mc(
                    query_image, [], letter_to_label
                )
                clip_pred_letter, clip_probs = evaluate_reference(
                    model, query_image, retrieval_dataset, dataset_name,
                    clip_example_idx, letter_to_label,
                )
                random_pred_letter, random_probs = (None, None)
                if run_random_control:
                    random_pred_letter, random_probs = evaluate_reference(
                        model, query_image, retrieval_dataset, dataset_name,
                        random_example_idx, letter_to_label,
                    )
                unrestricted_pred_letter, unrestricted_probs = (None, None)
                if run_unrestricted_control:
                    unrestricted_pred_letter, unrestricted_probs = evaluate_reference(
                        model, query_image, retrieval_dataset, dataset_name,
                        unrestricted_example_idx, letter_to_label,
                    )
                same_class_pred_letter, same_class_probs = (None, None)
                if run_same_class_control:
                    same_class_pred_letter, same_class_probs = evaluate_reference(
                        model, query_image, retrieval_dataset, dataset_name,
                        same_class_example_idx, letter_to_label,
                    )

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
                    zero_shot_correct=zero_shot_letter == dset.true_letter,
                    clip_correct=(clip_pred_letter == dset.true_letter) if clip_pred_letter else None,
                    reranker_correct=None,
                    pool_size=len(pool_indices),
                    pool_has_true_class=pool_has_true_class,
                    oracle_correct=None,
                    oracle_example_idx=None,
                    letter_seed=int(letter_seed),
                    random_example_idx=random_example_idx,
                    random_pred_letter=random_pred_letter,
                    random_probs=random_probs,
                    random_correct=(random_pred_letter == dset.true_letter) if random_pred_letter else None,
                    unrestricted_clip_example_idx=unrestricted_example_idx,
                    unrestricted_clip_pred_letter=unrestricted_pred_letter,
                    unrestricted_clip_probs=unrestricted_probs,
                    unrestricted_clip_correct=(
                        unrestricted_pred_letter == dset.true_letter
                        if unrestricted_pred_letter else None
                    ),
                    unrestricted_clip_example_in_options=unrestricted_in_options,
                    same_class_example_idx=same_class_example_idx,
                    same_class_pred_letter=same_class_pred_letter,
                    same_class_probs=same_class_probs,
                    same_class_correct=(
                        same_class_pred_letter == dset.true_letter if same_class_pred_letter else None
                    ),
                    clip_example_same_class=clip_example_same_class,
                    random_example_same_class=random_example_same_class,
                ))

                if compute_oracle and pool_indices:
                    oracle_context_by_k[k].append({
                        "record_idx": len(results_by_k[k]) - 1,
                        "query_idx": query_idx,
                        "pool_indices": pool_indices,
                        "clip_example_idx": clip_example_idx,
                        "clip_correct": clip_pred_letter == dset.true_letter,
                        "true_letter": dset.true_letter,
                        "letter_to_label": letter_to_label,
                    })

    # Save/upload 0-shot + CLIP-top-1 results immediately, before paying the
    # oracle pass's ~candidate_pool_k-per-query cost, so they're safe even if
    # the oracle pass fails or the notebook session ends partway through it.
    summarize_and_save(
        results_by_k, k_values, dataset_name, split, cfg.output.save_dir,
        results_kaggle_dataset, run_args, benchmark_scope,
    )

    if compute_oracle:
        print("\nRunning oracle pass (testing every pool candidate per query)...")
        for k in k_values:
            run_oracle_pass_for_k(
                model, query_dataset, retrieval_dataset, dataset_name, k,
                oracle_context_by_k[k], results_by_k[k],
            )

            # Save/upload after each K finishes, so oracle results for completed
            # K's survive even if a later K's oracle pass fails or times out.
            summarize_and_save(
                results_by_k, k_values, dataset_name, split, cfg.output.save_dir,
                results_kaggle_dataset, run_args, benchmark_scope,
            )


if __name__ == "__main__":
    main()
