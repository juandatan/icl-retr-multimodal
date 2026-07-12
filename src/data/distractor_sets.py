"""
Distractor-class-set construction for multiple-choice ICL evaluation.

Given a query image's SigLIP embedding and SigLIP text embeddings for every class
label, ranks classes by image-to-text (I2T) cosine similarity to identify the most
confusable distractors for that query, then materializes a K-way letter-labeled
multiple-choice set (with the true label force-included).

Ranking and materialization are split so the same SigLIP-derived ranking can be
sliced to any K without recomputing similarities -- reducing K just takes a prefix
of the same ranked list.
"""

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

# A-Z then a-z, giving up to 52 single-token option letters.
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]


@dataclass
class DistractorRanking:
    """All classes other than the true one, ranked by I2T similarity to the query."""
    query_idx: int
    true_class_idx: int
    ranked_class_indices: List[int]  # descending by I2T dot product, excludes true_class_idx


@dataclass
class DistractorSet:
    """A materialized K-way multiple-choice set with letter assignments."""
    query_idx: int
    k: int
    letter_to_class_idx: Dict[str, int]
    class_idx_to_letter: Dict[int, str]
    true_letter: str


def build_distractor_ranking(
    query_siglip_emb: np.ndarray,
    class_text_embeddings: np.ndarray,
    true_class_idx: int,
    query_idx: int,
) -> DistractorRanking:
    """
    Rank all classes other than the true one by I2T cosine similarity to the query.

    Args:
        query_siglip_emb: L2-normalized SigLIP image embedding, shape (D,)
        class_text_embeddings: L2-normalized SigLIP text embeddings for all classes,
            shape (C, D)
        true_class_idx: Index of the query's ground-truth class
        query_idx: Index of the query (carried through for downstream bookkeeping)

    Returns:
        DistractorRanking with ranked_class_indices sorted descending by similarity,
        excluding true_class_idx.
    """
    dot = class_text_embeddings @ query_siglip_emb  # (C,)

    order = np.argsort(dot)[::-1]
    ranked = [int(idx) for idx in order if int(idx) != true_class_idx]

    return DistractorRanking(
        query_idx=query_idx,
        true_class_idx=true_class_idx,
        ranked_class_indices=ranked,
    )


def materialize_distractor_set(
    ranking: DistractorRanking,
    k: int,
    base_seed: int = 42,
) -> DistractorSet:
    """
    Take the top-(k-1) distractor classes from a ranking, force-include the true
    class, and assign shuffled letters.

    Args:
        ranking: Precomputed DistractorRanking for this query
        k: Number of multiple-choice options (including the true label). Clamped
            to len(ranking.ranked_class_indices) + 1 if larger than available classes.
        base_seed: Base seed for the per-query, per-k letter shuffle

    Returns:
        DistractorSet of exactly min(k, available) options.
    """
    max_k = len(ranking.ranked_class_indices) + 1
    k = min(k, max_k)

    distractors = ranking.ranked_class_indices[:k - 1]
    class_indices = distractors + [ranking.true_class_idx]

    # Fold k into the seed so different K values in a sweep get independent,
    # still-reproducible shuffles for the same query.
    rng = np.random.default_rng(base_seed + ranking.query_idx + k * 100003)
    letters = LETTERS[:k]
    shuffled_letters = list(letters)
    rng.shuffle(shuffled_letters)

    letter_to_class_idx = {letter: cls for letter, cls in zip(shuffled_letters, class_indices)}
    class_idx_to_letter = {cls: letter for letter, cls in letter_to_class_idx.items()}
    true_letter = class_idx_to_letter[ranking.true_class_idx]

    return DistractorSet(
        query_idx=ranking.query_idx,
        k=k,
        letter_to_class_idx=letter_to_class_idx,
        class_idx_to_letter=class_idx_to_letter,
        true_letter=true_letter,
    )


def restrict_pool_to_distractor_classes(dataset, distractor_set: DistractorSet) -> Set[int]:
    """
    Map a distractor set's class indices to the set of dataset example indices
    whose label falls within those classes.

    Args:
        dataset: A BaseUtilityDataset (or subclass) with .examples: List[ClassificationExample]
        distractor_set: DistractorSet whose classes define the allowed label set

    Returns:
        Set of example indices (positions in dataset.examples) whose .label is
        one of distractor_set.class_idx_to_letter's keys.
    """
    allowed_classes = set(distractor_set.class_idx_to_letter.keys())
    return {ex.index for ex in dataset.examples if ex.label in allowed_classes}
