"""
Shared dataclass definitions used across the project.

This module provides standalone dataclass definitions that can be safely
imported without heavy dependencies, useful for pickle deserialization.
"""

from dataclasses import dataclass
import sys
from typing import Any, Dict, List, Optional

# Historical artifacts were serialized while ``src`` was inserted directly on
# sys.path, so their qualified module is ``data.dataclasses``. Register that
# name before any resume pickle is loaded.
sys.modules.setdefault("data.dataclasses", sys.modules[__name__])


@dataclass
class MCEvalResult:
    """Result of one query's multiple-choice evaluation (0-shot / CLIP-top-1 / reranker-top-1).

    Reranker fields are reserved for a follow-up phase that adds a reranker-top-1
    condition; they are always None until that phase wires in a trained checkpoint.

    Oracle fields measure the accuracy ceiling a perfect reranker could achieve by
    exhaustively testing every candidate in the retrieval pool; they are only
    populated when the eval script is run with oracle computation enabled (it costs
    ~pool_size forward passes per query instead of 1), otherwise left None.
    """
    query_idx: int
    true_class_idx: int
    true_letter: str
    k: int
    letter_to_class_idx: Dict[str, int]
    clip_example_idx: Optional[int]
    reranker_example_idx: Optional[int]
    zero_shot_pred_letter: str
    clip_pred_letter: Optional[str]
    reranker_pred_letter: Optional[str]
    zero_shot_probs: Dict[str, float]
    clip_probs: Optional[Dict[str, float]]
    reranker_probs: Optional[Dict[str, float]]
    zero_shot_correct: bool
    clip_correct: Optional[bool]
    reranker_correct: Optional[bool]
    pool_size: int
    pool_has_true_class: bool
    oracle_correct: Optional[bool] = None
    oracle_example_idx: Optional[int] = None
    # Schema-v2 fields. Defaults keep legacy pickles readable.
    letter_seed: int = 42
    random_example_idx: Optional[int] = None
    random_pred_letter: Optional[str] = None
    random_probs: Optional[Dict[str, float]] = None
    random_correct: Optional[bool] = None
    unrestricted_clip_example_idx: Optional[int] = None
    unrestricted_clip_pred_letter: Optional[str] = None
    unrestricted_clip_probs: Optional[Dict[str, float]] = None
    unrestricted_clip_correct: Optional[bool] = None
    unrestricted_clip_example_in_options: Optional[bool] = None
    same_class_example_idx: Optional[int] = None
    same_class_pred_letter: Optional[str] = None
    same_class_probs: Optional[Dict[str, float]] = None
    same_class_correct: Optional[bool] = None
    clip_example_same_class: Optional[bool] = None
    random_example_same_class: Optional[bool] = None


@dataclass
class FullLabelEvalResult:
    """All-class label-likelihood scores for one zero/one-shot query."""

    query_idx: int
    true_class_idx: int
    clip_example_idx: int
    clip_example_class_idx: int
    clip_similarity: float
    zero_shot_scores: List[float]
    clip_scores: List[float]
    distractor_class_indices: Dict[int, List[int]]


@dataclass
class FullLabelOracleResult:
    """Candidate-pool oracle diagnostics for one full-label query."""

    query_idx: int
    true_class_idx: int
    candidate_indices: List[int]
    candidate_class_indices: List[int]
    candidate_similarities: List[float]
    candidate_correct_by_k: Dict[int, List[bool]]
    candidate_margin_by_k: Dict[int, List[float]]
    candidate_true_rank_by_k: Dict[int, List[int]]


@dataclass
class LabelSpaceAuditResult:
    """Full-class scores used to audit restricted label-space targets."""

    query_idx: int
    true_class_idx: int
    zero_shot_scores: List[float]
    candidate_indices: List[int]
    candidate_class_indices: List[int]
    candidate_similarities: List[float]
    candidate_scores: List[List[float]]
    ranked_distractor_class_indices: List[int]


@dataclass
class RerankerTeacherQueryRecord:
    """Dense K-way Idefics2 targets for one query and its exemplar pool.

    Raw score arrays are canonical. Derived arrays are stored as well so model
    training can consume common targets without silently changing their
    definition; they can always be checked or regenerated from the raw scores.
    Array-valued fields use ``Any`` to keep this pickle schema importable without
    importing NumPy in this lightweight module.
    """

    query_split: str
    query_idx: int
    true_class_idx: int
    label_class_indices: Any
    label_siglip_similarities: Any
    ranked_distractor_class_indices: Any
    zero_shot_scores: Any
    zero_shot_metrics: Dict[str, Any]
    candidate_indices: Any
    candidate_class_indices: Any
    candidate_similarities: Any
    candidate_scores: Any
    candidate_metrics: Dict[str, Any]
    # Batch size is not part of the target definition, but recording it makes
    # any kernel-level floating-point variation across resumed runs auditable.
    scoring_batch_size: int = 8
    scoring_mode: str = "full_sequence_batch"
