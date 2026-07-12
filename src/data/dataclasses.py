"""
Shared dataclass definitions used across the project.

This module provides standalone dataclass definitions that can be safely
imported without heavy dependencies, useful for pickle deserialization.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MarginalUtilityResult:
    """Results for a single query-example pair."""
    query_idx: int
    example_idx: int
    query_label: str
    example_label: str
    baseline_log_prob: float
    oneshot_log_prob: float
    marginal_utility: float
    similarity_score: float
    same_class: bool


@dataclass
class MCEvalResult:
    """Result of one query's multiple-choice evaluation (0-shot / CLIP-top-1 / reranker-top-1).

    Reranker fields are reserved for a follow-up phase that adds a reranker-top-1
    condition; they are always None until that phase wires in a trained checkpoint.
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
