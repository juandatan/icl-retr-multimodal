"""
Shared dataclass definitions used across the project.

This module provides standalone dataclass definitions that can be safely
imported without heavy dependencies, useful for pickle deserialization.
"""

from dataclasses import dataclass


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
