"""Query-level listwise objectives for exemplar selection."""

import torch
from torch import nn

from src.losses.pairwise_ranking import PairwiseRankingLoss


class MultiplePositiveListwiseLoss(nn.Module):
    """Maximize softmax mass assigned to any teacher-correct exemplar.

    Candidate scores are normalized independently within each query. Queries
    with no teacher-correct candidate are excluded from the mean, as they have
    no positive selection under the current candidate pool and label space.
    """

    def forward(
        self,
        scores: torch.Tensor,
        teacher_correct: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, candidates]")
        if teacher_correct.shape != scores.shape:
            raise ValueError("teacher_correct must match scores")
        if candidate_mask.shape != scores.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean and match scores")
        if teacher_correct.dtype != torch.bool:
            raise ValueError("teacher_correct must be boolean")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("Every query must have at least one valid candidate")
        if not torch.isfinite(scores[candidate_mask]).all():
            raise ValueError("Valid scores contain non-finite values")

        positive_mask = teacher_correct & candidate_mask
        eligible_queries = positive_mask.any(dim=1)
        if not eligible_queries.any():
            return scores.sum() * 0.0

        valid_scores = scores[eligible_queries].masked_fill(
            ~candidate_mask[eligible_queries], -torch.inf
        )
        positive_scores = scores[eligible_queries].masked_fill(
            ~positive_mask[eligible_queries], -torch.inf
        )
        log_all_mass = torch.logsumexp(valid_scores, dim=1)
        log_positive_mass = torch.logsumexp(positive_scores, dim=1)
        return (log_all_mass - log_positive_mass).mean()


class HybridListwisePairwiseLoss(nn.Module):
    """Combine raw-margin pairwise ranking with correctness-set likelihood."""

    def __init__(
        self,
        *,
        listwise_weight: float = 0.25,
        min_target_gap: float = 0.02,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if listwise_weight < 0:
            raise ValueError("listwise_weight must be non-negative")
        self.listwise_weight = float(listwise_weight)
        self.pairwise = PairwiseRankingLoss(
            min_target_gap=min_target_gap,
            score_temperature=score_temperature,
        )
        self.listwise = MultiplePositiveListwiseLoss()

    def forward(
        self,
        scores: torch.Tensor,
        margin_targets: torch.Tensor,
        teacher_correct: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        pairwise_loss = self.pairwise(scores, margin_targets, candidate_mask)
        listwise_loss = self.listwise(scores, teacher_correct, candidate_mask)
        return pairwise_loss + self.listwise_weight * listwise_loss
