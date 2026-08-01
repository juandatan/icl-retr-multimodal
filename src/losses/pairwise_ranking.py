"""Noise-tolerant within-query pairwise ranking objective."""

import torch
from torch import nn
from torch.nn import functional as F


def _pairwise_differences(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    min_target_gap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.ndim != 2 or scores.shape != targets.shape:
        raise ValueError("scores and targets must have equal shape [batch, candidates]")
    if min_target_gap < 0:
        raise ValueError("min_target_gap must be non-negative")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    if candidate_mask.shape != scores.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean and match scores")
    if not torch.isfinite(scores[candidate_mask]).all():
        raise ValueError("Valid scores contain non-finite values")
    if not torch.isfinite(targets[candidate_mask]).all():
        raise ValueError("Valid targets contain non-finite values")

    candidate_count = scores.shape[1]
    upper_triangle = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            device=scores.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    valid_pairs = (
        candidate_mask.unsqueeze(2)
        & candidate_mask.unsqueeze(1)
        & upper_triangle.unsqueeze(0)
    )
    score_differences = scores.unsqueeze(2) - scores.unsqueeze(1)
    target_differences = targets.unsqueeze(2) - targets.unsqueeze(1)
    valid_pairs &= torch.abs(target_differences) > min_target_gap
    return score_differences, target_differences, valid_pairs


class PairwiseRankingLoss(nn.Module):
    """Rank candidates by teacher utility using a Bradley-Terry loss.

    Pairs whose teacher targets differ by at most ``min_target_gap`` are
    ignored. This prevents numerically unstable or effectively tied teacher
    preferences from contributing contradictory gradients.
    """

    def __init__(
        self,
        min_target_gap: float = 0.02,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min_target_gap < 0:
            raise ValueError("min_target_gap must be non-negative")
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive")
        self.min_target_gap = min_target_gap
        self.score_temperature = score_temperature

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        score_differences, target_differences, valid_pairs = _pairwise_differences(
            scores, targets, candidate_mask, self.min_target_gap
        )
        if not valid_pairs.any():
            return scores.sum() * 0.0
        preference = torch.sign(target_differences[valid_pairs])
        predicted_difference = score_differences[valid_pairs] / self.score_temperature
        return F.softplus(-preference * predicted_difference).mean()


@torch.no_grad()
def pairwise_ranking_accuracy(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    min_target_gap: float = 0.02,
) -> tuple[torch.Tensor, int]:
    """Return preference accuracy and the number of evaluated pairs."""
    score_differences, target_differences, valid_pairs = _pairwise_differences(
        scores, targets, candidate_mask, min_target_gap
    )
    pair_count = int(valid_pairs.sum().item())
    if pair_count == 0:
        return scores.new_tensor(float("nan")), 0
    correct = (
        torch.sign(score_differences[valid_pairs])
        == torch.sign(target_differences[valid_pairs])
    )
    return correct.float().mean(), pair_count
