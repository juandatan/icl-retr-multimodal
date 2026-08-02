"""Masked pointwise objectives for query-grouped reranker batches."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSoftLabelBCELoss(nn.Module):
    """Binary cross entropy between score logits and dense utilities in [0, 1]."""

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape != targets.shape:
            raise ValueError("scores and targets must have shape [batch, candidates]")
        if candidate_mask.shape != scores.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean and match scores")
        valid_targets = targets[candidate_mask]
        if not torch.isfinite(scores[candidate_mask]).all():
            raise ValueError("Valid scores contain non-finite values")
        if not torch.isfinite(valid_targets).all():
            raise ValueError("Valid targets contain non-finite values")
        if torch.any((valid_targets < 0) | (valid_targets > 1)):
            raise ValueError("Soft-label BCE targets must lie in [0, 1]")
        return F.binary_cross_entropy_with_logits(
            scores[candidate_mask], valid_targets
        )


class MaskedHuberLoss(nn.Module):
    """Huber regression for unbounded raw teacher utilities."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape != targets.shape:
            raise ValueError("scores and targets must have shape [batch, candidates]")
        if candidate_mask.shape != scores.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean and match scores")
        return F.huber_loss(
            scores[candidate_mask], targets[candidate_mask], delta=self.delta
        )
