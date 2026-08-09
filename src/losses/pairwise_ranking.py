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
        teacher_weight_temperature: float | None = None,
    ) -> None:
        super().__init__()
        if min_target_gap < 0:
            raise ValueError("min_target_gap must be non-negative")
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive")
        if teacher_weight_temperature is not None and teacher_weight_temperature <= 0:
            raise ValueError("teacher_weight_temperature must be positive or None")
        self.min_target_gap = min_target_gap
        self.score_temperature = score_temperature
        self.teacher_weight_temperature = teacher_weight_temperature

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
        losses = F.softplus(-preference * predicted_difference)
        if self.teacher_weight_temperature is None:
            return losses.mean()

        # Query-local teacher relevance emphasizes pairs containing candidates
        # near the top of the target ordering. Normalizing retained weights to
        # mean one preserves the overall loss scale.
        candidate_mask = (
            candidate_mask
            if candidate_mask is not None
            else torch.ones_like(scores, dtype=torch.bool)
        )
        relevance = torch.softmax(
            targets.masked_fill(~candidate_mask, -torch.inf)
            / self.teacher_weight_temperature,
            dim=1,
        )
        pair_weights = relevance.unsqueeze(2) + relevance.unsqueeze(1)
        retained_weights = pair_weights[valid_pairs]
        retained_weights = retained_weights / retained_weights.mean().clamp_min(
            torch.finfo(retained_weights.dtype).eps
        )
        return (losses * retained_weights).mean()


class TeacherCorrectnessCrossingLoss(nn.Module):
    """Query-balanced Bradley–Terry loss over positive/negative pairs only."""

    def __init__(self, score_temperature: float = 1.0) -> None:
        super().__init__()
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive")
        self.score_temperature = float(score_temperature)

    def forward(
        self,
        scores: torch.Tensor,
        teacher_correct: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_correct.shape != scores.shape or teacher_correct.dtype != torch.bool:
            raise ValueError("teacher_correct must be boolean and match scores")
        score_differences, target_differences, valid_pairs = _pairwise_differences(
            scores,
            teacher_correct.to(dtype=scores.dtype),
            candidate_mask,
            min_target_gap=0.0,
        )
        if not valid_pairs.any():
            return scores.sum() * 0.0
        preferences = torch.sign(target_differences[valid_pairs])
        losses = F.softplus(
            -preferences
            * score_differences[valid_pairs]
            / self.score_temperature
        )
        query_indices = valid_pairs.nonzero(as_tuple=False)[:, 0]
        loss_sums = scores.new_zeros(scores.shape[0])
        pair_counts = scores.new_zeros(scores.shape[0])
        loss_sums.scatter_add_(0, query_indices, losses)
        pair_counts.scatter_add_(0, query_indices, torch.ones_like(losses))
        eligible = pair_counts > 0
        return (loss_sums[eligible] / pair_counts[eligible]).mean()


class CorrectnessCrossingPairwiseLoss(nn.Module):
    """Prefer every teacher-correct candidate over every incorrect candidate.

    Comparisons within the correct set or within the incorrect set are omitted:
    neither can change top-1 correctness. Queries with no crossing pair produce
    zero correctness loss. A raw-margin auxiliary can optionally retain teacher
    utility ordering, including on all-incorrect queries.
    """

    def __init__(
        self,
        *,
        score_temperature: float = 1.0,
        margin_aux_weight: float = 0.0,
        margin_min_target_gap: float = 0.02,
    ) -> None:
        super().__init__()
        if margin_aux_weight < 0:
            raise ValueError("margin_aux_weight must be non-negative")
        self.margin_aux_weight = float(margin_aux_weight)
        self.correctness = TeacherCorrectnessCrossingLoss(score_temperature)
        self.margin = PairwiseRankingLoss(
            min_target_gap=margin_min_target_gap,
            score_temperature=score_temperature,
        )

    def forward(
        self,
        scores: torch.Tensor,
        margin_targets: torch.Tensor,
        teacher_correct: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_correct.shape != scores.shape or teacher_correct.dtype != torch.bool:
            raise ValueError("teacher_correct must be boolean and match scores")
        correctness_loss = self.correctness(
            scores,
            teacher_correct,
            candidate_mask,
        )
        if self.margin_aux_weight == 0:
            return correctness_loss
        margin_loss = self.margin(scores, margin_targets, candidate_mask)
        return correctness_loss + self.margin_aux_weight * margin_loss


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
