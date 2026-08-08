import torch

from src.losses.pairwise_ranking import (
    PairwiseRankingLoss,
    pairwise_ranking_accuracy,
)


def test_ordered_scores_have_lower_loss_than_reversed_scores():
    targets = torch.tensor([[3.0, 2.0, 1.0]])
    objective = PairwiseRankingLoss(min_target_gap=0.0)

    ordered = objective(torch.tensor([[3.0, 2.0, 1.0]]), targets)
    reversed_order = objective(torch.tensor([[1.0, 2.0, 3.0]]), targets)
    assert ordered < reversed_order

    accuracy, pair_count = pairwise_ranking_accuracy(
        torch.tensor([[3.0, 2.0, 1.0]]),
        targets,
        min_target_gap=0.0,
    )
    assert pair_count == 3
    assert accuracy.item() == 1.0


def test_mask_and_gap_threshold_exclude_invalid_or_near_tie_pairs():
    scores = torch.tensor([[3.0, 2.0, 100.0]])
    targets = torch.tensor([[1.0, 0.99, -100.0]])
    mask = torch.tensor([[True, True, False]])
    objective = PairwiseRankingLoss(min_target_gap=0.02)

    loss = objective(scores, targets, mask)
    assert loss.item() == 0.0

    accuracy, pair_count = pairwise_ranking_accuracy(
        scores, targets, mask, min_target_gap=0.02
    )
    assert pair_count == 0
    assert torch.isnan(accuracy)


def test_empty_pair_loss_remains_differentiable():
    scores = torch.tensor([[1.0]], requires_grad=True)
    targets = torch.tensor([[0.0]])
    loss = PairwiseRankingLoss()(scores, targets)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.item() == 0.0


def test_teacher_relevance_weighting_prioritizes_top_candidate_pairs():
    targets = torch.tensor([[1.0, 0.5, 0.0]])
    # Both pairs containing the top candidate are correct; the only incorrect
    # ordering is between the two lower-utility candidates.
    scores = torch.tensor([[1.0, -1.0, 0.0]])
    uniform = PairwiseRankingLoss(min_target_gap=0.0)(scores, targets)
    top_weighted = PairwiseRankingLoss(
        min_target_gap=0.0,
        teacher_weight_temperature=0.1,
    )(scores, targets)
    assert top_weighted < uniform
