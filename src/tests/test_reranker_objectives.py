import numpy as np
import pytest
import torch

from src.losses.listwise import MultiplePositiveListwiseLoss
from src.losses.pointwise import MaskedHuberLoss, MaskedSoftLabelBCELoss
from src.utils.reranker_metrics import reranker_selection_metrics


def test_masked_soft_label_bce_ignores_padding_and_backpropagates():
    scores = torch.tensor([[0.0, 1.0, 100.0]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.9, 0.0]])
    mask = torch.tensor([[True, True, False]])
    loss = MaskedSoftLabelBCELoss()(scores, targets, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad[0, 2] == 0


def test_soft_label_bce_rejects_unbounded_targets():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MaskedSoftLabelBCELoss()(
            torch.zeros(1, 2), torch.tensor([[0.0, 2.0]]),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_huber_accepts_raw_margin_targets():
    loss = MaskedHuberLoss()(torch.zeros(1, 2), torch.tensor([[-2.0, 3.0]]),
                             torch.ones(1, 2, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_multiple_positive_listwise_uses_within_query_positive_mass():
    scores = torch.zeros((2, 4), requires_grad=True)
    correct = torch.tensor([
        [True, True, False, False],
        [False, False, False, False],
    ])
    mask = torch.ones_like(correct)

    loss = MultiplePositiveListwiseLoss()(scores, correct, mask)
    loss.backward()

    assert loss.item() == pytest.approx(np.log(2.0))
    assert torch.all(scores.grad[0, :2] < 0)
    assert torch.all(scores.grad[0, 2:] > 0)
    assert torch.all(scores.grad[1] == 0)


def test_multiple_positive_listwise_ignores_padding_and_all_negative_batches():
    scores = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
    correct = torch.tensor([[True, False, True]])
    mask = torch.tensor([[True, True, False]])
    loss = MultiplePositiveListwiseLoss()(scores, correct, mask)
    loss.backward()
    assert loss.item() == pytest.approx(np.log(2.0))
    assert scores.grad[0, 2] == 0

    all_negative_scores = torch.randn(2, 3, requires_grad=True)
    zero = MultiplePositiveListwiseLoss()(
        all_negative_scores,
        torch.zeros(2, 3, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
    )
    zero.backward()
    assert zero.item() == 0.0
    assert torch.all(all_negative_scores.grad == 0)


def test_selection_metrics_report_accuracy_regret_and_rank_agreement():
    metrics = reranker_selection_metrics(
        scores=np.array([[0.1, 0.9, -5.0]]),
        targets=np.array([[0.2, 0.8, 0.0]]),
        margins=np.array([[-1.0, 2.0, 100.0]]),
        correct=np.array([[False, True, True]]),
        mask=np.array([[True, True, False]]),
    )
    assert metrics["restricted_selected_accuracy"] == 1.0
    assert metrics["mean_margin_regret"] == 0.0
    assert metrics["margin_oracle_agreement"] == 1.0
