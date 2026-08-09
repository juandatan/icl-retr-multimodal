import numpy as np
import torch

from scripts.export_reranker_predictions import _candidate_ranks, _load_checkpoint


def test_candidate_ranks_are_stable_and_leave_padding_at_minus_one():
    scores = np.asarray([[0.2, 0.9, 0.9, -100.0]], dtype=np.float32)
    mask = np.asarray([[True, True, True, False]])

    ranks = _candidate_ranks(scores, mask)

    # Equal scores retain candidate order, making the export reproducible.
    np.testing.assert_array_equal(ranks, [[2, 0, 1, -1]])


def test_checkpoint_loader_rejects_last_epoch_when_it_is_not_selected(tmp_path):
    checkpoint_path = tmp_path / "last.pt"
    torch.save({
        "model_state_dict": {},
        "model_config": {},
        "epoch": 12,
        "best_epoch": 7,
    }, checkpoint_path)

    try:
        _load_checkpoint(checkpoint_path, allow_non_best=False)
    except ValueError as error:
        assert "Pass best.pt" in str(error)
    else:
        raise AssertionError("Expected a non-best checkpoint to be rejected")

    loaded = _load_checkpoint(checkpoint_path, allow_non_best=True)
    assert loaded["epoch"] == 12
