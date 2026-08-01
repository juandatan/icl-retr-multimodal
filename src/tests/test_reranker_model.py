import torch

from src.models.reranker import LabelAwareReranker, RerankerConfig


def _inputs(batch=2, candidates=4):
    generator = torch.Generator().manual_seed(7)
    return {
        "query_clip": torch.randn(batch, 6, generator=generator),
        "candidate_clip": torch.randn(batch, candidates, 6, generator=generator),
        "query_siglip": torch.randn(batch, 8, generator=generator),
        "candidate_siglip": torch.randn(batch, candidates, 8, generator=generator),
        "candidate_label_siglip": torch.randn(
            batch, candidates, 8, generator=generator
        ),
        "clip_similarities": torch.randn(batch, candidates, generator=generator),
        "retrieval_ranks": torch.arange(candidates, dtype=torch.float32)
        .repeat(batch, 1)
        / max(candidates - 1, 1),
    }


def test_reranker_scores_every_candidate_and_backpropagates():
    model = LabelAwareReranker(
        RerankerConfig(clip_dim=6, siglip_dim=8, hidden_dim=16, metadata_dim=4)
    )
    scores = model(**_inputs())
    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()

    scores.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_candidate_permutation_only_permutes_scores():
    model = LabelAwareReranker(
        RerankerConfig(
            clip_dim=6,
            siglip_dim=8,
            hidden_dim=16,
            metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    inputs = _inputs(batch=1, candidates=4)
    permutation = torch.tensor([2, 0, 3, 1])
    candidate_keys = {
        "candidate_clip",
        "candidate_siglip",
        "candidate_label_siglip",
        "clip_similarities",
        "retrieval_ranks",
    }
    permuted = {
        key: value[:, permutation] if key in candidate_keys else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        original_scores = model(**inputs)
        permuted_scores = model(**permuted)
    torch.testing.assert_close(permuted_scores, original_scores[:, permutation])


def test_select_never_returns_a_padded_candidate():
    scores = torch.tensor([[0.1, 0.5, 100.0], [0.8, 0.2, 0.1]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    selected = LabelAwareReranker.select(scores, mask)
    assert selected.tolist() == [1, 0]
