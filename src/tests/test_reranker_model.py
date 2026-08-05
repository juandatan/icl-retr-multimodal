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
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
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
        "candidate_mask",
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


def test_pooled_transformer_scores_candidates_and_backpropagates():
    model = LabelAwareReranker(RerankerConfig(
        clip_dim=6,
        siglip_dim=8,
        architecture="pooled_transformer",
        hidden_dim=16,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=32,
    ))
    scores = model(**_inputs())
    scores.sum().backward()
    assert scores.shape == (2, 4)
    assert model.score_token.grad is not None


def _cross_candidate_model():
    return LabelAwareReranker(RerankerConfig(
        clip_dim=6,
        siglip_dim=8,
        architecture="cross_candidate_attention",
        hidden_dim=16,
        dropout=0.0,
        candidate_context_heads=4,
        candidate_context_layers=1,
        candidate_context_ff_dim=32,
    )).eval()


def test_cross_candidate_attention_is_permutation_equivariant():
    model = _cross_candidate_model()
    inputs = _inputs(batch=1, candidates=4)
    permutation = torch.tensor([2, 0, 3, 1])
    candidate_keys = {
        "candidate_clip",
        "candidate_siglip",
        "candidate_label_siglip",
        "clip_similarities",
        "retrieval_ranks",
        "candidate_mask",
    }
    permuted = {
        key: value[:, permutation] if key in candidate_keys else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        original_scores = model(**inputs)
        permuted_scores = model(**permuted)
    torch.testing.assert_close(permuted_scores, original_scores[:, permutation])


def test_cross_candidate_attention_scores_candidates_and_backpropagates():
    model = _cross_candidate_model().train()
    scores = model(**_inputs())
    scores.sum().backward()
    assert scores.shape == (2, 4)
    assert all(
        parameter.grad is not None
        for parameter in model.candidate_context.parameters()
        if parameter.requires_grad
    )


def test_cross_candidate_attention_masks_padding_and_uses_other_candidates():
    model = _cross_candidate_model()
    inputs = _inputs(batch=1, candidates=4)
    inputs["candidate_mask"][0, 3] = False

    changed_padding = {key: value.clone() for key, value in inputs.items()}
    changed_padding["candidate_siglip"][0, 3] = 1000
    changed_padding["candidate_label_siglip"][0, 3] = -1000
    with torch.no_grad():
        original_scores = model(**inputs)
        changed_padding_scores = model(**changed_padding)
    torch.testing.assert_close(
        changed_padding_scores[:, :3], original_scores[:, :3]
    )

    changed_candidate = {key: value.clone() for key, value in inputs.items()}
    changed_candidate["candidate_siglip"][0, 1] *= -3
    with torch.no_grad():
        changed_candidate_scores = model(**changed_candidate)
    assert not torch.isclose(
        changed_candidate_scores[0, 0], original_scores[0, 0]
    )


def test_optional_clip_and_metadata_branches_are_independent_ablation_flags():
    model = LabelAwareReranker(RerankerConfig(
        clip_dim=6,
        siglip_dim=8,
        hidden_dim=16,
        metadata_dim=4,
        use_clip_embeddings=True,
        use_clip_similarity=True,
        use_retrieval_rank=True,
        use_derived_siglip_similarities=True,
    ))
    assert model(**_inputs()).shape == (2, 4)


def _visual_token_model():
    return LabelAwareReranker(RerankerConfig(
        clip_dim=6,
        siglip_dim=8,
        architecture="visual_token_cross_encoder",
        hidden_dim=8,
        dropout=0.0,
        visual_token_dim=10,
        visual_token_count=3,
        visual_label_token_count=2,
        visual_token_heads=2,
        visual_token_layers=1,
        visual_token_ff_dim=16,
        visual_candidate_chunk_size=2,
    ))


def _visual_inputs(batch=2, candidates=4):
    generator = torch.Generator().manual_seed(19)
    label_mask = torch.ones(batch, candidates, 2, dtype=torch.bool)
    label_mask[:, :, 1] = False
    return {
        "query_visual_tokens": torch.randn(
            batch, 3, 10, generator=generator, dtype=torch.float16
        ),
        "candidate_visual_tokens": torch.randn(
            batch, candidates, 3, 10, generator=generator, dtype=torch.float16
        ),
        "candidate_label_tokens": torch.randn(
            batch, candidates, 2, 10, generator=generator, dtype=torch.float16
        ),
        "candidate_label_token_mask": label_mask,
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
    }


def test_visual_token_cross_encoder_scores_candidates_and_backpropagates():
    model = _visual_token_model().train()
    scores = model(**_visual_inputs())
    scores.sum().backward()

    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_visual_token_cross_encoder_is_candidate_independent_and_masks_labels():
    model = _visual_token_model().eval()
    inputs = _visual_inputs(batch=1, candidates=4)
    with torch.no_grad():
        original = model(**inputs)

    changed_padding = {key: value.clone() for key, value in inputs.items()}
    changed_padding["candidate_label_tokens"][:, :, 1] = 1000
    with torch.no_grad():
        padding_scores = model(**changed_padding)
    torch.testing.assert_close(padding_scores, original)

    changed_candidate = {key: value.clone() for key, value in inputs.items()}
    changed_candidate["candidate_visual_tokens"][:, 1] *= -5
    with torch.no_grad():
        changed_scores = model(**changed_candidate)
    torch.testing.assert_close(changed_scores[:, [0, 2, 3]], original[:, [0, 2, 3]])
    assert not torch.isclose(changed_scores[0, 1], original[0, 1])
