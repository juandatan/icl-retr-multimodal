from types import SimpleNamespace
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.idefics2_probe_dataset import (
    FrozenIdefics2ProbeDataset,
    PROBE_CACHE_METHOD,
    PROBE_CACHE_SCHEMA_VERSION,
    collate_frozen_idefics2_probe_queries,
)
from src.models.idefics2_probe import (
    FrozenIdefics2UtilityProbe,
    TextOnlyIdefics2Processor,
    dequantize_probe_representations,
    encode_frozen_idefics2_pairs,
    format_probe_prompt,
    quantize_probe_representations,
)
from src.utils.reranker_teacher_data import derive_candidate_metrics
from scripts.build_frozen_idefics2_probe_cache import _validate_probe_model_source
from scripts.train_frozen_idefics2_probe import _build_objective, _objective_loss
from src.losses.listwise import HybridListwisePairwiseLoss
from src.losses.pairwise_ranking import PairwiseRankingLoss
from src.losses.pointwise import MaskedSoftLabelBCELoss
from src.losses.pointwise import HybridPointwisePairwiseLoss


def _record(query_idx: int, split: str):
    candidate_indices = np.asarray([1, 2], dtype=np.int32)
    zero_scores = np.asarray([-0.6, -0.8, -0.9], dtype=np.float32)
    candidate_scores = np.asarray(
        [[-0.2, -0.8, -0.9], [-0.7, -0.5, -0.9]], dtype=np.float32
    )
    zero, candidates = derive_candidate_metrics(
        zero_scores, candidate_scores, true_local_idx=0
    )
    return SimpleNamespace(
        query_split=split,
        query_idx=query_idx,
        true_class_idx=0,
        label_class_indices=np.asarray([0, 1, 2], dtype=np.int16),
        label_siglip_similarities=np.asarray([1.0, 0.5, 0.2], dtype=np.float32),
        ranked_distractor_class_indices=np.asarray([1, 2], dtype=np.int16),
        zero_shot_scores=zero_scores,
        zero_shot_metrics=zero,
        candidate_indices=candidate_indices,
        candidate_class_indices=np.asarray([0, 1], dtype=np.int16),
        candidate_similarities=np.asarray([0.9, 0.8], dtype=np.float32),
        candidate_scores=candidate_scores,
        candidate_metrics=candidates,
    )


def _artifact():
    features = {
        "train": np.zeros((3, 2), dtype=np.float32),
        "val": np.zeros((3, 2), dtype=np.float32),
    }
    return {
        "method": "reranker_teacher_data",
        "immutable_args": {"schema_version": 2, "retrieval_split": "train"},
        "records": [_record(0, "train"), _record(0, "val")],
        "feature_tables": {
            "clip_image_embeddings_by_split": features,
            "siglip_image_embeddings_by_split": features,
            "siglip_class_text_embeddings": np.zeros((3, 2), dtype=np.float32),
            "class_names": ["class zero", "class one", "class two"],
        },
    }


def _probe_cache(tmp_path):
    cache = tmp_path / "probe"
    cache.mkdir()
    for split in ("train", "val"):
        values = np.arange(2 * 4, dtype=np.float32).reshape(1, 2, 4)
        quantized, scales = quantize_probe_representations(values.reshape(2, 4))
        np.save(cache / f"pair_representations_{split}.npy", quantized.reshape(1, 2, 4))
        np.save(cache / f"pair_representation_scales_{split}.npy", scales.reshape(1, 2, 1))
        np.save(cache / f"query_indices_{split}.npy", np.asarray([0], dtype=np.int32))
        np.save(cache / f"candidate_indices_{split}.npy", np.asarray([[1, 2]], dtype=np.int32))
    (cache / "metadata.json").write_text(json.dumps({
        "method": PROBE_CACHE_METHOD,
        "schema_version": PROBE_CACHE_SCHEMA_VERSION,
        "complete": True,
        "splits": ["train", "val"],
        "dtype": "int8",
    }))
    return cache


def test_probe_prompt_contains_only_exemplar_label():
    prompt = format_probe_prompt("Black footed albatross")
    assert prompt.count("<image>") == 2
    assert "Black footed albatross" in prompt
    assert "Query image:" in prompt
    assert "candidate" not in prompt.lower()
    assert "option" not in prompt.lower()


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            if text == "<image>":
                return {"input_ids": [32001]}
            return {"input_ids": [1]}
        return {
            "input_ids": text,
            "attention_mask": kwargs,
        }


def test_text_only_processor_expands_visual_placeholders_without_images():
    processor = TextOnlyIdefics2Processor(_FakeTokenizer(), image_seq_len=3)
    result = processor(
        text=["before <image> middle <image> after"],
        padding=True,
        return_tensors="pt",
    )
    expanded = result["input_ids"][0]
    assert expanded.count("<image>") == 6
    assert expanded.count("<fake_token_around_image>") == 4
    assert result["attention_mask"]["padding"] is True
    assert result["attention_mask"]["return_tensors"] == "pt"


def test_probe_representation_quantization_round_trip():
    values = np.asarray([[0.0, -2.0, 1.0], [3.0, 4.0, -1.0]], dtype=np.float32)
    quantized, scales = quantize_probe_representations(values)
    reconstructed = dequantize_probe_representations(quantized, scales)
    assert quantized.dtype == np.int8
    assert scales.shape == (2, 1)
    np.testing.assert_allclose(reconstructed, values, atol=0.04)


class _FakeProcessor:
    def __init__(self):
        self.prompts = None

    def __call__(self, *, text, padding, return_tensors):
        self.prompts = text
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }


class _FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_images = None

    def forward(self, input_ids, attention_mask, image_hidden_states, **kwargs):
        self.received_images = image_hidden_states.detach().cpu()
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
        return SimpleNamespace(last_hidden_state=hidden)


def test_pair_encoder_interleaves_images_and_reads_last_prompt_state():
    processor = _FakeProcessor()
    base = _FakeBase()
    model = SimpleNamespace(model=base)
    exemplar = torch.tensor([[[1.0]], [[2.0]]])
    query = torch.tensor([[[10.0]], [[20.0]]])
    encoded = encode_frozen_idefics2_pairs(
        model,
        processor,
        exemplar,
        query,
        ["a", "b"],
        device="cpu",
        use_amp=False,
    )
    assert base.received_images[:, 0, 0].tolist() == [1.0, 10.0, 2.0, 20.0]
    assert encoded[:, 0].tolist() == [3.0, 5.0]
    assert "a" in processor.prompts[0]
    assert "b" in processor.prompts[1]


def test_probe_dataset_aligns_cache_and_supports_probability_targets(tmp_path):
    cache = _probe_cache(tmp_path)
    direct = FrozenIdefics2ProbeDataset(
        _artifact(), cache, "train", "mean_token_probability"
    )
    incremental = FrozenIdefics2ProbeDataset(
        _artifact(),
        cache,
        "train",
        "normalized_incremental_mean_token_probability",
    )
    direct_item = direct[0]
    incremental_item = incremental[0]
    batch = collate_frozen_idefics2_probe_queries([direct_item])

    assert direct.input_dim == 4
    assert batch["pair_representations"].shape == (1, 2, 4)
    assert batch["candidate_mask"].tolist() == [[True, True]]
    assert torch.all((direct_item["targets"] >= 0) & (direct_item["targets"] <= 1))
    assert torch.all(
        (incremental_item["targets"] >= 0)
        & (incremental_item["targets"] <= 1)
    )
    assert not torch.equal(direct_item["targets"], incremental_item["targets"])

    record = _artifact()["records"][0]
    expected = np.exp(record.candidate_metrics["true_score"])
    np.testing.assert_allclose(direct_item["targets"].numpy(), expected)


def test_linear_probe_scores_each_candidate_and_masks_padding():
    model = FrozenIdefics2UtilityProbe(input_dim=4)
    representations = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    scores = model(representations, mask)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores[mask]).all()
    assert (scores[~mask] == torch.finfo(scores.dtype).min).all()


def test_layernorm_mlp_probe_scores_candidates_and_backpropagates():
    model = FrozenIdefics2UtilityProbe(
        input_dim=4,
        architecture="layernorm_mlp",
        hidden_dim=8,
        dropout=0.1,
    )
    representations = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    scores = model(representations, mask)
    scores[mask].sum().backward()

    assert scores.shape == (2, 3)
    assert torch.isfinite(scores[mask]).all()
    assert (scores[~mask] == torch.finfo(scores.dtype).min).all()
    assert representations.grad is not None
    assert torch.isfinite(representations.grad).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 57


def test_frozen_probe_objective_factory_supports_pointwise_and_pairwise():
    pointwise = _build_objective(OmegaConf.create({
        "objective": {
            "name": "pointwise_bce",
            "pairwise_min_target_gap": 0.02,
            "pairwise_score_temperature": 1.0,
        }
    }))
    pairwise = _build_objective(OmegaConf.create({
        "objective": {
            "name": "pairwise",
            "pairwise_min_target_gap": 0.03,
            "pairwise_score_temperature": 0.5,
        }
    }))
    hybrid = _build_objective(OmegaConf.create({
        "objective": {
            "name": "pointwise_pairwise",
            "hybrid_pairwise_weight": 0.1,
            "pairwise_min_target_gap": 0.02,
            "pairwise_score_temperature": 1.0,
            "pairwise_teacher_weight_temperature": None,
        }
    }))
    listwise_hybrid = _build_objective(OmegaConf.create({
        "objective": {
            "name": "hybrid_listwise_pairwise",
            "hybrid_listwise_weight": 0.1,
            "pairwise_min_target_gap": 0.02,
            "pairwise_score_temperature": 1.0,
            "pairwise_teacher_weight_temperature": None,
        }
    }))

    assert isinstance(pointwise, MaskedSoftLabelBCELoss)
    assert isinstance(pairwise, PairwiseRankingLoss)
    assert isinstance(hybrid, HybridPointwisePairwiseLoss)
    assert isinstance(listwise_hybrid, HybridListwisePairwiseLoss)
    assert pairwise.min_target_gap == 0.03
    assert pairwise.score_temperature == 0.5
    assert hybrid.pairwise_weight == 0.1
    assert listwise_hybrid.listwise_weight == 0.1


def test_hybrid_probe_loss_is_weighted_sum_of_components():
    objective = HybridPointwisePairwiseLoss(
        pairwise_weight=0.1,
        min_target_gap=0.0,
    )
    scores = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
    targets = torch.tensor([[0.8, 0.2, 0.6]])
    mask = torch.ones_like(scores, dtype=torch.bool)
    expected = objective.pointwise(scores, targets, mask) + 0.1 * objective.pairwise(
        scores, targets, mask
    )
    actual = objective(scores, targets, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad is not None


def test_frozen_probe_routes_correctness_to_listwise_hybrid():
    objective = HybridListwisePairwiseLoss(
        listwise_weight=0.1,
        min_target_gap=0.0,
    )
    scores = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
    batch = {
        "targets": torch.tensor([[0.8, 0.2, 0.6]]),
        "teacher_correct": torch.tensor([[True, False, True]]),
        "candidate_mask": torch.ones_like(scores, dtype=torch.bool),
    }
    expected = objective(
        scores,
        batch["targets"],
        batch["teacher_correct"],
        batch["candidate_mask"],
    )
    actual = _objective_loss(objective, scores, batch)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad is not None


def test_probe_model_source_accepts_exact_teacher_and_validated_awq():
    metadata = {
        "feature_source_model": "teacher-AWQ",
        "feature_equivalence_validation": {
            "architecture_matches_teacher": True,
            "quantization_method": "awq",
        },
    }
    assert _validate_probe_model_source(
        teacher_model="teacher",
        probe_model="teacher",
        visual_cache_metadata=metadata,
        load_in_8bit=True,
    ) == "exact_teacher_checkpoint"
    assert _validate_probe_model_source(
        teacher_model="teacher",
        probe_model="teacher-AWQ",
        visual_cache_metadata=metadata,
        load_in_8bit=False,
    ) == "validated_awq_approximation"
