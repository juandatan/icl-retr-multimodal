import numpy as np
import pytest
import torch
from types import SimpleNamespace

import scripts.build_reranker_visual_token_cache as cache_builder
from scripts.build_reranker_visual_token_cache import (
    AWQ_UNQUANTIZED_MODULES,
    _quantize_visual_tokens,
    _validate_feature_source,
)


def _config(quantization_config=None):
    config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=4,
            image_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            patch_size=2,
        ),
        perceiver_config=SimpleNamespace(
            resampler_depth=1,
            resampler_head_dim=2,
            resampler_n_heads=2,
            resampler_n_latents=3,
        ),
        text_config=SimpleNamespace(hidden_size=4, vocab_size=12),
    )
    if quantization_config is not None:
        config.quantization_config = quantization_config
    return config


class _Tokenizer:
    @staticmethod
    def encode(value, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in value]


class _Model:
    def __init__(self, config):
        self.config = config
        self.model = SimpleNamespace(
            vision_model=torch.nn.Linear(4, 4),
            connector=SimpleNamespace(
                modality_projection=torch.nn.Linear(4, 4),
                perceiver_resampler=torch.nn.Linear(4, 4),
            ),
        )
        self.embedding = torch.nn.Embedding(12, 4)

    def get_input_embeddings(self):
        return self.embedding


def _wrapper(config):
    return SimpleNamespace(
        model=_Model(config),
        processor=SimpleNamespace(tokenizer=_Tokenizer()),
    )


def test_per_token_int8_quantization_round_trips_and_reports_error():
    generator = np.random.default_rng(7)
    values = generator.normal(size=(4, 32)).astype(np.float32)
    quantized, scales, metrics = _quantize_visual_tokens(values)
    reconstructed = quantized.astype(np.float32) * scales.astype(np.float32)

    assert quantized.dtype == np.int8
    assert scales.dtype == np.float16
    assert scales.shape == (4, 1)
    assert np.max(np.abs(quantized)) <= 127
    np.testing.assert_allclose(
        metrics["mean_abs_error"],
        np.abs(values - reconstructed).mean(),
    )
    np.testing.assert_allclose(
        metrics["max_abs_error"],
        np.abs(values - reconstructed).max(),
    )
    assert metrics["cosine_similarity"] > 0.9999


def test_per_token_int8_quantization_handles_zero_tokens():
    quantized, scales, metrics = _quantize_visual_tokens(
        np.zeros((2, 8), dtype=np.float32)
    )

    assert not quantized.any()
    np.testing.assert_array_equal(scales, np.ones((2, 1), dtype=np.float16))
    assert metrics == {
        "mean_abs_error": 0.0,
        "max_abs_error": 0.0,
        "cosine_similarity": 1.0,
    }


def test_awq_feature_source_requires_and_validates_unquantized_visual_modules(
    monkeypatch,
):
    teacher_config = _config()
    source_config = _config({
        "quant_method": "awq",
        "modules_to_not_convert": sorted(AWQ_UNQUANTIZED_MODULES),
    })
    monkeypatch.setattr(
        cache_builder.AutoConfig,
        "from_pretrained",
        lambda _: teacher_config,
    )
    monkeypatch.setattr(
        cache_builder.AutoTokenizer,
        "from_pretrained",
        lambda _: _Tokenizer(),
    )

    result = _validate_feature_source(
        _wrapper(source_config),
        teacher_model="teacher",
        feature_source_model="teacher-AWQ",
        class_names=["bird one", "bird two"],
    )

    assert result["alternate_source"]
    assert result["quantization_method"] == "awq"
    assert set(result["runtime_unquantized_modules"]) == AWQ_UNQUANTIZED_MODULES


def test_awq_feature_source_rejects_missing_connector_exclusion(monkeypatch):
    teacher_config = _config()
    source_config = _config({
        "quant_method": "awq",
        "modules_to_not_convert": ["model.vision_model"],
    })
    monkeypatch.setattr(
        cache_builder.AutoConfig,
        "from_pretrained",
        lambda _: teacher_config,
    )
    monkeypatch.setattr(
        cache_builder.AutoTokenizer,
        "from_pretrained",
        lambda _: _Tokenizer(),
    )

    with pytest.raises(ValueError, match="missing="):
        _validate_feature_source(
            _wrapper(source_config),
            teacher_model="teacher",
            feature_source_model="teacher-AWQ",
            class_names=["bird"],
        )
