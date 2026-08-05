"""Unit tests for MC prompt construction; no model weights are loaded."""

import src.models.idefics2_wrapper as wrapper_module
from src.models.idefics2_wrapper import (
    Idefics2Wrapper,
    _build_idefics2_processor,
)


def _wrapper_without_model() -> Idefics2Wrapper:
    return Idefics2Wrapper.__new__(Idefics2Wrapper)


def test_concrete_processor_builder_bypasses_auto_image_processor(monkeypatch):
    tokenizer = object()
    image_processor = object()
    constructed = object()
    monkeypatch.setattr(
        wrapper_module.AutoTokenizer,
        "from_pretrained",
        lambda _: tokenizer,
    )
    monkeypatch.setattr(
        wrapper_module,
        "Idefics2ImageProcessor",
        lambda **_: image_processor,
    )

    def processor_factory(**kwargs):
        assert kwargs == {
            "image_processor": image_processor,
            "tokenizer": tokenizer,
            "image_seq_len": 64,
        }
        return constructed

    monkeypatch.setattr(wrapper_module, "Idefics2Processor", processor_factory)

    assert _build_idefics2_processor("checkpoint") is constructed


def test_zero_shot_prompt_requests_a_letter_for_the_query():
    prompt = _wrapper_without_model()._format_mc_prompt({"A": "albatross", "B": "auk"})

    assert "A. albatross" in prompt
    assert "B. auk" in prompt
    assert "Classify the query image. Output only the letter" in prompt
    assert prompt.endswith("<image>\nAnswer:")
    assert "Reference label:" not in prompt


def test_context_is_a_labeled_reference_not_a_conflicting_output():
    prompt = _wrapper_without_model()._format_mc_prompt(
        {"A": "albatross", "B": "auk"},
        example_labels=["a bird outside the option set"],
    )

    assert "Labeled reference examples:" in prompt
    assert "Reference label: a bird outside the option set" in prompt
    assert "Output: a bird outside the option set" not in prompt
    assert prompt.count("<image>") == 2
