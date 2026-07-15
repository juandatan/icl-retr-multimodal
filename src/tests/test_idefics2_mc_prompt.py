"""Unit tests for MC prompt construction; no model weights are loaded."""

from src.models.idefics2_wrapper import Idefics2Wrapper


def _wrapper_without_model() -> Idefics2Wrapper:
    return Idefics2Wrapper.__new__(Idefics2Wrapper)


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
