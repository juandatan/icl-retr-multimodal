from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_full_label_baselines import (
    _merge_records,
    _validate_resume_records,
    build_query_tasks,
    closed_set_metrics,
    stratified_query_indices,
    summarize_results,
)
from src.models.idefics2_wrapper import (
    PREFIX_KV_CACHE_SCORING,
    Idefics2Wrapper,
)
from src.utils.runtime import stratified_sample_indices


def test_stratified_sample_balances_classes_and_is_reproducible():
    examples = [
        SimpleNamespace(index=1000 + index, label=index // 4)
        for index in range(20)
    ]
    first = stratified_query_indices(examples, num_queries=8, seed=42)
    second = stratified_query_indices(examples, num_queries=8, seed=42)
    counts = np.bincount([examples[index].label for index in first], minlength=5)

    assert first == second
    assert len(first) == len(set(first)) == 8
    assert counts.max() - counts.min() <= 1


def test_proportional_stratified_rows_preserve_imbalanced_class_ratio():
    labels = [0] * 8 + [1] * 2
    first = stratified_sample_indices(labels, sample_count=5, seed=7)
    second = stratified_sample_indices(labels, sample_count=5, seed=7)
    selected_labels = np.asarray(labels)[first]

    assert first == second
    assert len(first) == len(set(first)) == 5
    assert np.sum(selected_labels == 0) == 4
    assert np.sum(selected_labels == 1) == 1


def test_closed_set_metrics_reports_rank_margin_and_probability():
    metrics = closed_set_metrics([0.0, 2.0, 1.0, -1.0], true_class_idx=2)

    assert metrics["predicted_class_idx"] == 1
    assert metrics["correct"] is False
    assert metrics["top5_correct"] is True
    assert metrics["true_rank"] == 2
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["true_margin"] == -1.0
    assert 0 < metrics["true_probability"] < 1
    assert np.isclose(metrics["true_log_probability"], np.log(metrics["true_probability"]))


def test_closed_set_log_probability_does_not_underflow():
    metrics = closed_set_metrics([0.0, -1000.0], true_class_idx=1)

    assert metrics["true_probability"] == 0.0
    assert np.isfinite(metrics["true_log_probability"])
    assert np.isclose(metrics["true_log_probability"], -1000.0)


def test_full_label_prompt_contains_no_candidate_option_list():
    wrapper = Idefics2Wrapper.__new__(Idefics2Wrapper)
    prompt = wrapper._format_full_label_scoring_prompt(["Laysan Albatross"])

    assert "Labeled reference examples:" in prompt
    assert "Label: Laysan Albatross" in prompt
    assert "possible classes" not in prompt.lower()
    assert prompt.endswith("Query image:\n<image>\nLabel:")
    assert prompt.count("<image>") == 2


def test_restricted_metrics_mask_fixed_scores_and_are_monotonic_in_k():
    record = SimpleNamespace(
        query_idx=7,
        true_class_idx=0,
        clip_example_class_idx=0,
        clip_similarity=0.9,
        zero_shot_scores=[3.0, 2.0, 4.0, 1.0, 0.0, -1.0],
        clip_scores=[5.0, 2.0, 4.0, 1.0, 0.0, -1.0],
        distractor_class_indices={
            2: [0, 1],
            4: [0, 1, 2, 3],
            6: [0, 1, 2, 3, 4, 5],
        },
    )

    results = summarize_results([record], temperature=1.0, k_values=[2, 4, 6])

    assert results["conditions"]["zero_shot"]["accuracy"] == 0.0
    assert [
        results["restricted_by_k"][k]["conditions"]["zero_shot"]["accuracy"]
        for k in (2, 4, 6)
    ] == [1.0, 0.0, 0.0]
    assert [
        results["restricted_by_k"][k]["conditions"]["clip_top1"]["accuracy"]
        for k in (2, 4, 6)
    ] == [1.0, 1.0, 1.0]


def test_query_tasks_build_nested_siglip_distractor_sets():
    dataset = SimpleNamespace(examples=[SimpleNamespace(label=0)])
    text_embeddings = np.eye(4)
    image_embeddings = np.array([[0.1, 0.9, 0.8, 0.7]])

    tasks = build_query_tasks(
        dataset,
        query_indices=[0],
        siglip_image_embeddings=image_embeddings,
        siglip_text_embeddings=text_embeddings,
        k_values=[2, 3, 4],
    )

    sets = tasks[0]["distractor_class_indices"]
    assert sets[2] == [0, 1]
    assert sets[3] == [0, 1, 2]
    assert sets[4] == [0, 1, 2, 3]


def test_worker_records_merge_in_original_query_order():
    records = {
        query_idx: SimpleNamespace(query_idx=query_idx)
        for query_idx in (11, 22, 33, 44)
    }

    merged = _merge_records(
        existing_records=[records[22]],
        shard_records=[[records[33], records[11]], [records[44]]],
        query_indices=[11, 22, 33, 44],
    )

    assert [record.query_idx for record in merged] == [11, 22, 33, 44]


def test_partial_progress_checkpoint_records_are_valid_for_declared_sample():
    records = [SimpleNamespace(query_idx=11), SimpleNamespace(query_idx=33)]

    _validate_resume_records(records, saved_query_indices=[11, 22, 33, 44])


def test_checkpoint_records_must_be_unique_and_within_declared_sample():
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        _validate_resume_records(
            [SimpleNamespace(query_idx=11), SimpleNamespace(query_idx=11)],
            saved_query_indices=[11, 22],
        )
    with np.testing.assert_raises_regex(ValueError, "outside"):
        _validate_resume_records(
            [SimpleNamespace(query_idx=33)],
            saved_query_indices=[11, 22],
        )


def test_image_feature_concatenation_supports_current_flattened_layout():
    context = torch.zeros(64, 16)
    query = torch.ones(64, 16)

    combined = Idefics2Wrapper.combine_full_label_scoring_image_features(context, query)

    assert combined.shape == (128, 16)
    assert torch.equal(combined[:64], context)
    assert torch.equal(combined[64:], query)


def test_image_encoding_uses_post_connector_pooler_output():
    raw_vision_features = torch.zeros(1, 729, 12)
    projected_features = torch.ones(64, 16)

    class FakeProcessor:
        @staticmethod
        def __call__(images, text, return_tensors):
            return {
                "pixel_values": torch.zeros(1, 1, 3, 4, 4),
                "pixel_attention_mask": torch.ones(1, 1, 4, 4),
            }

    class FakeModel:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))

        @staticmethod
        def get_image_features(**kwargs):
            return SimpleNamespace(
                last_hidden_state=raw_vision_features,
                pooler_output=projected_features,
            )

    wrapper = Idefics2Wrapper.__new__(Idefics2Wrapper)
    wrapper.device = "cpu"
    wrapper.processor = FakeProcessor()
    wrapper.model = FakeModel()

    result = wrapper._get_image_features(SimpleNamespace())

    assert torch.equal(result, projected_features)


def test_image_feature_concatenation_supports_legacy_layout():
    context = torch.zeros(1, 1, 64, 16)
    query = torch.ones(1, 1, 64, 16)

    combined = Idefics2Wrapper.combine_full_label_scoring_image_features(context, query)

    assert combined.shape == (1, 2, 64, 16)
    assert torch.equal(combined[:, :1], context)
    assert torch.equal(combined[:, 1:], query)


def test_precomputed_scoring_accepts_flattened_transformers_features():
    wrapper = Idefics2Wrapper.__new__(Idefics2Wrapper)
    calls = []

    def fake_score_batch(*, image_hidden_states_list, prompts, labels):
        calls.append((image_hidden_states_list, prompts, labels))
        return [-float(len(label)) for label in labels]

    wrapper._compute_label_probabilities_batch_with_features = fake_score_batch
    features = torch.zeros(64, 16)
    scores = wrapper.score_candidate_labels_with_image_features(
        features,
        example_labels=[],
        candidate_labels=["A", "Bird", "Long Bird"],
        batch_size=2,
    )

    assert scores == {"A": -1.0, "Bird": -4.0, "Long Bird": -9.0}
    assert [call[2] for call in calls] == [["A", "Bird"], ["Long Bird"]]
    assert all(call[0][0].shape == (64, 16) for call in calls)


def test_prefix_cache_prefills_once_and_branches_label_chunks():
    vocabulary_size = 16
    tokenizations = {
        "A": [1],
        "Bee": [2, 3, 4],
        "Cat": [5, 6],
    }

    class FakeTokenizer:
        pad_token_id = 0

        @staticmethod
        def encode(label, add_special_tokens=False):
            assert add_special_tokens is False
            return tokenizations[label]

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        @staticmethod
        def __call__(text, return_tensors):
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[7, 8]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }

    class FakeCache:
        def __init__(self):
            self.repeats = None

        def batch_repeat_interleave(self, repeats):
            self.repeats = repeats

    class FakeModel:
        def __init__(self):
            self.prefix_calls = 0
            self.continuation_batch_sizes = []

        def __call__(self, **inputs):
            if "past_key_values" not in inputs:
                self.prefix_calls += 1
                assert inputs["use_cache"] is True
                assert inputs["logits_to_keep"] == 1
                return SimpleNamespace(
                    logits=torch.zeros(1, 1, vocabulary_size),
                    past_key_values=FakeCache(),
                )
            batch_size, length = inputs["input_ids"].shape
            assert inputs["past_key_values"].repeats == batch_size
            assert inputs["attention_mask"].shape == (batch_size, 2 + length)
            self.continuation_batch_sizes.append(batch_size)
            return SimpleNamespace(
                logits=torch.zeros(batch_size, length, vocabulary_size),
            )

    wrapper = Idefics2Wrapper.__new__(Idefics2Wrapper)
    wrapper.device = "cpu"
    wrapper.processor = FakeProcessor()
    wrapper.model = FakeModel()
    wrapper._label_token_cache = {}
    wrapper._prompt_token_cache = {}
    scores = wrapper.score_candidate_labels_with_image_features(
        torch.zeros(64, 16),
        example_labels=[],
        candidate_labels=list(tokenizations),
        batch_size=2,
        scoring_mode=PREFIX_KV_CACHE_SCORING,
    )

    expected = -float(np.log(vocabulary_size))
    assert wrapper.model.prefix_calls == 1
    assert wrapper.model.continuation_batch_sizes == [2, 1]
    assert all(np.isclose(score, expected) for score in scores.values())
