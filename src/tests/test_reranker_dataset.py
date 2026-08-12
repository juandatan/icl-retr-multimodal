from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from src.data.reranker_dataset import (
    RerankerTeacherDataset,
    collate_reranker_queries,
)
from src.losses.pairwise_ranking import PairwiseRankingLoss
from src.models.reranker import LabelAwareReranker, RerankerConfig
from src.utils.reranker_teacher_data import derive_candidate_metrics


def _record(query_idx, candidate_indices, split="train"):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
    candidate_count = len(candidate_indices)
    zero_scores = np.asarray([1.0, 2.0, 0.0], dtype=np.float32)
    candidate_scores = np.stack([
        np.asarray([2.0 + position, 1.0, 0.0], dtype=np.float32)
        for position in range(candidate_count)
    ])
    zero, candidates = derive_candidate_metrics(
        zero_scores, candidate_scores, true_local_idx=0
    )
    return SimpleNamespace(
        query_split=split,
        query_idx=query_idx,
        true_class_idx=0,
        label_class_indices=np.asarray([0, 1, 2], dtype=np.int16),
        label_siglip_similarities=np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
        ranked_distractor_class_indices=np.asarray([1, 2, 3], dtype=np.int16),
        zero_shot_scores=zero_scores,
        zero_shot_metrics=zero,
        candidate_indices=candidate_indices,
        candidate_class_indices=np.arange(candidate_count, dtype=np.int16),
        candidate_similarities=np.linspace(
            0.9, 0.7, candidate_count, dtype=np.float32
        ),
        candidate_scores=candidate_scores,
        candidate_metrics=candidates,
    )


def _artifact():
    clip_train = np.arange(24, dtype=np.float32).reshape(6, 4)
    siglip_train = np.arange(30, dtype=np.float32).reshape(6, 5)
    label_features = np.arange(20, dtype=np.float32).reshape(4, 5)
    return {
        "method": "reranker_teacher_data",
        "immutable_args": {
            "schema_version": 2,
            "retrieval_split": "train",
        },
        "records": [
            _record(0, [1, 2, 3]),
            _record(1, [2, 3]),
        ],
        "feature_tables": {
            "clip_image_embeddings_by_split": {"train": clip_train},
            "siglip_image_embeddings_by_split": {"train": siglip_train},
            "siglip_class_text_embeddings": label_features,
            "class_names": ["a", "b", "c", "d"],
        },
    }


def _visual_token_cache(tmp_path, *, complete=True, dtype="float16"):
    cache_dir = tmp_path / "visual_tokens"
    cache_dir.mkdir()
    raw_tokens = np.arange(6 * 3 * 7, dtype=np.float32).reshape(6, 3, 7)
    if dtype == "int8":
        scales = (
            np.max(np.abs(raw_tokens), axis=-1, keepdims=True) / 127.0
        ).astype(np.float16)
        scales[scales == 0] = 1
        stored_tokens = np.clip(
            np.rint(raw_tokens / scales.astype(np.float32)), -127, 127
        ).astype(np.int8)
        np.save(cache_dir / "image_token_scales_train.npy", scales)
    else:
        stored_tokens = raw_tokens.astype(np.float16)
    np.save(cache_dir / "image_tokens_train.npy", stored_tokens)
    np.save(
        cache_dir / "label_token_embeddings.npy",
        np.arange(4 * 2 * 7, dtype=np.float16).reshape(4, 2, 7),
    )
    np.save(
        cache_dir / "label_token_mask.npy",
        np.asarray([[True, True], [True, False], [True, True], [True, False]]),
    )
    metadata = {
        "method": "idefics2_reranker_visual_token_cache",
        "schema_version": 2 if dtype == "int8" else 1,
        "complete": complete,
        "class_names": ["a", "b", "c", "d"],
        "split_rows": {"train": 6},
        "dtype": dtype,
        "quantization": (
            {"scheme": "symmetric_per_visual_token"}
            if dtype == "int8"
            else None
        ),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))
    return cache_dir


def test_dataset_resolves_query_candidate_and_label_features():
    dataset = RerankerTeacherDataset(_artifact(), split="train", target="margin")
    item = dataset[0]

    assert dataset.clip_dim == 4
    assert dataset.siglip_dim == 5
    assert item["candidate_clip"].shape == (3, 4)
    assert item["candidate_siglip"].shape == (3, 5)
    np.testing.assert_array_equal(
        item["candidate_label_siglip"].numpy(),
        _artifact()["feature_tables"]["siglip_class_text_embeddings"][:3],
    )
    np.testing.assert_allclose(item["retrieval_ranks"].numpy(), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(item["targets"].numpy(), [1.0, 2.0, 3.0])


def test_dataset_can_select_ranked_prefix_from_larger_artifact():
    dataset = RerankerTeacherDataset(
        _artifact(), split="train", target="margin", max_candidates=2
    )
    item = dataset[0]

    assert item["candidate_indices"].tolist() == [1, 2]
    assert item["candidate_class_indices"].tolist() == [0, 1]
    assert item["targets"].tolist() == [1.0, 2.0]
    assert item["teacher_margins"].shape == (2,)
    np.testing.assert_allclose(item["retrieval_ranks"].numpy(), [0.0, 1.0])


def test_dataset_rejects_candidate_limit_larger_than_artifact():
    with pytest.raises(ValueError, match="only 2 candidates"):
        RerankerTeacherDataset(
            _artifact(), split="train", max_candidates=3
        )


def test_dataset_rejects_incomplete_teacher_artifact():
    artifact = _artifact()
    artifact["complete"] = False
    with pytest.raises(ValueError, match="artifact is incomplete"):
        RerankerTeacherDataset(artifact, split="train")


def test_dataset_rejects_legacy_schema_v1():
    artifact = _artifact()
    artifact["immutable_args"]["schema_version"] = 1

    with pytest.raises(ValueError, match="schema version: 1"):
        RerankerTeacherDataset(artifact, split="train")


def test_collator_pads_variable_candidate_pools_and_builds_mask():
    dataset = RerankerTeacherDataset(_artifact(), split="train")
    batch = collate_reranker_queries([dataset[0], dataset[1]])

    assert batch["candidate_clip"].shape == (2, 3, 4)
    assert batch["candidate_mask"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert batch["candidate_indices"][1, 2].item() == -1
    assert batch["targets"][1, 2].item() == 0.0


def test_dataset_and_collator_expose_memory_mapped_visual_tokens(tmp_path):
    cache_dir = _visual_token_cache(tmp_path)
    dataset = RerankerTeacherDataset(
        _artifact(),
        split="train",
        visual_token_cache_path=cache_dir,
    )
    item = dataset[0]
    batch = collate_reranker_queries([dataset[0], dataset[1]])

    assert dataset.visual_token_dim == 7
    assert dataset.visual_token_count == 3
    assert dataset.label_token_count == 2
    assert item["query_visual_tokens"].shape == (3, 7)
    assert item["candidate_visual_tokens"].shape == (3, 3, 7)
    assert item["candidate_label_tokens"].shape == (3, 2, 7)
    assert batch["candidate_visual_tokens"].shape == (2, 3, 3, 7)
    assert batch["candidate_label_token_mask"].dtype == torch.bool
    assert not batch["candidate_label_token_mask"][1, 2].any()


def test_dataset_transparently_dequantizes_int8_visual_tokens(tmp_path):
    cache_dir = _visual_token_cache(tmp_path, dtype="int8")
    dataset = RerankerTeacherDataset(
        _artifact(),
        split="train",
        visual_token_cache_path=cache_dir,
    )
    item = dataset[0]

    expected = np.arange(6 * 3 * 7, dtype=np.float32).reshape(6, 3, 7)
    assert item["query_visual_tokens"].dtype == torch.float16
    np.testing.assert_allclose(
        item["query_visual_tokens"].float().numpy(), expected[0], atol=0.5
    )
    np.testing.assert_allclose(
        item["candidate_visual_tokens"].float().numpy(),
        expected[[1, 2, 3]],
        atol=0.5,
    )


def test_dataset_rejects_int8_cache_without_scales(tmp_path):
    cache_dir = _visual_token_cache(tmp_path, dtype="int8")
    (cache_dir / "image_token_scales_train.npy").unlink()

    with pytest.raises(FileNotFoundError, match="missing scales"):
        RerankerTeacherDataset(
            _artifact(),
            split="train",
            visual_token_cache_path=cache_dir,
        )


def test_dataset_rejects_incomplete_visual_token_cache(tmp_path):
    cache_dir = _visual_token_cache(tmp_path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        RerankerTeacherDataset(
            _artifact(),
            split="train",
            visual_token_cache_path=cache_dir,
        )


def test_dataset_rejects_unknown_target_and_missing_split():
    with pytest.raises(ValueError, match="Unsupported target"):
        RerankerTeacherDataset(_artifact(), split="train", target="not_a_target")
    with pytest.raises(ValueError, match="Missing feature tables"):
        RerankerTeacherDataset(_artifact(), split="val")


def test_dataset_derives_bounded_and_normalized_targets():
    bounded = RerankerTeacherDataset(
        _artifact(), split="train", target="bounded_margin", target_temperature=2.0
    )[0]["targets"]
    expected = torch.sigmoid(torch.tensor([1.0, 2.0, 3.0]) / 2.0)
    torch.testing.assert_close(bounded, expected)

    normalized = RerankerTeacherDataset(
        _artifact(),
        split="train",
        target="normalized_incremental_probability",
        incremental_lambda=1.0,
    )[0]["targets"]
    assert torch.all((0 <= normalized) & (normalized <= 1))


def test_dataset_exposes_teacher_metrics_only_for_evaluation():
    item = RerankerTeacherDataset(_artifact(), split="train")[0]
    assert item["teacher_margins"].shape == item["targets"].shape
    assert item["teacher_correct"].dtype == torch.bool
    assert "true_class_idx" not in item


def test_dataset_model_and_loss_contract_supports_one_training_step():
    dataset = RerankerTeacherDataset(_artifact(), split="train")
    batch = collate_reranker_queries([dataset[0], dataset[1]])
    model = LabelAwareReranker(
        RerankerConfig(
            clip_dim=dataset.clip_dim,
            siglip_dim=dataset.siglip_dim,
            hidden_dim=8,
            metadata_dim=4,
        )
    )
    scores = model(
        query_clip=batch["query_clip"],
        candidate_clip=batch["candidate_clip"],
        query_siglip=batch["query_siglip"],
        candidate_siglip=batch["candidate_siglip"],
        candidate_label_siglip=batch["candidate_label_siglip"],
        clip_similarities=batch["clip_similarities"],
        retrieval_ranks=batch["retrieval_ranks"],
        candidate_mask=batch["candidate_mask"],
    )
    loss = PairwiseRankingLoss(min_target_gap=0.0)(
        scores,
        batch["targets"],
        batch["candidate_mask"],
    )
    loss.backward()

    assert scores.shape == batch["candidate_mask"].shape
    assert torch.isfinite(loss)
