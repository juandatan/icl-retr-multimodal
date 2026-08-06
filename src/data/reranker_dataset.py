"""Dataset adapter for the bundled reranker teacher artifact."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Importing this module registers the historical ``data.dataclasses`` pickle
# alias before an artifact is deserialized.
from src.data import dataclasses as _artifact_dataclasses  # noqa: F401


RAW_TARGETS = frozenset({
    "margin",
    "incremental_margin",
    "true_probability",
    "true_log_probability",
    "incremental_true_probability",
    "incremental_true_log_probability",
})
SUPPORTED_TARGETS = RAW_TARGETS | frozenset({
    "bounded_margin",
    "bounded_incremental_margin",
    "mean_token_probability",
    "normalized_incremental_mean_token_probability",
    "normalized_incremental_probability",
})


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 feature table")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _load_visual_token_cache(
    cache_path: str | Path,
    *,
    payload: Mapping[str, Any],
    required_splits: set[str],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Open a completed memory-mapped Idefics2 visual-token sidecar."""
    cache_dir = Path(cache_path)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Visual-token cache metadata not found: {metadata_path}"
        )
    with metadata_path.open() as file:
        metadata = json.load(file)
    if metadata.get("method") != "idefics2_reranker_visual_token_cache":
        raise ValueError("Not an Idefics2 reranker visual-token cache")
    schema_version = int(metadata.get("schema_version", -1))
    if schema_version not in {1, 2}:
        raise ValueError(
            "Unsupported visual-token cache schema version: "
            f"{metadata.get('schema_version')}"
        )
    if not bool(metadata.get("complete", False)):
        raise ValueError(
            "Visual-token cache is incomplete; resume feature extraction first"
        )

    immutable = payload["immutable_args"]
    expected_model = immutable.get("idefics2_model")
    if expected_model and metadata.get("idefics2_model") != expected_model:
        raise ValueError(
            "Visual-token cache and teacher artifact use different Idefics2 models"
        )
    expected_split_hash = immutable.get("image_split_sha256")
    if (
        expected_split_hash
        and metadata.get("image_split_sha256") != expected_split_hash
    ):
        raise ValueError(
            "Visual-token cache and teacher artifact use different image splits"
        )

    tables = payload["feature_tables"]
    expected_class_names = list(tables.get("class_names", []))
    if list(metadata.get("class_names", [])) != expected_class_names:
        raise ValueError(
            "Visual-token cache class ordering differs from the teacher artifact"
        )

    image_tokens: dict[str, np.ndarray] = {}
    image_token_scales: dict[str, np.ndarray] | None = (
        {} if metadata.get("dtype") == "int8" else None
    )
    quantization = metadata.get("quantization")
    if image_token_scales is not None:
        if (
            schema_version < 2
            or not isinstance(quantization, Mapping)
            or quantization.get("scheme") != "symmetric_per_visual_token"
        ):
            raise ValueError("Visual-token cache has unsupported int8 quantization")
    split_rows = metadata.get("split_rows", {})
    for split in required_splits:
        path = cache_dir / f"image_tokens_{split}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Visual-token cache is missing split {split!r}: {path}"
            )
        values = np.load(path, mmap_mode="c")
        expected_rows = len(tables["siglip_image_embeddings_by_split"][split])
        if values.ndim != 3 or values.shape[0] != expected_rows:
            raise ValueError(
                f"Visual tokens for {split!r} must have shape "
                f"[{expected_rows}, tokens, hidden]"
            )
        if int(split_rows.get(split, -1)) != expected_rows:
            raise ValueError(
                f"Visual-token metadata row count differs for split {split!r}"
            )
        if image_token_scales is None:
            if not np.issubdtype(values.dtype, np.floating):
                raise ValueError(
                    f"Visual tokens for {split!r} are not floating point"
                )
        else:
            if values.dtype != np.int8:
                raise ValueError(f"Visual tokens for {split!r} are not int8")
            scales_path = cache_dir / f"image_token_scales_{split}.npy"
            if not scales_path.exists():
                raise FileNotFoundError(
                    f"Visual-token cache is missing scales for {split!r}"
                )
            scales = np.load(scales_path, mmap_mode="c")
            if (
                scales.shape != (*values.shape[:2], 1)
                or not np.issubdtype(scales.dtype, np.floating)
            ):
                raise ValueError(
                    f"Visual-token scales for {split!r} have invalid shape or dtype"
                )
            image_token_scales[split] = scales
        image_tokens[split] = values

    label_tokens_path = cache_dir / "label_token_embeddings.npy"
    label_mask_path = cache_dir / "label_token_mask.npy"
    if not label_tokens_path.exists() or not label_mask_path.exists():
        raise FileNotFoundError("Visual-token cache is missing label-token files")
    label_tokens = np.load(label_tokens_path, mmap_mode="c")
    label_mask = np.load(label_mask_path, mmap_mode="c")
    if (
        label_tokens.ndim != 3
        or label_tokens.shape[0] != len(expected_class_names)
        or label_mask.shape != label_tokens.shape[:2]
        or label_mask.dtype != np.bool_
    ):
        raise ValueError("Visual-token cache has invalid label-token tensors")
    hidden_dims = {values.shape[2] for values in image_tokens.values()}
    if hidden_dims != {label_tokens.shape[2]}:
        raise ValueError(
            "Image and label tokens in the visual-token cache have different widths"
        )
    return image_tokens, image_token_scales, label_tokens, label_mask, metadata


def _dequantize_visual_tokens(
    values: np.ndarray,
    scales: np.ndarray | None,
) -> np.ndarray:
    """Return model-ready FP16 visual tokens from either cache format."""
    values = np.asarray(values)
    if scales is None:
        return values
    reconstructed = values.astype(np.float32) * np.asarray(
        scales, dtype=np.float32
    )
    return reconstructed.astype(np.float16)


class RerankerTeacherDataset(Dataset):
    """Expose one query and its complete candidate pool per item.

    Query-level items are intentional: the ranking loss compares candidates
    only within a query and must never construct cross-query preference pairs.
    Feature tables remain shared in memory rather than being repeated for every
    query/exemplar pair.
    """

    def __init__(
        self,
        artifact: str | Path | Mapping[str, Any],
        split: str,
        target: str = "true_probability",
        target_temperature: float = 1.0,
        incremental_lambda: float = 1.0,
        visual_token_cache_path: str | Path | None = None,
    ) -> None:
        if target not in SUPPORTED_TARGETS:
            choices = ", ".join(sorted(SUPPORTED_TARGETS))
            raise ValueError(f"Unsupported target {target!r}; choose one of: {choices}")
        if target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        if not 0 <= incremental_lambda <= 1:
            raise ValueError("incremental_lambda must be in [0, 1]")

        if isinstance(artifact, (str, Path)):
            with Path(artifact).open("rb") as file:
                payload = pickle.load(file)
        elif isinstance(artifact, Mapping):
            payload = artifact
        else:
            raise TypeError("artifact must be a path or mapping")

        if payload.get("method") != "reranker_teacher_data":
            raise ValueError("Not a reranker teacher-data artifact")
        schema_version = int(payload.get("immutable_args", {}).get("schema_version", -1))
        if schema_version != 2:
            raise ValueError(f"Unsupported teacher artifact schema version: {schema_version}")

        tables = payload.get("feature_tables")
        if not isinstance(tables, Mapping):
            raise ValueError("Teacher artifact does not contain bundled feature tables")

        clip_by_split = tables.get("clip_image_embeddings_by_split")
        siglip_by_split = tables.get("siglip_image_embeddings_by_split")
        if not isinstance(clip_by_split, Mapping) or not isinstance(siglip_by_split, Mapping):
            raise ValueError("Teacher artifact has invalid image feature tables")

        all_feature_splits = set(clip_by_split) & set(siglip_by_split)
        retrieval_split = str(payload["immutable_args"]["retrieval_split"])
        required_splits = {split, retrieval_split}
        missing = required_splits - all_feature_splits
        if missing:
            raise ValueError(f"Missing feature tables for split(s): {sorted(missing)}")

        self.clip_features = {
            name: _as_float_tensor(values, f"CLIP features for {name}")
            for name, values in clip_by_split.items()
        }
        self.siglip_features = {
            name: _as_float_tensor(values, f"SigLIP features for {name}")
            for name, values in siglip_by_split.items()
        }
        self.label_features = _as_float_tensor(
            tables.get("siglip_class_text_embeddings"),
            "SigLIP class-text features",
        )
        self.visual_tokens: dict[str, np.ndarray] | None = None
        self.visual_token_scales: dict[str, np.ndarray] | None = None
        self.label_token_embeddings: np.ndarray | None = None
        self.label_token_mask: np.ndarray | None = None
        self.visual_token_cache_metadata: dict[str, Any] | None = None
        if visual_token_cache_path is not None:
            (
                self.visual_tokens,
                self.visual_token_scales,
                self.label_token_embeddings,
                self.label_token_mask,
                self.visual_token_cache_metadata,
            ) = _load_visual_token_cache(
                visual_token_cache_path,
                payload=payload,
                required_splits=required_splits,
            )
        self.records = [
            record for record in payload.get("records", [])
            if str(record.query_split) == split
        ]
        if not self.records:
            raise ValueError(f"Teacher artifact contains no records for split {split!r}")

        self.split = split
        self.retrieval_split = retrieval_split
        self.target = target
        self.target_temperature = float(target_temperature)
        self.incremental_lambda = float(incremental_lambda)
        self.clip_dim = self.clip_features[split].shape[1]
        self.siglip_dim = self.siglip_features[split].shape[1]
        self.visual_token_dim = (
            int(self.label_token_embeddings.shape[2])
            if self.label_token_embeddings is not None
            else 0
        )
        self.visual_token_count = (
            int(self.visual_tokens[split].shape[1])
            if self.visual_tokens is not None
            else 0
        )
        self.label_token_count = (
            int(self.label_token_embeddings.shape[1])
            if self.label_token_embeddings is not None
            else 0
        )
        self._validate_tables()
        for record in self.records:
            self._validate_record(record)

    def _validate_tables(self) -> None:
        clip_dims = {features.shape[1] for features in self.clip_features.values()}
        siglip_dims = {features.shape[1] for features in self.siglip_features.values()}
        if len(clip_dims) != 1:
            raise ValueError("CLIP feature dimensions differ across splits")
        if len(siglip_dims) != 1 or self.label_features.shape[1] != self.siglip_dim:
            raise ValueError("SigLIP image/text feature dimensions do not agree")
        for split in set(self.clip_features) & set(self.siglip_features):
            if len(self.clip_features[split]) != len(self.siglip_features[split]):
                raise ValueError(
                    f"CLIP and SigLIP row counts differ for split {split!r}"
                )

    def _validate_record(self, record: Any) -> None:
        query_idx = int(record.query_idx)
        if not 0 <= query_idx < len(self.clip_features[self.split]):
            raise ValueError(f"Query index {query_idx} is outside split {self.split!r}")

        candidate_indices = np.asarray(record.candidate_indices)
        candidate_classes = np.asarray(record.candidate_class_indices)
        similarities = np.asarray(record.candidate_similarities)
        targets = self._targets(record)
        candidate_count = len(candidate_indices)
        if candidate_count == 0:
            raise ValueError(f"Query {query_idx} has an empty candidate pool")
        if (
            not np.issubdtype(candidate_indices.dtype, np.integer)
            or not np.issubdtype(candidate_classes.dtype, np.integer)
        ):
            raise ValueError(f"Query {query_idx} has non-integer feature indices")
        if (
            candidate_classes.shape != (candidate_count,)
            or similarities.shape != (candidate_count,)
            or targets.shape != (candidate_count,)
        ):
            raise ValueError(f"Query {query_idx} has inconsistent candidate arrays")
        if (
            np.any(candidate_indices < 0)
            or np.any(candidate_indices >= len(self.clip_features[self.retrieval_split]))
            or np.any(candidate_classes < 0)
            or np.any(candidate_classes >= len(self.label_features))
        ):
            raise ValueError(f"Query {query_idx} has out-of-range feature indices")
        if not np.all(np.isfinite(similarities)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"Query {query_idx} has non-finite model inputs or targets")

    def _targets(self, record: Any) -> np.ndarray:
        metrics = record.candidate_metrics
        if self.target in RAW_TARGETS:
            return np.asarray(metrics.get(self.target), dtype=np.float32)
        if self.target == "bounded_margin":
            values = np.asarray(metrics["margin"], dtype=np.float64)
            scaled = np.clip(values / self.target_temperature, -80, 80)
            return (1 / (1 + np.exp(-scaled))).astype(np.float32)
        if self.target == "bounded_incremental_margin":
            values = np.asarray(metrics["incremental_margin"], dtype=np.float64)
            scaled = np.clip(values / self.target_temperature, -80, 80)
            return (1 / (1 + np.exp(-scaled))).astype(np.float32)
        if self.target in {
            "mean_token_probability",
            "normalized_incremental_mean_token_probability",
        }:
            # Idefics2 teacher scores are mean-token log likelihoods. Their
            # exponent is an output-label probability that does not normalize
            # over the K=32 label set, matching the selector's deployment
            # contract more closely than candidate-set softmax probability.
            one_shot_scores = np.asarray(metrics["true_score"], dtype=np.float64)
            zero_shot_score = float(record.zero_shot_metrics["true_score"])
            if np.any(one_shot_scores > 1e-5) or zero_shot_score > 1e-5:
                raise ValueError("Mean-token log probabilities must be non-positive")
            one_shot = np.exp(np.clip(one_shot_scores, -80, 0))
            if self.target == "mean_token_probability":
                return one_shot.astype(np.float32)
            zero_shot = float(np.exp(np.clip(zero_shot_score, -80, 0)))
            denominator = np.maximum(one_shot, zero_shot) ** self.incremental_lambda
            ratio = (one_shot - zero_shot) / np.maximum(denominator, 1e-12)
            return np.clip((ratio + 1) / 2, 0, 1).astype(np.float32)
        if self.target == "normalized_incremental_probability":
            one_shot = np.asarray(metrics["true_probability"], dtype=np.float64)
            zero_shot = float(record.zero_shot_metrics["true_probability"])
            denominator = np.maximum(one_shot, zero_shot) ** self.incremental_lambda
            ratio = (one_shot - zero_shot) / np.maximum(denominator, 1e-12)
            return np.clip((ratio + 1) / 2, 0, 1).astype(np.float32)
        raise AssertionError(f"Unhandled target: {self.target}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        query_idx = int(record.query_idx)
        candidate_indices = torch.as_tensor(
            np.asarray(record.candidate_indices), dtype=torch.long
        )
        candidate_classes = torch.as_tensor(
            np.asarray(record.candidate_class_indices), dtype=torch.long
        )
        candidate_count = len(candidate_indices)
        rank_denominator = max(candidate_count - 1, 1)

        item = {
            "query_split": self.split,
            "query_idx": query_idx,
            "candidate_indices": candidate_indices,
            "candidate_class_indices": candidate_classes,
            "query_clip": self.clip_features[self.split][query_idx],
            "candidate_clip": self.clip_features[self.retrieval_split][candidate_indices],
            "query_siglip": self.siglip_features[self.split][query_idx],
            "candidate_siglip": self.siglip_features[self.retrieval_split][candidate_indices],
            "candidate_label_siglip": self.label_features[candidate_classes],
            "clip_similarities": torch.as_tensor(
                np.asarray(record.candidate_similarities), dtype=torch.float32
            ),
            "retrieval_ranks": torch.arange(candidate_count, dtype=torch.float32)
            / rank_denominator,
            "targets": torch.as_tensor(self._targets(record), dtype=torch.float32),
            # Evaluation-only teacher quantities. None is a permitted model input.
            "teacher_margins": torch.as_tensor(
                np.asarray(record.candidate_metrics["margin"]), dtype=torch.float32
            ),
            "teacher_probabilities": torch.as_tensor(
                np.asarray(record.candidate_metrics["true_probability"]),
                dtype=torch.float32,
            ),
            "teacher_correct": torch.as_tensor(
                np.asarray(record.candidate_metrics["correct"]), dtype=torch.bool
            ),
        }
        if self.visual_tokens is not None:
            # mmap_mode="c" exposes writable, copy-on-write NumPy views, which
            # torch can wrap safely. Collation performs the actual batch copy.
            item.update({
                "query_visual_tokens": torch.from_numpy(
                    _dequantize_visual_tokens(
                        self.visual_tokens[self.split][query_idx],
                        (
                            self.visual_token_scales[self.split][query_idx]
                            if self.visual_token_scales is not None
                            else None
                        ),
                    )
                ),
                "candidate_visual_tokens": torch.from_numpy(
                    _dequantize_visual_tokens(
                        self.visual_tokens[self.retrieval_split][
                            candidate_indices.numpy()
                        ],
                        (
                            self.visual_token_scales[self.retrieval_split][
                                candidate_indices.numpy()
                            ]
                            if self.visual_token_scales is not None
                            else None
                        ),
                    )
                ),
                "candidate_label_tokens": torch.from_numpy(
                    np.asarray(
                        self.label_token_embeddings[candidate_classes.numpy()]
                    )
                ),
                "candidate_label_token_mask": torch.from_numpy(
                    np.asarray(self.label_token_mask[candidate_classes.numpy()])
                ),
            })
        return item


def collate_reranker_queries(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad candidate pools while retaining an explicit validity mask."""
    if not items:
        raise ValueError("Cannot collate an empty reranker batch")

    candidate_counts = torch.as_tensor(
        [len(item["candidate_indices"]) for item in items], dtype=torch.long
    )
    max_candidates = int(candidate_counts.max())
    candidate_mask = (
        torch.arange(max_candidates).unsqueeze(0) < candidate_counts.unsqueeze(1)
    )

    def stack(name: str) -> torch.Tensor:
        return torch.stack([item[name] for item in items])

    def pad(name: str, padding_value: float = 0.0) -> torch.Tensor:
        return pad_sequence(
            [item[name] for item in items],
            batch_first=True,
            padding_value=padding_value,
        )

    batch = {
        "query_split": [str(item["query_split"]) for item in items],
        "query_idx": torch.as_tensor([item["query_idx"] for item in items], dtype=torch.long),
        "candidate_counts": candidate_counts,
        "candidate_mask": candidate_mask,
        "candidate_indices": pad("candidate_indices", -1),
        "candidate_class_indices": pad("candidate_class_indices", -1),
        "query_clip": stack("query_clip"),
        "candidate_clip": pad("candidate_clip"),
        "query_siglip": stack("query_siglip"),
        "candidate_siglip": pad("candidate_siglip"),
        "candidate_label_siglip": pad("candidate_label_siglip"),
        "clip_similarities": pad("clip_similarities"),
        "retrieval_ranks": pad("retrieval_ranks"),
        "targets": pad("targets"),
        "teacher_margins": pad("teacher_margins"),
        "teacher_probabilities": pad("teacher_probabilities"),
        "teacher_correct": pad("teacher_correct"),
    }
    if "query_visual_tokens" in items[0]:
        if not all("query_visual_tokens" in item for item in items):
            raise ValueError("Batch mixes items with and without visual tokens")
        batch.update({
            "query_visual_tokens": stack("query_visual_tokens"),
            "candidate_visual_tokens": pad("candidate_visual_tokens"),
            "candidate_label_tokens": pad("candidate_label_tokens"),
            "candidate_label_token_mask": pad(
                "candidate_label_token_mask", False
            ),
        })
    return batch
