"""Dataset adapter for the bundled reranker teacher artifact."""

from collections.abc import Mapping, Sequence
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


SUPPORTED_TARGETS = frozenset({
    "margin",
    "incremental_margin",
    "true_probability",
    "true_log_probability",
    "incremental_true_probability",
    "incremental_true_log_probability",
})


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 feature table")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


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
        target: str = "margin",
    ) -> None:
        if target not in SUPPORTED_TARGETS:
            choices = ", ".join(sorted(SUPPORTED_TARGETS))
            raise ValueError(f"Unsupported target {target!r}; choose one of: {choices}")

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
        if schema_version != 1:
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
        self.records = [
            record for record in payload.get("records", [])
            if str(record.query_split) == split
        ]
        if not self.records:
            raise ValueError(f"Teacher artifact contains no records for split {split!r}")

        self.split = split
        self.retrieval_split = retrieval_split
        self.target = target
        self.clip_dim = self.clip_features[split].shape[1]
        self.siglip_dim = self.siglip_features[split].shape[1]
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
        targets = np.asarray(record.candidate_metrics.get(self.target))
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

        return {
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
            "targets": torch.as_tensor(
                np.asarray(record.candidate_metrics[self.target]), dtype=torch.float32
            ),
        }


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

    return {
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
    }
