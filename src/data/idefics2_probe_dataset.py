"""Dataset adapter for cached frozen-Idefics2 pair representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.reranker_dataset import RerankerTeacherDataset
from src.models.idefics2_probe import dequantize_probe_representations


PROBE_CACHE_METHOD = "frozen_idefics2_pair_probe_cache"
PROBE_CACHE_SCHEMA_VERSION = 2


class FrozenIdefics2ProbeDataset(Dataset):
    """Expose query groups of cached pair states and training-only utilities."""

    def __init__(
        self,
        artifact: str | Path | Mapping[str, Any],
        cache_path: str | Path,
        split: str,
        target: str,
        *,
        target_temperature: float = 1.0,
        incremental_lambda: float = 1.0,
    ) -> None:
        self.teacher = RerankerTeacherDataset(
            artifact=artifact,
            split=split,
            target=target,
            target_temperature=target_temperature,
            incremental_lambda=incremental_lambda,
        )
        self.split = split
        cache_dir = Path(cache_path)
        with (cache_dir / "metadata.json").open() as file:
            self.metadata = json.load(file)
        if self.metadata.get("method") != PROBE_CACHE_METHOD:
            raise ValueError("Not a frozen-Idefics2 probe cache")
        if int(self.metadata.get("schema_version", -1)) != PROBE_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported frozen-Idefics2 probe cache schema")
        if not bool(self.metadata.get("complete", False)):
            raise ValueError("Frozen-Idefics2 probe cache is incomplete")
        if split not in self.metadata.get("splits", []):
            raise ValueError(f"Probe cache does not contain split {split!r}")

        self.representations = np.load(
            cache_dir / f"pair_representations_{split}.npy", mmap_mode="c"
        )
        self.scales = None
        if self.metadata.get("dtype") == "int8":
            if self.representations.dtype != np.int8:
                raise ValueError("INT8 probe metadata has non-INT8 representations")
            self.scales = np.load(
                cache_dir / f"pair_representation_scales_{split}.npy",
                mmap_mode="c",
            )
            if self.scales.shape != (*self.representations.shape[:2], 1):
                raise ValueError("Probe representation scales have invalid shape")
        elif not np.issubdtype(self.representations.dtype, np.floating):
            raise ValueError("Probe representations must be floating point or INT8")

        query_indices = np.load(cache_dir / f"query_indices_{split}.npy")
        candidate_indices = np.load(cache_dir / f"candidate_indices_{split}.npy")
        expected_queries = np.asarray(
            [int(record.query_idx) for record in self.teacher.records]
        )
        expected_candidates = np.stack([
            np.asarray(record.candidate_indices, dtype=np.int64)
            for record in self.teacher.records
        ])
        if not np.array_equal(query_indices, expected_queries):
            raise ValueError("Probe cache query ordering differs from teacher artifact")
        if not np.array_equal(candidate_indices, expected_candidates):
            raise ValueError("Probe cache candidate ordering differs from teacher artifact")
        if self.representations.shape[:2] != expected_candidates.shape:
            raise ValueError("Probe cache shape differs from teacher candidate pools")
        self.input_dim = int(self.representations.shape[-1])

    def __len__(self) -> int:
        return len(self.teacher)

    def __getitem__(self, index: int) -> dict[str, Any]:
        teacher_item = self.teacher[index]
        values = dequantize_probe_representations(
            self.representations[index],
            self.scales[index] if self.scales is not None else None,
        )
        candidate_count = len(teacher_item["candidate_indices"])
        return {
            "query_split": self.split,
            "query_idx": teacher_item["query_idx"],
            "candidate_indices": teacher_item["candidate_indices"],
            "pair_representations": torch.from_numpy(np.asarray(values)),
            "targets": teacher_item["targets"],
            "teacher_margins": teacher_item["teacher_margins"],
            "teacher_correct": teacher_item["teacher_correct"],
            "candidate_mask": torch.ones(candidate_count, dtype=torch.bool),
        }


def collate_frozen_idefics2_probe_queries(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty probe batch")

    def pad(name: str, padding_value: float = 0.0) -> torch.Tensor:
        return pad_sequence(
            [item[name] for item in items],
            batch_first=True,
            padding_value=padding_value,
        )

    return {
        "query_split": [str(item["query_split"]) for item in items],
        "query_idx": torch.as_tensor([item["query_idx"] for item in items]),
        "candidate_indices": pad("candidate_indices", -1),
        "pair_representations": pad("pair_representations"),
        "targets": pad("targets"),
        "teacher_margins": pad("teacher_margins"),
        "teacher_correct": pad("teacher_correct"),
        "candidate_mask": pad("candidate_mask", False),
    }
