"""Cache pair-conditioned hidden states from a frozen Idefics2 language model."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.idefics2_probe_dataset import (  # noqa: E402
    PROBE_CACHE_METHOD,
    PROBE_CACHE_SCHEMA_VERSION,
)
from src.data.reranker_dataset import RerankerTeacherDataset  # noqa: E402
from src.models.idefics2_probe import (  # noqa: E402
    PROBE_PROMPT_TEMPLATE,
    encode_frozen_idefics2_pairs,
    load_frozen_idefics2_probe_backbone,
    quantize_probe_representations,
)
from src.utils.runtime import file_sha256, git_revision  # noqa: E402


def _atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _open_or_create(path: Path, *, dtype, shape) -> np.ndarray:
    if path.exists():
        values = np.load(path, mmap_mode="r+")
        if values.dtype != np.dtype(dtype) or values.shape != tuple(shape):
            raise ValueError(f"Existing cache array has incompatible shape/dtype: {path}")
        return values
    values = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    # In particular, a fresh completion bitmap must never inherit non-zero
    # bytes from a reused filesystem extent.
    values[...] = 0
    values.flush()
    return values


def _write_or_validate(path: Path, values: np.ndarray) -> None:
    if path.exists():
        existing = np.load(path)
        if not np.array_equal(existing, values):
            raise ValueError(f"Existing probe cache mapping differs: {path}")
    else:
        np.save(path, values)


def _validate_metadata(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(
                f"Probe cache metadata mismatch for {key}: "
                f"{existing.get(key)!r} != {value!r}"
            )


def _validate_probe_model_source(
    *,
    teacher_model: str,
    probe_model: str,
    visual_cache_metadata: dict[str, Any],
    load_in_8bit: bool,
) -> str:
    """Validate exact-teacher or explicitly provenance-matched AWQ probing."""
    if probe_model == teacher_model:
        return "exact_teacher_checkpoint"
    feature_source = str(visual_cache_metadata.get("feature_source_model", ""))
    validation = visual_cache_metadata.get("feature_equivalence_validation", {})
    if probe_model != feature_source:
        raise ValueError(
            "An alternate probe model must match the visual-token cache's "
            f"feature source ({feature_source!r})"
        )
    if (
        not isinstance(validation, dict)
        or not bool(validation.get("architecture_matches_teacher", False))
        or str(validation.get("quantization_method", "")).lower() != "awq"
    ):
        raise ValueError(
            "Alternate probe model lacks the visual cache's validated AWQ "
            "architecture-equivalence provenance"
        )
    if load_in_8bit:
        raise ValueError(
            "model.load_in_8bit must be false for a pre-quantized AWQ probe"
        )
    return "validated_awq_approximation"


def _validate_awq_runtime() -> None:
    """Fail early with actionable diagnostics for partial AWQ installations."""
    # Transformers 5 uses GPTQModel's shared quantized-linear implementation.
    # Import it here so an incomplete environment fails before model download.
    if importlib.util.find_spec("gptqmodel") is not None:
        try:
            importlib.import_module("gptqmodel.quantization")
        except Exception as error:
            missing = (
                f"; missing import {error.name!r}"
                if isinstance(error, ModuleNotFoundError) and error.name
                else ""
            )
            raise ImportError(
                "GPTQModel's AWQ backend failed its import check"
                f"{missing}. This can be caused by a missing or "
                "binary-incompatible torchvision build, or by a torch/torchao "
                "version mismatch. Install torchvision 0.23 from the same "
                "PyTorch wheel index as torch 2.8, then reinstall the pinned "
                "project dependency set with: pip install "
                "--upgrade-strategy only-if-needed -e '.[awq]'"
            ) from error
        return
    # Retain compatibility with older Transformers installations that dispatch
    # AWQ through AutoAWQ instead.
    if importlib.util.find_spec("awq") is not None:
        return
    raise ImportError(
        "Loading the storage-efficient AWQ probe requires a complete AWQ "
        "backend. For this project install it with: pip install -e '.[awq]'"
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="build_frozen_idefics2_probe_cache",
)
def main(cfg: DictConfig) -> None:
    artifact_path = Path(cfg.dataset.teacher_artifact_path)
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    teacher_model = str(artifact["immutable_args"].get("idefics2_model", ""))
    configured_teacher = str(cfg.model.idefics2_model)
    if teacher_model and teacher_model != configured_teacher:
        raise ValueError("Configured teacher model differs from the artifact")
    probe_model = str(cfg.model.probe_model)

    splits = [str(split) for split in cfg.dataset.splits]
    datasets = {
        split: RerankerTeacherDataset(
            artifact=artifact,
            split=split,
            target="true_probability",
            visual_token_cache_path=cfg.dataset.visual_token_cache_path,
        )
        for split in splits
    }
    candidate_counts = {
        split: {len(record.candidate_indices) for record in dataset.records}
        for split, dataset in datasets.items()
    }
    if any(len(counts) != 1 for counts in candidate_counts.values()):
        raise ValueError("Probe caching requires a fixed candidate count per split")
    hidden_dims = {dataset.visual_token_dim for dataset in datasets.values()}
    if len(hidden_dims) != 1:
        raise ValueError("Visual-token hidden dimensions differ across splits")
    hidden_dim = hidden_dims.pop()
    if hidden_dim <= 0:
        raise ValueError("Probe caching requires Idefics2 visual-token states")
    visual_token_counts = {
        dataset.visual_token_count for dataset in datasets.values()
    }
    if len(visual_token_counts) != 1:
        raise ValueError("Visual-token sequence lengths differ across splits")
    visual_token_count = visual_token_counts.pop()
    if visual_token_count <= 0:
        raise ValueError("Probe caching requires non-empty visual-token sequences")
    visual_cache_metadata = next(iter(datasets.values())).visual_token_cache_metadata
    probe_source = _validate_probe_model_source(
        teacher_model=teacher_model,
        probe_model=probe_model,
        visual_cache_metadata=visual_cache_metadata,
        load_in_8bit=bool(cfg.model.load_in_8bit),
    )
    if probe_source == "validated_awq_approximation":
        _validate_awq_runtime()

    dtype_name = str(cfg.output.dtype)
    if dtype_name not in {"int8", "float16"}:
        raise ValueError("output.dtype must be int8 or float16")
    cache_dir = Path(cfg.output.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    expected_metadata = {
        "method": PROBE_CACHE_METHOD,
        "schema_version": PROBE_CACHE_SCHEMA_VERSION,
        "idefics2_model": teacher_model,
        "probe_model": probe_model,
        "probe_model_source": probe_source,
        "load_in_8bit": bool(cfg.model.load_in_8bit),
        "awq_backend": (
            str(cfg.model.awq_backend)
            if probe_source == "validated_awq_approximation"
            else None
        ),
        "teacher_artifact_sha256": file_sha256(artifact_path),
        "visual_token_cache_path": str(Path(cfg.dataset.visual_token_cache_path).resolve()),
        "visual_token_cache_metadata": visual_cache_metadata,
        "splits": splits,
        "split_rows": {split: len(dataset) for split, dataset in datasets.items()},
        "candidate_counts": {
            split: next(iter(counts)) for split, counts in candidate_counts.items()
        },
        "hidden_dim": hidden_dim,
        "visual_token_count": visual_token_count,
        "dtype": dtype_name,
        "quantization": (
            {"scheme": "symmetric_per_pair"} if dtype_name == "int8" else None
        ),
        "prompt_template": PROBE_PROMPT_TEMPLATE,
    }
    existing_metadata = None
    if metadata_path.exists():
        with metadata_path.open() as file:
            existing_metadata = json.load(file)
        _validate_metadata(existing_metadata, expected_metadata)
        if bool(existing_metadata.get("complete", False)):
            print(f"✓ Complete frozen-Idefics2 probe cache exists: {cache_dir}")
            return

    tables: dict[str, dict[str, np.ndarray]] = {}
    for split, dataset in datasets.items():
        rows = len(dataset)
        candidates = next(iter(candidate_counts[split]))
        tables[split] = {
            "representations": _open_or_create(
                cache_dir / f"pair_representations_{split}.npy",
                dtype=np.int8 if dtype_name == "int8" else np.float16,
                shape=(rows, candidates, hidden_dim),
            ),
            "complete": _open_or_create(
                cache_dir / f"pair_complete_{split}.npy",
                dtype=np.bool_,
                shape=(rows, candidates),
            ),
        }
        if dtype_name == "int8":
            tables[split]["scales"] = _open_or_create(
                cache_dir / f"pair_representation_scales_{split}.npy",
                dtype=np.float16,
                shape=(rows, candidates, 1),
            )
        _write_or_validate(
            cache_dir / f"query_indices_{split}.npy",
            np.asarray([record.query_idx for record in dataset.records], dtype=np.int32),
        )
        _write_or_validate(
            cache_dir / f"candidate_indices_{split}.npy",
            np.stack([
                np.asarray(record.candidate_indices, dtype=np.int32)
                for record in dataset.records
            ]),
        )

    pending = sum(
        int((~table["complete"]).sum()) for table in tables.values()
    )
    if pending == 0:
        final_metadata = {
            **expected_metadata,
            "complete": True,
            "git_revision": git_revision(),
        }
        _atomic_json_dump(final_metadata, metadata_path)
        print(f"✓ Finalized complete frozen-Idefics2 probe cache: {cache_dir}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen Idefics2 pair encoding requires a CUDA GPU")

    print(
        f"Encoding {pending:,} pending exemplar/query pairs with frozen {probe_model}",
        flush=True,
    )
    try:
        backbone = load_frozen_idefics2_probe_backbone(
            probe_model,
            device=str(cfg.model.device),
            load_in_8bit=bool(cfg.model.load_in_8bit),
            image_seq_len=visual_token_count,
            awq_backend=(
                str(cfg.model.awq_backend)
                if probe_source == "validated_awq_approximation"
                else None
            ),
        )
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        if "Marlin" in str(exc) or "marlin" in str(exc):
            raise RuntimeError(
                "GPTQModel attempted to load its Marlin CUDA backend. The "
                "probe cache does not require Marlin; use "
                "model.awq_backend=gemm_triton (default), or the slower "
                "model.awq_backend=torch_awq fallback. If this message appears "
                "with the default, make sure the updated project checkout is "
                "the one being executed."
            ) from exc
        raise
    model_hidden = int(backbone.model.config.text_config.hidden_size)
    if model_hidden != hidden_dim:
        raise ValueError(
            f"Language hidden size {model_hidden} differs from visual states {hidden_dim}"
        )

    pair_batch_size = int(cfg.model.pair_batch_size)
    if pair_batch_size <= 0:
        raise ValueError("model.pair_batch_size must be positive")
    checkpoint_every = int(cfg.output.checkpoint_every_queries)
    if checkpoint_every < 0:
        raise ValueError("output.checkpoint_every_queries must be non-negative")
    class_names = list(artifact["feature_tables"]["class_names"])
    limits = OmegaConf.to_container(cfg.limits.max_queries_per_split, resolve=True)
    for split, dataset in datasets.items():
        table = tables[split]
        max_queries = limits.get(split) if isinstance(limits, dict) else None
        processed_queries = 0
        progress = tqdm(range(len(dataset)), desc=f"Frozen probe {split}")
        for row in progress:
            if bool(table["complete"][row].all()):
                continue
            if max_queries is not None and processed_queries >= int(max_queries):
                break
            item = dataset[row]
            pending_positions = np.flatnonzero(~np.asarray(table["complete"][row]))
            query_tokens = item["query_visual_tokens"]
            for start in range(0, len(pending_positions), pair_batch_size):
                positions = pending_positions[start:start + pair_batch_size]
                position_tensor = torch.as_tensor(positions, dtype=torch.long)
                exemplar_tokens = item["candidate_visual_tokens"][position_tensor]
                batch_query_tokens = query_tokens.unsqueeze(0).expand(
                    len(positions), -1, -1
                )
                class_indices = item["candidate_class_indices"][position_tensor]
                labels = [class_names[int(index)] for index in class_indices]
                encoded = encode_frozen_idefics2_pairs(
                    backbone.model,
                    backbone.processor,
                    exemplar_tokens,
                    batch_query_tokens,
                    labels,
                    device=backbone.device,
                    use_amp=bool(cfg.model.amp),
                ).numpy()
                if dtype_name == "int8":
                    quantized, scales = quantize_probe_representations(encoded)
                    table["representations"][row, positions] = quantized
                    table["scales"][row, positions] = scales
                else:
                    table["representations"][row, positions] = encoded.astype(np.float16)
                table["complete"][row, positions] = True
            processed_queries += 1
            if checkpoint_every > 0 and processed_queries % checkpoint_every == 0:
                for values in table.values():
                    values.flush()
                partial = {
                    **expected_metadata,
                    "complete": False,
                    "git_revision": git_revision(),
                    "completed_pairs": {
                        name: int(values["complete"].sum())
                        for name, values in tables.items()
                    },
                }
                _atomic_json_dump(partial, metadata_path)

    for table in tables.values():
        for values in table.values():
            values.flush()
    complete = all(bool(table["complete"].all()) for table in tables.values())
    final_metadata = {
        **expected_metadata,
        "complete": complete,
        "git_revision": git_revision(),
        "completed_pairs": {
            split: int(table["complete"].sum()) for split, table in tables.items()
        },
    }
    _atomic_json_dump(final_metadata, metadata_path)
    if complete:
        print(f"✓ Frozen-Idefics2 probe cache complete: {cache_dir}")
    else:
        print(f"Partial probe cache saved for resume: {cache_dir}")


if __name__ == "__main__":
    main()
