"""Build a resumable, memory-mapped Idefics2 token cache for reranking."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Register the historical pickle aliases before loading the teacher artifact.
from src.data import dataclasses as _artifact_dataclasses  # noqa: E402,F401
from src.data.dataset_registry import get_dataset_spec  # noqa: E402
from src.data.fine_grained_hf_dataset import FineGrainedHFDataset  # noqa: E402
from src.models.idefics2_wrapper import Idefics2Wrapper  # noqa: E402
from src.utils.runtime import file_sha256, git_revision  # noqa: E402


METHOD = "idefics2_reranker_visual_token_cache"
SCHEMA_VERSION = 2
INT8_SCHEME = "symmetric_per_visual_token"
AWQ_UNQUANTIZED_MODULES = frozenset({
    "model.vision_model",
    "model.connector.modality_projection",
    "model.connector.perceiver_resampler",
})
ARCHITECTURE_FIELDS = {
    "vision_config": (
        "hidden_size",
        "image_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "patch_size",
    ),
    "perceiver_config": (
        "resampler_depth",
        "resampler_head_dim",
        "resampler_n_heads",
        "resampler_n_latents",
    ),
    "text_config": ("hidden_size", "vocab_size"),
}


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _ordered_string_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_image_dataset(name: str, split: str, image_split_path: str):
    if name != "cub_200":
        raise ValueError("Visual-token caching currently supports only cub_200")
    spec = get_dataset_spec(name)
    return FineGrainedHFDataset(
        hf_repo_ids=list(spec.hf_repo_ids),
        split=split,
        data_dir=spec.data_dir,
        class_split_seed=42,
        image_split_path=image_split_path,
    )


def _normalize_image_tokens(features: torch.Tensor) -> np.ndarray:
    """Normalize one image's Transformers-version-dependent output layout."""
    features = features.detach()
    while features.ndim > 2 and features.shape[0] == 1:
        features = features.squeeze(0)
    if features.ndim != 2:
        raise ValueError(
            "Expected one image to produce [tokens, hidden] states; got "
            f"{tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("Idefics2 produced non-finite visual tokens")
    return features.to(device="cpu", dtype=torch.float32).numpy()


def _quantize_visual_tokens(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Symmetrically quantize each visual token across its hidden width."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Visual tokens must be a finite [tokens, hidden] array")
    maximum = np.max(np.abs(values), axis=-1, keepdims=True)
    scales = np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float16)
    # Quantize using the stored scale, so the reported error includes FP16
    # scale rounding and exactly matches training-time reconstruction.
    stored_scales = scales.astype(np.float32)
    quantized = np.clip(
        np.rint(values / stored_scales), -127, 127
    ).astype(np.int8)
    reconstructed = quantized.astype(np.float32) * stored_scales
    absolute_error = np.abs(values - reconstructed)
    flat_values = values.reshape(-1).astype(np.float64)
    flat_reconstructed = reconstructed.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(flat_values) * np.linalg.norm(flat_reconstructed)
    cosine = (
        float(np.dot(flat_values, flat_reconstructed) / denominator)
        if denominator > 0
        else 1.0
    )
    return quantized, scales, {
        "mean_abs_error": float(absolute_error.mean()),
        "max_abs_error": float(absolute_error.max()),
        "cosine_similarity": cosine,
    }


def _config_signature(config: Any) -> dict[str, dict[str, Any]]:
    signature: dict[str, dict[str, Any]] = {}
    for section_name, field_names in ARCHITECTURE_FIELDS.items():
        section = getattr(config, section_name, None)
        if section is None:
            raise ValueError(f"Idefics2 config is missing {section_name}")
        signature[section_name] = {
            field: getattr(section, field, None) for field in field_names
        }
    return signature


def _module_has_quantized_weights(module: torch.nn.Module) -> bool:
    for child in module.modules():
        class_name = type(child).__name__.lower()
        if "wqlinear" in class_name or "awqlinear" in class_name:
            return True
        if hasattr(child, "qweight"):
            return True
    return any(not parameter.is_floating_point() for parameter in module.parameters())


def _validate_feature_source(
    wrapper: Idefics2Wrapper,
    *,
    teacher_model: str,
    feature_source_model: str,
    class_names: list[str],
) -> dict[str, Any]:
    """Prove that an alternate checkpoint preserves extracted feature modules."""
    teacher_config = AutoConfig.from_pretrained(teacher_model)
    teacher_signature = _config_signature(teacher_config)
    source_signature = _config_signature(wrapper.model.config)
    if source_signature != teacher_signature:
        raise ValueError(
            "Feature-source and teacher Idefics2 architectures differ: "
            f"teacher={teacher_signature}, source={source_signature}"
        )

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    source_tokenizer = wrapper.processor.tokenizer
    mismatched_labels = [
        label for label in class_names
        if teacher_tokenizer.encode(label, add_special_tokens=False)
        != source_tokenizer.encode(label, add_special_tokens=False)
    ]
    if mismatched_labels:
        raise ValueError(
            "Feature-source tokenizer differs for class labels: "
            f"{mismatched_labels[:5]}"
        )

    result: dict[str, Any] = {
        "architecture_matches_teacher": True,
        "class_label_tokenization_matches_teacher": True,
        "alternate_source": feature_source_model != teacher_model,
    }
    if feature_source_model == teacher_model:
        result["quantization_method"] = None
        return result

    quantization_config = getattr(wrapper.model.config, "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, dict):
        raise ValueError(
            "An alternate feature source must declare a quantization_config"
        )
    quantization_method = str(quantization_config.get("quant_method", "")).lower()
    excluded = set(quantization_config.get("modules_to_not_convert", []))
    missing_exclusions = sorted(AWQ_UNQUANTIZED_MODULES - excluded)
    if quantization_method != "awq" or missing_exclusions:
        raise ValueError(
            "Alternate feature source must be AWQ with all visual/connector "
            f"modules excluded; missing={missing_exclusions}"
        )

    base_model = getattr(wrapper.model, "model", wrapper.model)
    checked_modules = {
        "model.vision_model": getattr(base_model, "vision_model", None),
        "model.connector.modality_projection": getattr(
            getattr(base_model, "connector", None), "modality_projection", None
        ),
        "model.connector.perceiver_resampler": getattr(
            getattr(base_model, "connector", None), "perceiver_resampler", None
        ),
    }
    missing_modules = [name for name, module in checked_modules.items() if module is None]
    if missing_modules:
        raise ValueError(f"Feature source is missing modules: {missing_modules}")
    quantized_modules = [
        name for name, module in checked_modules.items()
        if _module_has_quantized_weights(module)
    ]
    if quantized_modules:
        raise ValueError(
            "Feature-source visual components are unexpectedly quantized: "
            f"{quantized_modules}"
        )
    embedding = wrapper.model.get_input_embeddings()
    if embedding is None or not embedding.weight.is_floating_point():
        raise ValueError("Feature-source input embeddings are unexpectedly quantized")

    result.update({
        "quantization_method": quantization_method,
        "declared_unquantized_modules": sorted(excluded),
        "runtime_unquantized_modules": sorted(checked_modules),
        "input_embeddings_floating_point": True,
    })
    return result


def _label_token_tables(
    wrapper: Idefics2Wrapper,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    tokenizer = wrapper.processor.tokenizer
    token_ids = [
        tokenizer.encode(name, add_special_tokens=False)
        for name in class_names
    ]
    if any(not ids for ids in token_ids):
        empty = [name for name, ids in zip(class_names, token_ids) if not ids]
        raise ValueError(f"Class labels produced no tokens: {empty}")
    max_length = max(map(len, token_ids))
    embedding = wrapper.model.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise TypeError("Idefics2 did not expose its language input embeddings")
    embedding_device = next(embedding.parameters()).device
    hidden = int(embedding.weight.shape[1])
    values = np.zeros((len(class_names), max_length, hidden), dtype=np.float32)
    mask = np.zeros((len(class_names), max_length), dtype=np.bool_)
    with torch.no_grad():
        for class_idx, ids in enumerate(token_ids):
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=embedding_device)
            states = embedding(ids_tensor).detach().to(
                device="cpu", dtype=torch.float32
            ).numpy()
            values[class_idx, :len(ids)] = states
            mask[class_idx, :len(ids)] = True
    return values, mask, token_ids


def _cache_paths(cache_dir: Path, splits: list[str]) -> dict[str, Path]:
    paths = {
        "metadata": cache_dir / "metadata.json",
        "label_tokens": cache_dir / "label_token_embeddings.npy",
        "label_mask": cache_dir / "label_token_mask.npy",
    }
    for split in splits:
        paths[f"tokens_{split}"] = cache_dir / f"image_tokens_{split}.npy"
        paths[f"scales_{split}"] = cache_dir / f"image_token_scales_{split}.npy"
        paths[f"completed_{split}"] = cache_dir / f"completed_{split}.npy"
        paths[f"errors_{split}"] = cache_dir / f"quantization_errors_{split}.npy"
    return paths


def _validate_resume_metadata(
    existing: dict[str, Any], expected: dict[str, Any]
) -> None:
    immutable_keys = (
        "method",
        "schema_version",
        "idefics2_model",
        "feature_source_model",
        "feature_only",
        "load_in_8bit",
        "image_split_sha256",
        "teacher_artifact_sha256",
        "splits",
        "split_rows",
        "example_ids_sha256_by_split",
        "class_names",
        "dtype",
    )
    differences = {
        key: {"cached": existing.get(key), "requested": expected.get(key)}
        for key in immutable_keys
        if existing.get(key) != expected.get(key)
    }
    if differences:
        raise ValueError(
            "Existing visual-token cache is incompatible with this run: "
            f"{json.dumps(differences, indent=2)}"
        )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="build_reranker_visual_token_cache",
)
def main(cfg: DictConfig) -> None:
    splits = [str(split) for split in cfg.dataset.splits]
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("dataset.splits must contain unique split names")
    if "test" in splits:
        raise ValueError("Do not build training features from the held-out test split")
    dtype_name = str(cfg.output.dtype)
    dtype_by_name = {
        "int8": np.int8,
        "float16": np.float16,
        "float32": np.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError("output.dtype must be int8, float16, or float32")
    storage_dtype = dtype_by_name[dtype_name]

    artifact_path = Path(cfg.dataset.teacher_artifact_path)
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    if artifact.get("method") != "reranker_teacher_data":
        raise ValueError("dataset.teacher_artifact_path is not teacher data")
    artifact_splits = set(artifact["immutable_args"]["query_splits"])
    if not set(splits).issubset(artifact_splits):
        raise ValueError("Requested splits are absent from the teacher artifact")
    class_names = list(artifact["feature_tables"]["class_names"])
    artifact_teacher_model = artifact["immutable_args"].get("idefics2_model")
    teacher_model = str(cfg.model.idefics2_model)
    feature_source_model = str(cfg.model.feature_source_model)
    if artifact_teacher_model and teacher_model != artifact_teacher_model:
        raise ValueError("Configured Idefics2 model differs from the teacher artifact")
    if bool(cfg.model.load_in_8bit) and feature_source_model != teacher_model:
        raise ValueError(
            "Do not apply bitsandbytes 8-bit loading to an alternate quantized "
            "feature source"
        )
    if not bool(cfg.model.feature_only) and feature_source_model != teacher_model:
        print(
            "Warning: loading the complete alternate checkpoint; set "
            "model.feature_only=true to avoid quantized language-model shards",
            flush=True,
        )

    datasets = {
        split: _load_image_dataset(
            str(cfg.dataset.name), split, str(cfg.dataset.image_split_path)
        )
        for split in splits
    }
    for split, dataset in datasets.items():
        if list(dataset.class_names) != class_names:
            raise ValueError(f"Class ordering differs for split {split!r}")
        expected_rows = len(
            artifact["feature_tables"]["siglip_image_embeddings_by_split"][split]
        )
        if len(dataset) != expected_rows:
            raise ValueError(f"Image rows differ for split {split!r}")

    cache_dir = Path(cfg.output.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir, splits)
    expected_metadata = {
        "method": METHOD,
        "schema_version": SCHEMA_VERSION,
        "idefics2_model": teacher_model,
        "feature_source_model": feature_source_model,
        "feature_only": bool(cfg.model.feature_only),
        "load_in_8bit": bool(cfg.model.load_in_8bit),
        "image_split_sha256": file_sha256(cfg.dataset.image_split_path),
        "teacher_artifact_sha256": file_sha256(artifact_path),
        "splits": splits,
        "split_rows": {split: len(dataset) for split, dataset in datasets.items()},
        "example_ids_sha256_by_split": {
            split: _ordered_string_hash(
                [example.image_path for example in dataset.examples]
            )
            for split, dataset in datasets.items()
        },
        "class_names": class_names,
        "dtype": dtype_name,
    }

    existing_metadata = None
    if paths["metadata"].exists():
        with paths["metadata"].open() as file:
            existing_metadata = json.load(file)
        _validate_resume_metadata(existing_metadata, expected_metadata)
        if bool(existing_metadata.get("complete", False)):
            print(f"✓ Complete visual-token cache already exists: {cache_dir}")
            return

    if not torch.cuda.is_available():
        raise RuntimeError("Idefics2 visual-token extraction requires a CUDA GPU")
    wrapper = Idefics2Wrapper(
        model_name=feature_source_model,
        device="cuda:0",
        load_in_8bit=bool(cfg.model.load_in_8bit),
        feature_only=bool(cfg.model.feature_only),
    )
    feature_equivalence = _validate_feature_source(
        wrapper,
        teacher_model=teacher_model,
        feature_source_model=feature_source_model,
        class_names=class_names,
    )
    loaded_commit = getattr(wrapper.model.config, "_commit_hash", None)
    if (
        existing_metadata is not None
        and existing_metadata.get("feature_source_commit_hash")
        and loaded_commit != existing_metadata["feature_source_commit_hash"]
    ):
        raise ValueError(
            "Resolved feature-source revision differs from the partial cache"
        )

    if existing_metadata is None:
        first_split = splits[0]
        _, first_image = datasets[first_split][0]
        first_tokens = _normalize_image_tokens(
            wrapper.encode_full_label_scoring_images([first_image])
        )
        token_count, hidden_dim = first_tokens.shape
        label_values, label_mask, label_token_ids = _label_token_tables(
            wrapper, class_names
        )
        if label_values.shape[2] != hidden_dim:
            raise ValueError(
                "Idefics2 image and input-label states have different widths"
            )
        label_storage_dtype = np.float16 if dtype_name == "int8" else storage_dtype
        image_count = sum(len(dataset) for dataset in datasets.values())
        estimated_bytes = (
            image_count * token_count * hidden_dim * np.dtype(storage_dtype).itemsize
            + label_values.size * np.dtype(label_storage_dtype).itemsize
        )
        if dtype_name == "int8":
            estimated_bytes += image_count * token_count * np.dtype(np.float16).itemsize
        print(
            "Allocating visual-token sidecar: "
            f"{estimated_bytes / 1024 ** 3:.2f} GiB "
            f"({image_count} images, {dtype_name})",
            flush=True,
        )
        np.save(paths["label_tokens"], label_values.astype(label_storage_dtype))
        np.save(paths["label_mask"], label_mask)
        for split, dataset in datasets.items():
            np.lib.format.open_memmap(
                paths[f"tokens_{split}"],
                mode="w+",
                dtype=storage_dtype,
                shape=(len(dataset), token_count, hidden_dim),
            ).flush()
            if dtype_name == "int8":
                np.lib.format.open_memmap(
                    paths[f"scales_{split}"],
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(dataset), token_count, 1),
                ).flush()
                # Columns: mean absolute error, maximum absolute error, cosine.
                np.lib.format.open_memmap(
                    paths[f"errors_{split}"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(dataset), 3),
                ).flush()
            np.lib.format.open_memmap(
                paths[f"completed_{split}"],
                mode="w+",
                dtype=np.bool_,
                shape=(len(dataset),),
            ).flush()
        existing_metadata = {
            **expected_metadata,
            "visual_token_count": token_count,
            "hidden_dim": hidden_dim,
            "label_token_count": int(label_values.shape[1]),
            "label_token_ids": label_token_ids,
            "feature_source_commit_hash": loaded_commit,
            "feature_source_files": list(
                getattr(wrapper.model, "feature_source_files", [])
            ),
            "feature_equivalence_validation": feature_equivalence,
            "quantization": (
                {
                    "scheme": INT8_SCHEME,
                    "axis": "hidden_dimension_per_visual_token",
                    "qmin": -127,
                    "qmax": 127,
                    "scale_dtype": "float16",
                    "metrics": {},
                }
                if dtype_name == "int8"
                else None
            ),
            "git_revision": git_revision(),
            "completed_counts": {split: 0 for split in splits},
            "complete": False,
        }
        _atomic_json_dump(existing_metadata, paths["metadata"])
        prefetched = {(first_split, 0): first_tokens}
    else:
        prefetched = {}

    token_count = int(existing_metadata["visual_token_count"])
    hidden_dim = int(existing_metadata["hidden_dim"])
    token_arrays = {
        split: np.lib.format.open_memmap(
            paths[f"tokens_{split}"], mode="r+"
        )
        for split in splits
    }
    scale_arrays = (
        {
            split: np.lib.format.open_memmap(
                paths[f"scales_{split}"], mode="r+"
            )
            for split in splits
        }
        if dtype_name == "int8"
        else {}
    )
    error_arrays = (
        {
            split: np.lib.format.open_memmap(
                paths[f"errors_{split}"], mode="r+"
            )
            for split in splits
        }
        if dtype_name == "int8"
        else {}
    )
    completed_arrays = {
        split: np.lib.format.open_memmap(
            paths[f"completed_{split}"], mode="r+"
        )
        for split in splits
    }

    checkpoint_every = int(cfg.output.checkpoint_every_images)
    if checkpoint_every <= 0:
        raise ValueError("output.checkpoint_every_images must be positive")
    max_images = cfg.limits.get("max_images_per_split", None)
    max_images = None if max_images is None else int(max_images)
    if max_images is not None and max_images <= 0:
        raise ValueError("limits.max_images_per_split must be positive or null")

    def checkpoint() -> None:
        for values in token_arrays.values():
            values.flush()
        for values in scale_arrays.values():
            values.flush()
        for values in error_arrays.values():
            values.flush()
        for values in completed_arrays.values():
            values.flush()
        counts = {
            split: int(values.sum())
            for split, values in completed_arrays.items()
        }
        existing_metadata["completed_counts"] = counts
        existing_metadata["complete"] = all(
            counts[split] == len(datasets[split]) for split in splits
        )
        if dtype_name == "int8":
            quantization_metrics = {}
            for split in splits:
                completed = np.asarray(completed_arrays[split], dtype=np.bool_)
                metrics = np.asarray(error_arrays[split][completed])
                if len(metrics):
                    quantization_metrics[split] = {
                        "images": int(len(metrics)),
                        "mean_abs_error": float(metrics[:, 0].mean()),
                        "max_abs_error": float(metrics[:, 1].max()),
                        "mean_cosine_similarity": float(metrics[:, 2].mean()),
                        "min_cosine_similarity": float(metrics[:, 2].min()),
                    }
            existing_metadata["quantization"]["metrics"] = quantization_metrics
        _atomic_json_dump(existing_metadata, paths["metadata"])
        print(
            "Visual-token progress: "
            + ", ".join(
                f"{split}={counts[split]}/{len(datasets[split])}"
                for split in splits
            ),
            flush=True,
        )

    since_checkpoint = 0
    try:
        for split in splits:
            pending = np.flatnonzero(~completed_arrays[split])
            if max_images is not None:
                pending = pending[:max_images]
            progress = tqdm(pending, desc=f"Idefics2 visual tokens ({split})")
            for index_value in progress:
                index = int(index_value)
                features = prefetched.pop((split, index), None)
                if features is None:
                    _, image = datasets[split][index]
                    features = _normalize_image_tokens(
                        wrapper.encode_full_label_scoring_images([image])
                    )
                if features.shape != (token_count, hidden_dim):
                    raise ValueError(
                        f"Image {split}:{index} produced shape {features.shape}; "
                        f"expected {(token_count, hidden_dim)}"
                    )
                if dtype_name == "int8":
                    quantized, scales, errors = _quantize_visual_tokens(features)
                    token_arrays[split][index] = quantized
                    scale_arrays[split][index] = scales
                    error_arrays[split][index] = np.asarray([
                        errors["mean_abs_error"],
                        errors["max_abs_error"],
                        errors["cosine_similarity"],
                    ], dtype=np.float32)
                else:
                    token_arrays[split][index] = features.astype(
                        storage_dtype, copy=False
                    )
                completed_arrays[split][index] = True
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_every:
                    checkpoint()
                    since_checkpoint = 0
    finally:
        checkpoint()

    if existing_metadata["complete"]:
        if dtype_name == "int8":
            for split, metrics in existing_metadata["quantization"]["metrics"].items():
                print(
                    f"INT8 reconstruction ({split}): "
                    f"mean_abs_error={metrics['mean_abs_error']:.6g}, "
                    f"max_abs_error={metrics['max_abs_error']:.6g}, "
                    f"mean_cosine={metrics['mean_cosine_similarity']:.8f}, "
                    f"min_cosine={metrics['min_cosine_similarity']:.8f}",
                    flush=True,
                )
        print(f"✓ Visual-token cache complete: {cache_dir}", flush=True)
    else:
        print(
            "Visual-token cache remains incomplete because a limit was applied; "
            "rerun without the limit to resume.",
            flush=True,
        )


if __name__ == "__main__":
    main()
