"""Idefics2 scoring adapter for the retained CUB-200 pipeline."""

from __future__ import annotations

import copy
import json
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Idefics2ImageProcessor,
    Idefics2Processor,
)
from transformers.models.idefics2.modeling_idefics2 import (
    Idefics2Connector,
    Idefics2VisionTransformer,
)

FULL_SEQUENCE_SCORING = "full_sequence_batch"
PREFIX_KV_CACHE_SCORING = "prefix_kv_cache"
SUPPORTED_SCORING_MODES = frozenset({
    FULL_SEQUENCE_SCORING,
    PREFIX_KV_CACHE_SCORING,
})
FEATURE_WEIGHT_PREFIXES = (
    "model.vision_model.",
    "model.connector.",
)
FEATURE_EMBEDDING_KEY = "model.text_model.embed_tokens.weight"


def _build_idefics2_processor(model_name: str) -> Idefics2Processor:
    """Build the checkpoint processor without AutoImageProcessor inference.

    These values are the published ``HuggingFaceM4/idefics2-8b``
    ``preprocessor_config.json`` and ``processor_config.json`` settings. The
    retained evaluation pipeline disables image splitting immediately after
    construction, as it did when teacher targets were generated.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = Idefics2ImageProcessor(
        do_convert_rgb=True,
        do_image_splitting=True,
        do_normalize=True,
        do_pad=True,
        do_rescale=True,
        do_resize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=2,
        rescale_factor=1 / 255,
        size={"longest_edge": 980, "shortest_edge": 378},
    )
    return Idefics2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_seq_len=64,
    )


class _Idefics2FeatureOnlyModel(torch.nn.Module):
    """Idefics2 image path and input embeddings without the language model."""

    def __init__(self, config, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.vision_model = Idefics2VisionTransformer._from_config(
            config.vision_config
        ).to(dtype=dtype)
        self.connector = Idefics2Connector(config).to(dtype=dtype)
        self.embed_tokens = torch.nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            padding_idx=config.text_config.pad_token_id,
            dtype=dtype,
        )
        self.feature_source_files: list[str] = []

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vision_model.parameters()).dtype

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed_tokens

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_images = pixel_values.shape[:2]
        pixel_values = pixel_values.to(dtype=self.dtype)
        pixel_values = pixel_values.view(
            batch_size * num_images, *pixel_values.shape[2:]
        )
        values_per_image = pixel_values.shape[1:].numel()
        real_images = (
            (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != values_per_image
        )
        pixel_values = pixel_values[real_images].contiguous()
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                (pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            pixel_attention_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:]
            )[real_images].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches = pixel_attention_mask.unfold(1, patch_size, patch_size)
        patches = patches.unfold(2, patch_size, patch_size)
        patch_attention_mask = (
            patches.sum(dim=(-1, -2)) == patch_size * patch_size
        ).bool()
        image_outputs = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        return self.connector(
            image_outputs.last_hidden_state,
            attention_mask=patch_attention_mask.view(pixel_values.size(0), -1),
        )


def _load_module_prefix(
    handle: safe_open,
    module: torch.nn.Module,
    prefix: str,
) -> None:
    state = {
        key.removeprefix(prefix): handle.get_tensor(key)
        for key in handle.keys()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f"Feature checkpoint has no tensors under {prefix!r}")
    module.load_state_dict(state, strict=True)


def _load_idefics2_feature_only_model(
    model_name: str,
    *,
    device: str,
) -> _Idefics2FeatureOnlyModel:
    """Download and load only shards containing feature-extraction tensors."""
    config = AutoConfig.from_pretrained(model_name)
    index_path = hf_hub_download(
        repo_id=model_name,
        filename="model.safetensors.index.json",
    )
    with open(index_path) as file:
        weight_map = json.load(file)["weight_map"]
    required_keys = [
        key for key in weight_map
        if key.startswith(FEATURE_WEIGHT_PREFIXES) or key == FEATURE_EMBEDDING_KEY
    ]
    if FEATURE_EMBEDDING_KEY not in required_keys:
        raise ValueError("Feature checkpoint is missing the input embedding table")
    required_shards = sorted({weight_map[key] for key in required_keys})
    if not required_shards:
        raise ValueError("Feature checkpoint index has no Idefics2 visual tensors")

    model = _Idefics2FeatureOnlyModel(config, dtype=torch.float16)
    loaded_prefixes: set[str] = set()
    embedding_loaded = False
    for shard_name in required_shards:
        shard_path = hf_hub_download(repo_id=model_name, filename=shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if any(key.startswith(FEATURE_WEIGHT_PREFIXES[0]) for key in keys):
                _load_module_prefix(
                    handle, model.vision_model, FEATURE_WEIGHT_PREFIXES[0]
                )
                loaded_prefixes.add(FEATURE_WEIGHT_PREFIXES[0])
            if any(key.startswith(FEATURE_WEIGHT_PREFIXES[1]) for key in keys):
                _load_module_prefix(handle, model.connector, FEATURE_WEIGHT_PREFIXES[1])
                loaded_prefixes.add(FEATURE_WEIGHT_PREFIXES[1])
            if FEATURE_EMBEDDING_KEY in keys:
                model.embed_tokens.load_state_dict(
                    {"weight": handle.get_tensor(FEATURE_EMBEDDING_KEY)},
                    strict=True,
                )
                embedding_loaded = True
        model.feature_source_files.append(shard_name)
    if loaded_prefixes != set(FEATURE_WEIGHT_PREFIXES) or not embedding_loaded:
        raise ValueError("Feature checkpoint did not supply every required component")
    model.to(device)
    model.eval()
    return model


class Idefics2Wrapper:
    """Expose only the discriminative scoring paths used by this project."""

    def __init__(
        self,
        model_name: str = "HuggingFaceM4/idefics2-8b",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        scoring_mode: str = FULL_SEQUENCE_SCORING,
        feature_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.feature_only = feature_only
        if feature_only and load_in_8bit:
            raise ValueError("feature_only and load_in_8bit cannot be combined")
        if scoring_mode not in SUPPORTED_SCORING_MODES:
            raise ValueError(
                f"Unsupported scoring mode {scoring_mode!r}; expected one of "
                f"{sorted(SUPPORTED_SCORING_MODES)}"
            )
        self.scoring_mode = scoring_mode
        self.use_cache = False
        self._letter_token_id_cache: dict[str, int] = {}
        self._label_token_cache: dict[str, list[int]] = {}
        self._prompt_token_cache: dict[str, object] = {}

        print(f"Loading Idefics2 model: {model_name}")
        print(f"Device: {self.device}")
        if load_in_8bit:
            print("Using 8-bit quantization")

        # This wrapper is intentionally Idefics2-specific. Constructing the
        # concrete image processor bypasses ProcessorMixin's internal
        # AutoImageProcessor call, which can reject otherwise valid Idefics2
        # checkpoints under mixed Hub-cache/Transformers versions.
        self.processor = _build_idefics2_processor(model_name)
        self.processor.image_processor.do_image_splitting = False

        if feature_only:
            self.model = _load_idefics2_feature_only_model(
                model_name,
                device=self.device,
            )
        else:
            model_kwargs: dict = {}
            if load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                model_kwargs["device_map"] = (
                    {"": self.device}
                    if self.device.startswith("cuda:")
                    else "auto"
                )
            else:
                model_kwargs["dtype"] = (
                    torch.float16
                    if self.device.startswith("cuda")
                    else torch.float32
                )
                if self.device.startswith("cuda:"):
                    model_kwargs["device_map"] = {"": self.device}
                elif self.device.startswith("cuda"):
                    model_kwargs["device_map"] = "auto"

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **model_kwargs,
            )
            if not load_in_8bit and not self.device.startswith("cuda"):
                self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully\n")

    def _format_mc_prompt(
        self,
        letter_to_label: Dict[str, str],
        example_labels: Optional[List[str]] = None,
    ) -> str:
        """Build a deterministic hard-label prompt ending at ``Answer:``."""
        option_lines = [
            f"{letter}. {letter_to_label[letter]}"
            for letter in sorted(letter_to_label)
        ]
        prompt_parts = [
            "The goal of this task is to correctly classify an image. "
            "Choose the correct option from the list below.",
            "\n".join(option_lines),
            "",
        ]
        if example_labels:
            prompt_parts.append("Labeled reference examples:")
            for label in example_labels:
                prompt_parts.extend(
                    ["<image>", f"Reference label: {label}", "", ""]
                )
        prompt_parts.extend(
            [
                "Classify the query image. Output only the letter of the correct option.",
                "<image>",
                "Answer:",
            ]
        )
        return "\n".join(prompt_parts)

    def _get_image_features(self, image: Image.Image) -> torch.Tensor:
        """Encode one image into Idefics2's post-perceiver visual features."""
        inputs = self.processor(
            images=[image],
            text="<image>",
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        pixel_attention_mask = inputs.get("pixel_attention_mask")
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.to(self.device)

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.model.get_image_features(
                        pixel_values=pixel_values,
                        pixel_attention_mask=pixel_attention_mask,
                    )
            else:
                outputs = self.model.get_image_features(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                )
        if isinstance(outputs, torch.Tensor):
            image_features = outputs
        else:
            # Transformers 5 returns raw vision states in last_hidden_state and
            # post-connector, language-space states in pooler_output. Only the
            # latter are valid as Idefics2's image_hidden_states input.
            image_features = getattr(outputs, "pooler_output", None)
        if not isinstance(image_features, torch.Tensor):
            raise TypeError(
                "Idefics2 get_image_features returned an unsupported output "
                f"type: {type(outputs).__name__}"
            )
        expected_hidden_size = int(self.model.config.text_config.hidden_size)
        if image_features.shape[-1] != expected_hidden_size:
            raise ValueError(
                "Idefics2 image features are not in language-model space: "
                f"expected hidden size {expected_hidden_size}, got "
                f"{image_features.shape[-1]}"
            )
        return image_features

    def _get_image_features_multi(
        self,
        images: List[Image.Image],
    ) -> torch.Tensor:
        """Encode images independently, preserving their prompt order."""
        if not images:
            raise ValueError("images must not be empty")
        return self._concatenate_full_label_image_features(
            [self._get_image_features(image) for image in images]
        )

    @staticmethod
    def _concatenate_full_label_image_features(
        features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate flattened or dimension-preserving Transformers layouts."""
        if not features:
            raise ValueError("features must not be empty")
        rank = features[0].ndim
        if any(feature.ndim != rank for feature in features):
            raise ValueError("All image feature tensors must have the same rank")
        if any(
            feature.shape[-1] != features[0].shape[-1]
            for feature in features
        ):
            raise ValueError("All image feature tensors must have the same hidden size")
        image_dimension = 0 if rank == 2 else 1
        return torch.cat(features, dim=image_dimension)

    def _compute_label_probabilities_batch_with_features(
        self,
        image_hidden_states_list: List[torch.Tensor],
        prompts: List[str],
        labels: List[str],
    ) -> List[float]:
        """Return mean-token log likelihoods using precomputed image features."""
        batch_size = len(labels)
        if not batch_size:
            return []
        if len(prompts) != batch_size or len(image_hidden_states_list) != batch_size:
            raise ValueError("features, prompts, and labels must have equal lengths")

        all_label_tokens: list[list[int]] = []
        for label in labels:
            label_tokens = self._label_token_cache.get(label)
            if label_tokens is None:
                label_tokens = self.processor.tokenizer.encode(
                    label,
                    add_special_tokens=False,
                )
                self._label_token_cache[label] = label_tokens
            all_label_tokens.append(label_tokens)

        all_prompt_inputs = []
        for prompt in prompts:
            prompt_input = self._prompt_token_cache.get(prompt)
            if prompt_input is None:
                prompt_input = self.processor(text=prompt, return_tensors="pt")
                self._prompt_token_cache[prompt] = prompt_input
            all_prompt_inputs.append(prompt_input)

        if batch_size == 1:
            prompt_inputs = all_prompt_inputs[0]
        else:
            max_length = max(
                item["input_ids"].shape[1] for item in all_prompt_inputs
            )
            batched_input_ids = []
            batched_attention_mask = []
            for item in all_prompt_inputs:
                pad_length = max_length - item["input_ids"].shape[1]
                if pad_length:
                    input_ids = torch.cat(
                        [
                            item["input_ids"],
                            torch.full(
                                (1, pad_length),
                                self.processor.tokenizer.pad_token_id,
                                dtype=item["input_ids"].dtype,
                            ),
                        ],
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        [
                            item["attention_mask"],
                            torch.zeros(
                                (1, pad_length),
                                dtype=item["attention_mask"].dtype,
                            ),
                        ],
                        dim=1,
                    )
                else:
                    input_ids = item["input_ids"]
                    attention_mask = item["attention_mask"]
                batched_input_ids.append(input_ids)
                batched_attention_mask.append(attention_mask)
            prompt_inputs = {
                "input_ids": torch.cat(batched_input_ids, dim=0),
                "attention_mask": torch.cat(batched_attention_mask, dim=0),
            }

        prompt_inputs = {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in prompt_inputs.items()
        }
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_attention_mask = prompt_inputs["attention_mask"]

        max_label_length = max(len(tokens) for tokens in all_label_tokens)
        full_input_ids = []
        full_attention_mask = []
        label_start_positions = []
        for index, label_tokens in enumerate(all_label_tokens):
            prompt_ids = prompt_input_ids[index]
            prompt_mask = prompt_attention_mask[index]
            prompt_length = int(prompt_mask.sum().item())
            label_start_positions.append(prompt_length - 1)
            padded_label = label_tokens + [
                self.processor.tokenizer.pad_token_id
            ] * (max_label_length - len(label_tokens))
            full_input_ids.append(
                torch.cat(
                    [
                        prompt_ids,
                        torch.tensor(padded_label, device=self.device),
                    ]
                )
            )
            label_mask = [1] * len(label_tokens) + [0] * (
                max_label_length - len(label_tokens)
            )
            full_attention_mask.append(
                torch.cat(
                    [
                        prompt_mask,
                        torch.tensor(
                            label_mask,
                            device=self.device,
                            dtype=prompt_mask.dtype,
                        ),
                    ]
                )
            )

        model_inputs = {
            "input_ids": torch.stack(full_input_ids),
            "attention_mask": torch.stack(full_attention_mask),
            "image_hidden_states": torch.cat(image_hidden_states_list, dim=0),
            "use_cache": self.use_cache,
        }
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

        log_probabilities = torch.nn.functional.log_softmax(
            outputs.logits,
            dim=-1,
        )
        batch_log_probabilities = []
        for batch_index, label_tokens in enumerate(all_label_tokens):
            if not label_tokens:
                batch_log_probabilities.append(float("-inf"))
                continue
            start_position = label_start_positions[batch_index]
            log_probability = sum(
                log_probabilities[
                    batch_index,
                    start_position + token_offset,
                    token_id,
                ].item()
                for token_offset, token_id in enumerate(label_tokens)
            )
            batch_log_probabilities.append(
                log_probability / len(label_tokens)
            )
        return batch_log_probabilities

    def _compute_label_probabilities_with_prefix_cache(
        self,
        image_hidden_states: torch.Tensor,
        prompt: str,
        labels: List[str],
        batch_size: int,
    ) -> List[float]:
        """Score labels after one shared image-conditioned prefix prefill."""
        if not labels:
            return []

        all_label_tokens = []
        for label in labels:
            label_tokens = self._label_token_cache.get(label)
            if label_tokens is None:
                label_tokens = self.processor.tokenizer.encode(
                    label,
                    add_special_tokens=False,
                )
                self._label_token_cache[label] = label_tokens
            if not label_tokens:
                raise ValueError(f"Candidate label {label!r} produced no tokens")
            all_label_tokens.append(label_tokens)

        prompt_inputs = self._prompt_token_cache.get(prompt)
        if prompt_inputs is None:
            prompt_inputs = self.processor(text=prompt, return_tensors="pt")
            self._prompt_token_cache[prompt] = prompt_inputs
        prefix_input_ids = prompt_inputs["input_ids"].to(self.device)
        prefix_attention_mask = prompt_inputs["attention_mask"].to(self.device)
        prefix_model_inputs = {
            "input_ids": prefix_input_ids,
            "attention_mask": prefix_attention_mask,
            "image_hidden_states": image_hidden_states,
            "use_cache": True,
            # Only the final prefix position predicts the first label token.
            "logits_to_keep": 1,
        }
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    prefix_outputs = self.model(**prefix_model_inputs)
            else:
                prefix_outputs = self.model(**prefix_model_inputs)

        prefix_cache = getattr(prefix_outputs, "past_key_values", None)
        if prefix_cache is None or not hasattr(
            prefix_cache, "batch_repeat_interleave"
        ):
            raise TypeError(
                "Idefics2 did not return a branchable Transformers cache"
            )
        prefix_log_probabilities = torch.nn.functional.log_softmax(
            prefix_outputs.logits[0, -1],
            dim=-1,
        )
        scores = []
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The Idefics2 tokenizer must define a pad token")

        for start in range(0, len(labels), batch_size):
            chunk_tokens = all_label_tokens[start:start + batch_size]
            chunk_size = len(chunk_tokens)
            token_log_probability_sums = [
                float(prefix_log_probabilities[tokens[0]].item())
                for tokens in chunk_tokens
            ]
            continuation_length = max(len(tokens) - 1 for tokens in chunk_tokens)
            if continuation_length:
                continuation_ids = torch.full(
                    (chunk_size, continuation_length),
                    pad_token_id,
                    dtype=prefix_input_ids.dtype,
                    device=self.device,
                )
                continuation_mask = torch.zeros(
                    (chunk_size, continuation_length),
                    dtype=prefix_attention_mask.dtype,
                    device=self.device,
                )
                for row, tokens in enumerate(chunk_tokens):
                    continuation = tokens[:-1]
                    if continuation:
                        continuation_ids[row, :len(continuation)] = torch.tensor(
                            continuation,
                            dtype=prefix_input_ids.dtype,
                            device=self.device,
                        )
                        continuation_mask[row, :len(continuation)] = 1

                branched_cache = copy.deepcopy(prefix_cache)
                branched_cache.batch_repeat_interleave(chunk_size)
                continuation_inputs = {
                    "input_ids": continuation_ids,
                    "attention_mask": torch.cat(
                        [
                            prefix_attention_mask.repeat(chunk_size, 1),
                            continuation_mask,
                        ],
                        dim=1,
                    ),
                    "past_key_values": branched_cache,
                    "use_cache": True,
                }
                with torch.no_grad():
                    if self.device.startswith("cuda"):
                        with torch.autocast("cuda", dtype=torch.float16):
                            continuation_outputs = self.model(**continuation_inputs)
                    else:
                        continuation_outputs = self.model(**continuation_inputs)
                continuation_log_probabilities = torch.nn.functional.log_softmax(
                    continuation_outputs.logits,
                    dim=-1,
                )
                for row, tokens in enumerate(chunk_tokens):
                    token_log_probability_sums[row] += sum(
                        continuation_log_probabilities[
                            row,
                            token_offset - 1,
                            tokens[token_offset],
                        ].item()
                        for token_offset in range(1, len(tokens))
                    )
            scores.extend(
                total / len(tokens)
                for total, tokens in zip(token_log_probability_sums, chunk_tokens)
            )
        return scores

    def _format_full_label_scoring_prompt(
        self,
        example_labels: Optional[List[str]] = None,
    ) -> str:
        """Create the option-free prompt shared by all candidate labels."""
        prompt_parts = [
            "Classify the query image. Output only its exact class label.",
            "",
        ]
        if example_labels:
            prompt_parts.append("Labeled reference examples:")
            for label in example_labels:
                prompt_parts.extend(["<image>", f"Label: {label}", ""])
        prompt_parts.extend(["Query image:", "<image>", "Label:"])
        return "\n".join(prompt_parts)

    def encode_full_label_scoring_images(
        self,
        images: List[Image.Image],
    ) -> torch.Tensor:
        """Encode ordered prompt images once for candidate-label reuse."""
        return self._get_image_features_multi(images)

    @staticmethod
    def combine_full_label_scoring_image_features(
        context_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        """Combine context then query features in prompt order."""
        return Idefics2Wrapper._concatenate_full_label_image_features(
            [context_features, query_features]
        )

    def score_candidate_labels_with_image_features(
        self,
        image_hidden_states: torch.Tensor,
        example_labels: List[str],
        candidate_labels: List[str],
        batch_size: int = 8,
        scoring_mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """Score exact label continuations under one fixed prompt."""
        if not candidate_labels:
            raise ValueError("candidate_labels must not be empty")
        if len(set(candidate_labels)) != len(candidate_labels):
            raise ValueError("candidate_labels must be unique")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        scoring_mode = scoring_mode or getattr(
            self, "scoring_mode", FULL_SEQUENCE_SCORING
        )
        if scoring_mode not in SUPPORTED_SCORING_MODES:
            raise ValueError(
                f"Unsupported scoring mode {scoring_mode!r}; expected one of "
                f"{sorted(SUPPORTED_SCORING_MODES)}"
            )

        prompt = self._format_full_label_scoring_prompt(example_labels or None)
        if scoring_mode == PREFIX_KV_CACHE_SCORING:
            scores = self._compute_label_probabilities_with_prefix_cache(
                image_hidden_states,
                prompt,
                candidate_labels,
                batch_size,
            )
        else:
            scores = []
            for start in range(0, len(candidate_labels), batch_size):
                labels = candidate_labels[start:start + batch_size]
                scores.extend(
                    self._compute_label_probabilities_batch_with_features(
                        image_hidden_states_list=[
                            image_hidden_states
                        ] * len(labels),
                        prompts=[prompt] * len(labels),
                        labels=labels,
                    )
                )
        return {
            label: float(score)
            for label, score in zip(candidate_labels, scores)
        }

    def _letter_token_id(self, letter: str) -> int:
        """Return the tokenizer ID for an option letter in continuation context."""
        if letter not in self._letter_token_id_cache:
            token_ids = self.processor.tokenizer.encode(
                " " + letter,
                add_special_tokens=False,
            )
            if not token_ids:
                raise ValueError(f"Option letter {letter!r} produced no tokens")
            self._letter_token_id_cache[letter] = token_ids[-1]
        return self._letter_token_id_cache[letter]

    def classify_with_context_mc(
        self,
        query_image: Image.Image,
        context_examples: List[Tuple[Image.Image, str]],
        letter_to_label: Dict[str, str],
    ) -> Tuple[str, Dict[str, float]]:
        """Classify a K-way prompt from one next-token model pass."""
        if not letter_to_label:
            raise ValueError("letter_to_label must not be empty")
        example_labels = (
            [label for _, label in context_examples]
            if context_examples
            else None
        )
        prompt = self._format_mc_prompt(letter_to_label, example_labels)
        images = [image for image, _ in context_examples] + [query_image]
        image_hidden_states = self._get_image_features_multi(images)
        prompt_inputs = self.processor(text=prompt, return_tensors="pt")
        prompt_inputs = {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in prompt_inputs.items()
        }
        model_inputs = {
            "input_ids": prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
            "image_hidden_states": image_hidden_states,
            "use_cache": self.use_cache,
        }
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

        last_position = int(prompt_inputs["attention_mask"].sum().item()) - 1
        next_token_log_probabilities = torch.nn.functional.log_softmax(
            outputs.logits[0, last_position],
            dim=-1,
        )
        letters = list(letter_to_label)
        token_ids = torch.tensor(
            [self._letter_token_id(letter) for letter in letters],
            device=next_token_log_probabilities.device,
        )
        closed_set_probabilities = torch.nn.functional.softmax(
            next_token_log_probabilities[token_ids],
            dim=-1,
        )
        probabilities = {
            letter: closed_set_probabilities[index].item()
            for index, letter in enumerate(letters)
        }
        return max(probabilities, key=probabilities.get), probabilities
