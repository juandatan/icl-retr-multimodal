"""Idefics2 scoring adapter for the retained CUB-200 pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


class Idefics2Wrapper:
    """Expose only the discriminative scoring paths used by this project."""

    def __init__(
        self,
        model_name: str = "HuggingFaceM4/idefics2-8b",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.use_cache = False
        self._letter_token_id_cache: dict[str, int] = {}
        self._label_token_cache: dict[str, list[int]] = {}
        self._prompt_token_cache: dict[str, object] = {}

        print(f"Loading Idefics2 model: {model_name}")
        print(f"Device: {self.device}")
        if load_in_8bit:
            print("Using 8-bit quantization")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.image_processor.do_image_splitting = False

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
            model_kwargs["torch_dtype"] = (
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
            return outputs
        image_features = getattr(outputs, "last_hidden_state", None)
        if not isinstance(image_features, torch.Tensor):
            raise TypeError(
                "Idefics2 get_image_features returned an unsupported output "
                f"type: {type(outputs).__name__}"
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
    ) -> Dict[str, float]:
        """Score exact label continuations under one fixed prompt."""
        if not candidate_labels:
            raise ValueError("candidate_labels must not be empty")
        if len(set(candidate_labels)) != len(candidate_labels):
            raise ValueError("candidate_labels must be unique")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        prompt = self._format_full_label_scoring_prompt(example_labels or None)
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
