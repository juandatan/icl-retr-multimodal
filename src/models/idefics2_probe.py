"""Frozen Idefics2 pair encoder and lightweight utility probe."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


PROBE_PROMPT_TEMPLATE = (
    "Classify the query image. Output only its exact class label.\n\n"
    "Labeled reference examples:\n"
    "<image>\n"
    "Label: {exemplar_label}\n\n"
    "Query image:\n"
    "<image>\n"
    "Label:"
)


class TextOnlyIdefics2Processor:
    """Expand Idefics2 image placeholders without importing torchvision.

    Probe-cache generation supplies post-connector visual states directly to
    the language model, so running an image processor would be both redundant
    and incorrect.  This implements the text half of Idefics2Processor: every
    image placeholder becomes one boundary token, ``image_seq_len`` image
    tokens, and a closing boundary token before ordinary tokenization.
    """

    image_token = "<image>"
    image_boundary_token = "<fake_token_around_image>"

    def __init__(self, tokenizer, *, image_seq_len: int) -> None:
        if image_seq_len <= 0:
            raise ValueError("image_seq_len must be positive")
        self.tokenizer = tokenizer
        self.image_seq_len = int(image_seq_len)

        image_ids = tokenizer(
            self.image_token,
            add_special_tokens=False,
        )["input_ids"]
        if len(image_ids) != 1:
            raise ValueError(
                "The Idefics2 tokenizer must encode <image> as one special token"
            )
        self.image_token_id = int(image_ids[0])

    def __call__(
        self,
        *,
        text: str | Sequence[str],
        padding: bool | str = False,
        return_tensors: str | None = None,
        **tokenizer_kwargs: Any,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        image_block = (
            self.image_boundary_token
            + self.image_token * self.image_seq_len
            + self.image_boundary_token
        )
        expanded = [
            value.replace(self.image_token, image_block).replace(
                self.image_boundary_token * 2,
                self.image_boundary_token,
            )
            for value in texts
        ]
        return self.tokenizer(
            expanded,
            padding=padding,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )


@dataclass(frozen=True)
class FrozenIdefics2ProbeBackbone:
    model: nn.Module
    processor: TextOnlyIdefics2Processor
    device: torch.device


def load_frozen_idefics2_probe_backbone(
    model_name: str,
    *,
    device: str | torch.device,
    load_in_8bit: bool,
    image_seq_len: int,
    awq_backend: str | None = None,
) -> FrozenIdefics2ProbeBackbone:
    """Load only the frozen LM and tokenizer needed by the probe cache.

    Transformer imports are deliberately local. This path does not construct
    Idefics2ImageProcessor because cached visual states are supplied. The AWQ
    runtime currently still imports torchvision while registering unrelated
    GPTQModel architectures, so its binary must match the installed PyTorch.
    """
    from transformers import (  # noqa: PLC0415
        AutoModelForImageTextToText,
        AutoTokenizer,
        AwqConfig,
        BitsAndBytesConfig,
    )

    resolved_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = TextOnlyIdefics2Processor(
        tokenizer,
        image_seq_len=image_seq_len,
    )

    model_kwargs: dict[str, Any] = {}
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = (
            {"": str(resolved_device)}
            if resolved_device.type == "cuda" and resolved_device.index is not None
            else "auto"
        )
    else:
        if awq_backend is not None:
            if awq_backend not in {"gemm_triton", "torch_awq"}:
                raise ValueError(
                    "awq_backend must be gemm_triton or torch_awq for the "
                    "frozen probe"
                )
            # The checkpoint already contains its full AWQ configuration. When
            # Transformers merges this object with the stored config, only
            # loading attributes such as ``backend`` override the checkpoint.
            # An explicit backend prevents GPTQModel's AUTO mode from choosing
            # Marlin and trying to compile CUDA extensions with the host nvcc.
            model_kwargs["quantization_config"] = AwqConfig(
                backend=awq_backend,
            )
        model_kwargs["dtype"] = (
            torch.float16 if resolved_device.type == "cuda" else torch.float32
        )
        if resolved_device.type == "cuda":
            model_kwargs["device_map"] = (
                {"": str(resolved_device)}
                if resolved_device.index is not None
                else "auto"
            )

    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    if not load_in_8bit and resolved_device.type != "cuda":
        model = model.to(resolved_device)
    model.eval()

    configured_image_token_id = getattr(model.config, "image_token_id", None)
    if (
        configured_image_token_id is not None
        and int(configured_image_token_id) != processor.image_token_id
    ):
        raise ValueError(
            "Idefics2 tokenizer and model config disagree on the image token ID"
        )
    perceiver_config = getattr(model.config, "perceiver_config", None)
    configured_image_seq_len = getattr(
        perceiver_config,
        "resampler_n_latents",
        None,
    )
    if (
        configured_image_seq_len is not None
        and int(configured_image_seq_len) != image_seq_len
    ):
        raise ValueError(
            "Cached visual-token length differs from the Idefics2 perceiver config"
        )
    return FrozenIdefics2ProbeBackbone(model, processor, resolved_device)


def format_probe_prompt(exemplar_label: str) -> str:
    """Return the teacher-aligned one-shot prompt without a query answer."""
    if not exemplar_label or not str(exemplar_label).strip():
        raise ValueError("exemplar_label must be non-empty")
    return PROBE_PROMPT_TEMPLATE.format(exemplar_label=str(exemplar_label))


def _last_attended_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    masked = positions.masked_fill(~attention_mask.bool(), -1)
    last = masked.max(dim=1).values
    if (last < 0).any():
        raise ValueError("Every prompt must contain at least one attended token")
    return last


@torch.inference_mode()
def encode_frozen_idefics2_pairs(
    model,
    processor,
    exemplar_visual_tokens: torch.Tensor,
    query_visual_tokens: torch.Tensor,
    exemplar_labels: Sequence[str],
    *,
    device: str | torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """Encode labeled-exemplar/query pairs through Idefics2's frozen LM.

    Visual states must already be post-connector Idefics2 states. The two image
    sequences are interleaved in the same order as their ``<image>`` markers.
    Only the final prompt state is returned; no candidate output labels or
    teacher quantities are supplied to the model.
    """
    if exemplar_visual_tokens.ndim != 3 or query_visual_tokens.ndim != 3:
        raise ValueError("visual tokens must have shape [batch, tokens, hidden]")
    if exemplar_visual_tokens.shape != query_visual_tokens.shape:
        raise ValueError("exemplar and query visual-token shapes must match")
    batch_size = exemplar_visual_tokens.shape[0]
    if len(exemplar_labels) != batch_size:
        raise ValueError("exemplar_labels and visual-token batch must agree")

    prompts = [format_probe_prompt(label) for label in exemplar_labels]
    prompt_inputs = processor(
        text=prompts,
        padding=True,
        return_tensors="pt",
    )
    input_ids = prompt_inputs["input_ids"].to(device)
    attention_mask = prompt_inputs["attention_mask"].to(device)
    image_hidden_states = torch.stack(
        (exemplar_visual_tokens, query_visual_tokens), dim=1
    ).reshape(
        batch_size * 2,
        exemplar_visual_tokens.shape[1],
        exemplar_visual_tokens.shape[2],
    ).to(device)

    base_model = getattr(model, "model", model)
    amp_enabled = bool(use_amp) and torch.device(device).type == "cuda"
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.last_hidden_state
    last = _last_attended_positions(attention_mask)
    return hidden[torch.arange(batch_size, device=hidden.device), last].float().cpu()


def quantize_probe_representations(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically quantize each pair representation with one FP16 scale."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite rank-2 array")
    scales = np.max(np.abs(values), axis=-1, keepdims=True) / 127.0
    scales[scales == 0] = 1.0
    quantized = np.clip(np.rint(values / scales), -127, 127).astype(np.int8)
    return quantized, scales.astype(np.float16)


def dequantize_probe_representations(
    values: np.ndarray,
    scales: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(values)
    if scales is None:
        return values
    return (
        values.astype(np.float32) * np.asarray(scales, dtype=np.float32)
    ).astype(np.float16)


class FrozenIdefics2UtilityProbe(nn.Module):
    """A scalar linear probe over frozen, pair-conditioned Idefics2 states."""

    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.scorer = nn.Linear(self.input_dim, 1)

    def forward(
        self,
        pair_representations: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pair_representations.ndim != 3:
            raise ValueError(
                "pair_representations must have shape [batch, candidates, hidden]"
            )
        if pair_representations.shape[-1] != self.input_dim:
            raise ValueError("pair representation width differs from input_dim")
        if candidate_mask.shape != pair_representations.shape[:2]:
            raise ValueError("candidate_mask shape does not match representations")
        scores = self.scorer(self.dropout(pair_representations.float())).squeeze(-1)
        return scores.masked_fill(~candidate_mask.bool(), torch.finfo(scores.dtype).min)
