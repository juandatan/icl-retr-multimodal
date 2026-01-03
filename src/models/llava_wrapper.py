"""
LLaVA-1.5 wrapper for computing output probabilities and utilities.

This module provides a wrapper around the LLaVA-1.5-7B model to:
- Compute output probabilities for classification tasks
- Support 0-shot and n-shot in-context learning
- Calculate marginal utility of ICL examples
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
class LLaVAWrapper:
    """
    Wrapper for LLaVA-1.5-7B model to compute classification probabilities.

    Computes the probability of generating the correct class name by:
    - Concatenating prompt + label tokens
    - Single forward pass with causal attention
    - Extracting log probabilities for label tokens

    Supports batched processing for efficiency.

    Supports:
    - 0-shot prediction: P(class_name|image)
    - n-shot prediction: P(class_name|image, example1, ..., exampleN)
    - Marginal utility computation (normalized to [-1, 1])
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize LLaVA model.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        print(f"Loading LLaVA model: {model_name}")
        print(f"Device: {device}")
        if load_in_8bit:
            print("Using 8-bit quantization")
        elif load_in_4bit:
            print("Using 4-bit quantization")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        model_kwargs = {}
        if load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif load_in_4bit:
            model_kwargs['load_in_4bit'] = True
        else:
            model_kwargs['torch_dtype'] = torch.float16 if device == "cuda" else torch.float32

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
            device_map="auto" if device == "cuda" else None
        )

        if device != "cuda" and not (load_in_8bit or load_in_4bit):
            self.model = self.model.to(device)

        self.model.eval()
        print("✓ Model loaded successfully\n")

    def format_prompt(
        self,
        example_labels: Optional[List[str]] = None,
    ) -> str:
        """
        Format prompt for classification task.

        For 0-shot:
            "<image>\nQuestion: What is this?\nAnswer:"

        For n-shot:
            "<image>\nQuestion: What is this?\nAnswer: {label1}\n\n
             <image>\nQuestion: What is this?\nAnswer: {label2}\n\n
             <image>\nQuestion: What is this?\nAnswer:"

        Args:
            example_labels: List of example labels for ICL

        Returns:
            Formatted prompt string with <image> tokens
        """
        prompt_parts = []

        # Add examples if provided
        if example_labels is not None:
            for ex_label in example_labels:
                prompt_parts.append("<image>")
                prompt_parts.append("Question: What is this?")
                prompt_parts.append(f"Answer: {ex_label}")
                prompt_parts.append("")  # Blank line

        # Add query
        prompt_parts.append("<image>")
        prompt_parts.append("Question: What is this?")
        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)

    def _compute_label_probabilities_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        labels: List[str]
    ) -> List[float]:
        """
        Compute log probability of generating labels for a batch of inputs.

        Uses a single forward pass per batch by concatenating prompt + label tokens.
        The causal attention mask ensures proper autoregressive probability computation.

        Args:
            images: List of images (can be lists for multi-image inputs in ICL)
            prompts: List of prompt strings (ending in "Answer:")
            labels: List of ground-truth label strings

        Returns:
            List of log probabilities, one per input
        """
        batch_size = len(images)

        # Tokenize all labels first to know their lengths
        all_label_tokens = []
        for label in labels:
            label_tokens = self.processor.tokenizer.encode(
                label,
                add_special_tokens=False
            )
            all_label_tokens.append(label_tokens)

        # Process prompts (without labels)
        prompt_inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        prompt_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in prompt_inputs.items()}

        prompt_input_ids = prompt_inputs['input_ids']
        prompt_attention_mask = prompt_inputs['attention_mask']

        # Find max label length for padding
        max_label_len = max(len(tokens) for tokens in all_label_tokens)

        # Concatenate labels to prompts with padding
        full_input_ids = []
        full_attention_mask = []
        label_start_positions = []  # Where each label starts in the sequence

        for i in range(batch_size):
            # Get prompt for this sample
            prompt_ids = prompt_input_ids[i]
            prompt_mask = prompt_attention_mask[i]

            # Find actual prompt length (excluding padding)
            prompt_len = prompt_mask.sum().item()
            label_start_positions.append(prompt_len - 1)  # -1 for 0-indexed

            # Pad label tokens to max length
            label_tokens = all_label_tokens[i]
            padded_label = label_tokens + [self.processor.tokenizer.pad_token_id] * (max_label_len - len(label_tokens))

            # Concatenate
            full_ids = torch.cat([
                prompt_ids,
                torch.tensor(padded_label, device=self.device)
            ])

            # Create attention mask (1 for real tokens, 0 for padding)
            label_mask = [1] * len(label_tokens) + [0] * (max_label_len - len(label_tokens))
            full_mask = torch.cat([
                prompt_mask,
                torch.tensor(label_mask, device=self.device, dtype=prompt_mask.dtype)
            ])

            full_input_ids.append(full_ids)
            full_attention_mask.append(full_mask)

        # Stack into batch
        full_input_ids = torch.stack(full_input_ids)
        full_attention_mask = torch.stack(full_attention_mask)

        # Single forward pass for entire batch
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                pixel_values=prompt_inputs.get('pixel_values'),
            )

        # Extract log probabilities for each sample
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)

        batch_log_probs = []
        for i in range(batch_size):
            label_tokens = all_label_tokens[i]
            if len(label_tokens) == 0:
                batch_log_probs.append(-np.inf)
                continue

            # Extract log probs for this sample's label tokens
            start_pos = label_start_positions[i]
            log_prob_sum = 0.0

            for j, token_id in enumerate(label_tokens):
                pos = start_pos + j
                log_prob_sum += log_probs_all[i, pos, token_id].item()

            batch_log_probs.append(log_prob_sum)

        return batch_log_probs

    def compute_baseline_probabilities(
        self,
        dataset,
        batch_size: int = 8,
        save_path: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Compute 0-shot baseline log probabilities for all examples in dataset.

        Args:
            dataset: Dataset with examples (must have label_name attribute)
            batch_size: Batch size for processing
            save_path: Optional path to save computed probabilities

        Returns:
            Dictionary mapping example index -> 0-shot log probability
        """
        baseline_probs = {}

        print(f"Computing baseline (0-shot) probabilities for {len(dataset)} examples...")

        from tqdm import tqdm

        for start_idx in tqdm(range(0, len(dataset), batch_size)):
            end_idx = min(start_idx + batch_size, len(dataset))

            # Prepare batch
            batch_images = []
            batch_prompts = []
            batch_labels = []
            batch_indices = []

            for idx in range(start_idx, end_idx):
                example, image = dataset[idx]
                batch_images.append(image)  # Single image per sample for 0-shot
                batch_prompts.append(self.format_prompt(example_labels=None))
                batch_labels.append(example.label_name)
                batch_indices.append(idx)

            # Compute log probabilities for batch
            log_probs = self._compute_label_probabilities_batch(
                images=batch_images,
                prompts=batch_prompts,
                labels=batch_labels
            )

            # Store results
            for idx, log_prob in zip(batch_indices, log_probs):
                baseline_probs[idx] = log_prob

        if save_path:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(baseline_probs, f)
            print(f"✓ Saved baseline probabilities to {save_path}")

        return baseline_probs

    def compute_marginal_utilities_batch(
        self,
        query_images: List[Image.Image],
        query_labels: List[str],
        example_images: List[Image.Image],
        example_labels: List[str],
        baseline_log_probs: List[float],
    ) -> List[float]:
        """
        Compute marginal utilities for a batch of (query, example) pairs.

        Marginal utility = (u1 - u0) / max(abs(u1), abs(u0))
        where u1 = log P(label|image, example) and u0 = log P(label|image)

        This normalizes utility to [-1, 1] range.

        Args:
            query_images: List of query images
            query_labels: List of query labels
            example_images: List of example images (one per query)
            example_labels: List of example labels (one per query)
            baseline_log_probs: List of pre-computed 0-shot log probabilities

        Returns:
            List of marginal utilities, one per query
        """
        batch_size = len(query_images)

        # Prepare batch for 1-shot computation
        batch_images = []
        batch_prompts = []
        batch_labels = []

        for i in range(batch_size):
            # Each sample gets one example image + query image
            batch_images.append([example_images[i], query_images[i]])
            batch_prompts.append(self.format_prompt(example_labels=[example_labels[i]]))
            batch_labels.append(query_labels[i])

        # Compute 1-shot log probabilities
        oneshot_log_probs = self._compute_label_probabilities_batch(
            images=batch_images,
            prompts=batch_prompts,
            labels=batch_labels
        )

        # Compute normalized marginal utilities
        utilities = []
        for u1, u0 in zip(oneshot_log_probs, baseline_log_probs):
            # Normalized utility: (u1 - u0) / max(|u1|, |u0|)
            denominator = max(abs(u1), abs(u0), 1e-10)  # Avoid division by zero
            utility = (u1 - u0) / denominator
            utilities.append(utility)

        return utilities
