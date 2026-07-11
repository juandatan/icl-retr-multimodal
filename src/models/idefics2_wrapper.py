"""
Idefics2 wrapper for computing output probabilities and utilities.

This module provides a wrapper around Idefics2 models to:
- Compute output probabilities for classification tasks
- Support 0-shot and n-shot in-context learning with true multi-image interleaving
- Calculate marginal utility of ICL examples
"""

import re
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from torch.cuda.amp import autocast


class Idefics2Wrapper:
    """
    Wrapper for Idefics2 model to compute classification probabilities.

    Computes the probability of generating the correct class name by:
    - Concatenating prompt + label tokens
    - Single forward pass with causal attention
    - Extracting log probabilities for label tokens

    Supports batched processing for efficiency and proper multi-image interleaving.

    Supports:
    - 0-shot prediction: P(class_name|image)
    - n-shot prediction: P(class_name|image, example1, ..., exampleN)
    - Marginal utility computation (normalized to [-1, 1])
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceM4/idefics2-8b",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_cache: bool = False,  # Disabled for discriminative evaluation (no token-by-token generation)
        cache_vision_embeddings: bool = False,  # Disabled by default for Idefics2
        max_vision_cache_size: int = 5000,
        do_image_splitting: bool = False,
    ):
        """
        Initialize Idefics2 model.

        Args:
            model_name: HuggingFace model name (default: idefics2-8b)
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            use_cache: Whether to use KV cache (disabled by default for discriminative evaluation)
            cache_vision_embeddings: Whether to cache vision encoder outputs
            max_vision_cache_size: Maximum number of images to cache (0 = unlimited)
        """
        self.model_name = model_name
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        self.load_in_8bit = load_in_8bit
        self.do_image_splitting = do_image_splitting
        self.load_in_4bit = load_in_4bit
        self.use_cache = use_cache
        self.cache_vision_embeddings = cache_vision_embeddings
        self.max_vision_cache_size = max_vision_cache_size

        # Vision embedding cache: maps image hash -> vision embeddings
        self._vision_cache = {} if cache_vision_embeddings else None

        print(f"Loading Idefics2 model: {model_name}")
        print(f"Device: {device}")
        if load_in_8bit:
            print("Using 8-bit quantization")
        elif load_in_4bit:
            print("Using 4-bit quantization")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.image_processor.do_image_splitting = do_image_splitting

        # Configure model loading
        model_kwargs = {}
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs['quantization_config'] = quantization_config
            if device.startswith("cuda:"):
                model_kwargs['device_map'] = {"": device}
            else:
                model_kwargs['device_map'] = "auto"
        elif load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs['quantization_config'] = quantization_config
            if device.startswith("cuda:"):
                model_kwargs['device_map'] = {"": device}
            else:
                model_kwargs['device_map'] = "auto"
        else:
            model_kwargs['torch_dtype'] = torch.float16 if device.startswith("cuda") else torch.float32
            if device.startswith("cuda:"):
                model_kwargs['device_map'] = {"": device}
            elif device.startswith("cuda"):
                model_kwargs['device_map'] = "auto"

        # Load Idefics2 model
        print("Loading Idefics2 with multi-image interleaving support")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Only move to device if not using quantization and not already on a CUDA device
        if not (load_in_8bit or load_in_4bit) and not device.startswith("cuda"):
            self.model = self.model.to(device)

        self.model.eval()
        print("✓ Model loaded successfully\n")

    def format_prompt(
        self,
        example_labels: Optional[List[str]] = None,
        candidate_labels: Optional[List[str]] = None,
    ) -> str:
        """
        Format prompt for classification task with <image> tokens.

        Args:
            example_labels: List of example labels for ICL
            candidate_labels: List of candidate labels to choose from (for generative evaluation)

        Returns:
            String prompt with <image> tokens properly placed
        """
        # Build task description
        task_parts = []
        if candidate_labels is not None:
            num_classes = len(candidate_labels)
            task_parts.append(f"The goal of this task is to correctly classify an image. There are {num_classes} possible classes:")
            task_parts.append(self._format_candidate_list(candidate_labels))
            task_parts.append("Output only the exact class name from the list above.")
        else:
            task_parts.append("The goal of this task is to correctly classify an image.")

        task_description = "\n".join(task_parts)

        # Build prompt with <image> tokens
        prompt_parts = [task_description, ""]

        # Add examples with interleaved images
        if example_labels is not None and len(example_labels) > 0:
            for ex_label in example_labels:
                prompt_parts.append("<image>")
                prompt_parts.append(f"Output: {ex_label}")
                prompt_parts.append("")

        # Add query
        prompt_parts.append("<image>")
        prompt_parts.append("Output:")

        return "\n".join(prompt_parts)

    def _format_candidate_list(self, candidates: List[str], max_per_line: int = 5) -> str:
        """
        Format candidate list for readability.

        For many candidates (>20), uses numbered list with multiple per line.
        For few candidates (<=20), uses simple comma-separated list.

        Args:
            candidates: List of candidate labels
            max_per_line: Maximum candidates per line (default: 5)

        Returns:
            Formatted string
        """
        if len(candidates) <= 20:
            # Simple format for few candidates
            return ", ".join(candidates)

        # Numbered list format for many candidates
        lines = []

        for i in range(0, len(candidates), max_per_line):
            chunk = candidates[i:i + max_per_line]
            # Format as: "1. class1  2. class2  3. class3  4. class4  5. class5"
            line_items = [f"{i+j+1}. {c}" for j, c in enumerate(chunk)]
            lines.append("  " + "  ".join(line_items))

        return "\n".join(lines)

    def _compute_label_probabilities_batch(
        self,
        images: List[List[Image.Image]],  # List of image lists for multi-image ICL
        prompts: List[str],  # List of text prompts with <image> tokens
        labels: List[str]
    ) -> List[float]:
        """
        Compute log probability of generating labels for a batch of inputs.

        Uses a single forward pass per batch by concatenating prompt + label tokens.
        The causal attention mask ensures proper autoregressive probability computation.

        Args:
            images: List of image lists (e.g., [[img1, img2], [img3]] for batch_size=2)
            prompts: List of text prompts with <image> tokens
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

        # Process inputs: Idefics2 expects text strings and image lists
        # For batching, we process each sample individually then stack
        all_prompt_inputs = []
        for i in range(batch_size):
            prompt_input = self.processor(
                text=prompts[i],
                images=images[i],
                return_tensors="pt",
            )
            all_prompt_inputs.append(prompt_input)

        prompt_inputs = all_prompt_inputs[0]

        if batch_size > 1:
            # For batch processing, we need to pad and stack
            # This is a simplified version - may need refinement
            max_len = max(inp['input_ids'].shape[1] for inp in all_prompt_inputs)

            batched_input_ids = []
            batched_attention_mask = []
            batched_pixel_values = []

            for inp in all_prompt_inputs:
                # Pad input_ids
                pad_len = max_len - inp['input_ids'].shape[1]
                if pad_len > 0:
                    padded_ids = torch.cat([
                        inp['input_ids'],
                        torch.full((1, pad_len), self.processor.tokenizer.pad_token_id, dtype=inp['input_ids'].dtype)
                    ], dim=1)
                    padded_mask = torch.cat([
                        inp['attention_mask'],
                        torch.zeros((1, pad_len), dtype=inp['attention_mask'].dtype)
                    ], dim=1)
                else:
                    padded_ids = inp['input_ids']
                    padded_mask = inp['attention_mask']

                batched_input_ids.append(padded_ids)
                batched_attention_mask.append(padded_mask)
                batched_pixel_values.append(inp['pixel_values'])

            prompt_inputs = {
                'input_ids': torch.cat(batched_input_ids, dim=0),
                'attention_mask': torch.cat(batched_attention_mask, dim=0),
                'pixel_values': torch.cat(batched_pixel_values, dim=0),
            }

            # Add other keys if present
            if 'pixel_attention_mask' in all_prompt_inputs[0]:
                prompt_inputs['pixel_attention_mask'] = torch.cat([inp['pixel_attention_mask'] for inp in all_prompt_inputs], dim=0)

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

        # Single forward pass for entire batch with mixed precision
        model_inputs = {
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'pixel_values': prompt_inputs.get('pixel_values'),
            'use_cache': self.use_cache,
        }

        # Add any other inputs that Idefics2 might need
        if 'pixel_attention_mask' in prompt_inputs:
            model_inputs['pixel_attention_mask'] = prompt_inputs['pixel_attention_mask']
        if 'image_hidden_states' in prompt_inputs:
            model_inputs['image_hidden_states'] = prompt_inputs['image_hidden_states']

        with torch.no_grad():
            # Use autocast for CUDA devices to speed up computation
            if self.device.startswith("cuda"):
                with autocast(dtype=torch.float16):
                    outputs = self.model(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

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

            # Mean per-token log prob, not raw sum: summing penalizes longer label
            # strings regardless of correctness (e.g. "Olive sided Flycatcher" would
            # always score below "Mallard" under a raw sum), which breaks argmax-based
            # classification across candidates of varying token length. Dividing by
            # token count here is a no-op for marginal utility, since that's always a
            # ratio of two scores for the *same* label (same token count cancels out).
            batch_log_probs.append(log_prob_sum / len(label_tokens))

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
                batch_images.append([image])  # Single image per sample for 0-shot
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

    def _get_image_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract and cache vision features for a single image using Idefics2's vision encoder.

        This method computes the image_hidden_states (after vision tower + perceiver + MLP)
        which can be reused across multiple forward passes.

        Args:
            image: PIL Image

        Returns:
            image_hidden_states tensor of shape (1, 1, num_visual_tokens, hidden_size)
            For Idefics2: (1, 1, 64, 4096) ≈ 0.5 MB in fp16
        """
        # Process image to get pixel_values
        inputs = self.processor(
            images=[image],
            text="<image>",  # Dummy text, only need image processing
            return_tensors="pt"
        )

        # Move to device
        pixel_values = inputs['pixel_values'].to(self.device)
        pixel_attention_mask = inputs.get('pixel_attention_mask')
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.to(self.device)

        # Extract vision features
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with autocast(dtype=torch.float16):
                    image_hidden_states = self.model.get_image_features(
                        pixel_values=pixel_values,
                        pixel_attention_mask=pixel_attention_mask
                    )
            else:
                image_hidden_states = self.model.get_image_features(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask
                )

        return image_hidden_states

    def _get_image_features_multi(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract vision features for a list of images, one image at a time.

        Processes each image independently through _get_image_features to keep
        peak activation memory bounded to a single image, then concatenates.

        Returns image_hidden_states of shape (1, N, num_visual_tokens, hidden_size)
        where N = len(images), matching the format expected by
        _compute_label_probabilities_batch_with_features.
        """
        per_image_features = [self._get_image_features(img) for img in images]
        # Each is (1, 1, num_tokens, hidden); cat along dim=1 → (1, N, num_tokens, hidden)
        return torch.cat(per_image_features, dim=1)

    def compute_marginal_utilities_batch_cached(
        self,
        query_image: Image.Image,  # Single query image (will be cached)
        query_label: str,  # Single query label
        example_images: List[Image.Image],  # Multiple example images
        example_labels: List[str],  # Multiple example labels
        baseline_log_prob: float,  # Single baseline log prob for query
    ) -> List[float]:
        """
        Compute marginal utilities for multiple candidates of the SAME query.

        This is optimized for the common case where we have 1 query and N candidates:
        - Query image is encoded ONCE and cached
        - Each candidate image is encoded separately
        - Total: N+1 encodings instead of 2N

        Args:
            query_image: Query image (encoded once)
            query_label: Query label
            example_images: List of candidate example images
            example_labels: List of candidate example labels
            baseline_log_prob: Pre-computed 0-shot log probability for query

        Returns:
            List of marginal utilities, one per candidate
        """
        batch_size = len(example_images)

        # Cache query image features ONCE
        query_features = self._get_image_features(query_image)

        # Prepare batch for 1-shot computation
        batch_images_features = []  # Will hold combined features
        batch_prompts = []
        batch_labels = []

        for i in range(batch_size):
            # Cache each example image features
            example_features = self._get_image_features(example_images[i])

            # Combine: [example, query]
            # Shape: (1, 2, 64, 4096) for 2 images
            combined_features = torch.cat([example_features, query_features], dim=1)
            batch_images_features.append(combined_features)

            batch_prompts.append(self.format_prompt(example_labels=[example_labels[i]]))
            batch_labels.append(query_label)

        # Compute 1-shot log probabilities with cached features
        oneshot_log_probs = self._compute_label_probabilities_batch_with_features(
            image_hidden_states_list=batch_images_features,
            prompts=batch_prompts,
            labels=batch_labels
        )

        # Compute normalized marginal utilities
        utilities = []
        for u1 in oneshot_log_probs:
            # Normalized utility: (u1 - u0) / max(|u1|, |u0|)
            denominator = max(abs(u1), abs(baseline_log_prob), 1e-10)
            utility = (u1 - baseline_log_prob) / denominator
            utilities.append(utility)

        return utilities

    def _compute_label_probabilities_batch_with_features(
        self,
        image_hidden_states_list: List[torch.Tensor],  # Precomputed vision features
        prompts: List[str],
        labels: List[str]
    ) -> List[float]:
        """
        Compute log probabilities using precomputed image features.

        This is similar to _compute_label_probabilities_batch but accepts
        cached image_hidden_states instead of raw images, avoiding vision encoder overhead.

        Args:
            image_hidden_states_list: List of precomputed features, one per sample
                                      Each has shape (1, num_images, 64, 4096)
            prompts: List of text prompts with <image> tokens
            labels: List of ground-truth label strings

        Returns:
            List of log probabilities, one per input
        """
        batch_size = len(image_hidden_states_list)

        # Tokenize all labels first
        all_label_tokens = []
        for label in labels:
            label_tokens = self.processor.tokenizer.encode(
                label,
                add_special_tokens=False
            )
            all_label_tokens.append(label_tokens)

        all_prompt_inputs = []
        for i in range(batch_size):
            prompt_input = self.processor(
                text=prompts[i],
                return_tensors="pt",
            )
            all_prompt_inputs.append(prompt_input)

        # Stack the inputs
        if batch_size == 1:
            prompt_inputs = all_prompt_inputs[0]
        else:
            # Pad and stack
            max_len = max(inp['input_ids'].shape[1] for inp in all_prompt_inputs)
            batched_input_ids = []
            batched_attention_mask = []

            for inp in all_prompt_inputs:
                pad_len = max_len - inp['input_ids'].shape[1]
                if pad_len > 0:
                    padded_ids = torch.cat([
                        inp['input_ids'],
                        torch.full((1, pad_len), self.processor.tokenizer.pad_token_id, dtype=inp['input_ids'].dtype)
                    ], dim=1)
                    padded_mask = torch.cat([
                        inp['attention_mask'],
                        torch.zeros((1, pad_len), dtype=inp['attention_mask'].dtype)
                    ], dim=1)
                else:
                    padded_ids = inp['input_ids']
                    padded_mask = inp['attention_mask']

                batched_input_ids.append(padded_ids)
                batched_attention_mask.append(padded_mask)

            prompt_inputs = {
                'input_ids': torch.cat(batched_input_ids, dim=0),
                'attention_mask': torch.cat(batched_attention_mask, dim=0),
            }

        # Move to device
        prompt_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in prompt_inputs.items()}

        prompt_input_ids = prompt_inputs['input_ids']
        prompt_attention_mask = prompt_inputs['attention_mask']

        # Concatenate labels to prompts
        max_label_len = max(len(tokens) for tokens in all_label_tokens)
        full_input_ids = []
        full_attention_mask = []
        label_start_positions = []

        for i in range(batch_size):
            prompt_ids = prompt_input_ids[i]
            prompt_mask = prompt_attention_mask[i]
            prompt_len = prompt_mask.sum().item()
            label_start_positions.append(prompt_len - 1)

            label_tokens = all_label_tokens[i]
            padded_label = label_tokens + [self.processor.tokenizer.pad_token_id] * (max_label_len - len(label_tokens))

            full_ids = torch.cat([
                prompt_ids,
                torch.tensor(padded_label, device=self.device)
            ])

            label_mask = [1] * len(label_tokens) + [0] * (max_label_len - len(label_tokens))
            full_mask = torch.cat([
                prompt_mask,
                torch.tensor(label_mask, device=self.device, dtype=prompt_mask.dtype)
            ])

            full_input_ids.append(full_ids)
            full_attention_mask.append(full_mask)

        full_input_ids = torch.stack(full_input_ids)
        full_attention_mask = torch.stack(full_attention_mask)

        # Stack image features into batch
        # Each is (1, num_images, 64, 4096), stack along dim 0
        image_hidden_states = torch.cat(image_hidden_states_list, dim=0)

        # Forward pass with cached features
        model_inputs = {
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'image_hidden_states': image_hidden_states,  # Use cached features!
            'use_cache': self.use_cache,
        }

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with autocast(dtype=torch.float16):
                    outputs = self.model(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

        # Extract log probabilities
        logits = outputs.logits
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)

        batch_log_probs = []
        for i in range(batch_size):
            label_tokens = all_label_tokens[i]
            if len(label_tokens) == 0:
                batch_log_probs.append(-np.inf)
                continue

            start_pos = label_start_positions[i]
            log_prob_sum = 0.0

            for j, token_id in enumerate(label_tokens):
                pos = start_pos + j
                log_prob_sum += log_probs_all[i, pos, token_id].item()

            # Mean per-token log prob -- see _compute_label_probabilities_batch for why.
            batch_log_probs.append(log_prob_sum / len(label_tokens))

        return batch_log_probs

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

    def classify_with_context(
        self,
        query_image: Image.Image,
        context_examples: List[Tuple[Image.Image, str]],
        candidate_labels: List[str],
        batch_size: int = 8
    ) -> str:
        """
        Classify an image using in-context learning.

        Args:
            query_image: Query image to classify
            context_examples: List of (image, label_text) tuples for ICL context
            candidate_labels: List of candidate label names to choose from
            batch_size: Number of candidates to process in parallel (default: 8)
                       Lower this if you get OOM errors with many candidates

        Returns:
            Predicted label text (the candidate with highest log probability)
        """
        # Format prompt with context examples
        example_labels = [label for _, label in context_examples] if context_examples else None
        prompt = self.format_prompt(example_labels=example_labels)

        # Prepare images: context images + query image
        images = [img for img, _ in context_examples] + [query_image]

        # Encode images once and reuse the hidden states across all candidate chunks.
        # This avoids re-running the vision encoder once per candidate label.
        image_hidden_states = self._get_image_features_multi(images)

        # Process candidates in chunks to avoid OOM with many classes
        all_log_probs = []
        for i in range(0, len(candidate_labels), batch_size):
            chunk_labels = candidate_labels[i:i + batch_size]
            chunk_size = len(chunk_labels)

            chunk_log_probs = self._compute_label_probabilities_batch_with_features(
                image_hidden_states_list=[image_hidden_states] * chunk_size,
                prompts=[prompt] * chunk_size,
                labels=chunk_labels
            )
            all_log_probs.extend(chunk_log_probs)

        # Find label with highest log probability
        best_idx = max(range(len(all_log_probs)), key=lambda i: all_log_probs[i])
        return candidate_labels[best_idx]

    def classify_with_context_generative(
        self,
        query_image: Image.Image,
        context_examples: List[Tuple[Image.Image, str]],
        candidate_labels: List[str],
        max_new_tokens: int = 50
    ) -> str:
        """
        Classify an image using generative decoding (free-form generation).

        This method lets the model generate text freely, then matches the output
        to the closest candidate label.

        Args:
            query_image: Query image to classify
            context_examples: List of (image, label_text) tuples for ICL context
            candidate_labels: List of candidate label names to match against
            max_new_tokens: Maximum tokens to generate (default: 50)

        Returns:
            Predicted label text (matched to closest candidate)
        """
        # Format prompt with context examples and candidate labels
        example_labels = [label for _, label in context_examples] if context_examples else None
        prompt = self.format_prompt(example_labels=example_labels, candidate_labels=candidate_labels)

        # Prepare images: context images + query image
        images = [img for img, _ in context_examples] + [query_image]

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        ).to(self.device)

        # Encode stop sequences (newline to prevent continuing after answer)
        stop_str = "\n\n"
        stop_token_ids = self.processor.tokenizer.encode(stop_str, add_special_tokens=False)

        # Generate
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with autocast(dtype=torch.float16):
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding for consistency
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=[self.processor.tokenizer.eos_token_id] + stop_token_ids
                    )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=[self.processor.tokenizer.eos_token_id] + stop_token_ids
                )

        # Decode generated text
        # Remove the input prompt tokens to get only the generated part
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Log the raw generation for debugging
        print(f"[RAW GENERATION] '{generated_text}'")

        # Take only the first line (before any newline) to avoid prompt repetition
        if '\n' in generated_text:
            generated_text = generated_text.split('\n')[0].strip()
            print(f"[CLEANED] '{generated_text}'")

        generated_lower = generated_text.lower()

        # Match to closest candidate label
        # Try exact match first (case-insensitive)
        for label in candidate_labels:
            if label.lower().strip() == generated_lower:
                return label

        # Try numbered format match (e.g., "5" or "5." or "5. golden retriever")
        number_match = re.match(r'^(\d+)\.?\s*(.*)', generated_text)
        if number_match:
            number = int(number_match.group(1))
            # Check if number is valid index (1-indexed)
            if 1 <= number <= len(candidate_labels):
                return candidate_labels[number - 1]
            # If there's text after number, try matching that
            text_after = number_match.group(2).strip()
            if text_after:
                for label in candidate_labels:
                    if label.lower() == text_after.lower():
                        return label

        # Try substring match (check if label appears in generated text)
        for label in candidate_labels:
            if label.lower() in generated_lower:
                return label

        # Try reverse substring match (check if generated text appears in label)
        for label in candidate_labels:
            if generated_lower in label.lower():
                return label

        # Log warning if no match found
        print(f"Warning: Generated text '{generated_text}' didn't match any candidate. Using first candidate as fallback.")
        return candidate_labels[0] if candidate_labels else ""
