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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_cache: bool = True,
        cache_vision_embeddings: bool = False,  # Disabled by default for Idefics2
        max_vision_cache_size: int = 5000,
    ):
        """
        Initialize Idefics2 model.

        Args:
            model_name: HuggingFace model name (default: idefics2-8b)
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            use_cache: Whether to use KV cache for generation
            cache_vision_embeddings: Whether to cache vision encoder outputs
            max_vision_cache_size: Maximum number of images to cache (0 = unlimited)
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
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

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

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
            if device.startswith("cuda"):
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

        # Stack the inputs (simple approach for now - could optimize later)
        # For now, process one at a time to avoid complex batching logic
        prompt_inputs = all_prompt_inputs[0] if batch_size == 1 else None

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

        # Process candidates in chunks to avoid OOM with many classes
        all_log_probs = []
        for i in range(0, len(candidate_labels), batch_size):
            chunk_labels = candidate_labels[i:i + batch_size]

            # Batch compute log probabilities for this chunk
            batch_images = [images] * len(chunk_labels)
            batch_prompts = [prompt] * len(chunk_labels)

            chunk_log_probs = self._compute_label_probabilities_batch(
                images=batch_images,
                prompts=batch_prompts,
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
