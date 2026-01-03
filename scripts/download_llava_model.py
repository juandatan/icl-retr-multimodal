"""
Download LLaVA model ahead of time to avoid SSL issues during experiments.

This script pre-downloads the LLaVA-1.5-7B model to the HuggingFace cache.
Useful if you're experiencing network issues or want to run experiments offline.

Usage:
    python scripts/download_llava_model.py
    python scripts/download_llava_model.py --model llava-hf/llava-1.5-13b-hf
"""

import argparse
import os
from pathlib import Path
from transformers import AutoProcessor, LlavaForConditionalGeneration


def download_model(model_name: str):
    """Download LLaVA model and processor to HuggingFace cache."""
    # Get HuggingFace cache directory
    cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or Path.home() / '.cache' / 'huggingface'

    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("This may take a while (~14GB for 7B model)...\n")

    print("1. Downloading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    print("✓ Processor downloaded and cached\n")

    print("2. Downloading model weights...")
    print("   Note: Model will be cached automatically by HuggingFace")
    print("   Future runs will load from cache instead of re-downloading\n")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="float16",
        device_map="cpu",  # Just download, don't load to GPU
        low_cpu_mem_usage=True
    )
    print("✓ Model downloaded and cached\n")

    print(f"{'='*70}")
    print(f"✓ Successfully downloaded {model_name}")
    print(f"✓ Cached to: {cache_dir}")
    print(f"✓ Future runs will use cached model automatically")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LLaVA model")
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model name"
    )
    args = parser.parse_args()

    download_model(args.model)
