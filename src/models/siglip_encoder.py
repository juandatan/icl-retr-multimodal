"""
Standalone SigLIP encoder for image-to-text distractor-set construction.

Idefics2's vision tower architecturally matches SigLIP-SO400M, but Idefics2 has
no matched text tower -- so a standalone SigLIP checkpoint (with both towers) is
used here to compute image-to-text similarity for identifying confusable classes.
This is separate from Idefics2Wrapper (different model family, different purpose:
distractor-set construction, not classification) and separate from the CLIP
embeddings used for actual candidate-example retrieval.
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SiglipEncoder:
    """Wraps a HuggingFace SigLIP model for batched image/text embedding."""

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        """L2-normalized image embeddings, shape (N, D)."""
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
        return np.vstack(embeddings)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """L2-normalized text embeddings, shape (N, D)."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
        return np.vstack(embeddings)
