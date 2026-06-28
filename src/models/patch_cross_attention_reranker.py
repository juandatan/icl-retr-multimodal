"""
Patch-level cross-attention reranker for predicting marginal utilities.

Instead of operating on pooled CLIP embeddings, this model:
1. Extracts patch-level features from CLIP ViT (before pooling)
2. Applies cross-attention between query patches and example patches
3. Pools the attended representations to predict utility
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import clip

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class CLIPPatchExtractor(nn.Module):
    """Lightweight CLIP patch feature extractor (no trainable parameters)."""

    def __init__(self, clip_model_name: str = "ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features from CLIP ViT (before pooling)."""
        x = self.clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.ln_post(x)
        # Project each patch token into CLIP's semantic embedding space
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj.to(x.dtype)
        return x


class LearnablePooling(nn.Module):
    """
    Attention-weighted pooling over patch tokens.

    Learns a per-patch scalar weight conditioned on both query and example
    attended features, then returns a weighted sum over spatial patches.
    This lets the model suppress uninformative patches rather than averaging
    all of them equally.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Single linear layer maps each patch token to a scalar attention logit
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: (batch, seq_len, hidden_dim) — includes CLS at index 0

        Returns:
            Pooled vector: (batch, hidden_dim)
        """
        # Operate only on spatial patches (skip CLS token at index 0)
        spatial = patch_features[:, 1:, :]          # (batch, num_patches, hidden_dim)
        logits = self.score(spatial)                 # (batch, num_patches, 1)
        weights = torch.softmax(logits, dim=1)       # (batch, num_patches, 1)
        return (weights * spatial).sum(dim=1)        # (batch, hidden_dim)


class PatchCrossAttentionReranker(nn.Module):
    """
    Cross-attention reranker operating on image patches.

    Architecture:
        1. Extract patch features from CLIP ViT (before final pooling)
        2. Apply cross-attention between query and example patches
        3. Pool attended patches (e.g., mean/max pooling or CLS token)
        4. Predict utility from pooled representations
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
        feedforward_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
        freeze_clip: bool = True,
        pooling_method: str = "mean"
    ):
        """
        Initialize patch-level cross-attention reranker.

        Args:
            clip_model_name: CLIP model to use (e.g., "ViT-B/32", "ViT-L/14")
            hidden_dim: Hidden dimension for attention
            num_attention_heads: Number of attention heads
            num_attention_layers: Number of cross-attention layers
            feedforward_dims: FFN dimensions after pooling
            dropout: Dropout probability
            use_sigmoid: Apply sigmoid to output
            freeze_clip: If True, freeze CLIP weights
            pooling_method: How to pool patches ("mean", "max", "cls")
        """
        super().__init__()

        self.clip_model_name = clip_model_name
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        self.dropout = dropout
        self.use_sigmoid = use_sigmoid
        self.pooling_method = pooling_method
        feedforward_dims = feedforward_dims if feedforward_dims is not None else [256, 128]

        # Load CLIP model
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")

        # Freeze CLIP weights if requested
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Projected embedding dim (512 for ViT-B/32); falls back to transformer width
        # if the model has no projection (e.g. some non-standard checkpoints)
        proj = self.clip_model.visual.proj
        self.clip_embed_dim = proj.shape[1] if proj is not None else self.clip_model.visual.transformer.width

        # Project CLIP patch features to hidden dimension
        self.query_projection = nn.Linear(self.clip_embed_dim, hidden_dim)
        self.example_projection = nn.Linear(self.clip_embed_dim, hidden_dim)

        # Build cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_attention_layers)
        ])

        # Layer normalization after attention
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.example_norm = nn.LayerNorm(hidden_dim)

        # Learnable pooling (only instantiated when pooling_method == "learned")
        if pooling_method == "learned":
            self.query_pooler = LearnablePooling(hidden_dim)
            self.example_pooler = LearnablePooling(hidden_dim)

        # Feedforward network for utility prediction
        # Input: query_pooled + example_pooled + similarity
        combine_input_dim = 2 * hidden_dim + 1

        layers = []
        prev_dim = combine_input_dim
        for ff_dim in feedforward_dims:
            layers.extend([
                nn.Linear(prev_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = ff_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.feedforward = nn.Sequential(*layers)

    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from CLIP ViT.

        Args:
            images: Preprocessed images, shape (batch, 3, H, W)

        Returns:
            Patch features, shape (batch, num_patches, embed_dim)
        """
        # Get patch embeddings from CLIP ViT (before final pooling)
        x = self.clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Add CLS token and position embeddings
        x = torch.cat([
            self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)

        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer

        # Pass through transformer blocks
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.visual.ln_post(x)
        # Project each patch token into CLIP's semantic embedding space
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj.to(x.dtype)

        return x

    def pool_patches(self, patch_features: torch.Tensor, pooler: Optional['LearnablePooling'] = None) -> torch.Tensor:
        """Pool patch features into a single vector."""
        if self.pooling_method == "cls":
            return patch_features[:, 0, :]
        elif self.pooling_method == "mean":
            return patch_features[:, 1:, :].mean(dim=1)
        elif self.pooling_method == "max":
            return patch_features[:, 1:, :].max(dim=1)[0]
        elif self.pooling_method == "learned":
            assert pooler is not None
            return pooler(patch_features)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def forward(
        self,
        query_input: torch.Tensor,
        example_input: torch.Tensor,
        similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with patch-level cross-attention.

        Args:
            query_input: Either query images (batch, 3, H, W) or pre-extracted
                         patch features (batch, num_patches, embed_dim)
            example_input: Either example images (batch, 3, H, W) or pre-extracted
                           patch features (batch, num_patches, embed_dim)
            similarity: CLIP similarity scores, shape (batch, 1)

        Returns:
            Predicted utilities, shape (batch, 1)
        """
        # Detect if input is images (4D) or pre-extracted patches (3D)
        if query_input.dim() == 4:
            query_patches = self.extract_patch_features(query_input)
            example_patches = self.extract_patch_features(example_input)
        else:
            query_patches = query_input
            example_patches = example_input

        # Project to hidden dimension
        query_hidden = self.query_projection(query_patches)
        example_hidden = self.example_projection(example_patches)

        # Apply cross-attention layers
        for cross_attn_layer in self.cross_attention_layers:
            query_hidden, example_hidden = cross_attn_layer(query_hidden, example_hidden)

        # Apply layer normalization
        query_hidden = self.query_norm(query_hidden)
        example_hidden = self.example_norm(example_hidden)

        # Pool patches
        query_pooler = getattr(self, 'query_pooler', None)
        example_pooler = getattr(self, 'example_pooler', None)
        query_pooled = self.pool_patches(query_hidden, query_pooler)
        example_pooled = self.pool_patches(example_hidden, example_pooler)

        # Combine with similarity
        combined = torch.cat([query_pooled, example_pooled, similarity], dim=1)

        # Predict utility
        utility = self.feedforward(combined)

        if self.use_sigmoid:
            utility = torch.sigmoid(utility)

        return utility

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for patch-level features."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.query2example_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.example2query_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.example_norm = nn.LayerNorm(hidden_dim)

        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.example_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.query_ffn_norm = nn.LayerNorm(hidden_dim)
        self.example_ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_seq: torch.Tensor,
        example_seq: torch.Tensor
    ) -> tuple:
        """Apply bidirectional cross-attention."""
        # Query attends to example
        query_attended, _ = self.query2example_attn(
            query=query_seq,
            key=example_seq,
            value=example_seq
        )
        query_seq = self.query_norm(query_seq + query_attended)
        query_ffn_out = self.query_ffn(query_seq)
        query_seq = self.query_ffn_norm(query_seq + query_ffn_out)

        # Example attends to query
        example_attended, _ = self.example2query_attn(
            query=example_seq,
            key=query_seq,
            value=query_seq
        )
        example_seq = self.example_norm(example_seq + example_attended)
        example_ffn_out = self.example_ffn(example_seq)
        example_seq = self.example_ffn_norm(example_seq + example_ffn_out)

        return query_seq, example_seq
