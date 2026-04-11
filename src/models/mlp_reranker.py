"""
MLP-based reranker model for predicting marginal utilities.

Simple MLP that concatenates query and example CLIP embeddings (+ similarity)
and predicts the marginal utility score through feedforward layers.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.marginal_utility_dataset import InteractionFeaturesConfig


class MLPReranker(nn.Module):
    """
    MLP-based reranker for predicting marginal utility from CLIP embeddings.

    Architecture:
        Input: [query_emb (512-d), example_emb (512-d), similarity (1-d)] → 1025-d
        Hidden layers: [1025 → 512 → 256 → 128]
        Output: scalar utility prediction

    Uses ReLU activations and dropout for regularization.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        interaction_features: Optional[InteractionFeaturesConfig] = None,
        use_sigmoid: bool = False
    ):
        """
        Initialize reranker model.

        Args:
            embedding_dim: Dimension of CLIP embeddings (default: 512 for ViT-B/32)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            interaction_features: Configuration for interaction features
            use_sigmoid: If True, apply sigmoid activation to output (for BCE loss)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.interaction_features = interaction_features or InteractionFeaturesConfig()
        self.use_sigmoid = use_sigmoid

        # Calculate input dimension based on enabled features
        # Base: query_emb (512) + example_emb (512) + similarity (1) = 1025
        input_dim = 2 * embedding_dim + 1

        # Add interaction features
        input_dim += self.interaction_features.num_features

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (single utility score)
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        query_emb: torch.Tensor,
        example_emb: torch.Tensor,
        similarity: torch.Tensor,
        product: Optional[torch.Tensor] = None,
        difference: Optional[torch.Tensor] = None,
        l2_distance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query_emb: Query CLIP embeddings, shape (batch_size, embedding_dim)
            example_emb: Example CLIP embeddings, shape (batch_size, embedding_dim)
            similarity: CLIP similarity scores, shape (batch_size, 1)
            product: Element-wise product (optional), shape (batch_size, embedding_dim)
            difference: Element-wise difference (optional), shape (batch_size, embedding_dim)
            l2_distance: L2 distance (optional), shape (batch_size, 1)

        Returns:
            Predicted utilities, shape (batch_size, 1)
        """
        # Build input tensor dynamically
        inputs = [query_emb, example_emb, similarity]

        # Add interaction features in the same order as InteractionFeaturesConfig
        if self.interaction_features.use_product and product is not None:
            inputs.append(product)
        if self.interaction_features.use_difference and difference is not None:
            inputs.append(difference)
        if self.interaction_features.use_l2_distance and l2_distance is not None:
            inputs.append(l2_distance)

        # Concatenate all inputs
        x = torch.cat(inputs, dim=1)

        # Pass through MLP
        utility = self.mlp(x)  # (batch, 1)

        # Apply sigmoid if enabled (for BCE loss with normalized utilities)
        if self.use_sigmoid:
            utility = torch.sigmoid(utility)

        return utility

    def predict_utilities(
        self,
        query_emb: torch.Tensor,
        example_embs: torch.Tensor,
        similarities: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch predict utilities for multiple examples given a single query.

        Useful for ranking: given one query and K candidate examples,
        predict utilities for all K pairs.

        Args:
            query_emb: Single query embedding, shape (embedding_dim,) or (1, embedding_dim)
            example_embs: Multiple example embeddings, shape (K, embedding_dim)
            similarities: CLIP similarities, shape (K, 1) or (K,)

        Returns:
            Predicted utilities, shape (K, 1)
        """
        # Ensure batch dimension
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # (1, embedding_dim)

        if similarities.dim() == 1:
            similarities = similarities.unsqueeze(1)  # (K, 1)

        # Repeat query for all examples
        K = example_embs.size(0)
        query_emb_repeated = query_emb.repeat(K, 1)  # (K, embedding_dim)

        # Forward pass
        utilities = self.forward(query_emb_repeated, example_embs, similarities)

        return utilities

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing MLPReranker model...")
    print("=" * 70)

    # Create model
    model = MLPReranker(
        embedding_dim=512,
        hidden_dims=[512, 256, 128],
        dropout=0.1
    )

    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 4
    query_emb = torch.randn(batch_size, 512)
    example_emb = torch.randn(batch_size, 512)
    similarity = torch.randn(batch_size, 1)

    print(f"\nTest forward pass:")
    print(f"  Query embeddings: {query_emb.shape}")
    print(f"  Example embeddings: {example_emb.shape}")
    print(f"  Similarities: {similarity.shape}")

    output = model(query_emb, example_emb, similarity)
    print(f"  Output utilities: {output.shape}")
    print(f"  Sample values: {output[:3].squeeze().tolist()}")

    # Test batch prediction
    print(f"\nTest batch prediction (1 query, K examples):")
    single_query = torch.randn(512)
    K = 10
    multiple_examples = torch.randn(K, 512)
    multiple_sims = torch.randn(K, 1)

    utilities = model.predict_utilities(single_query, multiple_examples, multiple_sims)
    print(f"  Query: {single_query.shape}")
    print(f"  Examples: {multiple_examples.shape}")
    print(f"  Predicted utilities: {utilities.shape}")
    print(f"  Top-3 utilities: {utilities[:3].squeeze().tolist()}")

    print("\n✓ Model test passed!")
