"""
Cross-attention based reranker for predicting marginal utilities.

Unlike the MLP-based MLPReranker that concatenates embeddings,
this model uses cross-attention to allow query and example embeddings
to interact, learning which parts are relevant to each other.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class CrossAttentionReranker(nn.Module):
    """
    Cross-attention based reranker for predicting marginal utility.

    Architecture:
        1. Project query and example embeddings to a common hidden dimension
        2. Apply cross-attention layers:
           - Query attends to example (learns what in the example is relevant to the query)
           - Example attends to query (learns what in the query is relevant to the example)
        3. Combine attended representations with similarity score
        4. Pass through feedforward layers to predict utility

    Key advantage over MLP concatenation:
        - The model can focus on specific parts of embeddings rather than treating
          the entire concatenated vector uniformly
        - Attention weights provide interpretability (which features matter most)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        feedforward_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        use_sigmoid: bool = False
    ):
        """
        Initialize cross-attention reranker.

        Args:
            embedding_dim: Dimension of CLIP embeddings (default: 512 for ViT-B/32)
            hidden_dim: Hidden dimension for attention (must be divisible by num_attention_heads)
            num_attention_heads: Number of attention heads (typically 4, 8, or 16)
            num_layers: Number of cross-attention layers to stack
            feedforward_dims: List of feedforward layer dimensions after attention
            dropout: Dropout probability for regularization
            use_sigmoid: If True, apply sigmoid activation to output (for BCE loss)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_sigmoid = use_sigmoid

        # Validate that hidden_dim is divisible by num_attention_heads
        assert hidden_dim % num_attention_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_attention_heads ({num_attention_heads})"

        # Step 1: Project CLIP embeddings to hidden dimension
        # This allows the model to transform the frozen CLIP features
        self.query_projection = nn.Linear(embedding_dim, hidden_dim)
        self.example_projection = nn.Linear(embedding_dim, hidden_dim)

        # Step 2: Build cross-attention layers
        # Each layer performs bidirectional cross-attention
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Layer normalization after all attention layers
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.example_norm = nn.LayerNorm(hidden_dim)

        # Step 3: Combine attended representations with similarity score
        # Input: query_attended (hidden_dim) + example_attended (hidden_dim) + similarity (1)
        combine_input_dim = 2 * hidden_dim + 1

        # Step 4: Build feedforward layers for final utility prediction
        layers = []
        prev_dim = combine_input_dim

        for ff_dim in feedforward_dims:
            layers.extend([
                nn.Linear(prev_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = ff_dim

        # Output layer (single utility score)
        layers.append(nn.Linear(prev_dim, 1))

        self.feedforward = nn.Sequential(*layers)

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
        Forward pass with cross-attention.

        Args:
            query_emb: Query CLIP embeddings, shape (batch_size, embedding_dim)
            example_emb: Example CLIP embeddings, shape (batch_size, embedding_dim)
            similarity: CLIP similarity scores, shape (batch_size, 1)
            product: Unused (kept for API compatibility with MLPReranker)
            difference: Unused (kept for API compatibility with MLPReranker)
            l2_distance: Unused (kept for API compatibility with MLPReranker)

        Returns:
            Predicted utilities, shape (batch_size, 1)
        """
        batch_size = query_emb.size(0)

        # Step 1: Project embeddings to hidden dimension
        query_hidden = self.query_projection(query_emb)  # (batch, hidden_dim)
        example_hidden = self.example_projection(example_emb)  # (batch, hidden_dim)

        # Add sequence dimension for attention
        # PyTorch attention expects (batch, seq_len, hidden_dim)
        # We treat each embedding as a single token, so seq_len=1
        query_seq = query_hidden.unsqueeze(1)  # (batch, 1, hidden_dim)
        example_seq = example_hidden.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Step 2: Apply cross-attention layers
        # Each layer updates both query and example through bidirectional attention
        for cross_attn_layer in self.cross_attention_layers:
            query_seq, example_seq = cross_attn_layer(query_seq, example_seq)

        # Remove sequence dimension: (batch, 1, hidden_dim) → (batch, hidden_dim)
        query_attended = query_seq.squeeze(1)
        example_attended = example_seq.squeeze(1)

        # Apply final layer normalization
        query_attended = self.query_norm(query_attended)
        example_attended = self.example_norm(example_attended)

        # Step 3: Combine attended representations with similarity score
        combined = torch.cat([query_attended, example_attended, similarity], dim=1)

        # Step 4: Pass through feedforward network to predict utility
        utility = self.feedforward(combined)  # (batch, 1)

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


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with bidirectional attention.

    Performs:
        1. Query attends to example (query2example attention)
           - "What parts of the example are relevant given this query?"
        2. Example attends to query (example2query attention)
           - "What parts of the query make this example useful?"

    Each attended output is combined with residual connections and layer normalization,
    following the Transformer architecture pattern.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention layer.

        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-head attention: query attends to example
        # Query is the "question", example is the "knowledge base"
        self.query2example_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq_len, hidden_dim)
        )

        # Multi-head attention: example attends to query
        # Example is the "question", query is the "knowledge base"
        self.example2query_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization for residual connections (post-attention)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.example_norm = nn.LayerNorm(hidden_dim)

        # Feedforward network after attention (standard Transformer FFN)
        # Typically expands to 4x hidden_dim, then projects back
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

        # Layer normalization after feedforward (post-FFN)
        self.query_ffn_norm = nn.LayerNorm(hidden_dim)
        self.example_ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_seq: torch.Tensor,
        example_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional cross-attention.

        Args:
            query_seq: Query sequence, shape (batch, seq_len, hidden_dim)
            example_seq: Example sequence, shape (batch, seq_len, hidden_dim)

        Returns:
            (attended_query, attended_example): Both shape (batch, seq_len, hidden_dim)
        """
        # Part 1: Query attends to example
        # The query "asks" the example for relevant information
        query_attended, _ = self.query2example_attn(
            query=query_seq,      # What we want to update
            key=example_seq,      # What we attend over
            value=example_seq     # What we extract information from
        )

        # Residual connection + layer normalization (standard Transformer pattern)
        query_seq = self.query_norm(query_seq + query_attended)

        # Feedforward network (allows non-linear transformation)
        query_ffn_out = self.query_ffn(query_seq)
        query_seq = self.query_ffn_norm(query_seq + query_ffn_out)

        # Part 2: Example attends to query
        # The example "asks" the query what makes it useful
        example_attended, _ = self.example2query_attn(
            query=example_seq,    # What we want to update
            key=query_seq,        # What we attend over (updated query from step 1)
            value=query_seq       # What we extract information from
        )

        # Residual connection + layer normalization
        example_seq = self.example_norm(example_seq + example_attended)

        # Feedforward network
        example_ffn_out = self.example_ffn(example_seq)
        example_seq = self.example_ffn_norm(example_seq + example_ffn_out)

        return query_seq, example_seq


if __name__ == "__main__":
    # Quick test of the cross-attention architecture
    print("Testing CrossAttentionReranker model...")
    print("=" * 70)

    # Create model with reasonable defaults
    model = CrossAttentionReranker(
        embedding_dim=512,
        hidden_dim=256,
        num_attention_heads=8,
        num_layers=2,
        feedforward_dims=[256, 128],
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
