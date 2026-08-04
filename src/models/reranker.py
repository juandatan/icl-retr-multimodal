"""Frozen-feature architectures for multimodal ICL exemplar reranking."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RerankerConfig:
    """Serializable architecture parameters for a reranker checkpoint."""

    clip_dim: int
    siglip_dim: int
    architecture: str = "interaction_mlp"
    hidden_dim: int = 256
    metadata_dim: int = 64
    dropout: float = 0.1
    use_clip_embeddings: bool = False
    use_clip_similarity: bool = False
    use_retrieval_rank: bool = False
    use_derived_siglip_similarities: bool = False
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 1024
    candidate_context_layers: int = 1
    candidate_context_heads: int = 4
    candidate_context_ff_dim: int = 512

    def __post_init__(self) -> None:
        architectures = {
            "interaction_mlp",
            "pooled_transformer",
            "cross_candidate_attention",
        }
        if self.architecture not in architectures:
            raise ValueError(
                f"architecture must be one of {sorted(architectures)}"
            )
        dimensions = (
            self.clip_dim,
            self.siglip_dim,
            self.hidden_dim,
            self.metadata_dim,
            self.transformer_layers,
            self.transformer_heads,
            self.transformer_ff_dim,
            self.candidate_context_layers,
            self.candidate_context_heads,
            self.candidate_context_ff_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("All reranker dimensions and layer counts must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.hidden_dim % self.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if self.hidden_dim % self.candidate_context_heads:
            raise ValueError(
                "hidden_dim must be divisible by candidate_context_heads"
            )


class _ProjectionTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class _InteractionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def _pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat((left, right, left * right, torch.abs(left - right)), dim=-1)


class LabelAwareReranker(nn.Module):
    """Score each exemplar using only inference-available frozen features.

    The minimal controlled model consumes query-image, exemplar-image, and
    exemplar-label SigLIP embeddings. CLIP and scalar retrieval priors are
    independently switchable ablations. The default architectures score
    candidates independently; ``cross_candidate_attention`` adds masked,
    permutation-equivariant context across the candidate pool. No teacher
    target or query label enters any model input.
    """

    def __init__(self, config: RerankerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.siglip_image_tower = _ProjectionTower(
            config.siglip_dim, hidden, config.dropout
        )
        self.siglip_label_tower = _ProjectionTower(
            config.siglip_dim, hidden, config.dropout
        )
        if config.architecture in {
            "interaction_mlp",
            "cross_candidate_attention",
        }:
            self.siglip_interactions = _InteractionEncoder(
                12 * hidden, hidden, config.dropout
            )
            self.score_token = None
            self.role_embeddings = None
            self.transformer = None
        else:
            self.siglip_interactions = None
            self.score_token = nn.Parameter(torch.empty(1, 1, hidden))
            self.role_embeddings = nn.Parameter(torch.empty(1, 4, hidden))
            nn.init.normal_(self.score_token, std=0.02)
            nn.init.normal_(self.role_embeddings, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.transformer_heads,
                dim_feedforward=config.transformer_ff_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=config.transformer_layers,
                norm=nn.LayerNorm(hidden),
                enable_nested_tensor=False,
            )

        if config.architecture == "cross_candidate_attention":
            context_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.candidate_context_heads,
                dim_feedforward=config.candidate_context_ff_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.candidate_context = nn.TransformerEncoder(
                context_layer,
                num_layers=config.candidate_context_layers,
                norm=nn.LayerNorm(hidden),
                enable_nested_tensor=False,
            )
        else:
            self.candidate_context = None

        if config.use_clip_embeddings:
            self.clip_image_tower = _ProjectionTower(
                config.clip_dim, hidden, config.dropout
            )
            self.clip_interactions = _InteractionEncoder(
                4 * hidden, hidden, config.dropout
            )
        else:
            self.clip_image_tower = None
            self.clip_interactions = None

        metadata_count = sum((
            config.use_clip_similarity,
            config.use_retrieval_rank,
            2 * config.use_derived_siglip_similarities,
        ))
        if metadata_count:
            self.metadata_encoder = nn.Sequential(
                nn.LayerNorm(metadata_count),
                nn.Linear(metadata_count, config.metadata_dim),
                nn.GELU(),
            )
        else:
            self.metadata_encoder = None

        fusion_dim = hidden
        if config.use_clip_embeddings:
            fusion_dim += hidden
        if metadata_count:
            fusion_dim += config.metadata_dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 2 * hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )

    def _validate_inputs(
        self,
        query_clip: torch.Tensor,
        candidate_clip: torch.Tensor,
        query_siglip: torch.Tensor,
        candidate_siglip: torch.Tensor,
        candidate_label_siglip: torch.Tensor,
        clip_similarities: torch.Tensor,
        retrieval_ranks: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if query_clip.ndim != 2 or query_siglip.ndim != 2:
            raise ValueError("Query features must have shape [batch, feature_dim]")
        if (
            candidate_clip.ndim != 3
            or candidate_siglip.ndim != 3
            or candidate_label_siglip.ndim != 3
        ):
            raise ValueError(
                "Candidate features must have shape [batch, candidates, feature_dim]"
            )
        batch, candidates = candidate_siglip.shape[:2]
        expected_prefix = (batch, candidates)
        if (
            query_clip.shape != (batch, self.config.clip_dim)
            or candidate_clip.shape != (*expected_prefix, self.config.clip_dim)
            or query_siglip.shape != (batch, self.config.siglip_dim)
            or candidate_siglip.shape != (*expected_prefix, self.config.siglip_dim)
            or candidate_label_siglip.shape
            != (*expected_prefix, self.config.siglip_dim)
            or clip_similarities.shape != expected_prefix
            or retrieval_ranks.shape != expected_prefix
            or candidate_mask.shape != expected_prefix
        ):
            raise ValueError("Reranker input shapes do not match its configuration")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("Every query must have at least one valid candidate")
        return batch, candidates

    def _siglip_representation(
        self,
        query: torch.Tensor,
        candidate: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        candidates = candidate.shape[1]
        query_projected = self.siglip_image_tower(query).unsqueeze(1).expand(
            -1, candidates, -1
        )
        candidate_projected = self.siglip_image_tower(candidate)
        label_projected = self.siglip_label_tower(label)
        if self.config.architecture in {
            "interaction_mlp",
            "cross_candidate_attention",
        }:
            return self.siglip_interactions(torch.cat((
                _pair_features(query_projected, candidate_projected),
                _pair_features(query_projected, label_projected),
                _pair_features(candidate_projected, label_projected),
            ), dim=-1))

        batch = query.shape[0]
        score = self.score_token.expand(batch * candidates, -1, -1)
        content = torch.stack(
            (query_projected, candidate_projected, label_projected), dim=2
        ).reshape(batch * candidates, 3, -1)
        tokens = torch.cat((score, content), dim=1) + self.role_embeddings
        return self.transformer(tokens)[:, 0].reshape(batch, candidates, -1)

    def forward(
        self,
        *,
        query_clip: torch.Tensor,
        candidate_clip: torch.Tensor,
        query_siglip: torch.Tensor,
        candidate_siglip: torch.Tensor,
        candidate_label_siglip: torch.Tensor,
        clip_similarities: torch.Tensor,
        retrieval_ranks: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one unbounded utility score per candidate."""
        _, candidates = self._validate_inputs(
            query_clip,
            candidate_clip,
            query_siglip,
            candidate_siglip,
            candidate_label_siglip,
            clip_similarities,
            retrieval_ranks,
            candidate_mask,
        )
        query_siglip = F.normalize(query_siglip, dim=-1)
        candidate_siglip = F.normalize(candidate_siglip, dim=-1)
        candidate_label_siglip = F.normalize(candidate_label_siglip, dim=-1)
        siglip_representation = self._siglip_representation(
            query_siglip, candidate_siglip, candidate_label_siglip
        )
        if self.candidate_context is not None:
            siglip_representation = self.candidate_context(
                siglip_representation,
                src_key_padding_mask=~candidate_mask,
            )
            siglip_representation = siglip_representation.masked_fill(
                ~candidate_mask.unsqueeze(-1), 0.0
            )
        representations = [siglip_representation]

        if self.config.use_clip_embeddings:
            query_clip = F.normalize(query_clip, dim=-1)
            candidate_clip = F.normalize(candidate_clip, dim=-1)
            query_projected = self.clip_image_tower(query_clip).unsqueeze(1).expand(
                -1, candidates, -1
            )
            candidate_projected = self.clip_image_tower(candidate_clip)
            representations.append(self.clip_interactions(
                _pair_features(query_projected, candidate_projected)
            ))

        metadata = []
        if self.config.use_clip_similarity:
            metadata.append(clip_similarities)
        if self.config.use_retrieval_rank:
            metadata.append(retrieval_ranks)
        if self.config.use_derived_siglip_similarities:
            metadata.extend((
                torch.sum(
                    query_siglip.unsqueeze(1) * candidate_label_siglip, dim=-1
                ),
                torch.sum(candidate_siglip * candidate_label_siglip, dim=-1),
            ))
        if metadata:
            representations.append(self.metadata_encoder(torch.stack(metadata, dim=-1)))

        return self.scorer(torch.cat(representations, dim=-1)).squeeze(-1)

    @staticmethod
    def select(scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """Return each query's highest-scoring valid candidate position."""
        if scores.shape != candidate_mask.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("scores and boolean candidate_mask must have equal shapes")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("Every query must contain at least one valid candidate")
        return scores.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
