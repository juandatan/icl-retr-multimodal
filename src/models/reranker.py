"""Label-aware frozen-feature reranker for multimodal ICL exemplars."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RerankerConfig:
    """Serializable architecture parameters for a reranker checkpoint."""

    clip_dim: int
    siglip_dim: int
    hidden_dim: int = 256
    metadata_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if min(self.clip_dim, self.siglip_dim, self.hidden_dim, self.metadata_dim) <= 0:
            raise ValueError("All reranker dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


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
    """Score candidate exemplars without running an image encoder at training time.

    CLIP captures the retrieval geometry. SigLIP supplies a shared image/text
    space in which the exemplar's class label can interact with both images.
    Query and exemplar image towers share weights within each backbone so the
    score is not tied to two arbitrary embedding coordinate systems.
    """

    def __init__(self, config: RerankerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.clip_image_tower = _ProjectionTower(
            config.clip_dim, hidden, config.dropout
        )
        self.siglip_image_tower = _ProjectionTower(
            config.siglip_dim, hidden, config.dropout
        )
        self.siglip_label_tower = _ProjectionTower(
            config.siglip_dim, hidden, config.dropout
        )
        self.clip_interactions = _InteractionEncoder(
            4 * hidden, hidden, config.dropout
        )
        self.siglip_interactions = _InteractionEncoder(
            12 * hidden, hidden, config.dropout
        )
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, config.metadata_dim),
            nn.GELU(),
        )
        fusion_dim = 2 * hidden + config.metadata_dim
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
        batch, candidates = candidate_clip.shape[:2]
        expected_prefix = (batch, candidates)
        if (
            query_clip.shape != (batch, self.config.clip_dim)
            or candidate_clip.shape != (*expected_prefix, self.config.clip_dim)
            or query_siglip.shape != (batch, self.config.siglip_dim)
            or candidate_siglip.shape != (*expected_prefix, self.config.siglip_dim)
            or candidate_label_siglip.shape != (*expected_prefix, self.config.siglip_dim)
            or clip_similarities.shape != expected_prefix
            or retrieval_ranks.shape != expected_prefix
        ):
            raise ValueError("Reranker input shapes do not match its configuration")
        return batch, candidates

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
        )

        query_clip_normalized = F.normalize(query_clip, dim=-1)
        candidate_clip_normalized = F.normalize(candidate_clip, dim=-1)
        query_siglip_normalized = F.normalize(query_siglip, dim=-1)
        candidate_siglip_normalized = F.normalize(candidate_siglip, dim=-1)
        label_siglip_normalized = F.normalize(candidate_label_siglip, dim=-1)

        query_clip_projected = self.clip_image_tower(query_clip_normalized)
        candidate_clip_projected = self.clip_image_tower(candidate_clip_normalized)
        query_siglip_projected = self.siglip_image_tower(query_siglip_normalized)
        candidate_siglip_projected = self.siglip_image_tower(
            candidate_siglip_normalized
        )
        label_siglip_projected = self.siglip_label_tower(label_siglip_normalized)

        query_clip_projected = query_clip_projected.unsqueeze(1).expand(
            -1, candidates, -1
        )
        query_siglip_projected = query_siglip_projected.unsqueeze(1).expand(
            -1, candidates, -1
        )
        clip_representation = self.clip_interactions(
            _pair_features(query_clip_projected, candidate_clip_projected)
        )
        siglip_representation = self.siglip_interactions(torch.cat((
            _pair_features(query_siglip_projected, candidate_siglip_projected),
            _pair_features(query_siglip_projected, label_siglip_projected),
            _pair_features(candidate_siglip_projected, label_siglip_projected),
        ), dim=-1))

        query_label_similarity = torch.sum(
            query_siglip_normalized.unsqueeze(1) * label_siglip_normalized, dim=-1
        )
        exemplar_label_similarity = torch.sum(
            candidate_siglip_normalized * label_siglip_normalized, dim=-1
        )
        metadata = torch.stack((
            clip_similarities,
            retrieval_ranks,
            query_label_similarity,
            exemplar_label_similarity,
        ), dim=-1)
        metadata_representation = self.metadata_encoder(metadata)

        fused = torch.cat((
            clip_representation,
            siglip_representation,
            metadata_representation,
        ), dim=-1)
        return self.scorer(fused).squeeze(-1)

    @staticmethod
    def select(scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """Return each query's highest-scoring valid candidate position."""
        if scores.shape != candidate_mask.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("scores and boolean candidate_mask must have equal shapes")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("Every query must contain at least one valid candidate")
        return scores.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
