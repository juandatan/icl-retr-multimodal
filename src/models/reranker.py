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
    visual_token_dim: int = 0
    visual_token_count: int = 0
    visual_label_token_count: int = 0
    visual_token_layers: int = 2
    visual_token_heads: int = 4
    visual_token_ff_dim: int = 1024
    visual_candidate_chunk_size: int = 10

    def __post_init__(self) -> None:
        architectures = {
            "interaction_mlp",
            "pooled_transformer",
            "cross_candidate_attention",
            "visual_token_cross_encoder",
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
            self.visual_token_layers,
            self.visual_token_heads,
            self.visual_token_ff_dim,
            self.visual_candidate_chunk_size,
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
        if self.hidden_dim % self.visual_token_heads:
            raise ValueError("hidden_dim must be divisible by visual_token_heads")
        if self.architecture == "visual_token_cross_encoder" and min(
            self.visual_token_dim,
            self.visual_token_count,
            self.visual_label_token_count,
        ) <= 0:
            raise ValueError(
                "visual_token_cross_encoder requires positive visual-token dimensions"
            )
        if self.architecture == "visual_token_cross_encoder" and any((
            self.use_clip_embeddings,
            self.use_clip_similarity,
            self.use_retrieval_rank,
            self.use_derived_siglip_similarities,
        )):
            raise ValueError(
                "visual_token_cross_encoder does not combine pooled optional inputs"
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
    permutation-equivariant context across the candidate pool, while
    ``visual_token_cross_encoder`` jointly encodes one query/exemplar pair
    using frozen Idefics2 visual and label-token states. No teacher target or
    query label enters any model input.
    """

    def __init__(self, config: RerankerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        if config.architecture != "visual_token_cross_encoder":
            self.siglip_image_tower = _ProjectionTower(
                config.siglip_dim, hidden, config.dropout
            )
            self.siglip_label_tower = _ProjectionTower(
                config.siglip_dim, hidden, config.dropout
            )
        else:
            self.siglip_image_tower = None
            self.siglip_label_tower = None

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
        elif config.architecture == "pooled_transformer":
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
        else:
            self.siglip_interactions = None
            self.score_token = None
            self.role_embeddings = None
            self.transformer = None

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

        if config.architecture == "visual_token_cross_encoder":
            self.visual_image_projection = nn.Sequential(
                nn.LayerNorm(config.visual_token_dim),
                nn.Linear(config.visual_token_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.visual_label_projection = nn.Sequential(
                nn.LayerNorm(config.visual_token_dim),
                nn.Linear(config.visual_token_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.visual_utility_token = nn.Parameter(torch.empty(1, 1, hidden))
            self.visual_role_embeddings = nn.Parameter(torch.empty(3, hidden))
            self.visual_label_positions = nn.Parameter(torch.empty(
                1, config.visual_label_token_count, hidden
            ))
            nn.init.normal_(self.visual_utility_token, std=0.02)
            nn.init.normal_(self.visual_role_embeddings, std=0.02)
            nn.init.normal_(self.visual_label_positions, std=0.02)
            visual_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.visual_token_heads,
                dim_feedforward=config.visual_token_ff_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.visual_cross_encoder = nn.TransformerEncoder(
                visual_layer,
                num_layers=config.visual_token_layers,
                norm=nn.LayerNorm(hidden),
                enable_nested_tensor=False,
            )
            self.visual_scorer = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.visual_image_projection = None
            self.visual_label_projection = None
            self.visual_utility_token = None
            self.visual_role_embeddings = None
            self.visual_label_positions = None
            self.visual_cross_encoder = None
            self.visual_scorer = None

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
        self.scorer = (
            None
            if config.architecture == "visual_token_cross_encoder"
            else nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, 2 * hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(2 * hidden, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )
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

    def _visual_token_scores(
        self,
        query_visual_tokens: torch.Tensor,
        candidate_visual_tokens: torch.Tensor,
        candidate_label_tokens: torch.Tensor,
        candidate_label_token_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Jointly encode each query/exemplar pair without candidate-set context."""
        if query_visual_tokens.ndim != 3:
            raise ValueError(
                "query_visual_tokens must have shape [batch, tokens, hidden]"
            )
        if candidate_visual_tokens.ndim != 4:
            raise ValueError(
                "candidate_visual_tokens must have shape "
                "[batch, candidates, tokens, hidden]"
            )
        if candidate_label_tokens.ndim != 4:
            raise ValueError(
                "candidate_label_tokens must have shape "
                "[batch, candidates, tokens, hidden]"
            )
        batch, candidates, visual_tokens, token_dim = (
            candidate_visual_tokens.shape
        )
        expected_prefix = (batch, candidates)
        if (
            query_visual_tokens.shape
            != (batch, self.config.visual_token_count, self.config.visual_token_dim)
            or visual_tokens != self.config.visual_token_count
            or token_dim != self.config.visual_token_dim
            or candidate_label_tokens.shape
            != (
                batch,
                candidates,
                self.config.visual_label_token_count,
                self.config.visual_token_dim,
            )
            or candidate_label_token_mask.shape
            != (*expected_prefix, self.config.visual_label_token_count)
            or candidate_mask.shape != expected_prefix
        ):
            raise ValueError("Visual-token inputs do not match model configuration")
        if candidate_label_token_mask.dtype != torch.bool:
            raise ValueError("candidate_label_token_mask must be boolean")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("Every query must have at least one valid candidate")

        # Cached states are float16 to keep the sidecar compact. Outside CUDA
        # autocast, match the float32 trainable projections explicitly.
        if not torch.is_autocast_enabled():
            projection_dtype = self.visual_image_projection[1].weight.dtype
            query_visual_tokens = query_visual_tokens.to(projection_dtype)
            candidate_visual_tokens = candidate_visual_tokens.to(projection_dtype)
            candidate_label_tokens = candidate_label_tokens.to(projection_dtype)

        query_projected = self.visual_image_projection(query_visual_tokens)
        chunk_size = self.config.visual_candidate_chunk_size
        chunk_scores = []
        for start in range(0, candidates, chunk_size):
            stop = min(start + chunk_size, candidates)
            width = stop - start
            flat_count = batch * width
            exemplar = self.visual_image_projection(
                candidate_visual_tokens[:, start:stop]
            ).reshape(flat_count, visual_tokens, -1)
            label = self.visual_label_projection(
                candidate_label_tokens[:, start:stop]
            ).reshape(
                flat_count, self.config.visual_label_token_count, -1
            )
            label = (
                label
                + self.visual_role_embeddings[0]
                + self.visual_label_positions
            )
            query = query_projected.unsqueeze(1).expand(
                -1, width, -1, -1
            ).reshape(flat_count, visual_tokens, -1)
            query = query + self.visual_role_embeddings[1]
            exemplar = exemplar + self.visual_role_embeddings[2]
            utility = self.visual_utility_token.expand(flat_count, -1, -1)
            tokens = torch.cat((utility, label, query, exemplar), dim=1)

            label_valid = candidate_label_token_mask[:, start:stop].reshape(
                flat_count, self.config.visual_label_token_count
            )
            padding_mask = torch.cat((
                torch.zeros(
                    flat_count,
                    1,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                ~label_valid,
                torch.zeros(
                    flat_count,
                    2 * visual_tokens,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
            ), dim=1)
            encoded = self.visual_cross_encoder(
                tokens,
                src_key_padding_mask=padding_mask,
            )
            chunk_scores.append(
                self.visual_scorer(encoded[:, 0]).reshape(batch, width)
            )
        return torch.cat(chunk_scores, dim=1).masked_fill(~candidate_mask, 0.0)

    def forward(
        self,
        *,
        query_clip: torch.Tensor | None = None,
        candidate_clip: torch.Tensor | None = None,
        query_siglip: torch.Tensor | None = None,
        candidate_siglip: torch.Tensor | None = None,
        candidate_label_siglip: torch.Tensor | None = None,
        clip_similarities: torch.Tensor | None = None,
        retrieval_ranks: torch.Tensor | None = None,
        candidate_mask: torch.Tensor,
        query_visual_tokens: torch.Tensor | None = None,
        candidate_visual_tokens: torch.Tensor | None = None,
        candidate_label_tokens: torch.Tensor | None = None,
        candidate_label_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one unbounded utility score per candidate."""
        if self.config.architecture == "visual_token_cross_encoder":
            visual_inputs = (
                query_visual_tokens,
                candidate_visual_tokens,
                candidate_label_tokens,
                candidate_label_token_mask,
            )
            if any(value is None for value in visual_inputs):
                raise ValueError(
                    "visual_token_cross_encoder requires visual and label tokens"
                )
            return self._visual_token_scores(
                query_visual_tokens,
                candidate_visual_tokens,
                candidate_label_tokens,
                candidate_label_token_mask,
                candidate_mask,
            )

        pooled_inputs = (
            query_clip,
            candidate_clip,
            query_siglip,
            candidate_siglip,
            candidate_label_siglip,
            clip_similarities,
            retrieval_ranks,
        )
        if any(value is None for value in pooled_inputs):
            raise ValueError("Pooled reranker architecture requires pooled inputs")
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
