"""Invariant local pair features injected as an additive attention bias."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from baselines.se3bench.common import (
    AttentionStack,
    DenseSelfAttentionBase,
    GeometryPredictor,
    ResidualAttentionLayer,
)
from baselines.se3bench.geometry import (
    RBFEmbedding,
    build_local_frames,
    local_pair_vectors,
    pairwise_geometry,
    relative_frame_rotations,
    safe_normalize,
)


class LocalPairFeatureAttention(DenseSelfAttentionBase):
    """Content attention plus invariant distance/direction/orientation features."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        n_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 30.0,
        pair_hidden_dim: int = 32,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        self.rbf = RBFEmbedding(
            n_rbf=n_rbf,
            d_min=d_min,
            d_max=d_max,
            trainable_gamma=True,
        )
        # RBF distance + directed local unit vector + vec(R_i^T R_j).
        pair_input_dim = n_rbf + 3 + 9
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_input_dim, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, n_heads, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        *,
        coords_ca: Tensor,
        coords_n: Tensor,
        coords_c: Tensor,
        token_mask: Optional[Tensor] = None,
        coord_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        del edge_index
        query, key, value = self.project_qkv(x)
        logits = self.content_logits(query, key)

        frames, frame_mask = build_local_frames(
            coords_ca, coords_n, coords_c, coord_mask
        )
        _, distances, valid_pairs = pairwise_geometry(coords_ca, frame_mask)
        local_vectors = local_pair_vectors(
            coords_ca, frames, frame_mask
        )
        local_directions = safe_normalize(local_vectors)
        relative_rotations = relative_frame_rotations(
            frames, frame_mask
        ).flatten(start_dim=-2)
        pair_features = torch.cat(
            [
                self.rbf(distances),
                local_directions,
                relative_rotations,
            ],
            dim=-1,
        )
        bias = self.pair_bias(pair_features)
        bias = bias.masked_fill(~valid_pairs[..., None], 0.0)
        logits = logits + bias.permute(0, 3, 1, 2).contiguous()
        return self.aggregate(logits, value, token_mask)


class Model(GeometryPredictor):
    def __init__(
        self,
        input_ab_dim: int,
        input_ag_dim: int,
        *,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_ab: int = 4,
        n_layers_ag: int = 4,
        n_layers_cross: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
        use_cdr_type: bool = True,
        use_global_context: bool = True,
        use_pair_mil: bool = True,
        mil_weight: float = 0.75,
        geometry_config: Optional[dict[str, Any]] = None,
    ) -> None:
        geometry_config = dict(geometry_config or {})

        def encoder_factory(n_layers: int) -> AttentionStack:
            layers = []
            for _ in range(n_layers):
                attention = LocalPairFeatureAttention(
                    d_model,
                    n_heads,
                    dropout,
                    n_rbf=int(geometry_config.get("n_rbf", 16)),
                    d_min=float(geometry_config.get("d_min", 0.0)),
                    d_max=float(geometry_config.get("d_max", 30.0)),
                    pair_hidden_dim=int(
                        geometry_config.get("pair_hidden_dim", 32)
                    ),
                )
                layers.append(
                    ResidualAttentionLayer(attention, d_model, dim_ff, dropout)
                )
            return AttentionStack(layers)

        super().__init__(
            input_ab_dim,
            input_ag_dim,
            encoder_factory,
            d_model=d_model,
            n_heads=n_heads,
            n_layers_ab=n_layers_ab,
            n_layers_ag=n_layers_ag,
            n_layers_cross=n_layers_cross,
            dim_ff=dim_ff,
            dropout=dropout,
            use_cdr_type=use_cdr_type,
            use_global_context=use_global_context,
            use_pair_mil=use_pair_mil,
            mil_weight=mil_weight,
        )


__all__ = ["LocalPairFeatureAttention", "Model"]
