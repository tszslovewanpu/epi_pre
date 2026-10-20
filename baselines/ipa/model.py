"""Fixed-frame Invariant Point Attention for structure classification."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from baselines.se3bench.common import (
    AttentionStack,
    DenseSelfAttentionBase,
    GeometryPredictor,
    ResidualAttentionLayer,
)
from baselines.se3bench.geometry import build_local_frames, pairwise_mask


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class InvariantPointAttention(DenseSelfAttentionBase):
    """Scalar attention augmented with distances between learned local points."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        n_query_key_points: int = 4,
        n_value_points: int = 4,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if n_query_key_points < 1 or n_value_points < 1:
            raise ValueError("IPA point counts must be positive")
        self.n_query_key_points = int(n_query_key_points)
        self.n_value_points = int(n_value_points)
        self.query_points = nn.Linear(
            d_model, n_heads * self.n_query_key_points * 3
        )
        self.key_points = nn.Linear(
            d_model, n_heads * self.n_query_key_points * 3
        )
        self.value_points = nn.Linear(
            d_model, n_heads * self.n_value_points * 3
        )
        self.point_weights = nn.Parameter(
            torch.full((n_heads,), _inverse_softplus(1.0))
        )
        combined_dim = (
            d_model
            + n_heads * self.n_value_points * 3
            + n_heads * self.n_value_points
        )
        # Replace the scalar-only projection created by the base class.
        self.out_proj = nn.Linear(combined_dim, d_model)

    def _local_to_centered_global(
        self,
        local_points: Tensor,
        frames: Tensor,
        centered_coords: Tensor,
    ) -> Tensor:
        rotated = torch.einsum(
            "bnca,bnhpa->bnhpc", frames, local_points
        )
        return rotated + centered_coords[:, :, None, None, :]

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
        batch, length, _ = x.shape
        query, key, scalar_value = self.project_qkv(x)
        scalar_logits = self.content_logits(query, key)

        frames, frame_mask = build_local_frames(
            coords_ca, coords_n, coords_c, coord_mask
        )
        valid_pairs = pairwise_mask(frame_mask)
        clean_coords = torch.nan_to_num(
            coords_ca, nan=0.0, posinf=0.0, neginf=0.0
        )
        valid_weight = frame_mask.unsqueeze(-1).to(clean_coords.dtype)
        center = (clean_coords * valid_weight).sum(dim=1) / valid_weight.sum(
            dim=1
        ).clamp_min(1.0)
        centered_coords = clean_coords - center[:, None, :]

        query_local = self.query_points(x).view(
            batch,
            length,
            self.n_heads,
            self.n_query_key_points,
            3,
        )
        key_local = self.key_points(x).view(
            batch,
            length,
            self.n_heads,
            self.n_query_key_points,
            3,
        )
        query_global = self._local_to_centered_global(
            query_local, frames, centered_coords
        )
        key_global = self._local_to_centered_global(
            key_local, frames, centered_coords
        )

        # [B,H,P,N,3] formulation avoids materializing an N*N*P*3 tensor.
        query_by_point = query_global.permute(0, 2, 3, 1, 4)
        key_by_point = key_global.permute(0, 2, 3, 1, 4)
        query_norm = query_by_point.square().sum(dim=-1, keepdim=True)
        key_norm = (
            key_by_point.square().sum(dim=-1).unsqueeze(-2)
        )
        cross = torch.matmul(
            query_by_point, key_by_point.transpose(-1, -2)
        )
        squared_distance = (query_norm + key_norm - 2.0 * cross).clamp_min(0.0)
        squared_distance = squared_distance.sum(dim=2)
        point_scale = (
            F.softplus(self.point_weights)
            / (2.0 * float(self.n_query_key_points))
        )
        point_logits = -point_scale[None, :, None, None] * squared_distance
        point_logits = point_logits.masked_fill(
            ~valid_pairs[:, None], 0.0
        )
        logits = scalar_logits + point_logits
        if token_mask is not None:
            logits = logits.masked_fill(
                ~token_mask[:, None, None, :], -1.0e4
            )
        attention = F.softmax(logits, dim=-1)
        attention = self.attention_dropout(attention)

        scalar_output = torch.matmul(attention, scalar_value)
        scalar_output = scalar_output.transpose(1, 2).contiguous().view(
            batch, length, self.d_model
        )

        value_local = self.value_points(x).view(
            batch,
            length,
            self.n_heads,
            self.n_value_points,
            3,
        )
        value_global = self._local_to_centered_global(
            value_local, frames, centered_coords
        )
        geometric_attention = attention * valid_pairs[:, None].to(
            attention.dtype
        )
        aggregated_global = torch.einsum(
            "bhij,bjhpc->bihpc", geometric_attention, value_global
        )
        mass = geometric_attention.sum(dim=-1).permute(0, 2, 1)
        relative_global = aggregated_global - (
            mass[..., None, None]
            * centered_coords[:, :, None, None, :]
        )
        aggregated_local = torch.einsum(
            "bihpc,bica->bihpa", relative_global, frames
        )
        aggregated_local = aggregated_local.masked_fill(
            ~frame_mask[:, :, None, None, None], 0.0
        )
        point_norms = torch.linalg.vector_norm(
            aggregated_local, dim=-1
        )
        combined = torch.cat(
            [
                scalar_output,
                aggregated_local.flatten(start_dim=2),
                point_norms.flatten(start_dim=2),
            ],
            dim=-1,
        )
        output = self.out_proj(combined)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(output.dtype)
        return output


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
                attention = InvariantPointAttention(
                    d_model,
                    n_heads,
                    dropout,
                    n_query_key_points=int(
                        geometry_config.get("n_query_key_points", 4)
                    ),
                    n_value_points=int(
                        geometry_config.get("n_value_points", 4)
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


__all__ = ["InvariantPointAttention", "Model"]
