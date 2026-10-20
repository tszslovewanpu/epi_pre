"""GF3DRoPE self-encoder used by the rotation-augmentation baseline.

The attention calculation matches the supplied ``opensource_gf3drope`` model:
global-coordinate 3D rotary logits plus the companion distance-RBF and global
Fourier biases.  Rotation augmentation is deliberately implemented by the
training pipeline, not in this model.
"""

from __future__ import annotations

import math
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


class RBFDistanceBias(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 30.0,
    ) -> None:
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        spacing = (d_max - d_min) / max(n_rbf - 1, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0 / max(spacing, 1.0e-6) ** 2))
        self.projection = nn.Linear(n_rbf, n_heads, bias=False)

    def forward(self, distance: Tensor) -> Tensor:
        features = torch.exp(
            -self.gamma.abs()
            * (distance.unsqueeze(-1) - self.centers).square()
        )
        return self.projection(features).permute(0, 3, 1, 2).contiguous()


class GlobalFrameFourierBias(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_bands: int = 4,
        min_frequency: float = 0.1,
        max_frequency: float = 4.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.logspace(
                math.log10(min_frequency),
                math.log10(max_frequency),
                steps=n_bands,
            ),
        )
        self.projection = nn.Linear(3 * n_bands * 2, n_heads, bias=False)

    def forward(self, global_displacement: Tensor) -> Tensor:
        scaled = global_displacement[..., None] * self.frequencies
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        features = features.flatten(start_dim=-2)
        return self.projection(features).permute(0, 3, 1, 2).contiguous()


class GlobalFrame3DRoPEAttention(DenseSelfAttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        n_rope_freqs: int = 5,
        rope_min_freq: float = 0.08,
        rope_max_freq: float = 2.5,
        use_distance_bias: bool = True,
        use_fourier_bias: bool = True,
        n_rbf: int = 16,
        n_fourier_bands: int = 4,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        self.n_rope_freqs = min(int(n_rope_freqs), self.head_dim // 6)
        if self.n_rope_freqs < 1:
            raise ValueError("head dimension must contain at least one 3D RoPE frequency")
        self.rope_dim = 3 * self.n_rope_freqs * 2
        self.rest_dim = self.head_dim - self.rope_dim
        self.register_buffer(
            "rope_frequencies",
            torch.logspace(
                math.log10(rope_min_freq),
                math.log10(rope_max_freq),
                steps=self.n_rope_freqs,
            ),
        )
        self.distance_bias = (
            RBFDistanceBias(n_heads, n_rbf=n_rbf) if use_distance_bias else None
        )
        self.fourier_bias = (
            GlobalFrameFourierBias(n_heads, n_bands=n_fourier_bands)
            if use_fourier_bias
            else None
        )

    def rotary_logits(
        self, query: Tensor, key: Tensor, displacement: Tensor
    ) -> Tensor:
        batch, heads, length, _ = query.shape
        query_rope = query[..., : self.rope_dim].view(
            batch, heads, length, 3, self.n_rope_freqs, 2
        )
        key_rope = key[..., : self.rope_dim].view(
            batch, heads, length, 3, self.n_rope_freqs, 2
        )
        query_pair = query_rope.unsqueeze(3)
        key_pair = key_rope.unsqueeze(2)
        theta = (
            displacement[:, None, :, :, :, None]
            * self.rope_frequencies.view(1, 1, 1, 1, 1, -1)
        )
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        key_0, key_1 = key_pair[..., 0], key_pair[..., 1]
        rotated_0 = key_0 * cosine - key_1 * sine
        rotated_1 = key_0 * sine + key_1 * cosine
        logits = (
            query_pair[..., 0] * rotated_0 + query_pair[..., 1] * rotated_1
        ).sum(dim=(-1, -2))
        if self.rest_dim:
            logits = logits + torch.matmul(
                query[..., self.rope_dim :],
                key[..., self.rope_dim :].transpose(-2, -1),
            )
        return logits

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
        del coords_n, coords_c, edge_index
        query, key, value = self.project_qkv(x)
        clean = torch.nan_to_num(coords_ca, nan=0.0, posinf=0.0, neginf=0.0)
        displacement = clean[:, None, :, :] - clean[:, :, None, :]
        pair_mask = None
        if coord_mask is not None:
            pair_mask = coord_mask[:, :, None] & coord_mask[:, None, :]
            displacement = displacement.masked_fill(~pair_mask[..., None], 0.0)

        logits = self.rotary_logits(query, key, displacement) * self.scale
        distance = displacement.norm(dim=-1)
        if self.distance_bias is not None:
            bias = self.distance_bias(distance)
            if pair_mask is not None:
                bias = bias.masked_fill(~pair_mask[:, None], 0.0)
            logits = logits + bias
        if self.fourier_bias is not None:
            bias = self.fourier_bias(displacement)
            if pair_mask is not None:
                bias = bias.masked_fill(~pair_mask[:, None], 0.0)
            logits = logits + bias
        return self.aggregate(logits, value, token_mask)


class Model(GeometryPredictor):
    """Full supplied GF3DRoPE architecture; augmentation is external."""

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
                attention = GlobalFrame3DRoPEAttention(
                    d_model,
                    n_heads,
                    dropout,
                    n_rope_freqs=int(geometry_config.get("n_rope_freqs", 5)),
                    rope_min_freq=float(geometry_config.get("rope_min_freq", 0.08)),
                    rope_max_freq=float(geometry_config.get("rope_max_freq", 2.5)),
                    use_distance_bias=bool(
                        geometry_config.get("use_distance_bias", True)
                    ),
                    use_fourier_bias=bool(
                        geometry_config.get("use_fourier_bias", True)
                    ),
                    n_rbf=int(geometry_config.get("n_rbf", 16)),
                    n_fourier_bands=int(
                        geometry_config.get("n_fourier_bands", 4)
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


__all__ = ["GlobalFrame3DRoPEAttention", "Model"]
