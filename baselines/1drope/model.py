"""Sequence-index RoPE baseline with no three-dimensional information.

The antibody and antigen self-encoders use standard one-dimensional rotary
position embeddings over their respective zero-based tensor positions.  All
coordinate tensors, coordinate masks, and graph edges are accepted only to
match the shared benchmark interface and are deliberately ignored.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from baselines.se3bench.common import (
    AttentionStack,
    DenseSelfAttentionBase,
    GeometryPredictor,
    ResidualAttentionLayer,
)


class OneDRoPEAttention(DenseSelfAttentionBase):
    """Dense self-attention with standard sequence-index RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if rope_base <= 1.0:
            raise ValueError("rope_base must be greater than one")
        self.rope_base = float(rope_base)
        self.rope_dim = self.head_dim - self.head_dim % 2
        if self.rope_dim < 2:
            raise ValueError("head dimension must contain a rotary channel pair")
        inverse_frequencies = self.rope_base ** (
            -torch.arange(0, self.rope_dim, 2, dtype=torch.float32)
            / self.rope_dim
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies)

    def apply_rope(self, value: Tensor, positions: Tensor) -> Tensor:
        """Rotate channel pairs at the supplied scalar sequence positions."""

        rotary = value[..., : self.rope_dim]
        rest = value[..., self.rope_dim :]
        pairs = rotary.reshape(*rotary.shape[:-1], self.rope_dim // 2, 2)
        angles = positions.to(value.dtype).unsqueeze(-1) * self.inverse_frequencies.to(
            value.dtype
        )
        cosine = torch.cos(angles)[None, None]
        sine = torch.sin(angles)[None, None]
        first, second = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        ).flatten(start_dim=-2)
        return torch.cat((rotated, rest), dim=-1) if rest.numel() else rotated

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
        # These arguments are intentionally unused: this is the auditable
        # boundary that prevents any 3D or graph information entering 1D RoPE.
        del coords_ca, coords_n, coords_c, coord_mask, edge_index
        query, key, value = self.project_qkv(x)
        positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32)
        query = self.apply_rope(query, positions)
        key = self.apply_rope(key, positions)
        logits = self.content_logits(query, key)
        return self.aggregate(logits, value, token_mask)


class Model(GeometryPredictor):
    """1D-RoPE encoders with the original cross/context/MIL predictor."""

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
        use_cross_attention: bool = True,
        use_global_context: bool = True,
        use_pair_mil: bool = True,
        mil_weight: float = 0.75,
        geometry_config: Optional[dict[str, Any]] = None,
    ) -> None:
        if not use_cross_attention:
            raise ValueError("the 1D-RoPE protocol keeps cross-attention enabled")
        settings = dict(geometry_config or {})
        unexpected = sorted(set(settings) - {"rope_base"})
        if unexpected:
            raise ValueError(f"unexpected 1D-RoPE settings: {unexpected}")
        rope_base = float(settings.get("rope_base", 10_000.0))

        def encoder_factory(n_layers: int) -> AttentionStack:
            return AttentionStack(
                [
                    ResidualAttentionLayer(
                        OneDRoPEAttention(
                            d_model,
                            n_heads,
                            dropout,
                            rope_base=rope_base,
                        ),
                        d_model,
                        dim_ff,
                        dropout,
                    )
                    for _ in range(n_layers)
                ]
            )

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
        self.use_cross_attention = True


__all__ = ["Model", "OneDRoPEAttention"]
