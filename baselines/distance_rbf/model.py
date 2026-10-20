"""Distance-RBF attention-bias baseline.

Geometry affects attention only through a head-specific Gaussian RBF expansion
of pairwise CA distances.  There is deliberately no rotary, Fourier,
directional, local-frame, or equivariant-vector pathway.
"""

from __future__ import annotations

from typing import Any, Optional

from torch import Tensor

from baselines.se3bench.common import (
    AttentionStack,
    DenseSelfAttentionBase,
    GeometryPredictor,
    ResidualAttentionLayer,
)
from baselines.se3bench.geometry import (
    RBFDistanceBias,
    infer_coord_mask,
    pairwise_geometry,
)


class DistanceRBFAttention(DenseSelfAttentionBase):
    """Ordinary dense multi-head attention plus an invariant distance bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        n_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 30.0,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        self.distance_bias = RBFDistanceBias(
            n_heads=n_heads,
            n_rbf=n_rbf,
            d_min=d_min,
            d_max=d_max,
            trainable_gamma=True,
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

        geometry_mask = infer_coord_mask(coords_ca, coords_n, coords_c)
        if coord_mask is not None:
            geometry_mask = geometry_mask & coord_mask.to(
                device=geometry_mask.device,
                dtype=geometry_mask.dtype,
            )
        _, distances, valid_pairs = pairwise_geometry(
            coords_ca,
            geometry_mask,
        )

        logits = self.content_logits(query, key)
        # RBFDistanceBias applies valid_pairs after projection, so an invalid
        # pair receives exactly zero geometric bias rather than RBF(distance=0).
        logits = logits + self.distance_bias(distances, valid_pairs)
        return self.aggregate(logits, value, token_mask)


class Model(GeometryPredictor):
    """Shared AsEP predictor equipped with distance-biased self-attention."""

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
        n_rbf = int(geometry_config.get("n_rbf", 16))
        d_min = float(geometry_config.get("d_min", 0.0))
        d_max = float(geometry_config.get("d_max", 30.0))

        def encoder_factory(n_layers: int) -> AttentionStack:
            return AttentionStack(
                [
                    ResidualAttentionLayer(
                        DistanceRBFAttention(
                            d_model=d_model,
                            n_heads=n_heads,
                            dropout=dropout,
                            n_rbf=n_rbf,
                            d_min=d_min,
                            d_max=d_max,
                        ),
                        d_model=d_model,
                        dim_ff=dim_ff,
                        dropout=dropout,
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


__all__ = ["DistanceRBFAttention", "Model"]
