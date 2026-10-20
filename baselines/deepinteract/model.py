"""Shared epitope predictor with full-chain-context GeoT intra-chain encoders.

The non-geometric projections, cross attention, global context and prediction
heads are constructed by GeometryPredictor. Only the intra-chain encoders and
their geometric inputs differ. Full-chain context is mandatory: selected CDR
and surface nodes cannot be treated as a contiguous physical backbone.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from baselines.se3bench.common import (
    GeometryPredictor, masked_logmeanexp, masked_max, masked_mean,
)
from .geometry import build_geometry
from .layers import GeoTEncoder


class Model(GeometryPredictor):
    def __init__(
        self, input_ab_dim: int, input_ag_dim: int, *, d_model: int = 256,
        n_heads: int = 8, n_layers_ab: int = 4, n_layers_ag: int = 4,
        n_layers_cross: int = 2, dim_ff: int = 512, dropout: float = 0.1,
        use_cdr_type: bool = True, use_global_context: bool = True,
        use_pair_mil: bool = True, mil_weight: float = 0.75,
        geometry_config: Optional[dict[str, Any]] = None,
    ) -> None:
        config = dict(geometry_config or {})
        for key, expected in (
            ("deepinteract_frame_mode", "ca_trace"),
            ("deepinteract_relative_rotation", "corrected"),
            ("deepinteract_rbf_distance", "euclidean"),
            ("deepinteract_num_rbf", 18),
        ):
            if config.get(key, expected) != expected:
                raise ValueError(f"This documented adaptation requires {key}={expected!r}")
        self.knn = int(config.get("deepinteract_knn", 20))
        self.num_rbf = int(config.get("deepinteract_num_rbf", 18))
        self.geo_nbrhd_size = int(config.get("deepinteract_geo_nbrhd_size", 2))

        def encoder_factory(n_layers: int) -> GeoTEncoder:
            return GeoTEncoder(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                dropout=dropout,
                node_count_limit=int(config.get("deepinteract_node_count_limit", 2304)),
                norm_to_apply=str(config.get("deepinteract_norm", "batch")),
                checkpoint_min_tokens=int(config.get("deepinteract_checkpoint_min_tokens", 512)),
            )

        super().__init__(
            input_ab_dim, input_ag_dim, encoder_factory,
            d_model=d_model, n_heads=n_heads, n_layers_ab=n_layers_ab,
            n_layers_ag=n_layers_ag, n_layers_cross=n_layers_cross,
            dim_ff=dim_ff, dropout=dropout, use_cdr_type=use_cdr_type,
            use_global_context=use_global_context, use_pair_mil=use_pair_mil,
            mil_weight=mil_weight,
        )

    def forward(
        self, x_b: Tensor, x_g: Tensor, coords_ab: Tensor, coords_ag: Tensor,
        coords_n_ab: Tensor, coords_c_ab: Tensor, coords_n_ag: Tensor,
        coords_c_ag: Tensor, cdr_type: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None, mask_g: Optional[Tensor] = None,
        coord_mask_ab: Optional[Tensor] = None, coord_mask_ag: Optional[Tensor] = None,
        edge_index_ab: Optional[Tensor] = None, edge_index_ag: Optional[Tensor] = None,
        context_ab: Optional[dict[str, Tensor]] = None,
        context_ag: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        # The adapter synchronizes the selected coordinate aliases with these
        # full contexts after augmentation. Geometry uses the full contexts.
        del coords_ab, coords_ag, coords_n_ab, coords_c_ab, coords_n_ag, coords_c_ag
        del coord_mask_ab, coord_mask_ag, edge_index_ab, edge_index_ag
        if context_ab is None or context_ag is None:
            raise ValueError("DeepInteract requires full-chain context_ab/context_ag; use deepinteract.train")
        if x_b.ndim != 3 or x_g.ndim != 3:
            raise ValueError("Feature tensors must have shape [batch, selected residues, features]")
        batch_size, n_ab, _ = x_b.shape
        _, n_ag, _ = x_g.shape
        if mask_b is None:
            mask_b = torch.ones(batch_size, n_ab, dtype=torch.bool, device=x_b.device)
        if mask_g is None:
            mask_g = torch.ones(batch_size, n_ag, dtype=torch.bool, device=x_g.device)
        geometry_ab = build_geometry(context_ab, token_mask=mask_b, k=self.knn,
                                     num_rbf=self.num_rbf, geo_nbrhd_size=self.geo_nbrhd_size)
        geometry_ag = build_geometry(context_ag, token_mask=mask_g, k=self.knn,
                                     num_rbf=self.num_rbf, geo_nbrhd_size=self.geo_nbrhd_size)
        hidden_ab = self.ab_proj(x_b)
        hidden_ag = self.ag_proj(x_g)
        if self.use_cdr_type and cdr_type is not None:
            hidden_ab = hidden_ab + self.cdr_type_embed(cdr_type.clamp(0, 6))
        hidden_ab = self.ab_encoder(hidden_ab, geometry_ab)
        hidden_ag = self.ag_encoder(hidden_ag, geometry_ag)

        # Same partner-conditioning and heads as GeometryPredictor.forward.
        hidden_cross = hidden_ag
        for layer in self.cross_layers:
            hidden_cross = layer(hidden_cross, hidden_ab, target_mask=mask_g, memory_mask=mask_b)
        features = [hidden_ag, hidden_cross, hidden_ag * hidden_cross]
        if self.use_global_context:
            pooled = torch.cat([masked_mean(hidden_ab, mask_b, dim=1),
                                masked_max(hidden_ab, mask_b, dim=1)], dim=-1)
            global_context = self.global_proj(pooled).unsqueeze(1).expand(-1, n_ag, -1)
            features.extend([global_context, hidden_ag * global_context])
        node_logits = self.node_head(torch.cat(features, dim=-1)).squeeze(-1)
        if not self.use_pair_mil:
            return node_logits
        ag_pair = hidden_ag[:, :, None, :].expand(batch_size, n_ag, n_ab, self.d_model)
        ab_pair = hidden_ab[:, None, :, :].expand(batch_size, n_ag, n_ab, self.d_model)
        pair_logits = self.pair_head(torch.cat([ag_pair, ab_pair, ag_pair * ab_pair], dim=-1)).squeeze(-1)
        pair_mask = mask_b[:, None, :].expand(batch_size, n_ag, n_ab)
        mil_logits = masked_logmeanexp(pair_logits, pair_mask, dim=-1)
        logits = node_logits + self.logit_gate * mil_logits
        return logits * mask_g.to(logits.dtype)
