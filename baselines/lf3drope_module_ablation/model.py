"""LF3DRoPE with clean switches for three downstream prediction modules.

The LF3DRoPE intra-chain encoder is imported unchanged.  This isolated
predictor changes only the downstream antigen-to-antibody cross-attention,
global antibody context, and residue-pair MIL branches.  With module bits
``111`` its parameters and forward pass are exactly those of the benchmark
LF3DRoPE model.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from baselines.lf3drope.model import LocalFrame3DRoPEAttention
from baselines.se3bench.common import (
    AttentionStack,
    CrossAttentionLayer,
    ResidualAttentionLayer,
    masked_logmeanexp,
    masked_max,
    masked_mean,
)


class ModuleAblationPredictor(nn.Module):
    """Antibody-conditioned predictor with independently removable branches."""

    def __init__(
        self,
        input_ab_dim: int,
        input_ag_dim: int,
        encoder_factory: Any,
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
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_cdr_type = bool(use_cdr_type)
        self.use_cross_attention = bool(use_cross_attention)
        self.use_global_context = bool(use_global_context)
        self.use_pair_mil = bool(use_pair_mil)
        self.configured_cross_layers = int(n_layers_cross)
        if self.configured_cross_layers < 1:
            raise ValueError("n_layers_cross must be positive in this protocol")

        self.ab_proj = nn.Sequential(
            nn.Linear(input_ab_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ag_proj = nn.Sequential(
            nn.Linear(input_ag_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_cdr_type:
            self.cdr_type_embed = nn.Embedding(7, d_model)

        self.ab_encoder = encoder_factory(int(n_layers_ab))
        self.ag_encoder = encoder_factory(int(n_layers_ag))
        self.cross_layers = nn.ModuleList(
            CrossAttentionLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(
                self.configured_cross_layers if self.use_cross_attention else 0
            )
        )

        head_blocks = 1 + 2 * int(self.use_cross_attention)
        if self.use_global_context:
            self.global_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            head_blocks += 2

        self.node_head = nn.Sequential(
            nn.Linear(d_model * head_blocks, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        if self.use_pair_mil:
            self.pair_head = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            # This is a learned, unconstrained gate initialized to mil_weight.
            self.logit_gate = nn.Parameter(torch.tensor(float(mil_weight)))

    @staticmethod
    def _fallback_coord_mask(
        coords_ca: Tensor,
        coords_n: Tensor,
        coords_c: Tensor,
    ) -> Tensor:
        finite = (
            torch.isfinite(coords_ca).all(dim=-1)
            & torch.isfinite(coords_n).all(dim=-1)
            & torch.isfinite(coords_c).all(dim=-1)
        )
        nonzero = (
            coords_ca.abs().sum(dim=-1)
            + coords_n.abs().sum(dim=-1)
            + coords_c.abs().sum(dim=-1)
        ) > 0
        return finite & nonzero

    def forward(
        self,
        x_b: Tensor,
        x_g: Tensor,
        coords_ab: Tensor,
        coords_ag: Tensor,
        coords_n_ab: Tensor,
        coords_c_ab: Tensor,
        coords_n_ag: Tensor,
        coords_c_ag: Tensor,
        cdr_type: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
        mask_g: Optional[Tensor] = None,
        coord_mask_ab: Optional[Tensor] = None,
        coord_mask_ag: Optional[Tensor] = None,
        edge_index_ab: Optional[Tensor] = None,
        edge_index_ag: Optional[Tensor] = None,
    ) -> Tensor:
        if x_b.dim() == 2:
            x_b = x_b.unsqueeze(0)
            x_g = x_g.unsqueeze(0)
            coords_ab = coords_ab.unsqueeze(0)
            coords_ag = coords_ag.unsqueeze(0)
            coords_n_ab = coords_n_ab.unsqueeze(0)
            coords_c_ab = coords_c_ab.unsqueeze(0)
            coords_n_ag = coords_n_ag.unsqueeze(0)
            coords_c_ag = coords_c_ag.unsqueeze(0)
            cdr_type = None if cdr_type is None else cdr_type.unsqueeze(0)
            mask_b = None if mask_b is None else mask_b.unsqueeze(0)
            mask_g = None if mask_g is None else mask_g.unsqueeze(0)
            coord_mask_ab = (
                None if coord_mask_ab is None else coord_mask_ab.unsqueeze(0)
            )
            coord_mask_ag = (
                None if coord_mask_ag is None else coord_mask_ag.unsqueeze(0)
            )

        batch_size, n_ab, _ = x_b.shape
        _, n_ag, _ = x_g.shape
        if mask_b is None:
            mask_b = torch.ones(
                batch_size, n_ab, dtype=torch.bool, device=x_b.device
            )
        if mask_g is None:
            mask_g = torch.ones(
                batch_size, n_ag, dtype=torch.bool, device=x_g.device
            )
        if coord_mask_ab is None:
            coord_mask_ab = self._fallback_coord_mask(
                coords_ab, coords_n_ab, coords_c_ab
            )
        if coord_mask_ag is None:
            coord_mask_ag = self._fallback_coord_mask(
                coords_ag, coords_n_ag, coords_c_ag
            )

        hidden_ab = self.ab_proj(x_b)
        hidden_ag = self.ag_proj(x_g)
        if self.use_cdr_type and cdr_type is not None:
            hidden_ab = hidden_ab + self.cdr_type_embed(cdr_type.clamp(0, 6))

        hidden_ab = self.ab_encoder(
            hidden_ab,
            coords_ab,
            coords_n_ab,
            coords_c_ab,
            token_mask=mask_b,
            coord_mask=coord_mask_ab,
            edge_index=edge_index_ab,
        )
        hidden_ag = self.ag_encoder(
            hidden_ag,
            coords_ag,
            coords_n_ag,
            coords_c_ag,
            token_mask=mask_g,
            coord_mask=coord_mask_ag,
            edge_index=edge_index_ag,
        )

        # A disabled cross branch is removed rather than replaced by the
        # identity features [h_ag, h_ag, h_ag**2].
        features = [hidden_ag]
        if self.use_cross_attention:
            hidden_cross = hidden_ag
            for layer in self.cross_layers:
                hidden_cross = layer(
                    hidden_cross,
                    hidden_ab,
                    target_mask=mask_g,
                    memory_mask=mask_b,
                )
            features.extend([hidden_cross, hidden_ag * hidden_cross])

        if self.use_global_context:
            pooled = torch.cat(
                [
                    masked_mean(hidden_ab, mask_b, dim=1),
                    masked_max(hidden_ab, mask_b, dim=1),
                ],
                dim=-1,
            )
            global_context = self.global_proj(pooled).unsqueeze(1).expand(
                -1, n_ag, -1
            )
            features.extend([global_context, hidden_ag * global_context])

        node_logits = self.node_head(torch.cat(features, dim=-1)).squeeze(-1)
        logits = node_logits
        if self.use_pair_mil:
            ag_pair = hidden_ag[:, :, None, :].expand(
                batch_size, n_ag, n_ab, self.d_model
            )
            ab_pair = hidden_ab[:, None, :, :].expand(
                batch_size, n_ag, n_ab, self.d_model
            )
            pair_logits = self.pair_head(
                torch.cat([ag_pair, ab_pair, ag_pair * ab_pair], dim=-1)
            ).squeeze(-1)
            pair_mask = mask_b[:, None, :].expand(batch_size, n_ag, n_ab)
            mil_logits = masked_logmeanexp(pair_logits, pair_mask, dim=-1)
            logits = logits + self.logit_gate * mil_logits

        # Apply the antigen mask consistently, including the MIL-off path.
        return logits * mask_g.to(logits.dtype)


class Model(ModuleAblationPredictor):
    """LF3DRoPE encoder plus the controlled module-ablation predictor."""

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
        geometry_config = dict(geometry_config or {})

        def encoder_factory(n_layers: int) -> AttentionStack:
            layers = []
            for _ in range(n_layers):
                attention = LocalFrame3DRoPEAttention(
                    d_model,
                    n_heads,
                    dropout,
                    n_rope_freqs=int(geometry_config.get("n_rope_freqs", 5)),
                    rope_min_freq=float(
                        geometry_config.get("rope_min_freq", 0.08)
                    ),
                    rope_max_freq=float(
                        geometry_config.get("rope_max_freq", 2.5)
                    ),
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
                    ResidualAttentionLayer(
                        attention,
                        d_model,
                        dim_ff,
                        dropout,
                    )
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
            use_cross_attention=use_cross_attention,
            use_global_context=use_global_context,
            use_pair_mil=use_pair_mil,
            mil_weight=mil_weight,
        )


__all__ = ["Model", "ModuleAblationPredictor"]
