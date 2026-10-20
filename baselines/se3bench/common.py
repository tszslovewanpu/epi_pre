"""Architecture components shared by every geometric baseline.

Only the intra-chain encoder is method-specific.  The residue projections,
antigen-to-antibody cross-attention, antibody global context, node classifier,
and pair-MIL classifier intentionally match the supplied LF3DRoPE code.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def masked_mean(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.mean(dim=dim)
    weight = mask.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def masked_max(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.max(dim=dim).values
    expanded = mask.unsqueeze(-1)
    fill = torch.finfo(x.dtype).min
    result = x.masked_fill(~expanded, fill).max(dim=dim).values
    any_valid = mask.any(dim=dim, keepdim=False).unsqueeze(-1)
    return torch.where(any_valid, result, torch.zeros_like(result))


def masked_logmeanexp(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return torch.logsumexp(x, dim=dim) - math.log(x.size(dim))
    masked = x.masked_fill(~mask, -1.0e4)
    count = mask.sum(dim=dim).clamp_min(1).to(x.dtype)
    result = torch.logsumexp(masked, dim=dim) - torch.log(count)
    return torch.where(mask.any(dim=dim), result, torch.zeros_like(result))


class DenseSelfAttentionBase(nn.Module):
    """Projection and aggregation machinery for dense self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape

        def project(layer: nn.Linear) -> Tensor:
            return layer(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        return project(self.q_proj), project(self.k_proj), project(self.v_proj)

    def content_logits(self, q: Tensor, k: Tensor) -> Tensor:
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale

    def aggregate(self, logits: Tensor, values: Tensor, token_mask: Optional[Tensor]) -> Tensor:
        if token_mask is not None:
            logits = logits.masked_fill(~token_mask[:, None, None, :], -1.0e4)
        weights = F.softmax(logits, dim=-1)
        weights = self.attention_dropout(weights)
        output = torch.matmul(weights, values)
        output = output.transpose(1, 2).contiguous().view(
            values.size(0), values.size(2), self.d_model
        )
        output = self.out_proj(output)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(output.dtype)
        return output


class ResidualAttentionLayer(nn.Module):
    """Pre-normalized attention/FFN layer with a uniform geometry interface."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = attention
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Tensor,
        coords_c: Tensor,
        token_mask: Optional[Tensor] = None,
        coord_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        update = self.attention(
            self.norm1(x),
            coords_ca=coords_ca,
            coords_n=coords_n,
            coords_c=coords_c,
            token_mask=token_mask,
            coord_mask=coord_mask,
            edge_index=edge_index,
        )
        x = x + self.dropout(update)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(x.dtype)
        return x


class AttentionStack(nn.Module):
    """A stack of method-specific dense attention layers."""

    def __init__(self, layers: list[ResidualAttentionLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Tensor,
        coords_c: Tensor,
        token_mask: Optional[Tensor] = None,
        coord_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        min_tokens = int(os.environ.get("LF3DROPE_CHECKPOINT_MIN_TOKENS", "0"))
        use_checkpoint = (
            min_tokens > 0
            and self.training
            and torch.is_grad_enabled()
            and x.size(1) >= min_tokens
        )
        for layer in self.layers:
            args = (x, coords_ca, coords_n, coords_c)
            kwargs = dict(
                token_mask=token_mask, coord_mask=coord_mask, edge_index=edge_index
            )
            if use_checkpoint:
                # Recompute long-chain activations in FP32, preserving dropout RNG.
                x = checkpoint(
                    layer, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
                )
            else:
                x = layer(*args, **kwargs)
        return x


class CrossAttentionLayer(nn.Module):
    """Antigen-to-antibody content attention, matching the LF3DRoPE source."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        key_padding_mask = None if memory_mask is None else ~memory_mask.bool()
        attended, _ = self.cross_attn(
            self.norm_q(target),
            self.norm_mem(memory),
            self.norm_mem(memory),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = target + self.dropout(attended)
        output = output + self.dropout(self.ffn(self.norm2(output)))
        if target_mask is not None:
            output = output * target_mask.unsqueeze(-1).to(output.dtype)
        return output


EncoderFactory = Callable[[int], nn.Module]


class GeometryPredictor(nn.Module):
    """Common antibody-conditioned epitope predictor.

    ``encoder_factory`` receives the requested layer count and must return a
    chain encoder with the interface implemented by :class:`AttentionStack`.
    """

    def __init__(
        self,
        input_ab_dim: int,
        input_ag_dim: int,
        encoder_factory: EncoderFactory,
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
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_cdr_type = bool(use_cdr_type)
        self.use_global_context = bool(use_global_context)
        self.use_pair_mil = bool(use_pair_mil)

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
            for _ in range(n_layers_cross)
        )

        if self.use_global_context:
            self.global_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            head_input_dim = d_model * 5
        else:
            head_input_dim = d_model * 3

        self.node_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
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
            self.logit_gate = nn.Parameter(torch.tensor(float(mil_weight)))

    @staticmethod
    def _fallback_coord_mask(coords_ca: Tensor, coords_n: Tensor, coords_c: Tensor) -> Tensor:
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
            mask_b = torch.ones(batch_size, n_ab, dtype=torch.bool, device=x_b.device)
        if mask_g is None:
            mask_g = torch.ones(batch_size, n_ag, dtype=torch.bool, device=x_g.device)
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

        hidden_cross = hidden_ag
        for layer in self.cross_layers:
            hidden_cross = layer(
                hidden_cross,
                hidden_ab,
                target_mask=mask_g,
                memory_mask=mask_b,
            )

        features = [hidden_ag, hidden_cross, hidden_ag * hidden_cross]
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
        if not self.use_pair_mil:
            return node_logits

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
        logits = node_logits + self.logit_gate * mil_logits
        return logits * mask_g.to(logits.dtype)


__all__ = [
    "AttentionStack",
    "CrossAttentionLayer",
    "DenseSelfAttentionBase",
    "GeometryPredictor",
    "ResidualAttentionLayer",
    "masked_logmeanexp",
    "masked_max",
    "masked_mean",
]
