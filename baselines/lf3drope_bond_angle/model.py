"""LF3DRoPE with invariant N--CA--C bond-angle-conditioned Q/K/V.

Each intra-chain attention layer receives ``x + W[cos(theta), sin(theta)]``
before the original Q/K/V projections. W is a bias-free zero-initialized map.
Angles are recomputed from the current coordinates, including training jitter.
The original local-frame RoPE, RBF/Fourier biases, dense graph, residual blocks,
and prediction heads are unchanged. Query chunking bounds rotary intermediates;
the complete attention dropout draw preserves the original random-number use.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from baselines.lf3drope.model import (
    LocalFrame3DRoPEAttention,
    Model as OriginalLFModel,
    build_local_frames,
)


def backbone_angle_features(
    coords_ca: Tensor,
    coords_n: Tensor,
    coords_c: Tensor,
    coord_mask: Optional[Tensor] = None,
    token_mask: Optional[Tensor] = None,
    *,
    eps: float = 1.0e-8,
) -> tuple[Tensor, Tensor]:
    """Return invariant ``[cos(theta), sin(theta)]`` and its validity mask.

    The angle is between N-CA and C-CA, in [0, pi]. Missing/nonfinite atoms,
    zero-length bonds, and masked nodes contribute exactly zero. No cached or
    unperturbed geometry is used. Collinear nonzero bonds have valid angles even
    though their local frame (handled by the original LF code) is degenerate.
    """
    finite = (
        torch.isfinite(coords_ca).all(dim=-1)
        & torch.isfinite(coords_n).all(dim=-1)
        & torch.isfinite(coords_c).all(dim=-1)
    )
    clean = [torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
             for x in (coords_ca, coords_n, coords_c)]
    n_bond, c_bond = clean[1] - clean[0], clean[2] - clean[0]
    n_length, c_length = n_bond.norm(dim=-1), c_bond.norm(dim=-1)
    valid = (finite & torch.isfinite(n_length) & torch.isfinite(c_length)
             & (n_length > eps) & (c_length > eps))
    if coord_mask is not None:
        valid = valid & coord_mask.bool()
    if token_mask is not None:
        valid = valid & token_mask.bool()
    # Mask before normalization as well as after it: invalid coordinates must
    # neither inject a feature nor create a nonfinite backward intermediate.
    n_bond = torch.where(valid[..., None], n_bond, torch.zeros_like(n_bond))
    c_bond = torch.where(valid[..., None], c_bond, torch.zeros_like(c_bond))
    unit_n = n_bond / n_bond.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit_c = c_bond / c_bond.norm(dim=-1, keepdim=True).clamp_min(eps)
    cosine = (unit_n * unit_c).sum(dim=-1).clamp(-1.0, 1.0)
    sine = torch.cross(unit_n, unit_c, dim=-1).norm(dim=-1).clamp(0.0, 1.0)
    features = torch.stack((cosine, sine), dim=-1)
    return features.masked_fill(~valid[..., None], 0.0), valid


class _ZeroInitializedLinear(nn.Linear):
    """Construction does not consume random draws for the added branch."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class BondAngleAttention(LocalFrame3DRoPEAttention):
    """Existing LF attention plus a two-scalar bond-angle input projection."""

    def __init__(
        self,
        original: LocalFrame3DRoPEAttention,
        *,
        query_chunk_size: int = 128,
        checkpoint_attention: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if (isinstance(query_chunk_size, bool)
                or not isinstance(query_chunk_size, int) or query_chunk_size < 1):
            raise ValueError("attention_query_chunk_size must be a positive integer")
        if not isinstance(checkpoint_attention, bool):
            raise ValueError("attention_checkpoint must be boolean")
        self.query_chunk_size = query_chunk_size
        self.checkpoint_attention = checkpoint_attention
        for name in ("d_model", "n_heads", "head_dim", "scale", "n_rope_freqs", "rope_dim", "rest_dim"):
            setattr(self, name, getattr(original, name))
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "attention_dropout", "distance_bias", "fourier_bias"):
            setattr(self, name, getattr(original, name))
        self.register_buffer("rope_frequencies", original.rope_frequencies)
        self.angle_projection = _ZeroInitializedLinear(
            2, self.d_model, bias=False,
            device=original.q_proj.weight.device,
            dtype=original.q_proj.weight.dtype,
        )
        self.train(original.training)

    def _logits_chunk(
        self,
        query: Tensor,
        key: Tensor,
        coords: Tensor,
        frames: Tensor,
        coord_mask: Optional[Tensor],
        start: int,
        stop: int,
    ) -> Tensor:
        displacement = coords[:, None, :, :] - coords[:, start:stop, None, :]
        local = torch.einsum("bijc,bica->bija", displacement, frames[:, start:stop])
        pair_mask = None
        if coord_mask is not None:
            pair_mask = coord_mask[:, start:stop, None] & coord_mask[:, None, :]
            local = local.masked_fill(~pair_mask[..., None], 0.0)
            displacement = displacement.masked_fill(~pair_mask[..., None], 0.0)
        batch, heads, queries, _ = query.shape
        keys = key.shape[2]
        qr = query[..., :self.rope_dim].reshape(batch, heads, queries, 3, self.n_rope_freqs, 2)
        kr = key[..., :self.rope_dim].reshape(batch, heads, keys, 3, self.n_rope_freqs, 2)
        qp, kp = qr.unsqueeze(3), kr.unsqueeze(2)
        theta = local[:, None, :, :, :, None] * self.rope_frequencies.view(1, 1, 1, 1, 1, -1)
        cosine, sine = theta.cos(), theta.sin()
        rotated_0 = kp[..., 0] * cosine - kp[..., 1] * sine
        rotated_1 = kp[..., 0] * sine + kp[..., 1] * cosine
        logits = (qp[..., 0] * rotated_0 + qp[..., 1] * rotated_1).sum(dim=(-1, -2))
        if self.rest_dim:
            logits = logits + torch.matmul(query[..., self.rope_dim:], key[..., self.rope_dim:].transpose(-2, -1))
        logits = logits * self.scale
        if self.distance_bias is not None:
            bias = self.distance_bias(displacement.norm(dim=-1))
            if pair_mask is not None:
                bias = bias.masked_fill(~pair_mask[:, None], 0.0)
            logits = logits + bias
        if self.fourier_bias is not None:
            bias = self.fourier_bias(local)
            if pair_mask is not None:
                bias = bias.masked_fill(~pair_mask[:, None], 0.0)
            logits = logits + bias
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
        del edge_index  # The original complete intra-chain topology is retained.
        angles, angle_mask = backbone_angle_features(
            coords_ca, coords_n, coords_c, coord_mask, token_mask,
        )
        embedding = self.angle_projection(angles.to(dtype=x.dtype))
        embedding = embedding.masked_fill(~angle_mask[..., None], 0.0)
        query, key, value = self.project_qkv(x + embedding)
        frames = build_local_frames(coords_ca, coords_n, coords_c)
        coords = torch.nan_to_num(coords_ca, nan=0.0, posinf=0.0, neginf=0.0)
        chunks = []
        for start in range(0, x.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, x.shape[1])
            args = (query[:, :, start:stop], key, coords, frames, coord_mask, start, stop)
            if self.checkpoint_attention and self.training and torch.is_grad_enabled():
                logits = checkpoint(self._logits_chunk, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                logits = self._logits_chunk(*args)
            chunks.append(logits)
        # The original aggregation makes one complete dropout draw; splitting
        # this draw by query would change training randomness and zero-init parity.
        return self.aggregate(torch.cat(chunks, dim=2), value, token_mask)


class Model(OriginalLFModel):
    """Original predictor with one angle projection per intra-chain layer."""

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
        config = dict(geometry_config or {})
        # Construct every original module first, including shared heads. Adding
        # deterministic zero parameters afterward preserves its entire RNG stream.
        super().__init__(
            input_ab_dim, input_ag_dim, d_model=d_model, n_heads=n_heads,
            n_layers_ab=n_layers_ab, n_layers_ag=n_layers_ag,
            n_layers_cross=n_layers_cross, dim_ff=dim_ff, dropout=dropout,
            use_cdr_type=use_cdr_type, use_global_context=use_global_context,
            use_pair_mil=use_pair_mil, mil_weight=mil_weight, geometry_config=config,
        )
        for encoder in (self.ab_encoder, self.ag_encoder):
            for layer in encoder.layers:
                layer.attention = BondAngleAttention(
                    layer.attention,
                    query_chunk_size=config.get("attention_query_chunk_size", 128),
                    checkpoint_attention=config.get("attention_checkpoint", True),
                )


__all__ = ["BondAngleAttention", "Model", "backbone_angle_features"]
