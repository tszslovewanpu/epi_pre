"""Dense LF3DRoPE with an additional geometry-conditioned value message.

The original LF logits, geometric quantities, shared predictor, and initial
parameters are preserved. For head h this adapter adds

    sum_j dropout(softmax(A))_hij * W_h [RBF(d_ij), Fourier(u_ij)]

to the ordinary weighted values. The same attention weights serve both terms.
W starts at zero. Query chunking and non-reentrant activation checkpointing
bound geometric intermediates; linearity lets us aggregate the 40 geometric
features before projecting, so no [B,H,N,N,head_dim] message tensor is built.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from baselines.lf3drope.model import (
    LocalFrame3DRoPEAttention,
    Model as OriginalLFModel,
    build_local_frames,
)


class _ZeroInitializedLinear(nn.Linear):
    """A plain bias-free linear map whose construction consumes no RNG draws."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class GeometryValueAttention(LocalFrame3DRoPEAttention):
    """Wrap an existing LF attention without renaming/reinitializing its state.

    ``coord_mask`` has exactly the original attention's meaning: when supplied,
    false endpoints have no geometric contribution. The shared full predictor
    supplies its usual fallback coordinate mask when callers omit it. Padding
    additionally removes geometric values and masks the final query output.
    """

    def __init__(
        self,
        original: LocalFrame3DRoPEAttention,
        *,
        query_chunk_size: int = 128,
        checkpoint_attention: bool = True,
    ) -> None:
        # Calling the original constructor here would draw new random weights.
        nn.Module.__init__(self)
        if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError("attention_query_chunk_size must be a positive integer")
        if not isinstance(checkpoint_attention, bool):
            raise ValueError("attention_checkpoint must be boolean")
        if original.distance_bias is None or original.fourier_bias is None:
            raise ValueError("Geometry-value LF requires both original RBF and Fourier biases")
        self.query_chunk_size = query_chunk_size
        self.checkpoint_attention = checkpoint_attention
        for name in ("d_model", "n_heads", "head_dim", "scale", "n_rope_freqs", "rope_dim", "rest_dim"):
            setattr(self, name, getattr(original, name))
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "attention_dropout", "distance_bias", "fourier_bias"):
            setattr(self, name, getattr(original, name))
        self.register_buffer("rope_frequencies", original.rope_frequencies)
        feature_dim = self.distance_bias.centers.numel() + 6 * self.fourier_bias.frequencies.numel()
        self.geometry_value_projection = _ZeroInitializedLinear(
            feature_dim,
            self.d_model,
            bias=False,
            device=original.q_proj.weight.device,
            dtype=original.q_proj.weight.dtype,
        )
        self.train(original.training)

    def _pair_geometry(
        self,
        coords: Tensor,
        frames: Tensor,
        coord_mask: Optional[Tensor],
        start: int,
        stop: int,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        displacement = coords[:, None, :, :] - coords[:, start:stop, None, :]
        local = torch.einsum("bijc,bica->bija", displacement, frames[:, start:stop])
        pair_mask = None
        if coord_mask is not None:
            pair_mask = coord_mask[:, start:stop, None] & coord_mask[:, None, :]
            local = local.masked_fill(~pair_mask[..., None], 0.0)
            displacement = displacement.masked_fill(~pair_mask[..., None], 0.0)
        distance = displacement.norm(dim=-1)
        # These are the ORIGINAL LF basis, learned gamma, and fixed frequencies.
        rbf = torch.exp(
            -self.distance_bias.gamma.abs()
            * (distance[..., None] - self.distance_bias.centers).square()
        )
        scaled = local[..., None] * self.fourier_bias.frequencies
        fourier = torch.cat((scaled.sin(), scaled.cos()), dim=-1).flatten(start_dim=-2)
        return local, rbf, fourier, pair_mask

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
        local, rbf, fourier, pair_mask = self._pair_geometry(coords, frames, coord_mask, start, stop)
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
        distance_bias = self.distance_bias.projection(rbf).permute(0, 3, 1, 2).contiguous()
        fourier_bias = self.fourier_bias.projection(fourier).permute(0, 3, 1, 2).contiguous()
        if pair_mask is not None:
            distance_bias = distance_bias.masked_fill(~pair_mask[:, None], 0.0)
            fourier_bias = fourier_bias.masked_fill(~pair_mask[:, None], 0.0)
        return logits + distance_bias + fourier_bias

    def _geometry_value_chunk(
        self,
        weights: Tensor,
        coords: Tensor,
        frames: Tensor,
        coord_mask: Optional[Tensor],
        token_mask: Optional[Tensor],
        start: int,
        stop: int,
    ) -> Tensor:
        _, rbf, fourier, pair_mask = self._pair_geometry(coords, frames, coord_mask, start, stop)
        features = torch.cat((rbf, fourier), dim=-1)
        if token_mask is not None:
            token_pairs = token_mask[:, start:stop, None] & token_mask[:, None, :]
            pair_mask = token_pairs if pair_mask is None else pair_mask & token_pairs
        if pair_mask is not None:
            features = features.masked_fill(~pair_mask[..., None], 0.0)
        # sum_j a_hij W_h phi_ij = W_h sum_j a_hij phi_ij.
        # The full d_model pair message is never materialized.
        pooled = torch.einsum("bhij,bijf->bhif", weights, features)
        projection = self.geometry_value_projection.weight.reshape(self.n_heads, self.head_dim, -1)
        return torch.einsum("bhif,hdf->bhid", pooled, projection)

    def _run_chunk(self, operation: Callable[..., Tensor], *args: Any) -> Tensor:
        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            return checkpoint(operation, *args, use_reentrant=False, preserve_rng_state=True)
        return operation(*args)

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
        del edge_index  # This variant retains the complete dense LF topology.
        query, key, value = self.project_qkv(x)
        frames = build_local_frames(coords_ca, coords_n, coords_c)
        coords = torch.nan_to_num(coords_ca, nan=0.0, posinf=0.0, neginf=0.0)
        length = x.shape[1]
        intervals = [(start, min(start + self.query_chunk_size, length)) for start in range(0, length, self.query_chunk_size)]
        logits = torch.cat([
            self._run_chunk(self._logits_chunk, query[:, :, start:stop], key, coords, frames, coord_mask, start, stop)
            for start, stop in intervals
        ], dim=2)
        if token_mask is not None:
            logits = logits.masked_fill(~token_mask[:, None, None, :], -1.0e4)
        # One full-shaped draw preserves LF's dropout RNG consumption regardless
        # of query chunk size, and gives exactly the same mask to both messages.
        weights = self.attention_dropout(torch.softmax(logits, dim=-1))
        ordinary_values = torch.matmul(weights, value)
        geometric_values = torch.cat([
            self._run_chunk(self._geometry_value_chunk, weights[:, :, start:stop], coords, frames, coord_mask, token_mask, start, stop)
            for start, stop in intervals
        ], dim=2)
        output = ordinary_values + geometric_values
        output = output.transpose(1, 2).contiguous().view(x.shape[0], length, self.d_model)
        output = self.out_proj(output)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(output.dtype)
        return output


class Model(OriginalLFModel):
    """Original complete predictor plus zero-initialized geometric value maps."""

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
        if not config.get("use_distance_bias", True) or not config.get("use_fourier_bias", True):
            raise ValueError("Geometry-value LF preserves both original RBF and Fourier biases")
        # Build EVERY shared module before adding anything, including prediction
        # heads. Thus all original keys/values and the final RNG state match LF.
        super().__init__(
            input_ab_dim, input_ag_dim, d_model=d_model, n_heads=n_heads,
            n_layers_ab=n_layers_ab, n_layers_ag=n_layers_ag,
            n_layers_cross=n_layers_cross, dim_ff=dim_ff, dropout=dropout,
            use_cdr_type=use_cdr_type, use_global_context=use_global_context,
            use_pair_mil=use_pair_mil, mil_weight=mil_weight, geometry_config=config,
        )
        for encoder in (self.ab_encoder, self.ag_encoder):
            for layer in encoder.layers:
                layer.attention = GeometryValueAttention(
                    layer.attention,
                    query_chunk_size=config.get("attention_query_chunk_size", 128),
                    checkpoint_attention=config.get("attention_checkpoint", True),
                )


__all__ = ["GeometryValueAttention", "Model"]
