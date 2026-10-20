"""Fixed-graph LF3DRoPE with the original scalar attention and prediction head.

Only the attention domain changes: an edge ``[source, target]`` contributes
key/value ``source`` to query ``target``.  Supplied duplicates and self-loops
are retained, and no edges are added or rebuilt after coordinate jitter.
RoPE, distance RBF, Fourier bias, projections, residuals and FFNs retain the
original LF3DRoPE parameters and equations.  There are no GVP vector states.

As in the current GVP encoder, this interface supports B=1 and removes edges
touching padding or invalid backbone coordinates.  Unlike GVP's permissive
input conversion, absent graphs and malformed indices fail explicitly.
An explicitly empty graph is valid.  Isolated nodes have exactly zero
attention update (including the output-projection bias), while the shared LF
residual/FFN block still provides its ordinary local scalar update.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from baselines.lf3drope.model import (
    LocalFrame3DRoPEAttention,
    Model as DenseLFModel,
    build_local_frames,
)


def _execution_options(chunk_size: int, use_checkpoint: bool) -> tuple[int, bool]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("attention_query_chunk_size must be a positive integer")
    if not isinstance(use_checkpoint, bool):
        raise TypeError("attention_checkpoint must be bool")
    return chunk_size, use_checkpoint


class SparseLocalFrame3DRoPEAttention(LocalFrame3DRoPEAttention):
    """The original LF pair logits, evaluated only on supplied directed edges.

    Query chunks bound the number of simultaneously materialized pair rotary
    features.  During training each chunk can use non-reentrant activation
    checkpointing with preserved dropout RNG.  Only O(ND) node projections and
    O(E) graph indices persist outside these chunks; there is no N-by-N tensor.
    Attention dropout is applied to the stored edge probabilities.  Its random
    layout differs from dense LF, so finite complete-graph equivalence is for
    evaluation/dropout=0, not matched dense dropout realizations.
    """

    def __init__(
        self,
        *args: Any,
        attention_query_chunk_size: int = 128,
        attention_checkpoint: bool = True,
        **kwargs: Any,
    ) -> None:
        options = _execution_options(attention_query_chunk_size, attention_checkpoint)
        super().__init__(*args, **kwargs)
        self.attention_query_chunk_size, self.attention_checkpoint = options

    @classmethod
    def from_dense(
        cls,
        attention: LocalFrame3DRoPEAttention,
        *,
        attention_query_chunk_size: int,
        attention_checkpoint: bool,
    ) -> SparseLocalFrame3DRoPEAttention:
        """Adopt already-initialized modules without drawing any random numbers.

        The original Model is constructed in its exact original order first.
        Adopting its module state retains all parameter/buffer names, values,
        training flags and devices, including the initialization of subsequent
        cross-attention and prediction heads.  This is a normal importable
        subclass, so state-dict loading and full-module pickling both work.
        """
        if type(attention) is not LocalFrame3DRoPEAttention:
            raise TypeError("Expected an unmodified LF3DRoPE attention instance")
        options = _execution_options(attention_query_chunk_size, attention_checkpoint)
        converted = cls.__new__(cls)
        converted.__dict__ = attention.__dict__.copy()
        converted.attention_query_chunk_size, converted.attention_checkpoint = options
        return converted

    def _prepare_inputs(
        self,
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Tensor,
        coords_c: Tensor,
        token_mask: Optional[Tensor],
        coord_mask: Optional[Tensor],
        edge_index: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [1, N, {self.d_model}]")
        if x.shape[0] != 1:
            raise ValueError("Sparse LF3DRoPE currently supports batch size B=1 only")
        if x.shape[1] < 1 or not x.is_floating_point():
            raise ValueError("x must contain at least one node and have floating dtype")
        length = x.shape[1]
        backbone = []
        for name, coordinates in (("coords_ca", coords_ca), ("coords_n", coords_n), ("coords_c", coords_c)):
            if not isinstance(coordinates, Tensor) or coordinates.shape != (1, length, 3):
                raise ValueError(f"{name} must have shape [1, {length}, 3]")
            if not coordinates.is_floating_point():
                raise TypeError(f"{name} must have floating dtype")
            backbone.append(coordinates.to(device=x.device, dtype=x.dtype))
        ca, nitrogen, carbon = backbone
        # Match GVP's geometry mask even when an explicit coord_mask is given.
        finite = torch.stack([torch.isfinite(c).all(-1) for c in backbone]).all(0)
        nonzero = (ca.abs().sum(-1) + nitrogen.abs().sum(-1) + carbon.abs().sum(-1)) > 0
        geometry_mask = finite & nonzero
        for name, mask in (("token_mask", token_mask), ("coord_mask", coord_mask)):
            if mask is None:
                continue
            if not isinstance(mask, Tensor) or mask.shape != (1, length):
                raise ValueError(f"{name} must have shape [1, {length}]")
            if mask.dtype != torch.bool:
                raise TypeError(f"{name} must be bool")
            geometry_mask = geometry_mask & mask.to(device=x.device)
        if edge_index is None:
            raise ValueError("edge_index is required; use an explicit [2, 0] tensor for an empty graph")
        if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E], ordered [source, target]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("edge_index must have int32 or int64 dtype")
        edges = edge_index.to(device=x.device, dtype=torch.long)
        if edges.numel() and ((edges < 0).any() or (edges >= length).any()):
            raise ValueError(f"edge_index contains an index outside [0, {length})")
        valid = geometry_mask[0]
        edges = edges[:, valid[edges[0]] & valid[edges[1]]]
        return ca, nitrogen, carbon, edges

    def _edge_logits(
        self,
        query: Tensor,
        key: Tensor,
        local: Tensor,
        distance: Tensor,
    ) -> Tensor:
        """Return [E, H] logits from [E, H, D] query/key projections."""
        shape = (query.shape[0], self.n_heads, 3, self.n_rope_freqs, 2)
        q = query[..., :self.rope_dim].reshape(shape)
        k = key[..., :self.rope_dim].reshape(shape)
        theta = local[:, None, :, None] * self.rope_frequencies
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotated_0 = k[..., 0] * cosine - k[..., 1] * sine
        rotated_1 = k[..., 0] * sine + k[..., 1] * cosine
        logits = (q[..., 0] * rotated_0 + q[..., 1] * rotated_1).sum((-1, -2))
        if self.rest_dim:
            logits = logits + (query[..., self.rope_dim:] * key[..., self.rope_dim:]).sum(-1)
        logits = logits * self.scale
        # Use the original bias modules unchanged, with a singleton query axis.
        if self.distance_bias is not None:
            logits = logits + self.distance_bias(distance[None, None, :])[0, :, 0, :].transpose(0, 1)
        if self.fourier_bias is not None:
            logits = logits + self.fourier_bias(local[None, None, :, :])[0, :, 0, :].transpose(0, 1)
        return logits

    def _attend_chunk(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        ca: Tensor,
        frames: Tensor,
        source: Tensor,
        target: Tensor,
        query_start: int,
        query_count: int,
    ) -> Tensor:
        # Query=target; direction is key-minus-query in the query's local frame.
        displacement = ca[0, source] - ca[0, target]
        local = torch.einsum("ec,eca->ea", displacement, frames[0, target])
        distance = displacement.norm(dim=-1)
        q = query[0, :, target].transpose(0, 1)
        k = key[0, :, source].transpose(0, 1)
        logits = self._edge_logits(q, k, local, distance)
        local_target = target - query_start
        index = local_target[:, None].expand(-1, self.n_heads)
        # The max is a numerically stabilizing constant: detaching it preserves
        # the exact softmax derivative and avoids a redundant max backward.
        maximum = logits.new_full((query_count, self.n_heads), -torch.inf)
        maximum.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
        numerator = (logits - maximum[local_target]).exp()
        denominator = logits.new_zeros((query_count, self.n_heads)).index_add(0, local_target, numerator)
        # Every gathered target has at least one edge; there is no division by
        # an isolated node's zero denominator and no all-masked softmax row.
        weights = numerator / denominator[local_target]
        weights = self.attention_dropout(weights)
        messages = weights[..., None] * value[0, :, source].transpose(0, 1)
        aggregated = value.new_zeros((query_count, self.n_heads, self.head_dim)).index_add(0, local_target, messages)
        output = self.out_proj(aggregated.reshape(1, query_count, self.d_model))
        has_incoming = torch.zeros(query_count, dtype=torch.bool, device=target.device)
        has_incoming[local_target] = True
        return output.masked_fill(~has_incoming[None, :, None], 0.0)

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
        ca, nitrogen, carbon, edges = self._prepare_inputs(
            x, coords_ca, coords_n, coords_c, token_mask, coord_mask, edge_index
        )
        query, key, value = self.project_qkv(x)
        frames = build_local_frames(ca, nitrogen, carbon)
        outputs = []
        for start in range(0, x.shape[1], self.attention_query_chunk_size):
            count = min(self.attention_query_chunk_size, x.shape[1] - start)
            selected = (edges[1] >= start) & (edges[1] < start + count)
            source, target = edges[:, selected]
            arguments = (query, key, value, ca, frames, source, target, start, count)
            if self.attention_checkpoint and self.training and torch.is_grad_enabled():
                output = checkpoint(
                    self._attend_chunk, *arguments,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                output = self._attend_chunk(*arguments)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


class Model(DenseLFModel):
    """Original LF predictor with only its intra-chain attention domain changed."""

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
        chunk, use_checkpoint = _execution_options(
            config.get("attention_query_chunk_size", 128),
            config.get("attention_checkpoint", True),
        )
        super().__init__(
            input_ab_dim, input_ag_dim, d_model=d_model, n_heads=n_heads,
            n_layers_ab=n_layers_ab, n_layers_ag=n_layers_ag,
            n_layers_cross=n_layers_cross, dim_ff=dim_ff, dropout=dropout,
            use_cdr_type=use_cdr_type, use_global_context=use_global_context,
            use_pair_mil=use_pair_mil, mil_weight=mil_weight,
            geometry_config=config,
        )
        for encoder in (self.ab_encoder, self.ag_encoder):
            for layer in encoder.layers:
                layer.attention = SparseLocalFrame3DRoPEAttention.from_dense(
                    layer.attention, attention_query_chunk_size=chunk,
                    attention_checkpoint=use_checkpoint,
                )


__all__ = ["Model", "SparseLocalFrame3DRoPEAttention"]
