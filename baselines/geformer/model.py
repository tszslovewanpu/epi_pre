"""Geoformer IPE intra-chain encoder for the shared AsEP predictor.

Adapted from Microsoft AI2BMD's Geoformer branch, commit
20ba4c1fe3045de862f03ddbb84c10811884f5f8 (MIT; see LICENSE).  The spelling of
this package follows the existing ``geformer`` experiment directory; the
published architecture is Geoformer and IPE means Interatomic Positional
Encoding.

We replace atom-type embeddings with the benchmark's residue feature projection
and represent only active radius-graph pairs.  Each stored pair follows the
official dense equations: exponential-normal distance features, multiplicative
IPE attention with SiLU and cosine cutoff, equivariant direction aggregation,
and invariant gated pair updates.  The original post-LayerNorm/SiLU blocks are
retained.  Neither N--CA--C frames nor the dataset's fixed graph edges are used.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from baselines.se3bench.common import GeometryPredictor


class CosineCutoff(nn.Module):
    """Official smooth radial envelope, zero at and beyond the cutoff."""

    def __init__(self, cutoff: float = 5.0) -> None:
        super().__init__()
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        self.cutoff = float(cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        envelope = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return envelope * (distances < self.cutoff).to(distances.dtype)


class ExpNormalSmearing(nn.Module):
    """Geoformer's exponential-normal RBF, including its cosine envelope."""

    def __init__(
        self, cutoff: float = 5.0, num_rbf: int = 64, trainable: bool = False
    ) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf must be positive")
        self.cutoff_fn = CosineCutoff(cutoff)
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        self.alpha = 5.0 / self.cutoff
        means, betas = self._initial_params()
        if self.trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[Tensor, Tensor]:
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1.0, self.num_rbf)
        beta = (2.0 / self.num_rbf * (1.0 - start_value)) ** -2
        return means, beta.repeat(self.num_rbf)

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        with torch.no_grad():
            self.means.copy_(means)
            self.betas.copy_(betas)

    def forward(self, distance: Tensor) -> Tensor:
        distance = distance.unsqueeze(-1)
        return self.cutoff_fn(distance) * torch.exp(
            -self.betas * (torch.exp(-self.alpha * distance) - self.means).square()
        )


class VecLayerNorm(nn.Module):
    """Official optional vector normalization, with an explicit ``none`` mode.

    Vector axes occupy the penultimate dimension and feature channels the last;
    all operations commute with rotation.  The published QM9 example selects
    ``none``.  The non-trainable weights match the original IPE module.
    """

    def __init__(self, hidden_channels: int, norm_type: str = "none") -> None:
        super().__init__()
        if norm_type not in {"none", "max_min"}:
            raise ValueError("norm_type must be 'none' or 'max_min'")
        self.norm_type = norm_type
        self.eps = 1.0e-6
        self.register_buffer("weight", torch.ones(hidden_channels))

    def forward(self, vec: Tensor) -> Tensor:
        if vec.shape[-2] != 3:
            raise ValueError("vectors must have shape [..., 3, features]")
        if self.norm_type == "max_min":
            norm = torch.linalg.vector_norm(vec, dim=-2, keepdim=True)
            norm = norm.clamp_min(self.eps)
            direction = vec / norm
            maximum = norm.max(dim=-1, keepdim=True).values
            minimum = norm.min(dim=-1, keepdim=True).values
            span = maximum - minimum
            span = torch.where(span == 0, torch.ones_like(span), span)
            vec = ((norm - minimum) / span) * direction
        return vec * self.weight


def _reset_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class GeoformerMultiHeadAttention(nn.Module):
    """Sparse evaluation of the official dense IPE attention equations.

    ``edge_index[0]`` is query i and ``edge_index[1]`` is key j.  Its vector
    points from i to j.  All edges must be valid and within the cutoff, including
    self loops (whose vector is zero).  IPE is stored as [E, F], never [N, N, F].
    No softmax, head-dimension rescaling, or attention-probability dropout is
    applied: none occurs in the official Geoformer forward implementation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        cutoff: float = 5.0,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        self.embedding_dim = int(d_model)
        self.num_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.act = nn.SiLU()
        self.cutoff = CosineCutoff(cutoff)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dk_proj = nn.Linear(d_model, d_model)
        self.du_update_proj = nn.Linear(d_model, d_model)
        self.du_norm = VecLayerNorm(d_model, norm_type=norm_type)
        self.dihedral_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.edge_attr_update = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.q_proj, self.k_proj, self.v_proj, self.dk_proj,
            self.du_update_proj, self.dihedral_proj, self.edge_attr_update,
        ):
            _reset_linear(layer)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        vec: Tensor,
        dist: Tensor,
        edge_attr: Tensor,
        *,
        compute_ipe: bool = True,
    ) -> tuple[Tensor, Optional[Tensor]]:
        query, key = edge_index.unbind(0)
        n_nodes = x.shape[0]
        q = self.q_proj(x).reshape(n_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(n_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(n_nodes, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(edge_attr)).reshape(
            edge_attr.shape[0], self.num_heads, self.head_dim
        )
        score = (q[query] * k[key] * dk).sum(dim=-1)
        weights = self.act(score) * self.cutoff(dist).unsqueeze(-1)
        messages = (weights.unsqueeze(-1) * v[key]).reshape(
            edge_attr.shape[0], self.embedding_dim
        )
        attention = x.new_zeros((n_nodes, self.embedding_dim)).index_add(
            0, query, messages
        )

        # The final layer's pair update has no consumer in node prediction.
        # Do not evaluate that dead branch: its cubic products can overflow
        # even while every attention/node value remains finite.
        if not compute_ipe:
            return attention, None

        # Keep the biased du projection after attention weighting: moving it
        # before the weighting would change the official equations.
        vector_messages = self.du_update_proj(messages).unsqueeze(-2) * vec.unsqueeze(-1)
        du = x.new_zeros((n_nodes, 3, self.embedding_dim)).index_add(
            0, query, vector_messages
        )
        ws, wt = self.dihedral_proj(self.du_norm(du)).chunk(2, dim=-1)
        ipe = (ws[query] * wt[key]).sum(dim=-2)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe
        return attention, ipe


class GeoformerAttnBlock(nn.Module):
    """Official post-normalized attention / residual IPE / SiLU FFN block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
        *,
        cutoff: float = 5.0,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        self.self_attn = GeoformerMultiHeadAttention(
            d_model, n_heads, cutoff=cutoff, norm_type=norm_type
        )
        self.dropout_module = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        _reset_linear(self.ffn[0])
        _reset_linear(self.ffn[3])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        vec: Tensor,
        dist: Tensor,
        edge_attr: Tensor,
        *,
        compute_ipe: bool = True,
    ) -> tuple[Tensor, Tensor]:
        dx, dedge = self.self_attn(
            x, edge_index, vec, dist, edge_attr, compute_ipe=compute_ipe
        )
        x = self.attn_layer_norm(x + self.dropout_module(dx))
        if dedge is not None:
            edge_attr = edge_attr + dedge
        x = self.final_layer_norm(x + self.dropout_module(self.ffn(x)))
        return x, edge_attr


class GeoformerEncoder(nn.Module):
    """Residue-level IPE encoder with a freshly constructed CA radius graph.

    Coordinates use the dataset's Angstrom units, including any train-time
    jitter applied by the shared trainer.  Coordinate-invalid tokens have no
    geometric edges but retain their scalar representation and FFN updates.
    Padded tokens are zeroed.  Explicit coordinate masks allow valid CA atoms
    at the origin.  Only CA coordinates are needed by this encoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        dim_ff: int = 512,
        dropout: float = 0.1,
        cutoff: float = 5.0,
        num_rbf: int = 64,
        rbf_trainable: bool = False,
        norm_type: str = "none",
        checkpoint_min_tokens: int = 512,
    ) -> None:
        super().__init__()
        if n_layers < 0 or checkpoint_min_tokens < 0:
            raise ValueError("layer count and checkpoint threshold must be non-negative")
        self.d_model = int(d_model)
        self.cutoff = float(cutoff)
        self.checkpoint_min_tokens = int(checkpoint_min_tokens)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, rbf_trainable)
        self.dist_proj = nn.Linear(num_rbf, d_model)
        self.act = nn.SiLU()
        self.x_in_layernorm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            GeoformerAttnBlock(
                d_model, n_heads, dim_ff, dropout, cutoff=cutoff, norm_type=norm_type
            )
            for _ in range(n_layers)
        )
        _reset_linear(self.dist_proj)

    def _geometry(
        self, coords_ca: Tensor, geometry_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = coords_ca.shape
        clean_coords = torch.nan_to_num(coords_ca, nan=0.0, posinf=0.0, neginf=0.0)
        indices = []
        # Select topology without retaining a dense autograd graph.  Distances
        # and directions are then recomputed on the retained pairs, preserving
        # coordinate gradients away from radius boundaries.
        with torch.no_grad():
            for sample in range(batch):
                positions = clean_coords[sample]
                distance = torch.linalg.vector_norm(
                    positions.unsqueeze(0) - positions.unsqueeze(1), dim=-1
                )
                valid = geometry_mask[sample]
                adjacency = (distance < self.cutoff) & valid[:, None] & valid[None, :]
                query, key = adjacency.nonzero(as_tuple=True)
                indices.append(torch.stack((query, key)) + sample * length)
        edges = torch.cat(indices, dim=1)
        query, key = edges.unbind(0)
        flat_coords = clean_coords.reshape(batch * length, 3)
        displacement = flat_coords[key] - flat_coords[query]
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        # The official implementation uses d + 1e-8 and exact zero self vectors.
        direction = displacement / (distance.unsqueeze(-1) + 1.0e-8)
        return edges, direction, distance

    def forward_with_edges(
        self,
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Optional[Tensor] = None,
        coords_c: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        coord_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return node states, active edge indices, and final IPE for diagnostics.

        The input ``edge_index`` is deliberately ignored: IPE uses a radius
        graph recomputed from the current coordinates.  Returned indices address
        the flattened [B*N] nodes, with no edges between batch elements.

        This diagnostic computes the complete final IPE, unlike node-only
        ``forward``.  An overflowing diagnostic IPE is returned unchanged;
        it is never multiplied by zero and added to otherwise finite nodes.
        """
        return self._forward_with_edges(
            x, coords_ca, coords_n, coords_c, token_mask, coord_mask, edge_index,
            compute_final_ipe=True,
        )

    def _terminal_ipe_zero_dependency(self, x: Tensor) -> Tensor:
        """Keep exact zero gradients for parameters of the unused pair update.

        Summing an empty parameter view produces an autograd-connected zero
        without reading or reducing any potentially large activation.  This
        preserves the original finite-path optimizer/weight-decay behavior:
        the last IPE-only parameters receive zero gradients, not ``None``.
        """
        if self.layers:
            attention = self.layers[-1].self_attn
            modules = (
                attention.du_update_proj, attention.dihedral_proj,
                attention.edge_attr_update,
            )
        else:
            modules = (self.distance_expansion, self.dist_proj)
        zero = x.new_zeros(())
        for module in modules:
            for parameter in module.parameters():
                zero = zero + parameter.reshape(-1)[:0].sum()
        return zero

    def _forward_with_edges(
        self,
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Optional[Tensor] = None,
        coords_c: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        coord_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        *,
        compute_final_ipe: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del coords_n, coords_c, edge_index
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, N, {self.d_model}]")
        batch, length, channels = x.shape
        if batch < 1 or length < 1:
            raise ValueError("batch and sequence length must be positive")
        if coords_ca.shape != (batch, length, 3):
            raise ValueError("coords_ca must have shape [B, N, 3]")
        coords_ca = coords_ca.to(device=x.device, dtype=x.dtype)
        finite_coords = torch.isfinite(coords_ca).all(dim=-1)
        if token_mask is None:
            token_mask = torch.ones((batch, length), dtype=torch.bool, device=x.device)
        if token_mask.shape != (batch, length):
            raise ValueError("token_mask must have shape [B, N]")
        token_mask = token_mask.to(device=x.device, dtype=torch.bool)
        if coord_mask is None:
            coord_mask = finite_coords
        if coord_mask.shape != (batch, length):
            raise ValueError("coord_mask must have shape [B, N]")
        geometry_mask = coord_mask.to(device=x.device, dtype=torch.bool) & finite_coords & token_mask
        edges, direction, distance = self._geometry(coords_ca, geometry_mask)
        edge_attr = self.act(self.dist_proj(self.distance_expansion(distance)))
        node_mask = token_mask.reshape(batch * length, 1)
        x = self.x_in_layernorm(x).reshape(batch * length, channels)
        x = x.masked_fill(~node_mask, 0.0)
        use_checkpoint = (
            self.checkpoint_min_tokens > 0
            and length >= self.checkpoint_min_tokens
            and self.training
            and torch.is_grad_enabled()
        )
        for index, layer in enumerate(self.layers):
            args = (x, edges, direction, distance, edge_attr)
            compute_ipe = compute_final_ipe or index < len(self.layers) - 1
            if use_checkpoint:
                x, edge_attr = checkpoint(
                    layer, *args, use_reentrant=False, preserve_rng_state=True,
                    compute_ipe=compute_ipe,
                )
            else:
                x, edge_attr = layer(*args, compute_ipe=compute_ipe)
            x = x.masked_fill(~node_mask, 0.0)
        # Algebraically equivalent to the original zero dependency, without
        # inf*0 in the forward pass or 0*inf through a dead backward branch.
        x = x + self._terminal_ipe_zero_dependency(x)
        return x.reshape(batch, length, channels), edges, edge_attr

    def forward(
        self,
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Optional[Tensor] = None,
        coords_c: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        coord_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        return self._forward_with_edges(
            x, coords_ca, coords_n, coords_c, token_mask, coord_mask, edge_index,
            compute_final_ipe=False,
        )[0]


class Model(GeometryPredictor):
    """Shared epitope predictor with Geoformer IPE intra-chain modules."""

    @staticmethod
    def _fallback_coord_mask(
        coords_ca: Tensor, coords_n: Tensor, coords_c: Tensor
    ) -> Tensor:
        # IPE needs CA positions only.  Missing N/C atoms must not remove a
        # valid CA, and a physical atom at the origin is not missing geometry.
        # The dataset supplies explicit masks for zero-filled missing entries.
        del coords_n, coords_c
        return torch.isfinite(coords_ca).all(dim=-1)

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

        def encoder_factory(n_layers: int) -> GeoformerEncoder:
            return GeoformerEncoder(
                d_model, n_heads, n_layers,
                dim_ff=dim_ff,
                dropout=dropout,
                cutoff=float(config.get("geformer_cutoff", 5.0)),
                num_rbf=int(config.get("geformer_num_rbf", 64)),
                rbf_trainable=bool(config.get("geformer_rbf_trainable", False)),
                norm_type=str(config.get("geformer_norm_type", "none")),
                checkpoint_min_tokens=int(config.get("geformer_checkpoint_min_tokens", 512)),
            )

        super().__init__(
            input_ab_dim, input_ag_dim, encoder_factory,
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


__all__ = [
    "CosineCutoff", "ExpNormalSmearing", "VecLayerNorm",
    "GeoformerMultiHeadAttention", "GeoformerAttnBlock", "GeoformerEncoder", "Model",
]
