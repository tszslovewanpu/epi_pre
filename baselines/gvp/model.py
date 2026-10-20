"""Sparse GVP-style geometric encoder for the shared AsEP predictor.

Scalar channels are SE(3)-invariant and vector channels are SO(3)-equivariant
throughout the chain encoder.  Only the final scalar state is exposed to the
shared cross-attention and prediction heads.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from baselines.se3bench.common import GeometryPredictor


def _safe_unit(vector: Tensor, eps: float) -> Tensor:
    """Normalize vectors, mapping vectors below ``eps`` to exact zeros."""

    squared_norm = vector.square().sum(dim=-1, keepdim=True)
    inverse_norm = torch.rsqrt(squared_norm.clamp_min(eps * eps))
    valid = squared_norm > eps * eps
    return vector * inverse_norm * valid.to(vector.dtype)


class VectorLinear(nn.Module):
    """Linear channel mixing for vectors, with no bias in XYZ space."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if input_channels < 1 or output_channels < 1:
            raise ValueError("vector channel counts must be positive")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.weight = nn.Parameter(
            torch.empty(self.output_channels, self.input_channels)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, vectors: Tensor) -> Tensor:
        if vectors.shape[-2:] != (self.input_channels, 3):
            raise ValueError(
                "vectors must end in "
                f"[{self.input_channels}, 3], got {tuple(vectors.shape)}"
            )
        return torch.einsum("...ic,oi->...oc", vectors, self.weight)


class VectorLayerNorm(nn.Module):
    """Equivariant RMS normalization over vector channels."""

    def __init__(self, channels: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, vectors: Tensor) -> Tensor:
        if vectors.shape[-2:] != (self.channels, 3):
            raise ValueError(
                f"vectors must end in [{self.channels}, 3], "
                f"got {tuple(vectors.shape)}"
            )
        squared_rms = vectors.square().sum(dim=-1).mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(
            squared_rms.clamp_min(self.eps * self.eps)
        )
        normalized = vectors * inverse_rms.unsqueeze(-1)
        return normalized * self.weight[:, None]


class VectorDropout(nn.Module):
    """Drop whole vector channels while sharing the mask across XYZ."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("dropout probability must be in [0, 1]")
        self.probability = float(probability)

    def forward(self, vectors: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return vectors
        if self.probability == 1.0:
            return torch.zeros_like(vectors)
        keep = torch.rand_like(vectors[..., 0]) >= self.probability
        return vectors * keep.unsqueeze(-1).to(vectors.dtype) / (
            1.0 - self.probability
        )


class GeometricVectorPerceptron(nn.Module):
    """A scalar/vector GVP transformation.

    Vector norms enter the invariant scalar pathway.  The output vectors are
    obtained only through bias-free channel mixing and scalar-derived gates.
    """

    def __init__(
        self,
        scalar_input_dim: int,
        vector_input_channels: int,
        scalar_output_dim: int,
        vector_output_channels: int,
        *,
        vector_eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.vector_eps = float(vector_eps)
        hidden_vector_channels = max(
            int(vector_input_channels),
            int(vector_output_channels),
        )
        self.vector_hidden = VectorLinear(
            vector_input_channels,
            hidden_vector_channels,
        )
        self.scalar_projection = nn.Linear(
            scalar_input_dim + hidden_vector_channels,
            scalar_output_dim,
        )
        self.scalar_activation = nn.GELU()
        self.vector_projection = VectorLinear(
            hidden_vector_channels,
            vector_output_channels,
        )
        self.vector_gate = nn.Linear(
            scalar_output_dim,
            vector_output_channels,
        )

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden_vectors = self.vector_hidden(vectors)
        squared_norms = hidden_vectors.square().sum(dim=-1)
        vector_norms = torch.sqrt(
            squared_norms.clamp_min(self.vector_eps * self.vector_eps)
        )
        output_scalars = self.scalar_activation(
            self.scalar_projection(torch.cat([scalars, vector_norms], dim=-1))
        )
        output_vectors = self.vector_projection(hidden_vectors)
        gates = torch.sigmoid(self.vector_gate(output_scalars)).unsqueeze(-1)
        return output_scalars, output_vectors * gates


class RBFExpansion(nn.Module):
    """Invariant radial basis expansion of edge distances."""

    def __init__(
        self,
        n_rbf: int = 16,
        minimum: float = 0.0,
        maximum: float = 30.0,
    ) -> None:
        super().__init__()
        if n_rbf < 1:
            raise ValueError("n_rbf must be positive")
        if maximum <= minimum:
            raise ValueError("RBF maximum must be greater than minimum")
        centers = torch.linspace(minimum, maximum, int(n_rbf))
        spacing = (maximum - minimum) / max(int(n_rbf) - 1, 1)
        self.register_buffer("centers", centers)
        self.register_buffer(
            "gamma",
            torch.tensor(1.0 / max(spacing, 1.0e-6) ** 2),
        )

    def forward(self, distance: Tensor) -> Tensor:
        return torch.exp(
            -self.gamma
            * (distance.unsqueeze(-1) - self.centers).square()
        )


class SparseGVPLayer(nn.Module):
    """One sparse scalar/vector message-passing layer."""

    def __init__(
        self,
        scalar_dim: int,
        vector_channels: int,
        n_rbf: int,
        dropout: float,
        *,
        vector_eps: float,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.vector_eps = float(vector_eps)

        # Each directed message sees source and target node states, invariant
        # RBF edge features, and one equivariant target-to-source unit vector.
        self.message_gvp = GeometricVectorPerceptron(
            scalar_input_dim=2 * scalar_dim + n_rbf,
            vector_input_channels=2 * vector_channels + 1,
            scalar_output_dim=scalar_dim,
            vector_output_channels=vector_channels,
            vector_eps=vector_eps,
        )
        self.update_gvp = GeometricVectorPerceptron(
            scalar_input_dim=2 * scalar_dim,
            vector_input_channels=2 * vector_channels,
            scalar_output_dim=scalar_dim,
            vector_output_channels=vector_channels,
            vector_eps=vector_eps,
        )

        self.input_scalar_norm = nn.LayerNorm(scalar_dim)
        self.input_vector_norm = VectorLayerNorm(
            vector_channels,
            eps=vector_eps,
        )
        self.scalar_dropout = nn.Dropout(dropout)
        self.vector_dropout = VectorDropout(dropout)

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        coords_ca: Tensor,
        edge_index: Tensor,
        edge_rbf: Tensor,
        token_mask: Tensor,
        geometry_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized_scalars = self.input_scalar_norm(scalars)
        normalized_vectors = self.input_vector_norm(vectors)

        aggregate_scalars = torch.zeros_like(scalars)
        aggregate_vectors = torch.zeros_like(vectors)
        degree = scalars.new_zeros(scalars.shape[0])

        if edge_index.shape[1] > 0:
            source, target = edge_index[0], edge_index[1]
            displacement = coords_ca[source] - coords_ca[target]
            unit_displacement = _safe_unit(
                displacement,
                self.vector_eps,
            ).unsqueeze(-2)

            message_scalars, message_vectors = self.message_gvp(
                torch.cat(
                    [
                        normalized_scalars[source],
                        normalized_scalars[target],
                        edge_rbf,
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        normalized_vectors[source],
                        normalized_vectors[target],
                        unit_displacement,
                    ],
                    dim=-2,
                ),
            )
            aggregate_scalars.index_add_(0, target, message_scalars)
            aggregate_vectors.index_add_(0, target, message_vectors)
            degree.index_add_(
                0,
                target,
                torch.ones_like(target, dtype=scalars.dtype),
            )
            inverse_degree = degree.clamp_min(1.0).reciprocal()
            aggregate_scalars = aggregate_scalars * inverse_degree[:, None]
            aggregate_vectors = aggregate_vectors * inverse_degree[:, None, None]

        scalar_update, vector_update = self.update_gvp(
            torch.cat([normalized_scalars, aggregate_scalars], dim=-1),
            torch.cat([normalized_vectors, aggregate_vectors], dim=-2),
        )
        scalars = scalars + self.scalar_dropout(scalar_update)
        vectors = vectors + self.vector_dropout(vector_update)

        scalars = scalars * token_mask[:, None].to(scalars.dtype)
        vectors = vectors * geometry_mask[:, None, None].to(vectors.dtype)
        return scalars, vectors


class SparseGVPEncoder(nn.Module):
    """Sparse intra-chain GVP encoder with invariant scalar output."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        *,
        n_rbf: int = 16,
        vector_channels: int = 16,
        dropout: float = 0.1,
        rbf_min: float = 0.0,
        rbf_max: float = 30.0,
        vector_eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if vector_channels < 1:
            raise ValueError("vector_channels must be positive")
        if n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        self.d_model = int(d_model)
        self.vector_channels = int(vector_channels)
        self.vector_eps = float(vector_eps)
        self.initial_vector_projection = VectorLinear(3, vector_channels)
        self.rbf = RBFExpansion(n_rbf, rbf_min, rbf_max)
        self.layers = nn.ModuleList(
            SparseGVPLayer(
                d_model,
                vector_channels,
                n_rbf,
                dropout,
                vector_eps=vector_eps,
            )
            for _ in range(n_layers)
        )
        self.output_scalar_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _validate_shapes(
        x: Tensor,
        coords_ca: Tensor,
        coords_n: Tensor,
        coords_c: Tensor,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, N, S], got {tuple(x.shape)}")
        if x.shape[0] != 1:
            raise ValueError(
                "SparseGVPEncoder currently supports batch size B=1 only; "
                f"got B={x.shape[0]}"
            )
        expected = (1, x.shape[1], 3)
        for name, coordinates in (
            ("coords_ca", coords_ca),
            ("coords_n", coords_n),
            ("coords_c", coords_c),
        ):
            if coordinates.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, "
                    f"got {tuple(coordinates.shape)}"
                )

    @staticmethod
    def _prepare_mask(
        mask: Optional[Tensor],
        *,
        length: int,
        device: torch.device,
        default: Tensor,
        name: str,
    ) -> Tensor:
        if mask is None:
            return default
        if mask.shape != (1, length):
            raise ValueError(
                f"{name} must have shape [1, {length}], got {tuple(mask.shape)}"
            )
        return mask.to(device=device, dtype=torch.bool)[0]

    @staticmethod
    def _prepare_edges(
        edge_index: Optional[Tensor],
        *,
        length: int,
        device: torch.device,
        valid_nodes: Tensor,
    ) -> Tensor:
        if edge_index is None:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )
        edges = edge_index.to(device=device, dtype=torch.long)
        if edges.shape[1] == 0:
            return edges

        source, target = edges[0], edges[1]
        in_range = (
            (source >= 0)
            & (source < length)
            & (target >= 0)
            & (target < length)
        )
        edges = edges[:, in_range]
        if edges.shape[1] == 0:
            return edges
        source, target = edges[0], edges[1]
        valid_pairs = valid_nodes[source] & valid_nodes[target]
        return edges[:, valid_pairs]

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
        self._validate_shapes(x, coords_ca, coords_n, coords_c)
        length = x.shape[1]
        device, dtype = x.device, x.dtype
        coords_ca = coords_ca.to(device=device, dtype=dtype)
        coords_n = coords_n.to(device=device, dtype=dtype)
        coords_c = coords_c.to(device=device, dtype=dtype)

        finite_backbone = (
            torch.isfinite(coords_ca).all(dim=-1)
            & torch.isfinite(coords_n).all(dim=-1)
            & torch.isfinite(coords_c).all(dim=-1)
        )[0]
        nonzero_backbone = (
            coords_ca.abs().sum(dim=-1)
            + coords_n.abs().sum(dim=-1)
            + coords_c.abs().sum(dim=-1)
        )[0] > 0
        fallback_coord_mask = finite_backbone & nonzero_backbone
        token_mask_1d = self._prepare_mask(
            token_mask,
            length=length,
            device=device,
            default=torch.ones(length, dtype=torch.bool, device=device),
            name="token_mask",
        )
        coord_mask_1d = self._prepare_mask(
            coord_mask,
            length=length,
            device=device,
            default=fallback_coord_mask,
            name="coord_mask",
        )
        coord_mask_1d = (
            coord_mask_1d
            & finite_backbone
            & nonzero_backbone
        )
        geometry_mask = token_mask_1d & coord_mask_1d

        clean_ca = torch.nan_to_num(
            coords_ca[0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        clean_n = torch.nan_to_num(
            coords_n[0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        clean_c = torch.nan_to_num(
            coords_c[0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        n_direction = _safe_unit(clean_n - clean_ca, self.vector_eps)
        c_direction = _safe_unit(clean_c - clean_ca, self.vector_eps)
        normal_direction = torch.cross(n_direction, c_direction, dim=-1)
        initial_vectors = torch.stack(
            [n_direction, c_direction, normal_direction],
            dim=-2,
        )
        vectors = self.initial_vector_projection(initial_vectors)
        vectors = vectors * geometry_mask[:, None, None].to(dtype)
        scalars = x[0] * token_mask_1d[:, None].to(dtype)

        edges = self._prepare_edges(
            edge_index,
            length=length,
            device=device,
            valid_nodes=geometry_mask,
        )
        if edges.shape[1] > 0:
            source, target = edges[0], edges[1]
            distance = torch.linalg.vector_norm(
                clean_ca[source] - clean_ca[target],
                dim=-1,
            )
            edge_rbf = self.rbf(distance)
        else:
            edge_rbf = scalars.new_empty((0, self.rbf.centers.numel()))

        for layer in self.layers:
            scalars, vectors = layer(
                scalars,
                vectors,
                clean_ca,
                edges,
                edge_rbf,
                token_mask_1d,
                geometry_mask,
            )

        scalars = self.output_scalar_norm(scalars)
        scalars = scalars * token_mask_1d[:, None].to(dtype)
        return scalars.unsqueeze(0)


class Model(GeometryPredictor):
    """Shared AsEP predictor equipped with a sparse GVP chain encoder."""

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

        def encoder_factory(n_layers: int) -> SparseGVPEncoder:
            return SparseGVPEncoder(
                d_model,
                n_layers,
                n_rbf=int(geometry_config.get("n_rbf", 16)),
                vector_channels=int(
                    geometry_config.get("vector_channels", 16)
                ),
                dropout=dropout,
                rbf_min=float(
                    geometry_config.get(
                        "d_min",
                        geometry_config.get("rbf_min", 0.0),
                    )
                ),
                rbf_max=float(
                    geometry_config.get(
                        "d_max",
                        geometry_config.get("rbf_max", 30.0),
                    )
                ),
                vector_eps=float(geometry_config.get("vector_eps", 1.0e-8)),
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


__all__ = [
    "GeometricVectorPerceptron",
    "Model",
    "SparseGVPEncoder",
    "SparseGVPLayer",
    "VectorLinear",
]
