"""Pure PyTorch GeoT blocks adapted from DeepInteract (GPL-3.0).

Upstream: BioinfoMachineLearning/DeepInteract, commit
c78d205465f02ee4ef751dbbafb7ec8f30c75c9a, project/utils/deepinteract_modules.py.
See LICENSE.  DGL edge operations are replaced by gather/index_add; geometry is
provided separately.  The conformation gate order, shared normalization inside
each residual block, intermediate/final block distinction, and initialization
scheme are retained.  This is the intra-chain GeoT encoder, not DeepInteract's
complete pairwise ResNet predictor.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from math import sqrt
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


FEATURE_INDICES = {
    "edge_pos_enc": 0, "edge_weights": 1,
    "edge_dist_feats_start": 2, "edge_dist_feats_end": 20,
    "edge_dir_feats_start": 20, "edge_dir_feats_end": 23,
    "edge_orient_feats_start": 23, "edge_orient_feats_end": 27,
    "edge_amide_angles": 27,
}


def glorot_orthogonal(tensor: Tensor, scale: float = 2.0) -> None:
    """The upstream orthogonal initialization with variance rescaling."""
    with torch.no_grad():
        nn.init.orthogonal_(tensor)
        variance = tensor.var()
        factor = scale / ((tensor.size(-2) + tensor.size(-1)) * variance)
        tensor.mul_(factor.sqrt())


def _initialize_linear(layer: nn.Linear) -> None:
    glorot_orthogonal(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def get_geo_feats_from_edges(
    edges: Tensor, indices: Mapping[str, int] = FEATURE_INDICES
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        edges[:, indices["edge_dist_feats_start"]:indices["edge_dist_feats_end"]],
        edges[:, indices["edge_dir_feats_start"]:indices["edge_dir_feats_end"]],
        edges[:, indices["edge_orient_feats_start"]:indices["edge_orient_feats_end"]],
        edges[:, indices["edge_amide_angles"]],
    )


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Upstream BatchNorm, with defined behavior for empty/singleton graphs.

    A singleton training input uses the current running statistics and does not
    update them.  All inputs containing at least two rows use ordinary PyTorch
    BatchNorm.  This only extends the original implementation's short-graph
    boundary, where its training BatchNorm would raise an exception.
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return x
        if self.training and x.shape[0] == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=self.momentum or 0.0, eps=self.eps,
            )
        return super().forward(x)


def _make_norm(channels: int, norm_to_apply: str) -> nn.Module:
    if norm_to_apply == "layer":
        return nn.LayerNorm(channels)
    if norm_to_apply == "batch":
        return SafeBatchNorm1d(channels)
    raise ValueError("norm_to_apply must be 'batch' or 'layer'")


def _normalize(norm: nn.Module, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if mask is None:
        return norm(x)
    # Padding must not contribute to BatchNorm statistics. Coordinate-invalid
    # but real residues remain included through their independent token mask.
    rows = mask.nonzero(as_tuple=True)[0]
    return torch.zeros_like(x).index_copy(0, rows, norm(x[rows]))


@contextmanager
def _preserve_running_statistics(module: nn.Module):
    """Prevent checkpoint recomputation from updating BatchNorm state twice."""
    snapshots = [
        (value, value.detach().clone())
        for layer in module.modules()
        if isinstance(layer, nn.modules.batchnorm._BatchNorm)
        for value in (layer.running_mean, layer.running_var, layer.num_batches_tracked)
        if value is not None
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for value, original in snapshots:
                value.copy_(original)


class MultiHeadGeometricAttentionLayer(nn.Module):
    """Official multiplicative edge attention on directed source→target edges."""

    def __init__(
        self, num_input_feats: int, num_output_feats: int, num_heads: int,
        using_bias: bool = False, update_edge_feats: bool = True,
    ) -> None:
        super().__init__()
        self.num_output_feats = int(num_output_feats)
        self.num_heads = int(num_heads)
        self.using_bias = bool(using_bias)
        self.update_edge_feats = bool(update_edge_feats)
        output_dim = num_output_feats * num_heads
        self.Q = nn.Linear(num_input_feats, output_dim, bias=using_bias)
        self.K = nn.Linear(num_input_feats, output_dim, bias=using_bias)
        self.V = nn.Linear(num_input_feats, output_dim, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, output_dim, bias=using_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.Q, self.K, self.V, self.edge_feats_projection):
            _initialize_linear(layer)

    def forward(
        self, edge_index: Tensor, node_feats: Tensor, edge_feats: Tensor,
    ) -> tuple[Tensor, Optional[Tensor]]:
        source, target = edge_index.unbind(0)
        n_nodes, n_edges = node_feats.shape[0], edge_feats.shape[0]
        shape = (n_nodes, self.num_heads, self.num_output_feats)
        q, k, v = (layer(node_feats).reshape(shape) for layer in (self.Q, self.K, self.V))
        projected_edge = self.edge_feats_projection(edge_feats).reshape(
            n_edges, self.num_heads, self.num_output_feats
        )
        # Clipping before edge gating and again after feature summation are
        # separate operations in the official graph_utils attention functions.
        score = ((k[source] * q[target]) / sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        score = score * projected_edge
        edge_output = score if self.update_edge_feats else None
        weights = torch.exp(score.sum(dim=-1, keepdim=True).clamp(-5.0, 5.0))
        weighted_value = node_feats.new_zeros(shape).index_add(0, target, v[source] * weights)
        denominator = node_feats.new_zeros((n_nodes, self.num_heads, 1)).index_add(0, target, weights)
        return weighted_value / (denominator + 1.0e-6), edge_output


class InitEdgeModule(nn.Module):
    """Official geometric edge initializer with learned node-index embeddings."""

    def __init__(
        self, node_count_limit: int = 2304, num_edge_feats: int = 2,
        num_dist_feats: int = 18, num_dir_feats: int = 3,
        num_orient_feats: int = 4, num_amide_feats: int = 1,
        num_hidden_channels: int = 256, activ_fn: Optional[nn.Module] = None,
        feature_indices: Mapping[str, int] = FEATURE_INDICES,
    ) -> None:
        super().__init__()
        self.activ_fn = nn.SiLU() if activ_fn is None else activ_fn
        self.feature_indices = feature_indices
        self.node_embedding = nn.Embedding(node_count_limit, num_hidden_channels)
        for suffix in (0, 1):
            for name, size in (
                ("edge_messages", num_edge_feats), ("dist", num_dist_feats),
                ("dir", num_dir_feats), ("orient", num_orient_feats), ("amide", num_amide_feats),
            ):
                setattr(self, f"{name}_linear_{suffix}", nn.Linear(size, num_hidden_channels, bias=False))
        combined_size = num_edge_feats + num_dist_feats + num_dir_feats + num_orient_feats + num_amide_feats
        self.combined_linear_0 = nn.Linear(7 * num_hidden_channels, num_hidden_channels, bias=False)
        self.combined_linear_1 = nn.Linear(num_hidden_channels, combined_size, bias=False)
        self.combined_linear_2 = nn.Linear(combined_size, num_hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.node_embedding.weight, -sqrt(3), sqrt(3))
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                _initialize_linear(layer)

    def forward(self, edge_index: Tensor, edge_features: Tensor, node_indices: Tensor) -> Tensor:
        source, target = edge_index.unbind(0)
        node_feats = self.node_embedding(node_indices)
        inputs = (edge_features[:, :2], *get_geo_feats_from_edges(edge_features, self.feature_indices)[:3],
                  edge_features[:, self.feature_indices["edge_amide_angles"]].unsqueeze(-1))
        names = ("edge_messages", "dist", "dir", "orient", "amide")
        first = [getattr(self, f"{name}_linear_0")(value) for name, value in zip(names, inputs)]
        first[1:] = [self.activ_fn(value) for value in first[1:]]
        gate = self.activ_fn(self.combined_linear_0(torch.cat([node_feats[source], node_feats[target], *first], dim=-1)))
        second = [getattr(self, f"{name}_linear_1")(value) for name, value in zip(names, inputs)]
        second[1:] = [self.activ_fn(value) for value in second[1:]]
        combined = sum(value * gate for value in second)
        return self.combined_linear_2(self.combined_linear_1(combined))


class ResBlock(nn.Module):
    """Three-linear residual block sharing its norm, exactly as upstream."""

    def __init__(
        self, hidden_channels: int, activ_fn: Optional[nn.Module] = None,
        norm_to_apply: str = "batch",
    ) -> None:
        super().__init__()
        self.activ_fn = nn.SiLU() if activ_fn is None else activ_fn
        norm = _make_norm(hidden_channels, norm_to_apply)
        self.res_block = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels), norm, self.activ_fn,
            nn.Linear(hidden_channels, hidden_channels), norm, self.activ_fn,
            nn.Linear(hidden_channels, hidden_channels), norm, self.activ_fn,
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.res_block:
            if isinstance(layer, nn.Linear):
                _initialize_linear(layer)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for layer in self.res_block:
            x = layer(x)
        return residual + x


class ConformationModule(nn.Module):
    """Conformation update with distance→direction→orientation→amide gates."""

    def __init__(
        self, num_dist_feats: int = 18, num_dir_feats: int = 3,
        num_orient_feats: int = 4, num_amide_feats: int = 1,
        dist_embed_size: int = 8, dir_embed_size: int = 8,
        orient_embed_size: int = 8, amide_embed_size: int = 8,
        shared_embed_size: int = 64, num_hidden_channels: int = 256,
        num_pre_res_blocks: int = 2, num_post_res_blocks: int = 2,
        activ_fn: Optional[nn.Module] = None, norm_to_apply: str = "batch",
        feature_indices: Mapping[str, int] = FEATURE_INDICES,
    ) -> None:
        super().__init__()
        self.activ_fn = nn.SiLU() if activ_fn is None else activ_fn
        self.feature_indices = feature_indices
        for name, size, embedding in (
            ("dist", num_dist_feats, dist_embed_size), ("dir", num_dir_feats, dir_embed_size),
            ("orient", num_orient_feats, orient_embed_size), ("amide", num_amide_feats, amide_embed_size),
        ):
            setattr(self, f"{name}_linear_0", nn.Linear(size, embedding, bias=False))
            setattr(self, f"{name}_linear_1", nn.Linear(embedding, num_hidden_channels if name == "dist" else shared_embed_size, bias=False))
            setattr(self, f"final_{name}_linear", nn.Linear(size, num_hidden_channels, bias=False))
        self.nbr_linear = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.orig_msg_linear = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.downward_proj = nn.Linear(num_hidden_channels, shared_embed_size, bias=False)
        self.upward_proj = nn.Linear(shared_embed_size, num_hidden_channels, bias=False)
        self.res_connect_linear = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.pre_res_blocks = nn.ModuleList(ResBlock(num_hidden_channels, self.activ_fn, norm_to_apply) for _ in range(num_pre_res_blocks))
        self.post_res_blocks = nn.ModuleList(ResBlock(num_hidden_channels, self.activ_fn, norm_to_apply) for _ in range(num_post_res_blocks))
        self.final_linear = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                _initialize_linear(layer)

    def forward(self, edge_feats: Tensor, geometry: Mapping[str, Tensor]) -> Tensor:
        original = geometry["edge_features"]
        dist, direction, orientation, amide = get_geo_feats_from_edges(original, self.feature_indices)
        amide = amide.unsqueeze(-1)
        neighbor_ids = torch.cat((geometry["src_nbr_e_ids"], geometry["dst_nbr_e_ids"]), dim=1)
        mask = neighbor_ids >= 0
        if "src_nbr_e_mask" in geometry or "dst_nbr_e_mask" in geometry:
            mask = mask & torch.cat((
                geometry.get("src_nbr_e_mask", geometry["src_nbr_e_ids"] >= 0),
                geometry.get("dst_nbr_e_mask", geometry["dst_nbr_e_ids"] >= 0),
            ), dim=1)
        # Appending a true zero row permits -1 neighbors even when E == 0.
        safe_feats = torch.cat((edge_feats, edge_feats.new_zeros((1, edge_feats.shape[-1]))), dim=0)
        safe_ids = torch.where(mask, neighbor_ids, neighbor_ids.new_full((), edge_feats.shape[0]))
        neighbors = self.activ_fn(self.nbr_linear(safe_feats[safe_ids]))
        neighbors = neighbors * mask.unsqueeze(-1).to(neighbors.dtype)
        neighbors = neighbors * self.dist_linear_1(self.dist_linear_0(dist)).unsqueeze(1)
        neighbors = self.activ_fn(self.downward_proj(neighbors))
        for name, value in (("dir", direction), ("orient", orientation), ("amide", amide)):
            neighbors = neighbors * getattr(self, f"{name}_linear_1")(getattr(self, f"{name}_linear_0")(value)).unsqueeze(1)
        neighbors = self.activ_fn(self.upward_proj(neighbors.sum(dim=1)))
        updated = self.orig_msg_linear(edge_feats) + neighbors
        for layer in self.pre_res_blocks:
            updated = layer(updated)
        updated = edge_feats + self.activ_fn(self.res_connect_linear(updated))
        for layer in self.post_res_blocks:
            updated = layer(updated)
        gated = sum(getattr(self, f"final_{name}_linear")(value) * updated for name, value in (
            ("dist", dist), ("dir", direction), ("orient", orientation), ("amide", amide),
        ))
        return edge_feats + self.activ_fn(self.final_linear(gated))


class GeometricTransformerModule(nn.Module):
    """Intermediate GeoT block, including conformation and node/edge MLPs."""

    def __init__(
        self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1,
        norm_to_apply: str = "batch", residual: bool = True,
        *, update_edge_feats: bool = True, **conformation_kwargs,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        self.num_output_feats = int(d_model)
        self.dropout_rate = float(dropout)
        self.residual = bool(residual)
        self.update_edge_feats = bool(update_edge_feats)
        self.apply_layer_norm = norm_to_apply == "layer"
        self.conformation_module = ConformationModule(num_hidden_channels=d_model, norm_to_apply=norm_to_apply, **conformation_kwargs)
        norm_prefix = "layer_norm" if self.apply_layer_norm else "batch_norm"
        for stage, kind in ((1, "node"), (1, "edge"), (2, "node")):
            setattr(self, f"{norm_prefix}{stage}_{kind}_feats", _make_norm(d_model, norm_to_apply))
        if update_edge_feats:
            setattr(self, f"{norm_prefix}2_edge_feats", _make_norm(d_model, norm_to_apply))
        self.mha_module = MultiHeadGeometricAttentionLayer(d_model, d_model // n_heads, n_heads, False, update_edge_feats)
        self.O_node_feats = nn.Linear(d_model, d_model)
        activation, dropout_layer = nn.SiLU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(d_model, 2 * d_model, bias=False), activation, dropout_layer,
            nn.Linear(2 * d_model, d_model, bias=False),
        ])
        if update_edge_feats:
            self.O_edge_feats = nn.Linear(d_model, d_model)
            self.edge_feats_MLP = nn.ModuleList([
                nn.Linear(d_model, 2 * d_model, bias=False), activation, dropout_layer,
                nn.Linear(2 * d_model, d_model, bias=False),
            ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _initialize_linear(self.O_node_feats)
        if self.update_edge_feats:
            _initialize_linear(self.O_edge_feats)
        for mlp in (self.node_feats_MLP, getattr(self, "edge_feats_MLP", [])):
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    _initialize_linear(layer)

    def _norm(self, stage: int, kind: str, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        prefix = "layer_norm" if self.apply_layer_norm else "batch_norm"
        return _normalize(getattr(self, f"{prefix}{stage}_{kind}_feats"), x, mask)

    def forward(
        self, node_feats: Tensor, edge_feats: Tensor, geometry: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor] | Tensor:
        node_residual, edge_residual = node_feats, edge_feats
        conformed = self.conformation_module(edge_feats, geometry)
        mask = geometry.get("token_mask")
        mask = None if mask is None else mask.reshape(-1)
        node_attention, edge_attention = self.mha_module(
            geometry["edge_index"], self._norm(1, "node", node_feats, mask), self._norm(1, "edge", conformed),
        )
        node_feats = self.O_node_feats(F.dropout(node_attention.reshape(node_feats.shape), self.dropout_rate, self.training))
        if self.residual:
            node_feats = node_residual + node_feats
        if self.update_edge_feats:
            edge_feats = self.O_edge_feats(F.dropout(edge_attention.reshape(edge_feats.shape), self.dropout_rate, self.training))
            if self.residual:
                # Residual is the edge state BEFORE conformation, not conformed.
                edge_feats = edge_residual + edge_feats
        node_residual = node_feats
        node_feats = self._norm(2, "node", node_feats, mask)
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        if self.residual:
            node_feats = node_residual + node_feats
        if mask is not None:
            node_feats = node_feats.masked_fill(~mask.unsqueeze(-1), 0.0)
        if not self.update_edge_feats:
            return node_feats
        edge_residual = edge_feats
        edge_feats = self._norm(2, "edge", edge_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)
        if self.residual:
            edge_feats = edge_residual + edge_feats
        return node_feats, edge_feats


class FinalGeometricTransformerModule(GeometricTransformerModule):
    """Final GeoT block: conformation remains; attention only updates nodes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, update_edge_feats=False, **kwargs)


class GeoTEncoder(nn.Module):
    """Sparse GeoT on selected AsEP residues, accepting precomputed geometry.

    Geometry must contain ``edge_index[2,E]`` (source neighbor→target query),
    ``edge_features[E,28]``, ``node_features[B,N,7]``, and neighbor edge IDs
    ``src_nbr_e_ids``/``dst_nbr_e_ids`` with -1 padding. Optional ``token_mask``
    is independent of coordinate validity. Optional ``node_indices[B,N]``
    selects the learned index embeddings; by default indices restart at zero
    in every batch element. Node padding may use -1 indices.

    Shared PLM/aux/CDR features are concatenated with the seven structural node
    inputs and projected without bias, adapting the original input projection.
    ``forward_with_edges`` also returns the last retained intermediate edge
    state, like upstream DGLGeometricTransformer (the final block returns nodes).
    """

    def __init__(
        self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
        *, dropout: float = 0.1, node_count_limit: int = 2304,
        shared_embed_size: int = 64, dist_embed_size: int = 8,
        dir_embed_size: int = 8, orient_embed_size: int = 8,
        amide_embed_size: int = 8, num_pre_res_blocks: int = 2,
        num_post_res_blocks: int = 2, norm_to_apply: str = "batch",
        transformer_residual: bool = True, checkpoint_min_tokens: int = 0,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("GeoTEncoder requires at least one layer")
        if checkpoint_min_tokens < 0:
            raise ValueError("checkpoint_min_tokens must be nonnegative")
        self.d_model = int(d_model)
        self.checkpoint_min_tokens = int(checkpoint_min_tokens)
        self.node_in_embedding = nn.Linear(d_model + 7, d_model, bias=False)
        _initialize_linear(self.node_in_embedding)
        self.init_edge_module = InitEdgeModule(node_count_limit=node_count_limit, num_hidden_channels=d_model)
        options = dict(
            d_model=d_model, n_heads=n_heads, dropout=dropout,
            norm_to_apply=norm_to_apply, residual=transformer_residual,
            shared_embed_size=shared_embed_size, dist_embed_size=dist_embed_size,
            dir_embed_size=dir_embed_size, orient_embed_size=orient_embed_size,
            amide_embed_size=amide_embed_size, num_pre_res_blocks=num_pre_res_blocks,
            num_post_res_blocks=num_post_res_blocks,
        )
        self.gt_block = nn.ModuleList([
            *(GeometricTransformerModule(**options) for _ in range(n_layers - 1)),
            FinalGeometricTransformerModule(**options),
        ])

    def _call_layer(self, layer: nn.Module, nodes: Tensor, edges: Tensor, geometry: Mapping[str, Tensor], use_checkpoint: bool):
        if not use_checkpoint:
            return layer(nodes, edges, geometry)
        return checkpoint(
            layer, nodes, edges, geometry, use_reentrant=False, preserve_rng_state=True,
            context_fn=lambda: (nullcontext(), _preserve_running_statistics(layer)),
        )

    def forward_with_edges(self, x: Tensor, geometry: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B,N,{self.d_model}]")
        batch, length, _ = x.shape
        if batch < 1 or length < 1:
            raise ValueError("batch and sequence length must be positive")
        node_features = geometry["node_features"]
        if node_features.shape != (batch, length, 7):
            raise ValueError("geometry node_features must have shape [B,N,7]")
        token_mask = geometry.get("token_mask", torch.ones((batch, length), dtype=torch.bool, device=x.device))
        if token_mask.shape != (batch, length):
            raise ValueError("token_mask must have shape [B,N]")
        flat_mask = token_mask.reshape(-1)
        node_indices = geometry.get("node_indices")
        if node_indices is None:
            node_indices = torch.arange(length, device=x.device).expand(batch, -1)
        if node_indices.shape != (batch, length):
            raise ValueError("node_indices must have shape [B,N]")
        node_indices = node_indices.reshape(-1).masked_fill(~flat_mask, 0)
        capacity = self.init_edge_module.node_embedding.num_embeddings
        if bool(((node_indices < 0) | (node_indices >= capacity)).any()):
            raise ValueError(f"valid node indices must lie in [0,{capacity}); increase node_count_limit explicitly")
        edge_features, edge_index = geometry["edge_features"], geometry["edge_index"]
        if edge_features.ndim != 2 or edge_features.shape[-1] != 28:
            raise ValueError("edge_features must have shape [E,28]")
        if edge_index.shape != (2, edge_features.shape[0]):
            raise ValueError("edge_index must have shape [2,E]")
        x = self.node_in_embedding(torch.cat((x, node_features.to(x.dtype)), dim=-1)).reshape(batch * length, self.d_model)
        x = x.masked_fill(~flat_mask.unsqueeze(-1), 0.0)
        edges = self.init_edge_module(edge_index, edge_features, node_indices)
        local_geometry = dict(geometry, token_mask=token_mask)
        use_checkpoint = self.training and torch.is_grad_enabled() and self.checkpoint_min_tokens > 0 and length >= self.checkpoint_min_tokens
        for layer in self.gt_block[:-1]:
            x, edges = self._call_layer(layer, x, edges, local_geometry, use_checkpoint)
        x = self._call_layer(self.gt_block[-1], x, edges, local_geometry, use_checkpoint)
        return x.reshape(batch, length, self.d_model), edges

    def forward(self, x: Tensor, geometry: Mapping[str, Tensor]) -> Tensor:
        return self.forward_with_edges(x, geometry)[0]


__all__ = [
    "GeoTEncoder", "InitEdgeModule", "ConformationModule", "ResBlock",
    "MultiHeadGeometricAttentionLayer", "GeometricTransformerModule",
    "FinalGeometricTransformerModule", "SafeBatchNorm1d",
]
