"""DeepInteract geometric features on full physical chains, selected-node graphs.

This is an explicitly documented adaptation of the published geometric module,
not byte-equivalent feature preprocessing from the upstream repository. The
upstream row-frame quaternion order and squared-distance RBF input are corrected.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


GEOMETRY_POLICY = {
    "upstream_commit": "c78d205465f02ee4ef751dbbafb7ec8f30c75c9a",
    "distance_rbf": "18 Gaussian RBFs of true Euclidean CA distance, centers 0..20 Angstrom; upstream code uses squared distance, paper describes 16 RBFs",
    "relative_rotation": "O_query @ O_neighbor.T for row-stacked global-to-local frames; corrects upstream O_query.T @ O_neighbor",
    "frame_support": "complete physical-chain CA triplets before CDR/surface selection, never across gaps; both chain endpoints invalid",
    "torsions": "full-chain phi/psi/omega cosine and sine, independently masked at breaks and missing atoms",
    "missing_cb": "plane angle masked to zero, including Gly; no virtual CB",
    "sequence_features": "full-chain SEQRES index normalized within physical chain; edge sin(src-dst) uses original indices, zero for inter-chain pairs",
    "node_embedding_indices": "unique full-partner selected_indices including chain offsets and unresolved positions",
    "graph": "selected valid residues only; directed kNN k=20 including self, no radius cutoff; intra-partner inter-chain spatial edges permitted",
    "edge_neighborhood": "each edge independently samples two incoming edges at each endpoint without replacement, preserving self/central candidates; local generator seed 0 per sample/build, does not consume training RNG; -1 plus mask for missing neighbors",
    "jitter": "shared protocol perturbs full N/CA/C once before all features are rebuilt; CB unchanged",
}
ADAPTATION_METADATA = GEOMETRY_POLICY


def _normalize(vector: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector / norm.clamp_min(eps), norm.squeeze(-1) > eps


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Stable rotation matrix to canonical unit quaternion, ordered x/y/z/w.

    Select the largest diagonal candidate instead of dividing by a tiny scalar
    near 180 degrees. The positive-scalar convention resolves ordinary q/-q
    ambiguity; the largest vector component breaks the exact-pi tie.
    """
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                           1 - m00 + m11 - m22, 1 - m00 - m11 + m22), dim=-1)
    magnitude = squared.clamp_min(1e-12).sqrt()
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
    ), dim=-2) / (2 * magnitude[..., :, None].clamp_min(.1))
    index = magnitude.argmax(dim=-1)
    quaternion = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    quaternion, _ = _normalize(quaternion)
    scalar = quaternion[..., 0]
    largest_vector = quaternion[..., 1:].gather(-1, quaternion[..., 1:].abs().argmax(-1, keepdim=True)).squeeze(-1)
    sign_value = torch.where(scalar.abs() > 1e-12, scalar, largest_vector)
    quaternion = quaternion * torch.where(sign_value < 0, -1., 1.)[..., None]
    return torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)


def _torsion(a: Tensor, b: Tensor, c: Tensor, d: Tensor, valid: Tensor):
    first, first_ok = _normalize(b - a)
    axis, axis_ok = _normalize(c - b)
    last, last_ok = _normalize(d - c)
    normal1, normal1_ok = _normalize(torch.linalg.cross(first, axis))
    normal2, normal2_ok = _normalize(torch.linalg.cross(axis, last))
    cosine = (normal1 * normal2).sum(-1)
    sine = (torch.linalg.cross(normal1, normal2) * axis).sum(-1)
    valid = valid & first_ok & axis_ok & last_ok & normal1_ok & normal2_ok
    unit = torch.stack((cosine, sine), -1)
    unit, _ = _normalize(unit)
    return unit.masked_fill(~valid[..., None], 0.), valid


def full_chain_features(context: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Compute geometry before selection, honoring explicit physical links."""
    coords = context["coords"]
    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("context coords must have shape [B,L,4,3] in N/CA/C/CB order")
    batch, length = coords.shape[:2]
    if length < 1:
        raise ValueError("full-chain context cannot be empty")
    mask = context["atom_mask"].bool() & torch.isfinite(coords).all(-1)
    coords = torch.nan_to_num(coords, nan=0., posinf=0., neginf=0.)
    links = context["link_mask"].bool()
    if links.shape != (batch, max(length - 1, 0)):
        raise ValueError("link_mask must have shape [B,L-1]")
    chain = context["chain_id"]
    index = context["residue_index"]
    # Even a malformed supplied link cannot connect distinct physical chains.
    links = links & (chain[:, :-1] == chain[:, 1:]) & (chain[:, :-1] >= 0)
    links &= index[:, 1:] == index[:, :-1] + 1
    previous_link = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=coords.device), links), 1)
    next_link = torch.cat((links, torch.zeros(batch, 1, dtype=torch.bool, device=coords.device)), 1)
    n, ca, c, cb = coords.unbind(dim=-2)

    def previous(value):
        return torch.cat((torch.zeros_like(value[:, :1]), value[:, :-1]), dim=1)

    def following(value):
        return torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), dim=1)

    incoming, incoming_ok = _normalize(ca - previous(ca))
    outgoing, outgoing_ok = _normalize(following(ca) - ca)
    normal, normal_ok = _normalize(torch.linalg.cross(incoming, outgoing))
    bisector, bisector_ok = _normalize(incoming - outgoing)
    frames = torch.stack((bisector, normal, torch.linalg.cross(bisector, normal)), dim=-2)
    frame_mask = (previous_link & next_link & previous(mask[..., 1]) & mask[..., 1]
                  & following(mask[..., 1]) & incoming_ok & outgoing_ok & normal_ok & bisector_ok)
    frames = frames.masked_fill(~frame_mask[..., None, None], 0.)
    backbone = mask[..., :3].all(-1)
    phi, phi_mask = _torsion(previous(c), n, ca, c, previous_link & previous(mask[..., 2]) & backbone)
    psi, psi_mask = _torsion(n, ca, c, following(n), next_link & following(mask[..., 0]) & backbone)
    omega, omega_mask = _torsion(ca, c, following(n), following(ca),
                                 next_link & mask[..., 1] & mask[..., 2]
                                 & following(mask[..., 0]) & following(mask[..., 1]))
    angles = torch.stack((phi, psi, omega), dim=-2)
    torsions = torch.cat((angles[..., 0], angles[..., 1]), dim=-1)
    torsion_mask = torch.stack((phi_mask, psi_mask, omega_mask), dim=-1)
    plane_normal, plane_ok = _normalize(torch.linalg.cross(ca - cb, cb - n))
    plane_mask = mask[..., 0] & mask[..., 1] & mask[..., 3] & plane_ok
    plane_normal = plane_normal.masked_fill(~plane_mask[..., None], 0.)
    sequence_feature = coords.new_zeros(batch, length)
    for row in range(batch):
        for chain_number in chain[row].unique():
            if int(chain_number) < 0:
                continue
            belongs = chain[row] == chain_number
            denominator = index[row, belongs].max().clamp_min(1)
            sequence_feature[row, belongs] = index[row, belongs].to(coords.dtype) / denominator
    return {
        "coords": coords, "atom_mask": mask, "frames": frames, "frame_mask": frame_mask,
        "torsions": torsions, "torsion_mask": torsion_mask,
        "plane_normal": plane_normal, "plane_mask": plane_mask,
        "node_features": torch.cat((sequence_feature[..., None], torsions), dim=-1),
    }


def _scale_feature(values: Tensor) -> Tensor:
    if not values.numel():
        return values
    low, high = values.min(), values.max()
    span = high - low
    tolerance = max(1e-8, 32 * torch.finfo(values.dtype).eps)
    # A numerically constant angle field carries no pairwise information.
    # Dividing its roundoff by epsilon would amplify arbitrary frame rotation.
    return torch.where(span > tolerance, (values - low) / span.clamp_min(tolerance), torch.zeros_like(values))


def build_geometry(context: Mapping[str, Tensor], token_mask=None, k=20, num_rbf=18, geo_nbrhd_size=2):
    """Return selected-node geometric graph and edge-neighborhood tensors.

    Edges are neighbor src -> query dst, flattened over [B,N]. Learned endpoint
    embeddings should use node_indices (full-partner positions), never compressed
    selected indices. Spatial neighbors may cross chains within one partner;
    frames, torsions and sequence separation never cross physical chain breaks.
    """
    if int(k) < 1 or int(num_rbf) != 18 or int(geo_nbrhd_size) < 1:
        raise ValueError("k/neighborhood size must be positive; this 28-feature adapter requires 18 RBFs")
    full = full_chain_features(context)
    indices = context["selected_indices"]
    if indices.ndim != 2:
        raise ValueError("selected_indices must have shape [B,N]")
    batch, length = indices.shape
    if (indices >= context["coords"].shape[1]).any():
        raise ValueError("selected index is outside full context")
    rows = torch.arange(batch, device=indices.device)[:, None]
    safe_indices = indices.clamp_min(0)

    def selected(value):
        return value[rows, safe_indices]

    if token_mask is None:
        token_mask = indices >= 0
    else:
        token_mask = token_mask.to(device=indices.device, dtype=torch.bool) & (indices >= 0)
    coord_mask = selected(full["atom_mask"])[..., :3].all(-1) & token_mask
    if "selected_coord_mask" in context:
        coord_mask &= context["selected_coord_mask"]
    ca = selected(full["coords"])[..., 1, :]
    ca = ca.masked_fill(~coord_mask[..., None], 0.)
    frames = selected(full["frames"])
    frame_mask = selected(full["frame_mask"]) & coord_mask
    plane_normal = selected(full["plane_normal"])
    plane_mask = selected(full["plane_mask"]) & coord_mask
    sequence_index = selected(context["residue_index"])
    chain_id = selected(context["chain_id"])
    candidate_count = max(int(k), int(geo_nbrhd_size))
    edges, incoming = [], torch.full((batch * length, candidate_count), -1, dtype=torch.long, device=ca.device)
    edge_batches = []
    next_edge_id = 0
    # Topology is discrete; continuous edge quantities are recomputed below.
    with torch.no_grad():
        for row in range(batch):
            nodes = coord_mask[row].nonzero().flatten()
            if not nodes.numel():
                continue
            local_coords = ca[row, nodes]
            distance = torch.linalg.vector_norm(local_coords[:, None] - local_coords[None, :], dim=-1)
            # Guarantee self loops even when several residues have identical CA.
            distance.fill_diagonal_(-1.)
            nearest = torch.argsort(distance, dim=1, stable=True)[:, :min(int(k), nodes.numel())]
            src = nodes[nearest].reshape(-1) + row * length
            dst = nodes[:, None].expand_as(nearest).reshape(-1) + row * length
            edge = torch.stack((src, dst))
            edges.append(edge)
            edge_batches.append(torch.full((src.numel(),), row, device=ca.device, dtype=torch.long))
            local_eids = torch.arange(src.numel(), device=ca.device).reshape_as(nearest) + next_edge_id
            incoming[nodes + row * length, :nearest.shape[1]] = local_eids
            next_edge_id += src.numel()
    edge_index = torch.cat(edges, dim=1) if edges else torch.empty(2, 0, device=ca.device, dtype=torch.long)
    edge_batch = torch.cat(edge_batches) if edge_batches else torch.empty(0, device=ca.device, dtype=torch.long)
    src, dst = edge_index.unbind(0)
    flat_ca, flat_frames = ca.reshape(-1, 3), frames.reshape(-1, 3, 3)
    relative = flat_ca[src] - flat_ca[dst]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    relative_unit, _ = _normalize(relative)
    local_direction = torch.einsum("eij,ej->ei", flat_frames[dst], relative_unit)
    local_direction = local_direction.masked_fill(~frame_mask.reshape(-1)[dst, None], 0.)
    relative_rotation = flat_frames[dst] @ flat_frames[src].transpose(-1, -2)
    rotation_mask = frame_mask.reshape(-1)[src] & frame_mask.reshape(-1)[dst]
    identity = torch.eye(3, dtype=ca.dtype, device=ca.device)
    relative_rotation = torch.where(rotation_mask[:, None, None], relative_rotation, identity)
    quaternion = matrix_to_quaternion(relative_rotation).masked_fill(~rotation_mask[:, None], 0.)
    means = torch.linspace(0., 20., num_rbf, dtype=ca.dtype, device=ca.device)
    rbf = torch.exp(-((distance[:, None] - means) / (20. / num_rbf)).square())
    flat_index, flat_chain = sequence_index.reshape(-1), chain_id.reshape(-1)
    same_chain = flat_chain[src] == flat_chain[dst]
    sequence_separation = torch.sin((flat_index[src] - flat_index[dst]).to(ca.dtype)).masked_fill(~same_chain, 0.)
    normals = plane_normal.reshape(-1, 3)
    pair_plane_mask = plane_mask.reshape(-1)[src] & plane_mask.reshape(-1)[dst]
    sine = torch.linalg.vector_norm(torch.linalg.cross(normals[src], normals[dst]), dim=-1)
    cosine = (normals[src] * normals[dst]).sum(-1)
    raw_angle = torch.atan2(sine.masked_fill(~pair_plane_mask, 0.), cosine.masked_fill(~pair_plane_mask, 1.))
    edge_weight, plane_angle = torch.zeros_like(distance), torch.zeros_like(distance)
    for row in range(batch):
        belongs = edge_batch == row
        edge_weight[belongs] = _scale_feature(distance[belongs].square())
        plane_valid = belongs & pair_plane_mask
        plane_angle[plane_valid] = _scale_feature(raw_angle[plane_valid])
    features = torch.cat((sequence_separation[:, None], edge_weight[:, None], rbf,
                          local_direction, quaternion, plane_angle[:, None]), dim=-1)
    src_neighbors = torch.full((src.numel(), geo_nbrhd_size), -1, dtype=torch.long, device=ca.device)
    dst_neighbors = torch.full_like(src_neighbors, -1)
    for row in range(batch):
        belongs = edge_batch == row
        local_generator = torch.Generator(device=ca.device).manual_seed(0)
        for endpoint, output in ((src, src_neighbors), (dst, dst_neighbors)):
            candidates = incoming[endpoint[belongs]]
            priority = torch.rand(candidates.shape, device=ca.device, generator=local_generator)
            priority.masked_fill_(candidates < 0, float("inf"))
            choice = torch.argsort(priority, dim=-1, stable=True)[:, :geo_nbrhd_size]
            output[belongs] = candidates.gather(-1, choice)
    return {
        "edge_index": edge_index,
        "edge_features": features,
        "node_features": selected(full["node_features"]).masked_fill(~token_mask[..., None], 0.),
        "node_indices": indices.clamp_min(0),
        "src_nbr_e_ids": src_neighbors,
        "dst_nbr_e_ids": dst_neighbors,
        "src_nbr_e_mask": src_neighbors >= 0,
        "dst_nbr_e_mask": dst_neighbors >= 0,
        "selected_ca": ca,
        "coord_mask": coord_mask,
        "token_mask": token_mask,
        "frame_mask": frame_mask,
        "torsion_mask": selected(full["torsion_mask"]) & token_mask[..., None],
        "plane_mask": plane_mask,
        "edge_distances": distance,
    }
