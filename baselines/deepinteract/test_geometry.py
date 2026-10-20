"""CPU checks of full-context geometry, mapping, masks and augmentation."""
from __future__ import annotations

import copy

import torch

from baselines.deepinteract.data import (
    batch_context, jitter_context,
    selected_coordinates,
)
from baselines.deepinteract.geometry import build_geometry, full_chain_features, matrix_to_quaternion


def _context(indices=(1, 3, 5, 7, 8), dtype=torch.float64):
    t = torch.arange(10, dtype=dtype)
    ca = torch.stack((t * 1.5, torch.sin(t * .7), torch.cos(t * .4)), -1)
    coordinates = torch.stack((ca + torch.tensor([-.5, .6, .25]), ca,
                               ca + torch.tensor([.8, .1, -.2]), ca + torch.tensor([.2, -.9, .6])), -2)
    atom_mask = torch.ones(10, 4, dtype=torch.bool)
    atom_mask[2, 3] = False  # Gly: never construct a virtual CB.
    coordinates[2, 3] = 0.
    links = torch.ones(9, dtype=torch.bool)
    links[4] = False
    return {
        "coords": coordinates,
        "atom_mask": atom_mask,
        "chain_id": torch.tensor([0] * 5 + [1] * 5),
        "residue_index": torch.tensor(list(range(5)) * 2),
        "link_mask": links,
        "selected_indices": torch.tensor(indices),
        "selected_coord_mask": torch.ones(len(indices), dtype=torch.bool),
    }


def test_full_context_support_and_chain_breaks_are_used_before_selection():
    context = batch_context([_context()])
    features = full_chain_features(context)
    graph = build_geometry(context, k=3)
    # Selected i=1 has the unselected physical neighbors 0/2. i=3 uses 2/4.
    assert graph["frame_mask"].tolist() == [[True, True, False, True, True]]
    assert features["frame_mask"].tolist() == [[False, True, True, True, False, False, True, True, True, False]]
    # No phi at start of physical chain; no psi/omega at its end.
    assert not features["torsion_mask"][0, 5, 0]
    assert not features["torsion_mask"][0, 4, 1:].any()
    assert not features["plane_mask"][0, 2]
    assert torch.equal(graph["node_indices"], context["selected_indices"])
    expected = features["node_features"][0, context["selected_indices"][0]]
    torch.testing.assert_close(graph["node_features"][0], expected)
    # Changing an unselected frame-support atom must affect a selected frame.
    changed = copy.deepcopy(context)
    changed["coords"][0, 2, 1] += torch.tensor([.2, -.6, .9])
    assert not torch.allclose(full_chain_features(changed)["frames"][0, 1], features["frames"][0, 1])
    # A gap inside one chain invalidates only features using that peptide link.
    changed = copy.deepcopy(context)
    changed["link_mask"][0, 1] = False
    with_gap = full_chain_features(changed)
    assert not with_gap["frame_mask"][0, 1:3].any()
    assert not with_gap["torsion_mask"][0, 2, 0]
    assert not with_gap["torsion_mask"][0, 1, 1:].any()
    assert with_gap["frame_mask"][0, 3]


def test_features_are_rigid_invariant_and_quaternions_handle_pi():
    torch.manual_seed(51)
    context = batch_context([_context(indices=(1, 2, 3, 6, 7, 8))])
    original = build_geometry(context)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
    rotation[:, 0] *= torch.linalg.det(rotation)
    changed = copy.deepcopy(context)
    changed["coords"] = context["coords"] @ rotation.T + torch.tensor([17., -3., 8.])
    moved = build_geometry(changed)
    assert torch.equal(original["edge_index"], moved["edge_index"])
    torch.testing.assert_close(moved["node_features"], original["node_features"], atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(moved["edge_features"], original["edge_features"], atol=3e-11, rtol=3e-10)
    rotations = torch.stack((torch.eye(3), torch.diag(torch.tensor([1., -1., -1.])),
                              torch.diag(torch.tensor([-1., 1., -1.])))).double()
    quaternion = matrix_to_quaternion(rotations)
    torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(quaternion[0], torch.tensor([0., 0., 0., 1.], dtype=torch.float64))
    assert quaternion[1, 0] == 1 and quaternion[2, 1] == 1


def test_true_distance_rbf_graph_masks_neighbor_ids_and_no_cross_batch():
    left, right = _context(indices=(1, 2, 3)), _context(indices=(1,))
    right["selected_coord_mask"][0] = False
    context = batch_context([left, right])
    graph = build_geometry(context, k=2)
    assert graph["edge_features"].shape == (6, 28)
    src, dst = graph["edge_index"]
    assert (src < 3).all() and (dst < 3).all()
    assert torch.equal(src // 3, dst // 3)
    self_edges = src == dst
    assert int(self_edges.sum()) == 3
    distances = (graph["selected_ca"].reshape(-1, 3)[src] - graph["selected_ca"].reshape(-1, 3)[dst]).norm(dim=-1)
    centers = torch.linspace(0., 20., 18, dtype=torch.float64)
    expected = torch.exp(-((distances[:, None] - centers) / (20. / 18)).square())
    torch.testing.assert_close(graph["edge_features"][:, 2:20], expected)
    assert not torch.allclose(expected[~self_edges], torch.exp(-((distances[~self_edges, None].square() - centers) / (20. / 18)).square()))
    assert graph["token_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert not graph["coord_mask"][1].any()
    for endpoint, name in ((src, "src"), (dst, "dst")):
        ids = graph[f"{name}_nbr_e_ids"]
        valid = graph[f"{name}_nbr_e_mask"]
        assert valid.all()
        assert torch.equal(dst[ids], endpoint[:, None].expand_as(ids))
    single = build_geometry(batch_context([_context(indices=(1,))]))
    assert single["src_nbr_e_ids"].tolist() == [[0, -1]]
    assert single["src_nbr_e_mask"].tolist() == [[True, False]]
    empty = batch_context([right])
    assert build_geometry(empty)["edge_features"].shape == (0, 28)
    # Explicit self loops survive even when all CA coordinates coincide.
    coincident = batch_context([left])
    coincident["coords"][..., 1, :] = 0.
    graph = build_geometry(coincident, k=1)
    assert torch.equal(graph["edge_index"][0], graph["edge_index"][1])
    assert torch.isfinite(graph["edge_features"]).all()


def test_full_jitter_is_reproducible_masked_and_shared_with_selected_atoms():
    context = batch_context([_context()])
    before = copy.deepcopy(context)
    generator = torch.Generator().manual_seed(97)
    perturbed = jitter_context(context, .2, generator=generator)
    repeated = jitter_context(context, .2, generator=torch.Generator().manual_seed(97))
    assert torch.equal(perturbed["coords"], repeated["coords"])
    assert torch.equal(context["coords"], before["coords"])
    assert torch.equal(perturbed["coords"][..., 3, :], context["coords"][..., 3, :])
    assert not torch.equal(perturbed["coords"][..., :3, :], context["coords"][..., :3, :])
    selected = selected_coordinates(perturbed)
    torch.testing.assert_close(selected["coords_ca"][0], perturbed["coords"][0, context["selected_indices"][0], 1])
    clean, noisy = build_geometry(context), build_geometry(perturbed)
    assert torch.isfinite(noisy["node_features"]).all() and torch.isfinite(noisy["edge_features"]).all()
    assert not torch.allclose(clean["node_features"], noisy["node_features"])
    # Noise is applied also to unselected context positions used by frames.
    assert not torch.equal(perturbed["coords"][0, 2, 1], context["coords"][0, 2, 1])


def test_sampled_edge_neighborhoods_are_independent_reproducible_and_do_not_consume_rng():
    context = batch_context([_context(indices=(1, 2, 3, 6, 7, 8))])
    torch.manual_seed(61)
    rng_before = torch.random.get_rng_state()
    graph = build_geometry(context, k=6)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    repeated = build_geometry(context, k=6)
    for name in ("src_nbr_e_ids", "dst_nbr_e_ids"):
        assert torch.equal(graph[name], repeated[name])
        assert (graph[name][:, 0] != graph[name][:, 1]).all()
    # Edges sharing a query sample separately, not one shared pair per node.
    query_zero = graph["edge_index"][1] == 0
    choices = graph["dst_nbr_e_ids"][query_zero]
    assert torch.unique(choices, dim=0).shape[0] > 1
    # Adding another sample must not change this sample's sampled neighborhood.
    two = build_geometry(batch_context([_context(indices=(1, 2, 3, 6, 7, 8)), _context(indices=(1, 2, 3, 6, 7, 8))]), k=6)
    count = graph["edge_index"].shape[1]
    torch.testing.assert_close(two["src_nbr_e_ids"][:count], graph["src_nbr_e_ids"])
    torch.testing.assert_close(two["src_nbr_e_ids"][count:] - count, graph["src_nbr_e_ids"])


def test_coordinates_have_finite_gradients_and_missing_features_are_zero():
    context = batch_context([_context(indices=(1, 2, 3, 6, 7, 8))])
    context["coords"].requires_grad_()
    graph = build_geometry(context)
    node_probe = torch.linspace(-1., 1., graph["node_features"].numel(), dtype=torch.float64).reshape_as(graph["node_features"])
    edge_probe = torch.linspace(.4, -.3, graph["edge_features"].numel(), dtype=torch.float64).reshape_as(graph["edge_features"])
    ((graph["node_features"] * node_probe).sum() + (graph["edge_features"] * edge_probe).sum()).backward()
    assert torch.isfinite(context["coords"].grad).all()
    src, dst = graph["edge_index"]
    invalid_plane = ~graph["plane_mask"].reshape(-1)
    assert torch.count_nonzero(graph["edge_features"][invalid_plane[src] | invalid_plane[dst], 27]) == 0


