"""CPU checks of Geoformer geometry, masking, gradients and checkpointing."""

from __future__ import annotations

import copy

import pytest
import torch

from baselines.geformer import model as adapted


def _geometry():
    # Two independent molecules at overlapping positions; also exercise a
    # disconnected atom and a boundary-distance pair.
    positions = torch.tensor(
        [
            [[0., 0., 0.], [1.1, .2, 0.], [.3, 1.3, .4], [5., 0., 0.], [20., 0., 0.]],
            [[0., 0., 0.], [.2, 1.2, 0.], [1.5, .3, -.2], [3., 1., .6], [4., .2, 1.]],
        ], dtype=torch.float64,
    )
    relative = positions[:, None, :, :] - positions[:, :, None, :]
    distance = relative.norm(dim=-1)
    adjacent = distance < 5.0
    direction = relative / (distance[..., None] + 1e-8)
    direction = direction.masked_fill(~adjacent[..., None], 0.)
    batches, query, key = adjacent.nonzero(as_tuple=True)
    nodes = positions.shape[1]
    edge_index = torch.stack((batches * nodes + query, batches * nodes + key))
    return positions, distance, direction, adjacent, (batches, query, key), edge_index


def _encoder(norm_type="none", dropout=0.):
    return adapted.GeoformerEncoder(
        16, 4, 2, dim_ff=32, dropout=dropout, cutoff=5., num_rbf=8,
        norm_type=norm_type, checkpoint_min_tokens=0,
    ).double()


@pytest.mark.parametrize("norm_type", ["none", "max_min"])
def test_encoder_rigid_invariance_permutation_and_independent_batch(norm_type):
    torch.manual_seed(16)
    encoder = _encoder(norm_type).eval()
    positions = _geometry()[0]
    features = torch.randn(2, 5, 16, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    reference, edges, ipe = encoder.forward_with_edges(features, positions, coord_mask=mask)
    assert torch.equal(edges[0] // 5, edges[1] // 5)
    # Choose a proper, non-axis-aligned rotation.
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
    rotation[:, 0] *= torch.linalg.det(rotation)
    # Avoid testing radius-boundary rounding here: rotate/translate a cloud
    # with its points safely away from the cutoff instead of the exact-5 pair.
    stable_positions = positions * .91
    stable, stable_edges, stable_ipe = encoder.forward_with_edges(features, stable_positions)
    moved, moved_edges, moved_ipe = encoder.forward_with_edges(
        features, stable_positions @ rotation.T + torch.tensor([9., -3., 2.]),
    )
    assert torch.equal(stable_edges, moved_edges)
    torch.testing.assert_close(moved, stable, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(moved_ipe, stable_ipe, atol=2e-10, rtol=2e-9)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = encoder(features[:, permutation], positions[:, permutation])
    torch.testing.assert_close(permuted, reference[:, permutation], atol=2e-10, rtol=2e-9)
    for batch in range(2):
        separate, separate_edges, separate_ipe = encoder.forward_with_edges(
            features[batch:batch + 1], positions[batch:batch + 1],
        )
        selected = edges[0] // 5 == batch
        torch.testing.assert_close(separate, reference[batch:batch + 1], atol=2e-10, rtol=2e-9)
        assert torch.equal(separate_edges + 5 * batch, edges[:, selected])
        torch.testing.assert_close(separate_ipe, ipe[selected], atol=2e-10, rtol=2e-9)


def test_padding_missing_coordinates_coincident_atoms_and_cutoff_graph():
    torch.manual_seed(22)
    encoder = _encoder().eval()
    positions = torch.tensor([
        [[0., 0., 0.], [0., 0., 0.], [5., 0., 0.], [6., 0., 0.],
         [20., 0., 0.], [float("nan"), 1., 0.], [float("inf"), 0., 0.], [1., 0., 0.]],
    ], dtype=torch.float64)
    features = torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True)
    token_mask = torch.tensor([[True] * 7 + [False]])
    coord_mask = torch.tensor([[True, True, True, False, True, True, True, True]])
    output, edges, pair = encoder.forward_with_edges(
        features, positions, token_mask=token_mask, coord_mask=coord_mask,
        # Contradictory cached edges must not override the fresh radius graph.
        edge_index=torch.tensor([[0, 1, 4], [7, 6, 5]]),
    )
    expected_edges = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (4, 4)}
    assert set(map(tuple, edges.T.tolist())) == expected_edges
    assert torch.isfinite(output).all() and torch.isfinite(pair).all()
    assert torch.count_nonzero(output[~token_mask]) == 0
    (output.square().sum() + pair.square().sum()).backward()
    assert torch.isfinite(features.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    # Completely absent geometry must still retain finite scalar features.
    empty_output, empty_edges, empty_pair = encoder.forward_with_edges(
        features.detach(), positions, token_mask=token_mask,
        coord_mask=torch.zeros_like(coord_mask),
    )
    assert empty_edges.shape == (2, 0) and empty_pair.shape == (0, 16)
    assert torch.isfinite(empty_output).all()
    assert torch.count_nonzero(empty_output[~token_mask]) == 0


def test_checkpoint_preserves_nodes_ipe_gradients_and_dropout_rng(monkeypatch):
    torch.manual_seed(91)
    normal = _encoder(dropout=.2).train()
    recomputed = copy.deepcopy(normal)
    recomputed.checkpoint_min_tokens = 1
    checkpoint_calls = []
    original_checkpoint = adapted.checkpoint

    def tracking_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(function)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(adapted, "checkpoint", tracking_checkpoint)
    features = torch.randn(2, 5, 16, dtype=torch.float64)
    positions = _geometry()[0] * .91
    results = []
    for encoder in (normal, recomputed):
        x = features.clone().requires_grad_()
        coords = positions.clone().requires_grad_()
        torch.manual_seed(742)
        output, edges, pair = encoder.forward_with_edges(x, coords)
        probe = torch.linspace(-1., 1., output.numel(), dtype=output.dtype).reshape_as(output)
        ((output * probe).sum() + .03 * pair.square().sum()).backward()
        results.append((output, edges, pair, x.grad, coords.grad, torch.random.get_rng_state()))
    assert len(checkpoint_calls) == len(recomputed.layers)
    for reference, changed in zip(*results):
        if reference.dtype in (torch.long, torch.uint8):
            assert torch.equal(reference, changed)
        else:
            torch.testing.assert_close(changed, reference, atol=2e-11, rtol=2e-10)
            assert torch.isfinite(changed).all()
    for name, parameter in recomputed.named_parameters():
        expected = dict(normal.named_parameters())[name].grad
        assert (parameter.grad is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(parameter.grad, expected, atol=2e-10, rtol=2e-9, msg=name)


def _predictor_and_batch():
    torch.manual_seed(63)
    predictor = adapted.Model(
        8, 7, d_model=16, n_heads=4, n_layers_ab=2, n_layers_ag=2,
        n_layers_cross=1, dim_ff=32, dropout=0.,
        geometry_config={"geformer_num_rbf": 8, "geformer_checkpoint_min_tokens": 0},
    ).double().eval()
    positions = _geometry()[0][:1] * .91
    batch = {
        "x_b": torch.randn(1, 4, 8, dtype=torch.float64),
        "x_g": torch.randn(1, 5, 7, dtype=torch.float64),
        "coords_ab": positions[:, :4].clone(),
        "coords_ag": positions.clone(),
        "cdr_type": torch.tensor([[0, 1, 2, 3]]),
        "mask_b": torch.ones(1, 4, dtype=torch.bool),
        "mask_g": torch.ones(1, 5, dtype=torch.bool),
        "coord_mask_ab": torch.ones(1, 4, dtype=torch.bool),
        "coord_mask_ag": torch.ones(1, 5, dtype=torch.bool),
    }
    for chain in ("ab", "ag"):
        batch[f"coords_n_{chain}"] = batch[f"coords_{chain}"] + torch.tensor([.1, .9, 0.])
        batch[f"coords_c_{chain}"] = batch[f"coords_{chain}"] + torch.tensor([1., .2, .1])
    return predictor, batch


def test_prediction_uses_ca_without_backbone_frames_and_train_jitter_changes_geometry():
    from baselines.se3bench.train import apply_train_augmentation

    predictor, batch = _predictor_and_batch()
    expected = predictor(**batch)
    absent_frames = dict(batch)
    for chain in ("ab", "ag"):
        for atom in ("n", "c"):
            absent_frames[f"coords_{atom}_{chain}"] = torch.full_like(batch[f"coords_{chain}"], float("nan"))
    observed = predictor(**absent_frames)
    torch.testing.assert_close(observed, expected, atol=0., rtol=0.)
    for chain in ("ab", "ag"):
        del absent_frames[f"coord_mask_{chain}"]
    torch.testing.assert_close(predictor(**absent_frames), expected, atol=0., rtol=0.)
    torch.manual_seed(11)
    augmented = apply_train_augmentation(batch, {"augmentation": {
        "enabled": True, "coord_jitter_std": .2, "coord_jitter_prob": 1.,
        "global_rotation_prob": 0.,
    }})
    for chain in ("ab", "ag"):
        assert not torch.equal(augmented[f"coords_{chain}"], batch[f"coords_{chain}"])
    noisy = predictor(**augmented)
    assert torch.isfinite(noisy).all()
    assert (noisy - expected).abs().max() > 1e-9
    noisy.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in predictor.parameters())
