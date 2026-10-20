from __future__ import annotations

import torch

from baselines.se3bench.geometry import (
    RBFDistanceBias,
    RBFEmbedding,
    build_local_frames,
    infer_coord_mask,
    local_pair_vectors,
    pairwise_displacements,
    pairwise_distances,
    pairwise_geometry,
    pairwise_mask,
    relative_frame_rotations,
    safe_normalize,
)


def _proper_rotation(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    axis = torch.tensor([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / axis.norm()
    angle = torch.tensor(0.73, dtype=dtype)
    x, y, z = axis
    skew = torch.stack(
        (
            torch.stack((x * 0, -z, y)),
            torch.stack((z, y * 0, -x)),
            torch.stack((-y, x, z * 0)),
        )
    )
    identity = torch.eye(3, dtype=dtype)
    rotation = (
        identity
        + torch.sin(angle) * skew
        + (1 - torch.cos(angle)) * (skew @ skew)
    )
    torch.testing.assert_close(
        rotation.transpose(-1, -2) @ rotation,
        identity,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones((), dtype=dtype),
        atol=1e-12,
        rtol=1e-12,
    )
    return rotation


def _backbone(dtype: torch.dtype = torch.float64):
    ca = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [-3.0, 0.5, 4.0],
            [2.5, -2.0, 0.25],
            [0.0, 0.0, 0.0],  # all-zero missing sentinel
            [4.0, 4.0, 4.0],  # collinear backbone
        ],
        dtype=dtype,
    )
    c = ca + torch.tensor(
        [
            [1.2, 0.1, 0.0],
            [0.2, 1.0, -0.1],
            [-0.1, 0.3, 1.1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    n = ca + torch.tensor(
        [
            [-0.2, 0.9, 0.3],
            [0.4, -0.2, 1.0],
            [0.8, -0.4, 0.2],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    return ca.unsqueeze(0), n.unsqueeze(0), c.unsqueeze(0)


def _rigid_transform(
    coords: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return coords @ rotation.transpose(-1, -2) + translation


def test_safe_normalize_is_finite_and_zero_safe():
    vectors = torch.tensor(
        [[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [float("nan"), 1.0, 2.0]]
    )
    normalized = safe_normalize(vectors)
    torch.testing.assert_close(normalized[0], torch.tensor([0.6, 0.0, 0.8]))
    assert torch.equal(normalized[1:], torch.zeros_like(normalized[1:]))
    assert torch.isfinite(normalized).all()


def test_coord_mask_rejects_nonfinite_zero_and_degenerate_backbones():
    ca, n, c = _backbone()
    ca = torch.cat((ca, ca[:, :1]), dim=1)
    n = torch.cat((n, n[:, :1]), dim=1)
    c = torch.cat((c, c[:, :1]), dim=1)
    n[0, -1, 1] = float("nan")

    valid = infer_coord_mask(ca, n, c)
    assert torch.equal(
        valid,
        torch.tensor([[True, True, True, False, False, False]]),
    )


def test_frames_are_right_handed_orthonormal_and_rotation_covariant():
    ca, n, c = _backbone()
    frames, valid = build_local_frames(ca, n, c)
    identity = torch.eye(3, dtype=frames.dtype).expand(3, 3, 3)
    gram = frames[0, valid[0]].transpose(-1, -2) @ frames[0, valid[0]]
    torch.testing.assert_close(gram, identity, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.linalg.det(frames[0, valid[0]]),
        torch.ones(3, dtype=frames.dtype),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.equal(frames[0, ~valid[0]], torch.zeros_like(frames[0, ~valid[0]]))

    rotation = _proper_rotation()
    translation = torch.tensor([7.0, -11.0, 2.5], dtype=ca.dtype)
    ca_rt = _rigid_transform(ca, rotation, translation)
    n_rt = _rigid_transform(n, rotation, translation)
    c_rt = _rigid_transform(c, rotation, translation)
    transformed_frames, transformed_valid = build_local_frames(ca_rt, n_rt, c_rt)

    assert torch.equal(transformed_valid, valid)
    expected = torch.einsum("ab,...nbc->...nac", rotation, frames)
    torch.testing.assert_close(
        transformed_frames,
        expected,
        atol=2e-12,
        rtol=2e-12,
    )


def test_pair_geometry_masks_invalid_pairs_and_obeys_se3_transform_rules():
    ca, n, c = _backbone()
    _, valid = build_local_frames(ca, n, c)
    displacement, distance, pairs = pairwise_geometry(ca, valid)

    assert torch.equal(pairs, pairwise_mask(valid))
    assert torch.equal(
        displacement[~pairs],
        torch.zeros_like(displacement[~pairs]),
    )
    assert torch.equal(distance[~pairs], torch.zeros_like(distance[~pairs]))
    torch.testing.assert_close(
        displacement,
        pairwise_displacements(ca, valid),
    )
    torch.testing.assert_close(distance, pairwise_distances(ca, valid))

    rotation = _proper_rotation()
    translation = torch.tensor([-3.0, 0.25, 8.0], dtype=ca.dtype)
    ca_rt = _rigid_transform(ca, rotation, translation)
    displacement_rt, distance_rt, pairs_rt = pairwise_geometry(ca_rt, valid)
    expected_displacement = displacement @ rotation.transpose(-1, -2)

    assert torch.equal(pairs_rt, pairs)
    torch.testing.assert_close(
        displacement_rt,
        expected_displacement,
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(distance_rt, distance, atol=2e-12, rtol=2e-12)


def test_local_vectors_and_relative_rotations_are_exactly_se3_invariant():
    ca, n, c = _backbone()
    frames, valid = build_local_frames(ca, n, c)
    local = local_pair_vectors(ca, frames, valid)
    relative = relative_frame_rotations(frames, valid)

    pairs = pairwise_mask(valid)
    assert torch.equal(local[~pairs], torch.zeros_like(local[~pairs]))
    assert torch.equal(relative[~pairs], torch.zeros_like(relative[~pairs]))
    diagonal = relative[0, valid[0], valid[0]]
    torch.testing.assert_close(
        diagonal,
        torch.eye(3, dtype=relative.dtype).expand_as(diagonal),
        atol=1e-12,
        rtol=1e-12,
    )

    rotation = _proper_rotation()
    translation = torch.tensor([2.0, -5.0, 13.0], dtype=ca.dtype)
    ca_rt = _rigid_transform(ca, rotation, translation)
    n_rt = _rigid_transform(n, rotation, translation)
    c_rt = _rigid_transform(c, rotation, translation)
    frames_rt, valid_rt = build_local_frames(ca_rt, n_rt, c_rt)

    local_rt = local_pair_vectors(ca_rt, frames_rt, valid_rt)
    relative_rt = relative_frame_rotations(frames_rt, valid_rt)
    torch.testing.assert_close(local_rt, local, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(relative_rt, relative, atol=3e-12, rtol=3e-12)


def test_rbf_embedding_and_bias_are_distance_invariant_and_masked():
    ca, n, c = _backbone()
    _, valid = build_local_frames(ca, n, c)
    pairs = pairwise_mask(valid)
    distance = pairwise_distances(ca, valid)

    embedding = RBFEmbedding(n_rbf=8, d_min=0.0, d_max=12.0).double()
    features = embedding(distance)
    assert features.shape == (1, 5, 5, 8)
    assert torch.isfinite(features).all()

    bias_module = RBFDistanceBias(
        n_heads=3,
        n_rbf=8,
        d_min=0.0,
        d_max=12.0,
    ).double()
    bias = bias_module(distance, pairs)
    assert bias.shape == (1, 3, 5, 5)
    expanded_pairs = pairs[:, None].expand_as(bias)
    assert torch.equal(bias[~expanded_pairs], torch.zeros_like(bias[~expanded_pairs]))

    rotation = _proper_rotation()
    translation = torch.tensor([19.0, -4.0, 0.75], dtype=ca.dtype)
    distance_rt = pairwise_distances(
        _rigid_transform(ca, rotation, translation),
        valid,
    )
    torch.testing.assert_close(
        embedding(distance_rt),
        features,
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        bias_module(distance_rt, pairs),
        bias,
        atol=2e-12,
        rtol=2e-12,
    )
