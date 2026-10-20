from __future__ import annotations

import pytest
import torch

from baselines.gf3drope_rotaug.augmentation import (
    apply_random_global_rotation,
    rotate_complex_coordinates,
    sample_uniform_so3,
)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _example_batch(batch_size: int = 3):
    base_ab = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-2.0, 0.5, 4.0],
            [90.0, 91.0, 92.0],
        ]
    )
    base_ag = torch.tensor(
        [
            [-1.0, 3.0, 0.0],
            [2.5, -4.0, 1.0],
            [80.0, 81.0, 82.0],
            [0.25, 0.5, 0.75],
        ]
    )
    coords_ab = base_ab.unsqueeze(0).repeat(batch_size, 1, 1)
    coords_ag = base_ag.unsqueeze(0).repeat(batch_size, 1, 1)
    return {
        "coords_ab": coords_ab,
        "coords_n_ab": coords_ab + 0.25,
        "coords_c_ab": coords_ab - 0.5,
        "coord_mask_ab": torch.tensor(
            [[True, True, False]], dtype=torch.bool
        ).repeat(batch_size, 1),
        "coords_ag": coords_ag,
        "coords_n_ag": coords_ag + 0.75,
        "coords_c_ag": coords_ag - 1.25,
        "coord_mask_ag": torch.tensor(
            [[True, True, False, True]], dtype=torch.bool
        ).repeat(batch_size, 1),
        "x_b": torch.arange(batch_size, dtype=torch.float32),
    }


def test_sampled_rotations_are_proper_orthogonal_matrices():
    rotation = sample_uniform_so3(
        512,
        dtype=torch.float64,
        generator=_generator(101),
    )
    identity = torch.eye(3, dtype=torch.float64).expand_as(rotation)

    torch.testing.assert_close(
        rotation.transpose(1, 2) @ rotation,
        identity,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones(512, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_gaussian_quaternion_sampling_has_isotropic_moments():
    rotation = sample_uniform_so3(
        20_000,
        dtype=torch.float64,
        generator=_generator(202),
    )
    north = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    rotated = torch.matmul(rotation, north)

    torch.testing.assert_close(
        rotated.mean(dim=0),
        torch.zeros(3, dtype=torch.float64),
        atol=0.015,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rotated.square().mean(dim=0),
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
        atol=0.015,
        rtol=0.0,
    )


def test_one_rotation_is_shared_across_all_coordinate_tensors():
    batch = _example_batch(batch_size=3)
    expected_rotation = sample_uniform_so3(
        3,
        generator=_generator(303),
    )
    rotated = apply_random_global_rotation(
        batch,
        probability=1.0,
        generator=_generator(303),
    )

    for prefix in ("ab", "ag"):
        mask = batch[f"coord_mask_{prefix}"]
        for key in (
            f"coords_{prefix}",
            f"coords_n_{prefix}",
            f"coords_c_{prefix}",
        ):
            expected = torch.bmm(
                batch[key],
                expected_rotation.transpose(1, 2),
            )
            torch.testing.assert_close(rotated[key][mask], expected[mask])

    assert rotated["x_b"] is batch["x_b"]


def test_invalid_rows_are_preserved_exactly_and_input_is_not_mutated():
    batch = _example_batch(batch_size=2)
    snapshots = {key: value.clone() for key, value in batch.items()}
    rotation = sample_uniform_so3(2, generator=_generator(404))

    rotated = rotate_complex_coordinates(batch, rotation)

    for prefix in ("ab", "ag"):
        invalid = ~batch[f"coord_mask_{prefix}"]
        for key in (
            f"coords_{prefix}",
            f"coords_n_{prefix}",
            f"coords_c_{prefix}",
        ):
            assert torch.equal(rotated[key][invalid], batch[key][invalid])
            assert torch.equal(batch[key], snapshots[key])
            assert rotated[key] is not batch[key]


def test_rotation_preserves_distances_after_any_prior_jitter():
    batch = _example_batch(batch_size=2)
    jittered = dict(batch)
    noise_generator = _generator(505)
    for prefix in ("ab", "ag"):
        mask = batch[f"coord_mask_{prefix}"]
        for key in (
            f"coords_{prefix}",
            f"coords_n_{prefix}",
            f"coords_c_{prefix}",
        ):
            noise = torch.randn(
                batch[key].shape,
                generator=noise_generator,
                dtype=batch[key].dtype,
            )
            jittered[key] = batch[key] + 0.2 * noise * mask.unsqueeze(-1)

    rotated = apply_random_global_rotation(
        jittered,
        probability=1.0,
        generator=_generator(506),
    )
    for prefix in ("ab", "ag"):
        mask = batch[f"coord_mask_{prefix}"]
        before = torch.cdist(jittered[f"coords_{prefix}"], jittered[f"coords_{prefix}"])
        after = torch.cdist(rotated[f"coords_{prefix}"], rotated[f"coords_{prefix}"])
        pair_mask = mask[:, :, None] & mask[:, None, :]
        torch.testing.assert_close(after[pair_mask], before[pair_mask])


def test_probability_zero_is_an_exact_no_op_without_consuming_rng():
    batch = _example_batch(batch_size=2)
    generator = _generator(606)
    state_before = generator.get_state().clone()

    result = apply_random_global_rotation(
        batch,
        probability=0.0,
        generator=generator,
    )

    assert result is not batch
    assert torch.equal(generator.get_state(), state_before)
    for key, value in batch.items():
        assert result[key] is value


def test_configurable_probability_selects_complexes_independently():
    batch = _example_batch(batch_size=64)
    rotated = apply_random_global_rotation(
        batch,
        probability=0.5,
        generator=_generator(707),
    )
    valid_point = batch["coord_mask_ab"][:, 0]
    changed = (
        rotated["coords_ab"][:, 0] - batch["coords_ab"][:, 0]
    ).abs().amax(dim=-1) > 1e-6

    assert valid_point.all()
    assert changed.any()
    assert (~changed).any()

    # Within a selected complex, AB and AG use the same rigid transform, so a
    # cross-chain displacement keeps its norm.
    cross_before = batch["coords_ab"][:, 0] - batch["coords_ag"][:, 0]
    cross_after = rotated["coords_ab"][:, 0] - rotated["coords_ag"][:, 0]
    torch.testing.assert_close(
        cross_after.norm(dim=-1),
        cross_before.norm(dim=-1),
    )


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\]"):
        apply_random_global_rotation(
            _example_batch(batch_size=1),
            probability=probability,
        )
