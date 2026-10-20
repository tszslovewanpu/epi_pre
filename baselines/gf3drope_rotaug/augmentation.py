"""Global rigid-body rotation augmentation for GF3DRoPE coordinate batches.

The functions in this module operate on the batch dictionary produced by the
GF3DRoPE training pipeline.  A single proper rotation is sampled per complex
and reused for antibody/antigen N, CA, and C coordinates.  Coordinate masks
are treated as missing-data masks: invalid rows are preserved exactly.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor


_COORDINATE_GROUPS = (
    ("coords_ab", "coords_n_ab", "coords_c_ab", "coord_mask_ab"),
    ("coords_ag", "coords_n_ag", "coords_c_ag", "coord_mask_ag"),
)


def sample_uniform_so3(
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample Haar-uniform proper 3D rotations.

    Normalizing four independent standard Gaussian variables gives a uniform
    point on the unit 3-sphere.  Interpreting that point as a unit quaternion
    induces the Haar measure on SO(3).

    Returns:
        A tensor with shape ``[batch_size, 3, 3]``.  Matrices use the standard
        column-vector convention; row-vector coordinates should therefore be
        transformed with ``coords @ rotation.transpose(-1, -2)``.
    """

    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if not dtype.is_floating_point:
        raise TypeError(f"rotation dtype must be floating point, got {dtype}")

    quaternion = torch.randn(
        batch_size,
        4,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(dtype).tiny
    )
    w, x, y, z = quaternion.unbind(dim=-1)

    rotation = torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return rotation.reshape(batch_size, 3, 3)


def rotate_valid_coordinates(coords: Tensor, coord_mask: Tensor, rotation: Tensor) -> Tensor:
    """Rotate valid coordinate rows while preserving invalid rows exactly."""

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, N, 3], got {tuple(coords.shape)}")
    if coord_mask.shape != coords.shape[:-1]:
        raise ValueError(
            "coord_mask must match the leading coordinate dimensions, "
            f"got mask={tuple(coord_mask.shape)} coords={tuple(coords.shape)}"
        )
    if rotation.shape != (coords.shape[0], 3, 3):
        raise ValueError(
            f"rotation must have shape [{coords.shape[0]}, 3, 3], "
            f"got {tuple(rotation.shape)}"
        )
    if rotation.device != coords.device or rotation.dtype != coords.dtype:
        raise ValueError("rotation must have the same device and dtype as coords")

    rotated = torch.bmm(coords, rotation.transpose(1, 2))
    return torch.where(coord_mask.bool().unsqueeze(-1), rotated, coords)


def rotate_complex_coordinates(batch: Mapping[str, Tensor], rotation: Tensor) -> Dict[str, Tensor]:
    """Apply one shared rotation per complex to all AB/AG backbone atoms.

    The returned dictionary is a shallow copy.  Coordinate entries are
    replaced with rotated tensors; the input dictionary and its coordinate
    tensors are not modified in place.
    """

    result = dict(batch)
    for ca_key, n_key, c_key, mask_key in _COORDINATE_GROUPS:
        if mask_key not in batch:
            raise KeyError(f"missing coordinate mask {mask_key!r}")
        mask = batch[mask_key]
        for coord_key in (ca_key, n_key, c_key):
            if coord_key not in batch:
                raise KeyError(f"missing coordinate tensor {coord_key!r}")
            result[coord_key] = rotate_valid_coordinates(
                batch[coord_key],
                mask,
                rotation,
            )
    return result


def apply_random_global_rotation(
    batch: Mapping[str, Tensor],
    probability: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    """Randomly rotate each complex in a GF3DRoPE batch.

    Args:
        batch: Batch mapping containing AB/AG ``coords``, ``coords_n``,
            ``coords_c``, and coordinate-mask entries.
        probability: Independent probability of applying rotation to each
            complex.  ``0`` performs no random draws and returns a shallow copy.
        generator: Optional torch random generator for reproducible sampling.

    Rotation is intentionally independent of coordinate jitter ordering.  The
    training caller should invoke this function after its existing jitter step.
    """

    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if "coords_ab" not in batch:
        raise KeyError("missing reference coordinate tensor 'coords_ab'")

    reference = batch["coords_ab"]
    if reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError(
            f"coords_ab must have shape [B, N, 3], got {tuple(reference.shape)}"
        )
    if probability == 0.0:
        return dict(batch)

    batch_size = reference.shape[0]
    rotation = sample_uniform_so3(
        batch_size,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )

    if probability < 1.0:
        apply_mask = torch.rand(
            batch_size,
            device=reference.device,
            generator=generator,
        ) < probability
        identity = torch.eye(
            3,
            device=reference.device,
            dtype=reference.dtype,
        ).expand(batch_size, -1, -1)
        rotation = torch.where(apply_mask[:, None, None], rotation, identity)

    return rotate_complex_coordinates(batch, rotation)


__all__ = [
    "apply_random_global_rotation",
    "rotate_complex_coordinates",
    "rotate_valid_coordinates",
    "sample_uniform_so3",
]
