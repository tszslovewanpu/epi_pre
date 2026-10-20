"""Torch-only geometric primitives shared by the SE(3) baselines.

Frames use column vectors: ``frames[..., i, :, a]`` is local axis ``a`` of
residue ``i`` expressed in global coordinates.  Pair displacements are
directed from residue ``i`` to residue ``j``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "safe_normalize",
    "infer_coord_mask",
    "build_local_frames",
    "pairwise_mask",
    "pair_mask",
    "pairwise_displacements",
    "pairwise_distances",
    "pairwise_geometry",
    "local_pair_vectors",
    "relative_frame_rotations",
    "RBFEmbedding",
    "RBFDistanceBias",
]


def _check_xyz(name: str, value: Tensor) -> None:
    if value.ndim < 2 or value.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape [..., N, 3], got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")


def _check_same_shape(**values: Tensor) -> None:
    shapes = {name: tuple(value.shape) for name, value in values.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"coordinate tensors must have identical shapes: {shapes}")


def _combine_mask(
    inferred: Tensor,
    supplied: Optional[Tensor],
    *,
    name: str = "coord_mask",
) -> Tensor:
    if supplied is None:
        return inferred
    if supplied.shape != inferred.shape:
        raise ValueError(
            f"{name} must have shape {tuple(inferred.shape)}, "
            f"got {tuple(supplied.shape)}"
        )
    return inferred & supplied.to(device=inferred.device, dtype=torch.bool)


def safe_normalize(
    vector: Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Normalize vectors without producing NaNs for zero or non-finite rows.

    A zero, NaN, or infinite vector maps to zero.  Finite vectors with norm
    below ``eps`` are divided by ``eps`` rather than an unstable tiny norm.
    """

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    finite = torch.isfinite(vector).all(dim=dim, keepdim=True)
    clean = torch.where(finite, vector, torch.zeros_like(vector))
    norm = torch.linalg.vector_norm(clean, dim=dim, keepdim=True)
    normalized = clean / norm.clamp_min(eps)
    return torch.where(finite, normalized, torch.zeros_like(normalized))


def infer_coord_mask(
    coords_ca: Tensor,
    coords_n: Tensor,
    coords_c: Tensor,
    *,
    eps: float = 1e-8,
    collinearity_tol: float = 1e-4,
) -> Tensor:
    """Infer which residues have a usable N--CA--C backbone frame.

    Valid residues must have finite coordinates, must not be the all-zero
    missing-coordinate sentinel, must have nonzero CA--N and CA--C bonds, and
    must not have collinear N--CA--C atoms.  The sentinel check is performed
    jointly over the three atoms so a legitimate atom at the global origin is
    not rejected.
    """

    _check_xyz("coords_ca", coords_ca)
    _check_xyz("coords_n", coords_n)
    _check_xyz("coords_c", coords_c)
    _check_same_shape(coords_ca=coords_ca, coords_n=coords_n, coords_c=coords_c)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not 0 <= collinearity_tol < 1:
        raise ValueError(
            "collinearity_tol must lie in [0, 1), "
            f"got {collinearity_tol}"
        )

    finite = (
        torch.isfinite(coords_ca).all(dim=-1)
        & torch.isfinite(coords_n).all(dim=-1)
        & torch.isfinite(coords_c).all(dim=-1)
    )
    ca = torch.nan_to_num(coords_ca, nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.nan_to_num(coords_n, nan=0.0, posinf=0.0, neginf=0.0)
    c = torch.nan_to_num(coords_c, nan=0.0, posinf=0.0, neginf=0.0)

    nonzero_triplet = (
        ca.abs().sum(dim=-1) + n.abs().sum(dim=-1) + c.abs().sum(dim=-1)
    ) > eps

    ca_to_c = c - ca
    ca_to_n = n - ca
    c_norm = torch.linalg.vector_norm(ca_to_c, dim=-1)
    n_norm = torch.linalg.vector_norm(ca_to_n, dim=-1)
    cross_norm = torch.linalg.vector_norm(
        torch.cross(ca_to_c, ca_to_n, dim=-1),
        dim=-1,
    )
    sine_angle = cross_norm / (c_norm * n_norm).clamp_min(eps * eps)
    nondegenerate = (
        (c_norm > eps)
        & (n_norm > eps)
        & (sine_angle > collinearity_tol)
    )
    return finite & nonzero_triplet & nondegenerate


def build_local_frames(
    coords_ca: Tensor,
    coords_n: Tensor,
    coords_c: Tensor,
    coord_mask: Optional[Tensor] = None,
    *,
    eps: float = 1e-8,
    collinearity_tol: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    """Build right-handed residue frames and return ``(frames, valid_mask)``.

    The first axis points from CA to C.  The second is the component of CA to N
    orthogonal to the first, and the third completes a right-handed basis.
    Invalid frames are exactly zero.  Under a proper global rotation ``Q``,
    valid frames transform covariantly as ``R_i -> Q R_i``.
    """

    valid = infer_coord_mask(
        coords_ca,
        coords_n,
        coords_c,
        eps=eps,
        collinearity_tol=collinearity_tol,
    )
    valid = _combine_mask(valid, coord_mask)

    ca = torch.nan_to_num(coords_ca, nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.nan_to_num(coords_n, nan=0.0, posinf=0.0, neginf=0.0)
    c = torch.nan_to_num(coords_c, nan=0.0, posinf=0.0, neginf=0.0)

    e1 = safe_normalize(c - ca, eps=eps)
    n_direction = n - ca
    n_orthogonal = (
        n_direction
        - (n_direction * e1).sum(dim=-1, keepdim=True) * e1
    )
    e2 = safe_normalize(n_orthogonal, eps=eps)
    e3 = safe_normalize(torch.cross(e1, e2, dim=-1), eps=eps)
    # Recompute e2 to reduce accumulated Gram--Schmidt error.
    e2 = safe_normalize(torch.cross(e3, e1, dim=-1), eps=eps)
    frames = torch.stack((e1, e2, e3), dim=-1)
    frames = frames.masked_fill(~valid[..., None, None], 0.0)
    return frames, valid


def pairwise_mask(valid_mask: Tensor) -> Tensor:
    """Return the directed valid-pair mask ``[..., i, j]``."""

    if valid_mask.ndim < 1:
        raise ValueError("valid_mask must have at least one dimension")
    valid = valid_mask.to(dtype=torch.bool)
    return valid[..., :, None] & valid[..., None, :]


# Short alias useful in attention implementations.
pair_mask = pairwise_mask


def _coord_validity(
    coords: Tensor,
    coord_mask: Optional[Tensor],
) -> Tensor:
    _check_xyz("coords", coords)
    finite = torch.isfinite(coords).all(dim=-1)
    return _combine_mask(finite, coord_mask)


def pairwise_displacements(
    coords: Tensor,
    coord_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return masked global displacements ``x_j - x_i``.

    The output has shape ``[..., N, N, 3]``.  Every invalid pair is exactly
    zero, including pairs containing a non-finite coordinate.
    """

    valid = _coord_validity(coords, coord_mask)
    clean = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    displacement = clean[..., None, :, :] - clean[..., :, None, :]
    return displacement.masked_fill(
        ~pairwise_mask(valid)[..., None],
        0.0,
    )


def pairwise_distances(
    coords: Tensor,
    coord_mask: Optional[Tensor] = None,
    *,
    keepdim: bool = False,
) -> Tensor:
    """Return masked Euclidean pair distances."""

    displacement = pairwise_displacements(coords, coord_mask)
    return torch.linalg.vector_norm(
        displacement,
        dim=-1,
        keepdim=keepdim,
    )


def pairwise_geometry(
    coords: Tensor,
    coord_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(displacements, distances, valid_pair_mask)``."""

    valid = _coord_validity(coords, coord_mask)
    pairs = pairwise_mask(valid)
    displacement = pairwise_displacements(coords, valid)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    return displacement, distance, pairs


def _frame_validity(frames: Tensor, eps: float = 1e-8) -> Tensor:
    if frames.ndim < 3 or frames.shape[-2:] != (3, 3):
        raise ValueError(
            "frames must have shape [..., N, 3, 3], "
            f"got {tuple(frames.shape)}"
        )
    if not torch.is_floating_point(frames):
        raise TypeError("frames must be a floating-point tensor")
    finite = torch.isfinite(frames).all(dim=(-1, -2))
    clean = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    # Local frames are required to be non-degenerate and right-handed.
    proper = torch.linalg.det(clean) > eps
    return finite & proper


def local_pair_vectors(
    coords: Tensor,
    frames: Tensor,
    coord_mask: Optional[Tensor] = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Return invariant pair vectors ``R_i^T (x_j - x_i)``.

    Frames and coordinates may be batched or unbatched.  Invalid pair outputs
    are exactly zero.
    """

    _check_xyz("coords", coords)
    if frames.shape[:-2] != coords.shape[:-1] or frames.shape[-2:] != (3, 3):
        raise ValueError(
            "frames must have shape [..., N, 3, 3] matching coords; "
            f"got coords={tuple(coords.shape)}, frames={tuple(frames.shape)}"
        )
    valid = _coord_validity(coords, coord_mask) & _frame_validity(frames, eps)
    displacement = pairwise_displacements(coords, valid)
    clean_frames = torch.nan_to_num(
        frames,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    local = torch.einsum(
        "...ijc,...ica->...ija",
        displacement,
        clean_frames,
    )
    return local.masked_fill(~pairwise_mask(valid)[..., None], 0.0)


def relative_frame_rotations(
    frames: Tensor,
    frame_mask: Optional[Tensor] = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Return invariant relative rotations ``R_i^T R_j``.

    The output has shape ``[..., N, N, 3, 3]`` and is zero for invalid pairs.
    """

    valid = _frame_validity(frames, eps)
    valid = _combine_mask(valid, frame_mask, name="frame_mask")
    clean = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    relative = torch.einsum(
        "...ica,...jcb->...ijab",
        clean,
        clean,
    )
    return relative.masked_fill(
        ~pairwise_mask(valid)[..., None, None],
        0.0,
    )


class RBFEmbedding(nn.Module):
    """Gaussian radial-basis embedding of scalar distances."""

    def __init__(
        self,
        n_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 30.0,
        *,
        gamma: Optional[float] = None,
        trainable_gamma: bool = False,
    ) -> None:
        super().__init__()
        if n_rbf < 1:
            raise ValueError(f"n_rbf must be positive, got {n_rbf}")
        if d_max <= d_min:
            raise ValueError(
                f"d_max must be greater than d_min, got {d_min}, {d_max}"
            )
        self.n_rbf = int(n_rbf)
        centers = torch.linspace(float(d_min), float(d_max), self.n_rbf)
        self.register_buffer("centers", centers)

        spacing = (float(d_max) - float(d_min)) / max(self.n_rbf - 1, 1)
        initial_gamma = 1.0 / max(spacing, 1e-12) ** 2
        if gamma is not None:
            if gamma <= 0:
                raise ValueError(f"gamma must be positive, got {gamma}")
            initial_gamma = float(gamma)
        gamma_tensor = torch.tensor(initial_gamma)
        if trainable_gamma:
            self.gamma = nn.Parameter(gamma_tensor)
        else:
            self.register_buffer("gamma", gamma_tensor)

    def forward(self, distances: Tensor) -> Tensor:
        """Embed a scalar distance field of shape ``[...]`` as ``[..., K]``."""

        centers = self.centers.to(
            device=distances.device,
            dtype=distances.dtype,
        )
        gamma = self.gamma.to(device=distances.device, dtype=distances.dtype)
        gamma = gamma.abs().clamp_min(torch.finfo(distances.dtype).eps)
        offset = distances.unsqueeze(-1) - centers
        return torch.exp(-gamma * offset.square())


class RBFDistanceBias(nn.Module):
    """Project distance RBFs to a head-specific additive attention bias."""

    def __init__(
        self,
        n_heads: int,
        n_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 30.0,
        *,
        gamma: Optional[float] = None,
        trainable_gamma: bool = True,
    ) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        self.n_heads = int(n_heads)
        self.rbf = RBFEmbedding(
            n_rbf=n_rbf,
            d_min=d_min,
            d_max=d_max,
            gamma=gamma,
            trainable_gamma=trainable_gamma,
        )
        self.proj = nn.Linear(n_rbf, self.n_heads, bias=False)

    @property
    def centers(self) -> Tensor:
        return self.rbf.centers

    @property
    def gamma(self) -> Tensor:
        return self.rbf.gamma

    def forward(
        self,
        distances: Tensor,
        valid_pair_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return bias with shape ``[..., H, N, N]``.

        ``distances`` should have shape ``[..., N, N]``.  Supplying
        ``valid_pair_mask`` is essential when zero is also the sentinel value
        for an invalid distance: the projected invalid-pair bias is then
        exactly zero.
        """

        if not torch.is_floating_point(distances):
            raise TypeError("distances must be a floating-point tensor")
        if distances.ndim < 2:
            raise ValueError(
                "distances must have shape [..., N, N], "
                f"got {tuple(distances.shape)}"
            )
        if valid_pair_mask is not None and valid_pair_mask.shape != distances.shape:
            raise ValueError(
                "valid_pair_mask must match distances, got "
                f"{tuple(valid_pair_mask.shape)} and {tuple(distances.shape)}"
            )

        bias = self.proj(self.rbf(distances)).movedim(-1, -3)
        if valid_pair_mask is not None:
            bias = bias.masked_fill(
                ~valid_pair_mask.to(device=bias.device, dtype=torch.bool)[
                    ..., None, :, :
                ],
                0.0,
            )
        return bias
