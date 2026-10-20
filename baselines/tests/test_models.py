from __future__ import annotations

import importlib

import pytest
import torch


INVARIANT_METHODS = (
    "lf3drope",
    "geformer",
    "distance_rbf",
    "local_geometry",
    "ipa",
    "gvp",
)
GLOBAL_FRAME_METHODS = ("gf3drope", "gf3drope_rotaug")
ALL_METHODS = INVARIANT_METHODS + GLOBAL_FRAME_METHODS


def _rotation(dtype=torch.float32):
    axis = torch.tensor([1.0, -2.0, 0.75], dtype=dtype)
    axis = axis / axis.norm()
    angle = torch.tensor(0.83, dtype=dtype)
    x, y, z = axis
    skew = torch.stack(
        (
            torch.stack((x * 0, -z, y)),
            torch.stack((z, y * 0, -x)),
            torch.stack((-y, x, z * 0)),
        )
    )
    identity = torch.eye(3, dtype=dtype)
    return identity + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (
        skew @ skew
    )


def _chain(length: int, offset: float):
    index = torch.arange(length, dtype=torch.float32)
    ca = torch.stack(
        (
            index * 1.7 + offset,
            torch.sin(index * 0.7) + 0.3 * offset,
            torch.cos(index * 0.4) - 0.2 * offset,
        ),
        dim=-1,
    )
    n = ca + torch.tensor([-0.35, 0.95, 0.20])
    c = ca + torch.tensor([1.20, 0.10, -0.15])
    return ca.unsqueeze(0), n.unsqueeze(0), c.unsqueeze(0)


def _edges(length: int):
    source = []
    target = []
    for index in range(length - 1):
        source.extend((index, index + 1))
        target.extend((index + 1, index))
    return torch.tensor([source, target], dtype=torch.long)


def _batch():
    generator = torch.Generator().manual_seed(11)
    ca_ab, n_ab, c_ab = _chain(5, 0.5)
    ca_ag, n_ag, c_ag = _chain(6, 8.0)
    return {
        "x_b": torch.randn(1, 5, 8, generator=generator),
        "x_g": torch.randn(1, 6, 7, generator=generator),
        "coords_ab": ca_ab,
        "coords_ag": ca_ag,
        "coords_n_ab": n_ab,
        "coords_c_ab": c_ab,
        "coords_n_ag": n_ag,
        "coords_c_ag": c_ag,
        "cdr_type": torch.tensor([[0, 1, 2, 3, 4]]),
        "mask_b": torch.ones(1, 5, dtype=torch.bool),
        "mask_g": torch.ones(1, 6, dtype=torch.bool),
        "coord_mask_ab": torch.ones(1, 5, dtype=torch.bool),
        "coord_mask_ag": torch.ones(1, 6, dtype=torch.bool),
        "edge_index_ab": _edges(5),
        "edge_index_ag": _edges(6),
    }


def _model(method: str):
    module = importlib.import_module(f"baselines.{method}.model")
    return module.Model(
        input_ab_dim=8,
        input_ag_dim=7,
        d_model=32,
        n_heads=4,
        n_layers_ab=1,
        n_layers_ag=1,
        n_layers_cross=1,
        dim_ff=64,
        dropout=0.0,
        use_cdr_type=True,
        use_global_context=True,
        use_pair_mil=True,
        geometry_config={
            "n_rbf": 8,
            "pair_hidden_dim": 12,
            "n_query_key_points": 2,
            "n_value_points": 2,
            "vector_channels": 4,
            "n_rope_freqs": 1,
            "n_fourier_bands": 2,
        },
    )


def _transform(batch, rotation, translation):
    output = dict(batch)
    for key in (
        "coords_ab",
        "coords_ag",
        "coords_n_ab",
        "coords_c_ab",
        "coords_n_ag",
        "coords_c_ag",
    ):
        output[key] = batch[key] @ rotation.T + translation
    return output


@pytest.mark.parametrize("method", ALL_METHODS)
def test_all_models_have_finite_forward_and_backward(method):
    torch.manual_seed(7)
    model = _model(method)
    batch = _batch()
    output = model(**batch)
    assert output.shape == (1, 6)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("method", INVARIANT_METHODS)
def test_invariant_models_preserve_logits_under_rigid_transform(method):
    torch.manual_seed(19)
    model = _model(method).eval()
    batch = _batch()
    rotation = _rotation()
    translation = torch.tensor([17.0, -8.0, 3.5])
    transformed = _transform(batch, rotation, translation)
    with torch.no_grad():
        reference = model(**batch)
        changed = model(**transformed)
    torch.testing.assert_close(changed, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("method", GLOBAL_FRAME_METHODS)
def test_global_frame_rope_remains_rotation_sensitive(method):
    torch.manual_seed(23)
    model = _model(method).eval()
    batch = _batch()
    transformed = _transform(
        batch, _rotation(), torch.tensor([17.0, -8.0, 3.5])
    )
    with torch.no_grad():
        reference = model(**batch)
        changed = model(**transformed)
    assert (changed - reference).abs().max().item() > 1e-5


def test_unaugmented_and_rotation_augmented_gf_models_are_identical():
    torch.manual_seed(27)
    unaugmented = _model("gf3drope").eval()
    torch.manual_seed(27)
    augmented = _model("gf3drope_rotaug").eval()
    assert list(unaugmented.state_dict()) == list(augmented.state_dict())
    for left, right in zip(
        unaugmented.state_dict().values(),
        augmented.state_dict().values(),
    ):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    with torch.no_grad():
        torch.testing.assert_close(
            unaugmented(**_batch()),
            augmented(**_batch()),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("method", ALL_METHODS)
def test_degenerate_and_missing_geometry_stays_finite(method):
    torch.manual_seed(29)
    model = _model(method).eval()
    batch = _batch()
    batch["coords_ab"] = batch["coords_ab"].clone()
    batch["coords_n_ab"] = batch["coords_n_ab"].clone()
    batch["coords_c_ab"] = batch["coords_c_ab"].clone()
    batch["coords_ab"][0, 1] = 0
    batch["coords_n_ab"][0, 1] = 0
    batch["coords_c_ab"][0, 1] = 0
    batch["coord_mask_ab"] = batch["coord_mask_ab"].clone()
    batch["coord_mask_ab"][0, 1] = False
    # A supplied-valid but collinear frame must also be handled safely.
    batch["coords_n_ab"][0, 2] = batch["coords_ab"][0, 2] + torch.tensor(
        [1.0, 0.0, 0.0]
    )
    batch["coords_c_ab"][0, 2] = batch["coords_ab"][0, 2] + torch.tensor(
        [2.0, 0.0, 0.0]
    )
    with torch.no_grad():
        output = model(**batch)
    assert torch.isfinite(output).all()


def test_gvp_all_zero_vector_path_has_finite_backward():
    """Regression test for the undefined derivative of sqrt at zero."""

    torch.manual_seed(31)
    model = _model("gvp")
    batch = _batch()
    for key in (
        "coords_ab",
        "coords_ag",
        "coords_n_ab",
        "coords_c_ab",
        "coords_n_ag",
        "coords_c_ag",
    ):
        batch[key] = torch.zeros_like(batch[key])
    batch["coord_mask_ab"] = torch.zeros_like(batch["coord_mask_ab"])
    batch["coord_mask_ag"] = torch.zeros_like(batch["coord_mask_ag"])
    batch["edge_index_ab"] = torch.empty(2, 0, dtype=torch.long)
    batch["edge_index_ag"] = torch.empty(2, 0, dtype=torch.long)

    output = model(**batch)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
