"""CPU scientific checks: no training, optimizer, dataset, or saved weights."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from baselines.lf3drope.model import (
    LocalFrame3DRoPEAttention,
    Model as OriginalLFModel,
    build_local_frames,
)
from baselines.lf3drope_bond_angle.model import (
    BondAngleAttention,
    Model,
    backbone_angle_features,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def geometry(length=7, batch=2):
    ca = torch.randn(batch, length, 3, dtype=torch.float64) * 3 + 8
    n = ca + torch.randn_like(ca) * .15 + ca.new_tensor([-.4, 1.2, .2])
    c = ca + torch.randn_like(ca) * .15 + ca.new_tensor([1.5, .1, .05])
    return dict(coords_ca=ca, coords_n=n, coords_c=c,
                token_mask=torch.ones(batch, length, dtype=torch.bool),
                coord_mask=torch.ones(batch, length, dtype=torch.bool))


def attention(*, chunk=3, checkpoint=True, dropout=.2, nonzero=True, config=None):
    original = LocalFrame3DRoPEAttention(32, 4, dropout, **(config or {})).double()
    result = BondAngleAttention(original, query_chunk_size=chunk,
                                checkpoint_attention=checkpoint)
    if nonzero:
        with torch.no_grad():
            result.angle_projection.weight.normal_(std=.08)
    return result


def dense_reference(module, x, g):
    """Use the original unchunked LF forward after angle input addition."""
    features, valid = backbone_angle_features(
        g['coords_ca'], g['coords_n'], g['coords_c'],
        g.get('coord_mask'), g.get('token_mask'),
    )
    embedding = module.angle_projection(features).masked_fill(~valid[..., None], 0.)
    return LocalFrame3DRoPEAttention.forward(module, x + embedding, **g)


def inputs_with_grad(x, g):
    xx = x.detach().clone().requires_grad_()
    gg = {key: value.detach().clone().requires_grad_(value.is_floating_point())
          for key, value in g.items()}
    return xx, gg


def compare_gradients(left, right, x1, x2, g1, g2):
    torch.testing.assert_close(x1.grad, x2.grad, atol=3e-10, rtol=3e-9)
    for key in ('coords_ca', 'coords_n', 'coords_c'):
        assert torch.isfinite(g1[key].grad).all()
        torch.testing.assert_close(g1[key].grad, g2[key].grad, atol=3e-10, rtol=3e-9)
    other_parameters = dict(right.named_parameters())
    for name, parameter in left.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other_parameters[name].grad,
                                   atol=3e-10, rtol=3e-9, msg=name)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('bias_config', [{}, {'use_distance_bias': False},
                                        {'use_fourier_bias': False}])
def test_chunked_forward_backward_matches_original_formula_and_rng(training, bias_config):
    torch.manual_seed(201)
    model = attention(config=bias_config).train(training)
    reference = copy.deepcopy(model)
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    g = geometry()
    g['coord_mask'][0, 2] = False
    g['token_mask'][1, -2:] = False
    x1, g1 = inputs_with_grad(x, g)
    x2, g2 = inputs_with_grad(x, g)
    torch.manual_seed(401)
    actual = model(x1, **g1)
    after_actual = torch.get_rng_state()
    torch.manual_seed(401)
    expected = dense_reference(reference, x2, g2)
    assert torch.equal(after_actual, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-11)
    readout = torch.randn_like(actual)
    before_backward = torch.get_rng_state()
    (actual * readout).sum().backward()
    (expected * readout).sum().backward()
    assert torch.equal(torch.get_rng_state(), before_backward)
    compare_gradients(model, reference, x1, x2, g1, g2)


@pytest.mark.parametrize('chunk,checkpoint_enabled', [(1, False), (3, False),
                                                     (1, True), (3, True), (128, True)])
def test_chunk_checkpoint_outputs_gradients_and_rng(chunk, checkpoint_enabled):
    torch.manual_seed(13)
    reference = attention(chunk=128, checkpoint=False, dropout=.35).train()
    model = copy.deepcopy(reference)
    model.query_chunk_size = chunk
    model.checkpoint_attention = checkpoint_enabled
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    g = geometry()
    x1, g1 = inputs_with_grad(x, g)
    x2, g2 = inputs_with_grad(x, g)
    torch.manual_seed(53)
    actual = model(x1, **g1)
    actual_rng = torch.get_rng_state()
    torch.manual_seed(53)
    expected = reference(x2, **g2)
    expected_rng = torch.get_rng_state()
    assert torch.equal(actual_rng, expected_rng)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-11)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    compare_gradients(model, reference, x1, x2, g1, g2)


def full_batch():
    ab, ag = geometry(4, 1), geometry(6, 1)
    result = dict(x_b=torch.randn(1, 4, 5, dtype=torch.float64),
                  x_g=torch.randn(1, 6, 7, dtype=torch.float64),
                  cdr_type=torch.tensor([[1, 2, 4, 6]]))
    for suffix, g, token in (('ab', ab, 'b'), ('ag', ag, 'g')):
        result['coords_' + suffix] = g['coords_ca']
        result['coords_n_' + suffix] = g['coords_n']
        result['coords_c_' + suffix] = g['coords_c']
        result['coord_mask_' + suffix] = g['coord_mask']
        result['mask_' + token] = g['token_mask']
    return result


def model_kwargs():
    return dict(input_ab_dim=5, input_ag_dim=7, d_model=32, n_heads=4,
                n_layers_ab=2, n_layers_ag=2, n_layers_cross=1, dim_ff=64, dropout=.2)


@pytest.mark.parametrize('training', [False, True])
def test_zero_init_preserves_full_lf_state_rng_output_and_existing_gradients(training, monkeypatch):
    monkeypatch.delenv('LF3DROPE_CHECKPOINT_MIN_TOKENS', raising=False)
    torch.manual_seed(42)
    original = OriginalLFModel(**model_kwargs()).double().train(training)
    original_rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = Model(**model_kwargs(), geometry_config={'attention_query_chunk_size': 2}).double().train(training)
    assert torch.equal(original_rng, torch.get_rng_state())
    state, baseline = model.state_dict(), original.state_dict()
    for name, value in baseline.items():
        assert torch.equal(state[name], value), name
    extra = set(state) - set(baseline)
    assert len(extra) == 4
    assert all(name.endswith('.angle_projection.weight') for name in extra)
    assert all(torch.count_nonzero(state[name]) == 0 for name in extra)
    batch = full_batch()
    torch.manual_seed(29)
    expected = original(**batch)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(29)
    actual = model(**batch)
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-11)
    actual.sum().backward()
    expected.sum().backward()
    baseline_parameters = dict(original.named_parameters())
    for name, parameter in model.named_parameters():
        if name in baseline_parameters:
            torch.testing.assert_close(parameter.grad, baseline_parameters[name].grad,
                                       atol=3e-10, rtol=3e-9, msg=name)
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().max() > 1e-8, name


def test_default_parameter_increment_and_exact_initial_state():
    torch.manual_seed(90)
    original = OriginalLFModel(5, 7)
    original_rng = torch.get_rng_state()
    torch.manual_seed(90)
    model = Model(5, 7)
    assert torch.equal(original_rng, torch.get_rng_state())
    for name, value in original.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    for encoder in (model.ab_encoder, model.ag_encoder):
        for layer in encoder.layers:
            assert layer.attention.angle_projection.weight.shape == (256, 2)
            assert layer.attention.angle_projection.bias is None
            assert layer.attention.query_chunk_size == 128
            assert layer.attention.checkpoint_attention is True
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in original.parameters()) == 4096


def test_default_head_allocation_fp32_dense_parity_and_nonzero_angle_gradient():
    torch.manual_seed(27)
    original = LocalFrame3DRoPEAttention(256, 8, .1).train()
    model = BondAngleAttention(copy.deepcopy(original), query_chunk_size=3).train()
    assert model.rope_dim == 30 and model.rest_dim == 2
    g = {key: value.float() if value.is_floating_point() else value
         for key, value in geometry(7, 1).items()}
    x = torch.randn(1, 7, 256)
    torch.manual_seed(28)
    expected = original(x, **g)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(28)
    actual = model(x, **g)
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    assert torch.isfinite(model.angle_projection.weight.grad).all()
    assert model.angle_projection.weight.grad.abs().max() > 1e-6


def test_angle_definition_and_current_coordinates_without_cache():
    theta = torch.tensor([0., math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi], dtype=torch.float64)
    ca = torch.zeros(1, 5, 3, dtype=torch.float64)
    c = ca + ca.new_tensor([1.5, 0., 0.])
    n = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), dim=-1)[None] * 1.45
    features, valid = backbone_angle_features(ca, n, c)
    expected = torch.stack((theta.cos(), theta.sin()), dim=-1)[None]
    torch.testing.assert_close(features, expected, atol=1e-14, rtol=1e-14)
    assert valid.all()
    changed_n = n.clone()
    changed_n[0, 2, 0] += .2
    perturbed, _ = backbone_angle_features(ca, changed_n, c)
    assert (perturbed - features).abs().max() > .1
    # A common translation of all three atoms in each residue leaves its angle.
    offset = torch.randn_like(ca)
    shifted, _ = backbone_angle_features(ca + offset, n + offset, c + offset)
    torch.testing.assert_close(shifted, features, atol=1e-14, rtol=1e-14)


def test_nonzero_angles_distinguish_fixed_ca_and_fixed_frames():
    torch.manual_seed(12)
    model = attention(nonzero=False, dropout=0.).eval()
    x = torch.randn(1, 1, 32, dtype=torch.float64).expand(1, 7, 32).clone()
    g = geometry(7, 1)
    ca = g['coords_ca']
    g['coords_c'] = ca + ca.new_tensor([1.5, 0., 0.])
    theta_a = torch.full((1, 7), math.radians(100), dtype=torch.float64)
    theta_b = torch.full((1, 7), math.radians(125), dtype=torch.float64)
    g['coords_n'] = ca + torch.stack((theta_a.cos(), theta_a.sin(), theta_a * 0), -1) * 1.45
    changed = dict(g)
    changed['coords_n'] = ca + torch.stack((theta_b.cos(), theta_b.sin(), theta_b * 0), -1) * 1.45
    torch.testing.assert_close(build_local_frames(ca, g['coords_n'], g['coords_c']),
                               build_local_frames(ca, changed['coords_n'], changed['coords_c']),
                               atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(model(x, **g), model(x, **changed), atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        model.angle_projection.weight.normal_(std=.1)
    assert (model(x, **g) - model(x, **changed)).abs().max() > 1e-3


def test_invalid_angle_features_are_zero_and_backward_finite():
    g = geometry(9, 1)
    g['coord_mask'][0, 0] = False
    g['token_mask'][0, 1] = False
    g['coords_n'][0, 2] = g['coords_ca'][0, 2]  # zero N bond
    g['coords_c'][0, 3] = g['coords_ca'][0, 3]  # zero C bond
    g['coords_n'][0, 4, 0] = float('nan')
    g['coords_c'][0, 5, 1] = float('inf')
    g['coords_ca'][0, 6, 2] = -float('inf')
    for key in ('coords_ca', 'coords_n', 'coords_c'):
        g[key].requires_grad_()
    features, valid = backbone_angle_features(g['coords_ca'], g['coords_n'], g['coords_c'],
                                               g['coord_mask'], g['token_mask'])
    assert not valid[0, :7].any() and valid[0, 7:].all()
    assert torch.count_nonzero(features[0, :7]) == 0
    assert torch.isfinite(features).all()
    features.sum().backward()
    for key in ('coords_ca', 'coords_n', 'coords_c'):
        assert torch.isfinite(g[key].grad).all(), key
        assert torch.count_nonzero(g[key].grad[0, :7]) == 0


def test_invalid_angle_projection_never_changes_original_attention_and_padding_is_zero():
    torch.manual_seed(8)
    model = attention(dropout=0.).eval()
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    g = geometry()
    g['coord_mask'].zero_()
    g['token_mask'][:, -1] = False
    g['coords_n'][:, 2] = float('nan')
    actual = model(x, **g)
    expected = LocalFrame3DRoPEAttention.forward(model, x, **g)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, -1]) == 0


def test_independent_partner_se3_invariance_of_full_model():
    torch.manual_seed(16)
    model = Model(**model_kwargs(), geometry_config={'attention_query_chunk_size': 2}).double().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('angle_projection.weight'):
                parameter.normal_(std=.1)
    batch = full_batch()
    moved = dict(batch)
    for partner in ('ab', 'ag'):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        rotation[:, 2] *= torch.linalg.det(rotation)
        translation = torch.randn(3, dtype=torch.float64) * 20
        for key in ('coords_' + partner, 'coords_n_' + partner, 'coords_c_' + partner):
            moved[key] = batch[key] @ rotation.T + translation
    torch.testing.assert_close(model(**batch), model(**moved), atol=3e-11, rtol=3e-10)


@pytest.mark.parametrize('config', [{'attention_query_chunk_size': 0},
                                  {'attention_query_chunk_size': True},
                                  {'attention_query_chunk_size': 1.5},
                                  {'attention_checkpoint': 'false'}])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        Model(**model_kwargs(), geometry_config=config)


def test_nested_existing_layer_checkpoint_keeps_dropout_rng(monkeypatch):
    torch.manual_seed(128)
    model = Model(**model_kwargs(), geometry_config={'attention_query_chunk_size': 2}).double().train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('angle_projection.weight'):
                parameter.normal_(std=.1)
    reference = copy.deepcopy(model)
    batch = full_batch()
    monkeypatch.delenv('LF3DROPE_CHECKPOINT_MIN_TOKENS', raising=False)
    torch.manual_seed(45)
    expected = reference(**batch)
    expected.square().sum().backward()
    expected_rng = torch.get_rng_state()
    monkeypatch.setenv('LF3DROPE_CHECKPOINT_MIN_TOKENS', '1')
    torch.manual_seed(45)
    actual = model(**batch)
    actual.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, atol=3e-10, rtol=3e-9, msg=name)
