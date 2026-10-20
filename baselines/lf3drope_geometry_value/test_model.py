"""CPU unit checks only: no optimizer, dataset, trained weights, or queue."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from baselines.lf3drope.model import (
    LocalFrame3DRoPEAttention,
    Model as OriginalLFModel,
    build_local_frames,
    local_pair_vectors,
)
from baselines.lf3drope_geometry_value.model import GeometryValueAttention, Model


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


def attention(*, chunk=3, checkpoint=True, dropout=.2, nonzero=True):
    original = LocalFrame3DRoPEAttention(32, 4, dropout).double()
    result = GeometryValueAttention(original, query_chunk_size=chunk, checkpoint_attention=checkpoint)
    if nonzero:
        with torch.no_grad():
            result.geometry_value_projection.weight.normal_(std=.08)
    return result


def dense_reference(module, x, geometry):
    """Materialize g_ij and multiply a_ij*(v_j+g_ij) independently of chunks."""
    q, k, v = module.project_qkv(x)
    ca = torch.nan_to_num(geometry['coords_ca'], nan=0., posinf=0., neginf=0.)
    frames = build_local_frames(geometry['coords_ca'], geometry['coords_n'], geometry['coords_c'])
    coordinate_mask, token_mask = geometry.get('coord_mask'), geometry.get('token_mask')
    local = local_pair_vectors(ca, frames, coordinate_mask)
    displacement = ca[:, None, :, :] - ca[:, :, None, :]
    pairs = None if coordinate_mask is None else coordinate_mask[:, :, None] & coordinate_mask[:, None, :]
    if pairs is not None:
        displacement = displacement.masked_fill(~pairs[..., None], 0.)
    distance = displacement.norm(dim=-1)
    logits = LocalFrame3DRoPEAttention.rotary_logits(module, q, k, local) * module.scale
    distance_bias = module.distance_bias(distance)
    fourier_bias = module.fourier_bias(local)
    if pairs is not None:
        distance_bias = distance_bias.masked_fill(~pairs[:, None], 0.)
        fourier_bias = fourier_bias.masked_fill(~pairs[:, None], 0.)
    logits = logits + distance_bias + fourier_bias
    if token_mask is not None:
        logits = logits.masked_fill(~token_mask[:, None, None, :], -1.e4)
    weights = module.attention_dropout(logits.softmax(-1))
    rbf = torch.exp(-module.distance_bias.gamma.abs() * (distance[..., None]-module.distance_bias.centers).square())
    phase = local[..., None]*module.fourier_bias.frequencies
    fourier = torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)
    features = torch.cat((rbf, fourier), -1)
    if token_mask is not None:
        token_pairs = token_mask[:, :, None] & token_mask[:, None, :]
        pairs = token_pairs if pairs is None else pairs & token_pairs
    if pairs is not None:
        features = features.masked_fill(~pairs[..., None], 0.)
    batch, length = x.shape[:2]
    messages = module.geometry_value_projection(features).reshape(batch, length, length, module.n_heads, module.head_dim)
    messages = messages.permute(0, 3, 1, 2, 4)
    out = (weights[..., None] * (v[:, :, None, :, :] + messages)).sum(dim=3)
    out = module.out_proj(out.transpose(1, 2).reshape(batch, length, module.d_model))
    return out if token_mask is None else out * token_mask[..., None]


def inputs_with_grad(x, g):
    xx = x.detach().clone().requires_grad_()
    gg = {key: value.detach().clone().requires_grad_(value.is_floating_point()) for key, value in g.items()}
    return xx, gg


def compare_gradients(left, right, x1, x2, g1, g2):
    torch.testing.assert_close(x1.grad, x2.grad, atol=3e-10, rtol=3e-9)
    for key in ('coords_ca', 'coords_n', 'coords_c'):
        assert torch.isfinite(g1[key].grad).all()
        torch.testing.assert_close(g1[key].grad, g2[key].grad, atol=3e-10, rtol=3e-9)
    right_parameters = dict(right.named_parameters())
    for name, p in left.named_parameters():
        other = right_parameters[name]
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, other.grad, atol=3e-10, rtol=3e-9, msg=name)


@pytest.mark.parametrize('training', [False, True])
def test_dense_message_formula_forward_backward_and_same_dropout(training):
    torch.manual_seed(201)
    model = attention(checkpoint=True).train(training)
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
    after_expected = torch.get_rng_state()
    assert torch.equal(after_actual, after_expected)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-11)
    readout = torch.randn_like(actual)
    (actual*readout).sum().backward()
    rng_before_reference_backward = torch.get_rng_state()
    (expected*readout).sum().backward()
    assert torch.equal(torch.get_rng_state(), rng_before_reference_backward)
    compare_gradients(model, reference, x1, x2, g1, g2)


@pytest.mark.parametrize('chunk,checkpoint_enabled', [(1, False), (3, False), (1, True), (3, True), (128, True)])
def test_chunk_checkpoint_outputs_gradients_and_rng(chunk, checkpoint_enabled):
    torch.manual_seed(13)
    ref = attention(chunk=128, checkpoint=False, dropout=.35).train()
    model = copy.deepcopy(ref)
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
    expected = ref(x2, **g2)
    expected_rng = torch.get_rng_state()
    assert torch.equal(actual_rng, expected_rng)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-11)
    actual.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    expected.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    compare_gradients(model, ref, x1, x2, g1, g2)


def full_batch():
    ab, ag = geometry(4, 1), geometry(6, 1)
    result = dict(x_b=torch.randn(1,4,5,dtype=torch.float64), x_g=torch.randn(1,6,7,dtype=torch.float64),
                  cdr_type=torch.tensor([[1,2,4,6]]))
    for suffix, g, token in (('ab',ab,'b'), ('ag',ag,'g')):
        result['coords_'+suffix] = g['coords_ca']
        result['coords_n_'+suffix] = g['coords_n']
        result['coords_c_'+suffix] = g['coords_c']
        result['coord_mask_'+suffix] = g['coord_mask']
        result['mask_'+token] = g['token_mask']
    return result


def model_kwargs():
    return dict(input_ab_dim=5, input_ag_dim=7, d_model=32, n_heads=4,
                n_layers_ab=2, n_layers_ag=2, n_layers_cross=1, dim_ff=64, dropout=.2)


@pytest.mark.parametrize('training', [False, True])
def test_zero_initialization_preserves_full_lf_parameters_buffers_rng_and_output(training, monkeypatch):
    monkeypatch.delenv('LF3DROPE_CHECKPOINT_MIN_TOKENS', raising=False)
    torch.manual_seed(42)
    original = OriginalLFModel(**model_kwargs()).double().train(training)
    original_rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = Model(**model_kwargs(), geometry_config={'attention_query_chunk_size':2}).double().train(training)
    assert torch.equal(original_rng, torch.get_rng_state())
    state, baseline_state = model.state_dict(), original.state_dict()
    for name, value in baseline_state.items():
        assert torch.equal(state[name], value), name
    extra = set(state)-set(baseline_state)
    assert len(extra) == 4
    assert all(name.endswith('.geometry_value_projection.weight') for name in extra)
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
    for name, p in model.named_parameters():
        if name in baseline_parameters:
            torch.testing.assert_close(p.grad, baseline_parameters[name].grad, atol=3e-10, rtol=3e-9, msg=name)
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    # The zero branch is trainable; it is not disconnected by an init shortcut.
    assert any(p.grad.abs().max() > 1e-8 for name,p in model.named_parameters() if name in extra)


def test_real_default_head_allocation_and_parameter_increment():
    torch.manual_seed(90)
    original = OriginalLFModel(5,7)
    original_rng = torch.get_rng_state()
    torch.manual_seed(90)
    model = Model(5,7)
    assert torch.equal(original_rng, torch.get_rng_state())
    for name, value in original.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    for encoder in (model.ab_encoder, model.ag_encoder):
        for layer in encoder.layers:
            assert layer.attention.head_dim == 32
            assert layer.attention.geometry_value_projection.weight.shape == (256,40)
            assert layer.attention.query_chunk_size == 128
            assert layer.attention.checkpoint_attention is True
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in original.parameters()) == 81920


def test_nonzero_geometry_values_break_constant_feature_symmetry():
    torch.manual_seed(12)
    model = attention(nonzero=False, dropout=0.).eval()
    x = torch.randn(1,1,32,dtype=torch.float64).expand(1,7,32).clone()
    a, b = geometry(7,1), geometry(7,1)
    torch.testing.assert_close(model(x, **a), model(x, **b), atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        model.geometry_value_projection.weight.normal_(std=.1)
    out_a, out_b = model(x, **a), model(x, **b)
    assert (out_a-out_b).abs().max() > 1e-3
    assert (out_a-out_a[:, :1]).abs().max() > 1e-3


def test_invalid_geometry_zero_values_and_padding_do_not_contribute():
    torch.manual_seed(8)
    model = attention(dropout=0.).eval()
    with torch.no_grad():
        model.v_proj.weight.zero_(); model.v_proj.bias.zero_()
        model.out_proj.weight.copy_(torch.eye(32,dtype=torch.float64)); model.out_proj.bias.zero_()
    x = torch.randn(2,7,32,dtype=torch.float64)
    g = geometry()
    g['coord_mask'][:,2] = False
    for key in ('coords_ca','coords_n','coords_c'):
        g[key][:,2] = float('nan')
    g['token_mask'][:,-1] = False
    out = model(x, **g)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[:,2]) == 0
    assert torch.count_nonzero(out[:,-1]) == 0
    assert out[:,:2].abs().max() > 1e-5
    changed = {k:v.clone() for k,v in g.items()}
    for key in ('coords_ca','coords_n','coords_c'):
        changed[key][:,-1] = 9000
    xx = x.clone(); xx[:,-1] = 9000
    torch.testing.assert_close(model(xx, **changed), out, atol=1e-12, rtol=1e-12)
    g['coord_mask'].zero_()
    assert torch.count_nonzero(model(x, **g)) == 0
    g['token_mask'].zero_()
    assert torch.count_nonzero(model(x, **g)) == 0


def test_independent_partner_se3_invariance_of_full_model():
    torch.manual_seed(16)
    model = Model(**model_kwargs(), geometry_config={'attention_query_chunk_size':2}).double().eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.endswith('geometry_value_projection.weight'):
                p.normal_(std=.04)
    batch = full_batch()
    moved = dict(batch)
    for partner in ('ab','ag'):
        q, _ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        q[:,2] *= torch.linalg.det(q)
        translation = torch.randn(3,dtype=torch.float64)*20
        for key in ('coords_'+partner,'coords_n_'+partner,'coords_c_'+partner):
            moved[key] = batch[key]@q.T+translation
    torch.testing.assert_close(model(**batch), model(**moved), atol=3e-11, rtol=3e-10)


@pytest.mark.parametrize('config', [
    {'attention_query_chunk_size':0}, {'attention_query_chunk_size':True},
    {'attention_checkpoint':'false'}, {'use_distance_bias':False}, {'use_fourier_bias':False},
])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        Model(**model_kwargs(), geometry_config=config)


def test_nested_existing_layer_checkpoint_keeps_dropout_rng(monkeypatch):
    torch.manual_seed(128)
    model = Model(**model_kwargs(), geometry_config={'attention_query_chunk_size':2}).double().train()
    reference = copy.deepcopy(model)
    batch = full_batch()
    monkeypatch.delenv('LF3DROPE_CHECKPOINT_MIN_TOKENS', raising=False)
    torch.manual_seed(45)
    expected = reference(**batch)
    expected.square().sum().backward()
    expected_rng = torch.get_rng_state()
    monkeypatch.setenv('LF3DROPE_CHECKPOINT_MIN_TOKENS','1')
    torch.manual_seed(45)
    actual = model(**batch)
    actual.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for (name,p),(other_name,other) in zip(model.named_parameters(),reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad,other.grad,atol=3e-10,rtol=3e-9,msg=name)
