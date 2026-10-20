"""CPU checks against the unmodified LF equations and GVP graph convention."""

from __future__ import annotations

import copy
import io

import pytest
import torch

from baselines.gvp.model import SparseGVPEncoder
from baselines.lf3drope.model import (
    LocalFrame3DRoPEAttention,
    Model as OriginalModel,
    build_local_frames,
    local_pair_vectors,
)
from baselines.se3bench.common import ResidualAttentionLayer
from baselines.lf3drope_sparse_edge.model import Model, SparseLocalFrame3DRoPEAttention


def _coordinates(length):
    ca = torch.randn(1, length, 3, dtype=torch.float64) + 2.0
    nitrogen = ca + torch.tensor([-.4, .8, .2], dtype=ca.dtype)
    carbon = ca + torch.tensor([1.1, .1, -.2], dtype=ca.dtype)
    return dict(coords_ca=ca, coords_n=nitrogen, coords_c=carbon)


def _complete_graph(length):
    # target-major order, including the diagonal, matches dense attention rows.
    target, source = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    return torch.stack((source.flatten(), target.flatten()))


def _attention(*, dropout=0.0, chunk=2, checkpoint=False):
    return SparseLocalFrame3DRoPEAttention(
        28, 2, dropout, n_rope_freqs=2,
        attention_query_chunk_size=chunk, attention_checkpoint=checkpoint,
    ).double()


def _model(cls, *, chunk=2, checkpoint=False, dropout=0.0):
    return cls(7, 9, d_model=28, n_heads=2, n_layers_ab=2, n_layers_ag=2,
               n_layers_cross=1, dim_ff=36, dropout=dropout,
               geometry_config={"attention_query_chunk_size":chunk,
                                "attention_checkpoint":checkpoint}).double()


def _batch():
    ab, ag = _coordinates(4), _coordinates(5)
    return dict(
        x_b=torch.randn(1, 4, 7, dtype=torch.float64),
        x_g=torch.randn(1, 5, 9, dtype=torch.float64),
        coords_ab=ab["coords_ca"], coords_n_ab=ab["coords_n"], coords_c_ab=ab["coords_c"],
        coords_ag=ag["coords_ca"], coords_n_ag=ag["coords_n"], coords_c_ag=ag["coords_c"],
        cdr_type=torch.tensor([[0, 1, 3, 6]]),
        edge_index_ab=_complete_graph(4), edge_index_ag=_complete_graph(5),
    )


def _gradients_match(reference, observed, atol=2e-10, rtol=2e-9):
    left, right = dict(reference.named_parameters()), dict(observed.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        expected, actual = left[name].grad, right[name].grad
        assert (expected is None) == (actual is None), name
        if expected is not None:
            assert torch.isfinite(actual).all(), name
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)


def _dense_restricted_reference(attention, x, coordinates, edges):
    """Use original dense LF formula, then independently select each row's keys.

    Iterating the explicit edge list also preserves duplicated keys, unlike a
    boolean adjacency matrix.  This reference uses neither sparse logits nor
    sparse aggregation from the implementation under test.
    """
    q, k, v = attention.project_qkv(x)
    frames = build_local_frames(**coordinates)
    local = local_pair_vectors(coordinates["coords_ca"], frames)
    logits = attention.rotary_logits(q, k, local) * attention.scale
    distance = (coordinates["coords_ca"][:, None] - coordinates["coords_ca"][:, :, None]).norm(dim=-1)
    if attention.distance_bias is not None:
        logits = logits + attention.distance_bias(distance)
    if attention.fourier_bias is not None:
        logits = logits + attention.fourier_bias(local)
    rows = []
    for target in range(x.shape[1]):
        source = edges[0, edges[1] == target]
        if source.numel():
            weights = logits[0, :, target, source].softmax(-1)
            aggregate = (weights[..., None] * v[0, :, source]).sum(1).reshape(1, 1, -1)
            rows.append(attention.out_proj(aggregate))
        else:
            rows.append(x.new_zeros(1, 1, x.shape[-1]))
    return torch.cat(rows, dim=1)


@pytest.mark.parametrize("chunk", [1, 128])
def test_complete_graph_matches_original_full_predictor_state_outputs_and_gradients(chunk):
    torch.manual_seed(152)
    original = _model(OriginalModel).eval()
    original_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(152)
    sparse = _model(Model, chunk=chunk).eval()
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    assert original.state_dict().keys() == sparse.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(sparse.state_dict()[name], value, atol=0, rtol=0)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in sparse.parameters())
    batch = _batch()
    batch_left = {key: value.clone().requires_grad_(value.is_floating_point()) for key, value in batch.items()}
    batch_right = {key: value.clone().requires_grad_(value.is_floating_point()) for key, value in batch.items()}
    expected, actual = original(**batch_left), sparse(**batch_right)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    probe = torch.linspace(-.5, 1.3, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    _gradients_match(original, sparse)
    for name in batch_left:
        if batch_left[name].is_floating_point():
            assert batch_left[name].grad is not None, name
            torch.testing.assert_close(batch_right[name].grad, batch_left[name].grad, atol=2e-10, rtol=2e-9, msg=name)


def test_asymmetric_edges_and_duplicate_edges_match_independent_dense_formula():
    torch.manual_seed(77)
    sparse = _attention().eval()
    reference = LocalFrame3DRoPEAttention(28, 2, 0.0, n_rope_freqs=2).double().eval()
    reference.load_state_dict(sparse.state_dict(), strict=True)
    x = torch.randn(1, 5, 28, dtype=torch.float64)
    coordinates = _coordinates(5)
    # Includes a duplicate, a self-loop, unordered targets and an isolated node.
    edges = torch.tensor([[0, 1, 1, 3, 4, 0], [2, 2, 2, 1, 4, 1]])
    expected = _dense_restricted_reference(reference, x, coordinates, edges)
    actual = sparse(x, **coordinates, edge_index=edges)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    assert torch.equal(actual[:, 0], torch.zeros_like(actual[:, 0]))
    assert torch.equal(actual[:, 3], torch.zeros_like(actual[:, 3]))
    # A single 0->2 edge must deliver exactly V(0) to query 2, regardless of
    # distance or self-loop availability.  Reversing the convention fails this.
    single = sparse(x, **coordinates, edge_index=torch.tensor([[0], [2]]))
    torch.testing.assert_close(single[:, 2], sparse.out_proj(sparse.v_proj(x[:, 0])), atol=2e-15, rtol=2e-14)
    assert torch.count_nonzero(single[:, [0, 1, 3, 4]]) == 0
    expected.square().sum().backward()
    actual.square().sum().backward()
    _gradients_match(reference, sparse)


def test_gvp_geometry_filtering_empty_graph_and_local_ffn_are_finite_and_do_not_mutate_inputs():
    torch.manual_seed(28)
    sparse = _attention().eval()
    x = torch.randn(1, 7, 28, dtype=torch.float64, requires_grad=True)
    coordinates = _coordinates(7)
    # Explicit true mask cannot resurrect NaN or the all-zero backbone sentinel.
    coordinates["coords_n"][0, 2, 0] = torch.nan
    for value in coordinates.values():
        value[0, 3] = 0
    for name in coordinates:
        coordinates[name].requires_grad_()
    token_mask = torch.tensor([[True, True, True, True, True, False, True]])
    coord_mask = torch.tensor([[True, True, True, True, False, True, True]])
    edges = _complete_graph(7)
    saved = {name: value.detach().clone() for name, value in coordinates.items()}
    saved_edges = edges.clone()
    actual = sparse(x, **coordinates, token_mask=token_mask, coord_mask=coord_mask, edge_index=edges)
    valid = torch.tensor([True, True, False, False, False, False, True])
    filtered = SparseGVPEncoder._prepare_edges(edges, length=7, device=x.device, valid_nodes=valid)
    cleaned = {name: torch.nan_to_num(value) for name, value in coordinates.items()}
    reference = _dense_restricted_reference(sparse, x, cleaned, filtered)
    torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-11)
    assert torch.count_nonzero(actual[:, ~valid]) == 0
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(value.grad).all() for value in coordinates.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sparse.parameters())
    for name, value in saved.items():
        torch.testing.assert_close(coordinates[name], value, atol=0, rtol=0, equal_nan=True)
    assert torch.equal(edges, saved_edges)

    sparse.zero_grad(set_to_none=True)
    empty = torch.empty(2, 0, dtype=torch.long)
    output = sparse(x.detach(), **cleaned, edge_index=empty)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sparse.parameters())
    block = ResidualAttentionLayer(sparse, 28, 32, 0.0).double().eval()
    local_input = x.detach()
    expected_local = local_input + block.ffn(block.norm2(local_input))
    observed_local = block(local_input, **cleaned, edge_index=empty)
    torch.testing.assert_close(observed_local, expected_local, atol=0, rtol=0)


def test_independent_partner_se3_invariance_and_fixed_graph_jitter():
    torch.manual_seed(320)
    model = _model(Model).eval()
    batch = _batch()
    batch["edge_index_ab"] = torch.tensor([[0, 1, 2, 3, 3], [1, 2, 3, 0, 2]])
    batch["edge_index_ag"] = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
    expected = model(**batch)
    moved = {name: value.clone() for name, value in batch.items()}
    for partner in ("ab", "ag"):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        rotation[:, -1] *= torch.det(rotation)
        shift = 5 * torch.randn(3, dtype=torch.float64)
        for key in (f"coords_{partner}", f"coords_n_{partner}", f"coords_c_{partner}"):
            moved[key] = batch[key] @ rotation + shift
    torch.testing.assert_close(model(**moved), expected, atol=2e-11, rtol=2e-10)
    jittered = {name: value.clone() for name, value in batch.items()}
    for name in jittered:
        if name.startswith("coords_"):
            jittered[name] = jittered[name] + .2 * torch.randn_like(jittered[name])
    observed = model(**jittered)
    assert torch.isfinite(observed).all()
    assert (observed - expected).abs().max() > 1e-8
    assert torch.equal(jittered["edge_index_ab"], batch["edge_index_ab"])
    assert torch.equal(jittered["edge_index_ag"], batch["edge_index_ag"])


@pytest.mark.parametrize("outer_checkpoint", ["0", "1"])
def test_checkpoint_preserves_dropout_outputs_all_gradients_and_rng(monkeypatch, outer_checkpoint):
    monkeypatch.setenv("LF3DROPE_CHECKPOINT_MIN_TOKENS", outer_checkpoint)
    torch.manual_seed(125)
    plain = _model(Model, checkpoint=False, dropout=.2).train()
    recompute = copy.deepcopy(plain)
    for encoder in (recompute.ab_encoder, recompute.ag_encoder):
        for layer in encoder.layers:
            layer.attention.attention_checkpoint = True
    batch = _batch()
    results = []
    for model in (plain, recompute):
        torch.manual_seed(415)
        prediction = model(**batch)
        prediction.square().sum().backward()
        results.append((prediction, torch.random.get_rng_state().clone()))
    torch.testing.assert_close(results[1][0], results[0][0], atol=0, rtol=0)
    assert torch.equal(results[1][1], results[0][1])
    _gradients_match(plain, recompute, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["missing", "edge_shape", "float_edge", "negative", "out_of_range", "float_token_mask", "float_coord_mask", "mask_shape", "batch_two", "coordinate_shape"])
def test_malformed_inputs_fail_explicitly(case):
    attention = _attention()
    x = torch.randn(1, 4, 28, dtype=torch.float64)
    kwargs = dict(**_coordinates(4), edge_index=_complete_graph(4))
    if case == "missing":
        kwargs["edge_index"] = None
    elif case == "edge_shape":
        kwargs["edge_index"] = torch.tensor([0, 1])
    elif case == "float_edge":
        kwargs["edge_index"] = kwargs["edge_index"].float()
    elif case == "negative":
        kwargs["edge_index"] = torch.tensor([[-1], [0]])
    elif case == "out_of_range":
        kwargs["edge_index"] = torch.tensor([[0], [4]])
    elif case == "float_token_mask":
        kwargs["token_mask"] = torch.ones(1, 4)
    elif case == "float_coord_mask":
        kwargs["coord_mask"] = torch.ones(1, 4)
    elif case == "mask_shape":
        kwargs["coord_mask"] = torch.ones(4, dtype=torch.bool)
    elif case == "batch_two":
        x = x.expand(2, -1, -1)
    elif case == "coordinate_shape":
        kwargs["coords_ca"] = kwargs["coords_ca"][0]
    with pytest.raises((ValueError, TypeError)):
        attention(x, **kwargs)


def test_state_dict_checkpoint_and_full_module_pickle_roundtrip():
    torch.manual_seed(771)
    model = _model(Model, checkpoint=True).eval()
    batch = _batch()
    expected = model(**batch)
    state = io.BytesIO()
    torch.save({"model_state_dict":model.state_dict()}, state)
    state.seek(0)
    loaded = _model(Model, checkpoint=True).eval()
    loaded.load_state_dict(torch.load(state, weights_only=True)["model_state_dict"], strict=True)
    torch.testing.assert_close(loaded(**batch), expected, atol=0, rtol=0)
    package = io.BytesIO()
    torch.save(model, package)
    package.seek(0)
    pickled = torch.load(package, weights_only=False)
    assert isinstance(pickled.ab_encoder.layers[0].attention, SparseLocalFrame3DRoPEAttention)
    assert pickled.ab_encoder.layers[0].attention.attention_checkpoint is True
    torch.testing.assert_close(pickled(**batch), expected, atol=0, rtol=0)
