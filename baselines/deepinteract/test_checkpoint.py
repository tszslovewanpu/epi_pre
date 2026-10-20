"""CPU checks for GeoT checkpoint equivalence and very short graphs."""

import copy

import pytest
import torch

from .layers import GeoTEncoder


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_checkpoint_preserves_forward_gradients_rng_and_batchnorm(norm):
    torch.manual_seed(17)
    plain = GeoTEncoder(
        16, 4, 2, dropout=0.2, norm_to_apply=norm, node_count_limit=8,
        shared_embed_size=8, num_pre_res_blocks=1, num_post_res_blocks=1,
    )
    recomputed = copy.deepcopy(plain)
    recomputed.checkpoint_min_tokens = 1
    x = torch.randn(1, 5, 16, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    geometry = {
        "edge_index": torch.tensor([
            [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        ]),
        "edge_features": torch.randn(10, 28),
        "node_features": torch.randn(1, 5, 7),
        "src_nbr_e_ids": torch.tensor([[i, (i + 1) % 10] for i in range(10)]),
        "dst_nbr_e_ids": torch.tensor([[i, (i + 2) % 10] for i in range(10)]),
    }
    torch.manual_seed(9)
    expected = plain(x, geometry)
    expected.square().mean().backward()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(9)
    actual = recomputed(other_x, geometry)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(other_x.grad, x.grad)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
    other_parameters = dict(recomputed.named_parameters())
    for name, parameter in plain.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other_parameters[name].grad)
    other_buffers = dict(recomputed.named_buffers())
    for name, buffer in plain.named_buffers():
        torch.testing.assert_close(buffer, other_buffers[name])


@pytest.mark.parametrize("num_nodes,num_edges", [(1, 1), (1, 0), (3, 0)])
def test_short_graphs_have_finite_forward_and_backward(num_nodes, num_edges):
    model = GeoTEncoder(
        8, 2, 1, node_count_limit=8,
        num_pre_res_blocks=1, num_post_res_blocks=1,
    )
    x = torch.randn(1, num_nodes, 8, requires_grad=True)
    geometry = {
        "edge_index": torch.zeros(2, num_edges, dtype=torch.long),
        "edge_features": torch.randn(num_edges, 28),
        "node_features": torch.randn(1, num_nodes, 7),
        "src_nbr_e_ids": torch.full((num_edges, 2), -1),
        "dst_nbr_e_ids": torch.full((num_edges, 2), -1),
    }
    output = model(x, geometry)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
