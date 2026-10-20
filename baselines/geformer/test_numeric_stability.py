"""Synthetic regression for the unused terminal IPE branch overflowing."""

from __future__ import annotations

import pytest
import torch

from baselines.geformer import model as repaired


def _inputs():
    coordinates = torch.tensor([
        [0.0, 0.0, 0.0], [1.2, 0.2, 0.1], [0.3, 1.4, 0.5],
        [1.1, 1.3, 0.7], [2.0, 0.4, 1.0],
    ], dtype=torch.float64).unsqueeze(0)
    return torch.randn(1, 5, 16, dtype=torch.float64), coordinates


@pytest.mark.parametrize("checkpoint", [0, 1])
def test_node_only_forward_survives_terminal_ipe_overflow(checkpoint):
    """The essential regression runs without external source or dataset files."""
    torch.manual_seed(601)
    encoder = repaired.GeoformerEncoder(
        16, 4, 1, dim_ff=32, dropout=0., num_rbf=8,
        checkpoint_min_tokens=checkpoint,
    ).train()
    with torch.no_grad():
        encoder.layers[-1].self_attn.dihedral_proj.weight.fill_(1e25)
    features, positions = _inputs()
    features, positions = features.float(), positions.float()
    calls = []
    handle = encoder.layers[-1].self_attn.dihedral_proj.register_forward_hook(
        lambda *_: calls.append("terminal_ipe")
    )
    try:
        prediction = encoder(features, positions)
        assert calls == []
        assert torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
        assert calls == []
        with torch.no_grad():
            nodes, _, pairs = encoder.forward_with_edges(features, positions)
        assert calls == ["terminal_ipe"]
        assert not torch.isfinite(pairs).all()
        torch.testing.assert_close(nodes, prediction.detach(), atol=0, rtol=0)
    finally:
        handle.remove()
