"""Predictor and shared-training integration checks on synthetic full contexts."""
from __future__ import annotations

import copy
from unittest.mock import patch

import torch
from torch import nn

from baselines.se3bench.common import GeometryPredictor
from .data import selected_coordinates
from .model import Model
from .train import apply_train_augmentation, sync_selected_coordinates


def context(seed=1):
    generator = torch.Generator().manual_seed(seed)
    ca = torch.randn(1, 12, 3, generator=generator).cumsum(1) * 2
    coords = ca[:, :, None, :] + torch.randn(1, 12, 4, 3, generator=generator)
    coords[:, :, 1] = ca
    return {"coords": coords, "atom_mask": torch.ones(1, 12, 4, dtype=torch.bool),
            "chain_id": torch.zeros(1, 12, dtype=torch.long),
            "residue_index": torch.arange(12).unsqueeze(0),
            "link_mask": torch.ones(1, 11, dtype=torch.bool),
            "selected_indices": torch.tensor([[1, 2, 5, 8, 9]]),
            "selected_coord_mask": torch.ones(1, 5, dtype=torch.bool)}


def batch():
    generator = torch.Generator().manual_seed(91)
    result = {"context_ab": context(1), "context_ag": context(2),
              "x_b": torch.randn(1, 5, 8, generator=generator),
              "x_g": torch.randn(1, 5, 7, generator=generator),
              "cdr_type": torch.tensor([[0, 1, 2, 3, 4]]),
              "mask_b": torch.ones(1, 5, dtype=torch.bool),
              "mask_g": torch.ones(1, 5, dtype=torch.bool)}
    return sync_selected_coordinates(result)


def kwargs():
    return dict(input_ab_dim=8, input_ag_dim=7, d_model=16, n_heads=4,
                n_layers_ab=2, n_layers_ag=2, n_layers_cross=1, dim_ff=32,
                dropout=0., use_cdr_type=True, use_global_context=True, use_pair_mil=True)


def model():
    return Model(**kwargs(), geometry_config={"deepinteract_node_count_limit": 32})


def test_complete_predictor_has_finite_forward_backward_and_independent_pose_invariance():
    torch.manual_seed(7)
    predictor = model()
    original = batch()
    logits = predictor(**original)
    assert logits.shape == (1, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in predictor.parameters())
    transformed = copy.deepcopy(original)
    for part in ("ab", "ag"):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        rotation[:, 0] *= torch.linalg.det(rotation)
        transformed[f"context_{part}"]["coords"] = transformed[f"context_{part}"]["coords"] @ rotation.T + torch.randn(3) * 10
    sync_selected_coordinates(transformed)
    predictor.eval()
    with torch.no_grad():
        torch.testing.assert_close(predictor(**original), predictor(**transformed), atol=3e-4, rtol=3e-4)


def test_full_context_augmentation_is_consistent_and_does_not_mutate_inputs():
    original = batch()
    snapshot = copy.deepcopy(original)
    hp = {"augmentation": {"enabled": True, "coord_jitter_std": 0.2, "coord_jitter_prob": 1.0,
                           "global_rotation_prob": 0., "feature_dropout_prob": 0., "feature_noise_std": 0.}}
    augmented = apply_train_augmentation(original, hp)
    for part in ("ab", "ag"):
        current = augmented[f"context_{part}"]
        assert not torch.equal(current["coords"][..., :3, :], original[f"context_{part}"]["coords"][..., :3, :])
        torch.testing.assert_close(current["coords"][..., 3, :], original[f"context_{part}"]["coords"][..., 3, :], atol=0, rtol=0)
        torch.testing.assert_close(original[f"context_{part}"]["coords"], snapshot[f"context_{part}"]["coords"], atol=0, rtol=0)
        selected = selected_coordinates(current)
        torch.testing.assert_close(augmented[f"coords_{part}"], selected["coords_ca"], atol=0, rtol=0)
    predictor = model().eval()
    with torch.no_grad():
        assert not torch.allclose(predictor(**original), predictor(**augmented), atol=1e-6, rtol=1e-6)


class Passthrough(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_shared_partner_conditioning_and_heads_are_unchanged():
    # Replacing both encoders with identities isolates the common projections,
    # cross attention, antibody context, node head and pair-MIL computations.
    torch.manual_seed(31)
    predictor = model().eval()
    common = GeometryPredictor(**kwargs(), encoder_factory=lambda n: Passthrough()).eval()
    predictor.ab_encoder = Passthrough()
    predictor.ag_encoder = Passthrough()
    common.load_state_dict(predictor.state_dict(), strict=True)
    inputs = batch()
    with patch("baselines.deepinteract.model.build_geometry", return_value={}), torch.no_grad():
        actual = predictor(**inputs)
        expected = common(**{k: v for k, v in inputs.items() if not k.startswith("context_")})
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
