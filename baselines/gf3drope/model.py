"""Unaugmented supplied GF3DRoPE model.

GF3DRoPE and the rotation-augmentation baseline have the same architecture.
The controlled difference is entirely in the training pipeline:
``gf3drope`` uses no global rotation, whereas ``gf3drope_rotaug`` applies one
Haar-uniform global rotation to every training complex.
"""

from baselines.gf3drope_rotaug.model import (
    GlobalFrame3DRoPEAttention,
    Model,
)

__all__ = ["GlobalFrame3DRoPEAttention", "Model"]
