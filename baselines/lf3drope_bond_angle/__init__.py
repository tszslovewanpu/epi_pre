"""Invariant bond-angle-conditioned local-frame 3D RoPE baseline."""

__all__ = ["BondAngleAttention", "Model", "backbone_angle_features"]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        return getattr(import_module(f"{__name__}.model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
