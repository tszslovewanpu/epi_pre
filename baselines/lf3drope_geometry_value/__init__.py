"""Dense LF3DRoPE geometry-conditioned value ablation."""

__all__ = ["GeometryValueAttention", "Model"]


def __getattr__(name):
    # Preparing an experiment grid should load only the deployment helpers.
    if name in __all__:
        from importlib import import_module
        return getattr(import_module(f"{__name__}.model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
