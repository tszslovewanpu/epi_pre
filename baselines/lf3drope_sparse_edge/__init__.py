"""LF3DRoPE restricted to the dataset's supplied directed intra-chain edges."""

__all__ = ["Model", "SparseLocalFrame3DRoPEAttention"]


def __getattr__(name):
    # Configuration/queue inspection must not load Torch or initialize a model.
    if name in __all__:
        from importlib import import_module
        return getattr(import_module(f"{__name__}.model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
