"""Shared implementation for SE(3)-aware AsEP baselines.

Dataset symbols live in :mod:`baselines.se3bench.data`.  Keeping package
initialization minimal lets the torch-only models be imported without eagerly
requiring the isolated BioPython dependency.
"""

from .common import GeometryPredictor

__all__ = ["GeometryPredictor"]
