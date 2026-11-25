"""Horizons and outcome labels."""

from pm.targets.base import TargetSpec
from pm.targets.registry import TARGET_REGISTRY, get_target, list_target_ids
from pm.targets.ternary_displacement import (
    DEFAULT_HORIZONS,
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    TernaryDisplacementTarget,
)

__all__ = [
    "TargetSpec",
    "TARGET_REGISTRY",
    "get_target",
    "list_target_ids",
    "TernaryDisplacementTarget",
    "DEFAULT_HORIZONS",
    "LABEL_DOWN",
    "LABEL_FLAT",
    "LABEL_UP",
]
