"""Target registry — ternary displacement on 3/10/30 with per-h tau."""

from __future__ import annotations

from pm.targets.base import TargetSpec
from pm.targets.ternary_displacement import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_FLAT_TAU_BY_H,
    DEFAULT_HORIZONS,
    TernaryDisplacementTarget,
)

TARGET_REGISTRY: dict[str, TargetSpec] = {}


def _register_defaults() -> None:
    tgt = TernaryDisplacementTarget(
        horizons=DEFAULT_HORIZONS,
        atr_period=DEFAULT_ATR_PERIOD,
        flat_tau=DEFAULT_FLAT_TAU_BY_H,
        target_id="ternary_displacement",
    )
    TARGET_REGISTRY[tgt.id] = tgt


_register_defaults()


def list_target_ids() -> list[str]:
    return sorted(TARGET_REGISTRY.keys())


def get_target(target_id: str) -> TargetSpec:
    if target_id not in TARGET_REGISTRY:
        raise KeyError(
            f"unknown target id {target_id!r} (known={list_target_ids()})"
        )
    return TARGET_REGISTRY[target_id]
