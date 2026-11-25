"""Target registry — first-barrier primary; touch + ternary legacy."""

from __future__ import annotations

from pm.targets.atr_barrier import (
    DEFAULT_ATR_PERIOD as BARRIER_ATR_PERIOD,
    DEFAULT_BARRIER_TAU,
    DEFAULT_HORIZON as BARRIER_HORIZON,
    AtrBarrierFirstTarget,
    AtrBarrierTouchTarget,
)
from pm.targets.base import TargetSpec
from pm.targets.ternary_displacement import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_FLAT_TAU_BY_H,
    DEFAULT_HORIZONS,
    TernaryDisplacementTarget,
)

TARGET_REGISTRY: dict[str, TargetSpec] = {}


def _register_defaults() -> None:
    first = AtrBarrierFirstTarget(
        horizon=BARRIER_HORIZON,
        atr_period=BARRIER_ATR_PERIOD,
        barrier_tau=DEFAULT_BARRIER_TAU,
        target_id="atr_barrier_first",
    )
    TARGET_REGISTRY[first.id] = first

    touch = AtrBarrierTouchTarget(
        horizon=BARRIER_HORIZON,
        atr_period=BARRIER_ATR_PERIOD,
        barrier_tau=DEFAULT_BARRIER_TAU,
        target_id="atr_barrier_touch",
    )
    TARGET_REGISTRY[touch.id] = touch

    ternary = TernaryDisplacementTarget(
        horizons=DEFAULT_HORIZONS,
        atr_period=DEFAULT_ATR_PERIOD,
        flat_tau=DEFAULT_FLAT_TAU_BY_H,
        target_id="ternary_displacement",
    )
    TARGET_REGISTRY[ternary.id] = ternary


_register_defaults()


def list_target_ids() -> list[str]:
    return sorted(TARGET_REGISTRY.keys())


def get_target(target_id: str) -> TargetSpec:
    if target_id not in TARGET_REGISTRY:
        raise KeyError(
            f"unknown target id {target_id!r} (known={list_target_ids()})"
        )
    return TARGET_REGISTRY[target_id]
