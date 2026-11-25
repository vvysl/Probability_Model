"""Target registry — empty until step 2 (horizons)."""

from __future__ import annotations

from pm.targets.base import TargetSpec

TARGET_REGISTRY: dict[str, TargetSpec] = {}


def list_target_ids() -> list[str]:
    return sorted(TARGET_REGISTRY.keys())


def get_target(target_id: str) -> TargetSpec:
    if target_id not in TARGET_REGISTRY:
        raise KeyError(
            f"unknown target id {target_id!r}; registry empty until step 2 "
            f"(known={list_target_ids()})"
        )
    return TARGET_REGISTRY[target_id]
