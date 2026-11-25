"""State variable registry — empty until step 3 (market constants)."""

from __future__ import annotations

from pm.state.base import StateVar

STATE_REGISTRY: dict[str, StateVar] = {}


def list_state_ids() -> list[str]:
    return sorted(STATE_REGISTRY.keys())


def get_state_var(state_id: str) -> StateVar:
    if state_id not in STATE_REGISTRY:
        raise KeyError(
            f"unknown state id {state_id!r}; registry empty until step 3 "
            f"(known={list_state_ids()})"
        )
    return STATE_REGISTRY[state_id]


def build_state_matrix(bars, state_ids: list[str]):
    """Concatenate registered state vars; step 3 will populate registry."""
    import pandas as pd

    if not state_ids:
        return pd.DataFrame(index=bars.index)

    parts = []
    for sid in state_ids:
        var = get_state_var(sid)
        block = var.compute(bars)
        if isinstance(block, pd.Series):
            block = block.to_frame(name=sid)
        parts.append(block)
    return pd.concat(parts, axis=1)
