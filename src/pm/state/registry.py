"""State variable registry — v1 catalog blocks A–E."""

from __future__ import annotations

from pm.state.base import StateVar
from pm.state.candle import default_candle_vars
from pm.state.catalog import V1_STATE_IDS
from pm.state.returns import default_return_vars
from pm.state.trend import default_trend_vars
from pm.state.volatility import default_volatility_vars
from pm.state.volume_feats import default_volume_vars

STATE_REGISTRY: dict[str, StateVar] = {}


def _register_defaults() -> None:
    for factory in (
        default_return_vars,
        default_volatility_vars,
        default_candle_vars,
        default_trend_vars,
        default_volume_vars,
    ):
        for var in factory():
            if var.id in STATE_REGISTRY:
                raise ValueError(f"duplicate state id {var.id!r}")
            STATE_REGISTRY[var.id] = var
    missing = [sid for sid in V1_STATE_IDS if sid not in STATE_REGISTRY]
    if missing:
        raise RuntimeError(f"v1 catalog ids not registered: {missing}")


_register_defaults()


def list_state_ids() -> list[str]:
    return sorted(STATE_REGISTRY.keys())


def get_state_var(state_id: str) -> StateVar:
    if state_id not in STATE_REGISTRY:
        raise KeyError(
            f"unknown state id {state_id!r} (known={list_state_ids()})"
        )
    return STATE_REGISTRY[state_id]


def build_state_matrix(bars, state_ids: list[str]):
    """Concatenate registered state vars into one frame."""
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
