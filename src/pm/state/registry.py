"""State variable registry — v1 + alphabet wave-1."""

from __future__ import annotations

from pm.state.alphabet_wave1 import (
    WAVE1_STATE_IDS,
    compute_alphabet_wave1,
    default_alphabet_wave1_vars,
)
from pm.state.base import StateVar
from pm.state.candle import default_candle_vars
from pm.state.catalog import DEFAULT_STATE_IDS, V1_STATE_IDS
from pm.state.returns import default_return_vars
from pm.state.trend import default_trend_vars
from pm.state.volatility import default_volatility_vars
from pm.state.volume_feats import default_volume_vars

STATE_REGISTRY: dict[str, StateVar] = {}
_WAVE1_SET = set(WAVE1_STATE_IDS)


def _register_defaults() -> None:
    for factory in (
        default_return_vars,
        default_volatility_vars,
        default_candle_vars,
        default_trend_vars,
        default_volume_vars,
        default_alphabet_wave1_vars,
    ):
        for var in factory():
            if var.id in STATE_REGISTRY:
                raise ValueError(f"duplicate state id {var.id!r}")
            STATE_REGISTRY[var.id] = var
    missing = [sid for sid in DEFAULT_STATE_IDS if sid not in STATE_REGISTRY]
    if missing:
        raise RuntimeError(f"default catalog ids not registered: {missing}")


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
    """Concatenate registered state vars; wave-1 computed once as a batch."""
    import pandas as pd

    if not state_ids:
        return pd.DataFrame(index=bars.index)

    wave1_cols = [s for s in state_ids if s in _WAVE1_SET]
    other = [s for s in state_ids if s not in _WAVE1_SET]
    wave1_frame = compute_alphabet_wave1(bars) if wave1_cols else None

    parts_by_id: dict[str, pd.Series] = {}
    for sid in other:
        var = get_state_var(sid)
        block = var.compute(bars)
        if isinstance(block, pd.DataFrame):
            for col in block.columns:
                parts_by_id[str(col)] = block[col]
        else:
            parts_by_id[sid] = block.rename(sid)
    if wave1_frame is not None:
        for sid in wave1_cols:
            parts_by_id[sid] = wave1_frame[sid]

    ordered = [parts_by_id[sid] for sid in state_ids]
    return pd.concat(ordered, axis=1)
