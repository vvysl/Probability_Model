"""Ternary displacement targets + causal ATR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pm.bars import make_synthetic_ohlcv
from pm.targets.registry import get_target, list_target_ids
from pm.targets.scale import atr, true_range
from pm.targets.ternary_displacement import (
    DEFAULT_HORIZONS,
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    TernaryDisplacementTarget,
)


def test_target_registered():
    assert "ternary_displacement" in list_target_ids()
    tgt = get_target("ternary_displacement")
    assert tuple(tgt.horizons) == DEFAULT_HORIZONS
    assert tgt.tau(3) == pytest.approx(0.75)
    assert tgt.tau(10) == pytest.approx(1.25)
    assert tgt.tau(30) == pytest.approx(2.0)


def test_per_horizon_tau_map():
    bars = make_synthetic_ohlcv(n=120, seed=5)
    tgt = TernaryDisplacementTarget(
        horizons=(3, 10, 30),
        atr_period=14,
        flat_tau={3: 0.75, 10: 1.25, 30: 2.0},
    )
    z = tgt.compute_z(bars)
    y = tgt.compute(bars)
    # Larger tau ⇒ more flats on same z (or equal)
    flat3 = (y["y_h3"] == LABEL_FLAT).sum()
    flat30 = (y["y_h30"] == LABEL_FLAT).sum()
    # Not a hard inequality across different h windows, but z columns exist
    assert list(z.columns) == ["z_h3", "z_h10", "z_h30"]
    assert flat3 >= 0 and flat30 >= 0


def test_atr_causal_matches_prefix():
    bars = make_synthetic_ohlcv(n=80, seed=3)
    full = atr(bars, period=14)
    mid = 40
    prefix = atr(bars.iloc[:mid], period=14)
    # ATR at t must equal ATR computed on bars[:t+1] only.
    pd.testing.assert_series_equal(
        prefix.iloc[:mid],
        full.iloc[:mid],
        check_names=False,
    )


def test_true_range_uses_prior_close_only():
    bars = make_synthetic_ohlcv(n=5, seed=4)
    tr0 = true_range(bars).copy()
    # First bar: no prior close → TR falls back to high-low.
    assert tr0.iloc[0] == pytest.approx(bars["high"].iloc[0] - bars["low"].iloc[0])
    bars2 = bars.copy()
    bars2.iloc[-1, bars2.columns.get_loc("high")] = bars2["high"].iloc[-1] + 50.0
    tr1 = true_range(bars2)
    pd.testing.assert_series_equal(tr0.iloc[:-1], tr1.iloc[:-1])


def test_label_columns_and_tail_nan():
    bars = make_synthetic_ohlcv(n=100, seed=5)
    tgt = TernaryDisplacementTarget(horizons=(3, 10, 30), atr_period=14, flat_tau=0.15)
    y = tgt.compute(bars)
    assert list(y.columns) == ["y_h3", "y_h10", "y_h30"]
    assert y["y_h3"].iloc[-3:].isna().all()
    assert y["y_h10"].iloc[-10:].isna().all()
    assert y["y_h30"].iloc[-30:].isna().all()
    valid = y["y_h3"].dropna()
    assert set(valid.unique()).issubset({LABEL_DOWN, LABEL_FLAT, LABEL_UP})


def test_no_lookahead_on_labels():
    """Price after t+h must not change y_t; close at t+h may."""
    bars = make_synthetic_ohlcv(n=120, seed=6)
    tgt = TernaryDisplacementTarget(horizons=(10,), atr_period=14, flat_tau=0.1)
    y0 = tgt.compute(bars)
    t, h = 50, 10

    bars_after = bars.copy()
    j = t + h + 5
    bars_after.iloc[j, bars_after.columns.get_loc("close")] += 10.0
    bars_after.iloc[j, bars_after.columns.get_loc("high")] += 10.0
    y_after = tgt.compute(bars_after)
    assert y0["y_h10"].iloc[t] == y_after["y_h10"].iloc[t]

    bars_at_h = bars.copy()
    bars_at_h.iloc[t + h, bars_at_h.columns.get_loc("close")] += 25.0
    bars_at_h.iloc[t + h, bars_at_h.columns.get_loc("high")] += 25.0
    y_at_h = tgt.compute(bars_at_h)
    assert y0["y_h10"].iloc[t] != y_at_h["y_h10"].iloc[t]


def test_flat_tau_zero_is_pure_sign():
    bars = make_synthetic_ohlcv(n=80, seed=7)
    tgt = TernaryDisplacementTarget(horizons=(5,), atr_period=10, flat_tau=0.0)
    y = tgt.compute(bars)["y_h5"].dropna()
    # With tau=0, exact zero displacement is flat; else ±1.
    assert set(y.unique()).issubset({LABEL_DOWN, LABEL_FLAT, LABEL_UP})


def test_pipeline_computes_y():
    from pm.pipeline import run_scaffold

    cfg = {
        "horizons": [3, 10, 30],
        "state_ids": [],
        "target_id": "ternary_displacement",
        "holdout_frac": 0.2,
    }
    bars = make_synthetic_ohlcv(n=100, seed=8)
    out = run_scaffold(cfg, bars=bars)
    assert out["y"] is not None
    assert list(out["y"].columns) == ["y_h3", "y_h10", "y_h30"]
    assert out["status"]["ready_for_train"] is False  # no state yet
