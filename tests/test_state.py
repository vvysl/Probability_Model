"""v1 + alphabet wave-1 continuous state catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pm.bars import make_synthetic_ohlcv
from pm.pipeline import describe_readiness, load_config, run_scaffold
from pm.state.catalog import DEFAULT_STATE_IDS, V1_STATE_IDS
from pm.state.registry import build_state_matrix, get_state_var, list_state_ids
from pm.vol import atr


def test_default_catalog_registered():
    assert set(DEFAULT_STATE_IDS).issubset(set(list_state_ids()))
    for sid in DEFAULT_STATE_IDS:
        assert get_state_var(sid).id == sid


def test_build_full_default_matrix():
    bars = make_synthetic_ohlcv(n=400, seed=11)
    X = build_state_matrix(bars, list(DEFAULT_STATE_IDS))
    assert list(X.columns) == list(DEFAULT_STATE_IDS)
    assert len(X) == len(bars)
    # After atr_240 warmup, core v1 cols finite; wave1 has own warmups
    tail = X[list(V1_STATE_IDS)].iloc[250:]
    assert tail.notna().all().all()


def test_atr_state_matches_vol_helper():
    bars = make_synthetic_ohlcv(n=300, seed=12)
    s = get_state_var("atr_60").compute(bars)
    pd.testing.assert_series_equal(s, atr(bars, 60).rename("atr_60"))


def test_ret_atr_causal_prefix():
    bars = make_synthetic_ohlcv(n=200, seed=13)
    full = get_state_var("ret_atr_5").compute(bars)
    mid = 120
    prefix = get_state_var("ret_atr_5").compute(bars.iloc[:mid])
    pd.testing.assert_series_equal(prefix, full.iloc[:mid], check_names=False)


def test_efficiency_bounds():
    bars = make_synthetic_ohlcv(n=200, seed=15)
    eff = get_state_var("eff_30").compute(bars).dropna()
    assert (eff >= 0).all() and (eff <= 1.0 + 1e-9).all()


def test_close_pos_flat_bar():
    bars = make_synthetic_ohlcv(n=50, seed=16)
    i = 40
    px = float(bars["close"].iloc[i])
    for col in ("open", "high", "low", "close"):
        bars.iloc[i, bars.columns.get_loc(col)] = px
    pos = get_state_var("close_pos").compute(bars)
    assert pos.iloc[i] == pytest.approx(0.5)


def test_pipeline_default_states():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    status = describe_readiness(cfg)
    assert status["state_ids"] == list(DEFAULT_STATE_IDS)
    assert status["ready_for_train"] is True

    bars = make_synthetic_ohlcv(n=400, seed=14)
    out = run_scaffold(cfg, bars=bars)
    assert list(out["X"].columns) == list(DEFAULT_STATE_IDS)
    assert out["y"] is not None
