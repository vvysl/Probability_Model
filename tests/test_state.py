"""v1 continuous state catalog (blocks A–E)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pm.bars import make_synthetic_ohlcv
from pm.pipeline import describe_readiness, load_config, run_scaffold
from pm.state.catalog import V1_STATE_IDS
from pm.state.registry import build_state_matrix, get_state_var, list_state_ids
from pm.vol import atr


def test_v1_catalog_registered():
    assert len(V1_STATE_IDS) == 30
    assert set(V1_STATE_IDS).issubset(set(list_state_ids()))
    for sid in V1_STATE_IDS:
        assert get_state_var(sid).id == sid


def test_build_full_v1_matrix():
    bars = make_synthetic_ohlcv(n=400, seed=11)
    X = build_state_matrix(bars, list(V1_STATE_IDS))
    assert list(X.columns) == list(V1_STATE_IDS)
    assert len(X) == len(bars)
    # After atr_240 warmup, most rows should be finite for all columns
    tail = X.iloc[250:]
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
    # Force a zero-range bar
    i = 40
    px = float(bars["close"].iloc[i])
    for col in ("open", "high", "low", "close"):
        bars.iloc[i, bars.columns.get_loc(col)] = px
    pos = get_state_var("close_pos").compute(bars)
    assert pos.iloc[i] == pytest.approx(0.5)


def test_atr_ratio_expansion():
    bars = make_synthetic_ohlcv(n=300, seed=17)
    ratio = get_state_var("atr_ratio_14_60").compute(bars)
    a14 = get_state_var("atr_14").compute(bars)
    a60 = get_state_var("atr_60").compute(bars)
    expected = a14 / a60.where(a60 > 0)
    pd.testing.assert_series_equal(ratio, expected.rename("atr_ratio_14_60"))


def test_dir_consist_range():
    bars = make_synthetic_ohlcv(n=100, seed=18)
    d = get_state_var("dir_consist_15").compute(bars).dropna()
    assert (d >= 0).all() and (d <= 1).all()


def test_no_raw_volume_in_v1():
    assert "volume" not in V1_STATE_IDS
    assert all(not sid.startswith("volume") for sid in V1_STATE_IDS)


def test_pipeline_default_v1():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    status = describe_readiness(cfg)
    assert status["state_ids"] == list(V1_STATE_IDS)
    assert status["ready_for_train"] is True

    bars = make_synthetic_ohlcv(n=400, seed=14)
    out = run_scaffold(cfg, bars=bars)
    assert list(out["X"].columns) == list(V1_STATE_IDS)
    assert out["y"] is not None
