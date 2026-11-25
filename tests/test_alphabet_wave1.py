"""Alphabet wave-1 state features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pm.bars import make_synthetic_ohlcv
from pm.pipeline import load_config
from pm.state.alphabet_wave1 import WAVE1_STATE_IDS, compute_alphabet_wave1
from pm.state.catalog import DEFAULT_STATE_IDS, V1_STATE_IDS
from pm.state.registry import build_state_matrix, list_state_ids


def test_wave1_ids_registered():
    assert set(WAVE1_STATE_IDS).issubset(set(list_state_ids()))
    assert len(DEFAULT_STATE_IDS) == 43
    assert DEFAULT_STATE_IDS[:30] == V1_STATE_IDS


def test_wave1_compute_shape_and_causal():
    bars = make_synthetic_ohlcv(n=400, seed=31)
    # Spread across a few hours so session segments exist
    X = compute_alphabet_wave1(bars)
    assert list(X.columns) == list(WAVE1_STATE_IDS)
    assert len(X) == len(bars)
    # Warmup: gk needs W_YZ=48
    assert X["gk_rv_ratio"].iloc[:47].isna().all()
    assert X["gk_rv_ratio"].iloc[47:].notna().any()
    # session_range_pos in [0,1]
    pos = X["session_range_pos"].dropna()
    assert (pos >= 0).all() and (pos <= 1).all()


def test_wave1_no_lookahead_prefix():
    bars = make_synthetic_ohlcv(n=300, seed=32)
    full = compute_alphabet_wave1(bars)
    mid = 180
    prefix = compute_alphabet_wave1(bars.iloc[:mid])
    # Session hist for vol_frac can differ slightly at segment edges when
    # truncated; compare non-session-hist columns strictly.
    cols = [
        "gk_rv_ratio",
        "range_compression",
        "ret_skew",
        "ret_accel",
        "sma_open_approach",
        "vol_pct",
    ]
    pd.testing.assert_frame_equal(
        prefix[cols],
        full[cols].iloc[:mid],
        check_names=True,
        rtol=1e-10,
        atol=1e-10,
    )


def test_build_matrix_includes_wave1_once():
    bars = make_synthetic_ohlcv(n=250, seed=33)
    ids = list(V1_STATE_IDS[:3]) + list(WAVE1_STATE_IDS[:2])
    X = build_state_matrix(bars, ids)
    assert list(X.columns) == ids


def test_default_config_has_43_states():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    assert cfg["state_ids"] == list(DEFAULT_STATE_IDS)
