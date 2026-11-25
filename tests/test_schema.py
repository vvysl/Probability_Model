"""Synthetic OHLCV schema + scaffold readiness smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pm.bars import REQUIRED_COLUMNS, load_ohlcv, make_synthetic_ohlcv, normalize_ohlcv
from pm.eval.splits import time_holdout_cut
from pm.pipeline import describe_readiness, load_config, run_scaffold
from pm.state.registry import build_state_matrix, list_state_ids
from pm.targets.registry import list_target_ids


def test_synthetic_ohlcv_schema():
    bars = make_synthetic_ohlcv(n=32, seed=1)
    assert list(bars.columns) == list(REQUIRED_COLUMNS)
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert bars.index.tz is not None
    assert bars.index.is_monotonic_increasing
    assert not bars.index.has_duplicates
    assert bars["high"].ge(bars[["open", "close"]].max(axis=1)).all()
    assert bars["low"].le(bars[["open", "close"]].min(axis=1)).all()


def test_normalize_rejects_missing_columns():
    raw = make_synthetic_ohlcv(n=8).reset_index().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing OHLCV"):
        normalize_ohlcv(raw)


def test_load_ohlcv_roundtrip(tmp_path: Path):
    bars = make_synthetic_ohlcv(n=16, seed=2)
    path = tmp_path / "sample.csv"
    bars.reset_index().to_csv(path, index=False)
    loaded = load_ohlcv(path)
    # CSV round-trip drops DatetimeIndex.freq; compare values only.
    pd.testing.assert_frame_equal(
        loaded, bars, check_freq=False, check_exact=False, rtol=1e-12
    )


def test_registries_empty_in_step1():
    assert list_state_ids() == []
    assert list_target_ids() == []


def test_build_state_matrix_empty_ids():
    bars = make_synthetic_ohlcv(n=10)
    X = build_state_matrix(bars, [])
    assert X.empty
    assert len(X) == len(bars)


def test_time_holdout_cut():
    assert time_holdout_cut(100, 0.2) == 80
    with pytest.raises(ValueError):
        time_holdout_cut(100, 0.0)


def test_default_config_empty_horizons_and_state():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    status = describe_readiness(cfg)
    assert status["horizons"] == []
    assert status["state_ids"] == []
    assert status["ready_for_train"] is False


def test_run_scaffold_without_train():
    cfg = {
        "horizons": [],
        "state_ids": [],
        "target_id": None,
        "holdout_frac": 0.2,
    }
    bars = make_synthetic_ohlcv(n=50)
    out = run_scaffold(cfg, bars=bars)
    assert out["status"]["ready_for_train"] is False
    assert out["holdout_cut"] == 40
    assert out["X"] is not None and out["X"].empty
    assert out["y"] is None
