"""ATR barrier targets + train (first = primary, touch = legacy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pm.bars import make_synthetic_ohlcv
from pm.eval.binary_metrics import binary_metrics
from pm.model.binary import HgbBinaryPredictor
from pm.pipeline import load_config
from pm.state.catalog import DEFAULT_STATE_IDS
from pm.targets.atr_barrier import (
    LABEL_DOWN_FIRST,
    LABEL_NONE,
    LABEL_UP_FIRST,
    AtrBarrierFirstTarget,
    AtrBarrierTouchTarget,
)
from pm.targets.registry import list_target_ids
from pm.train import run_train


def test_targets_registered():
    ids = list_target_ids()
    assert "atr_barrier_first" in ids
    assert "atr_barrier_touch" in ids


def test_first_barrier_labels():
    bars = make_synthetic_ohlcv(n=200, seed=41)
    tgt = AtrBarrierFirstTarget(horizon=30, atr_period=14, barrier_tau=2.0)
    y = tgt.compute(bars)
    assert list(y.columns) == ["first_h30"]
    assert y["first_h30"].iloc[-30:].isna().all()
    valid = y["first_h30"].dropna()
    assert set(valid.unique()).issubset(
        {LABEL_DOWN_FIRST, LABEL_NONE, LABEL_UP_FIRST}
    )


def test_touch_still_works():
    bars = make_synthetic_ohlcv(n=200, seed=41)
    tgt = AtrBarrierTouchTarget(horizon=30, barrier_tau=2.0)
    y = tgt.compute(bars)
    assert list(y.columns) == ["hit_up_h30", "hit_down_h30"]


def test_first_no_lookahead():
    bars = make_synthetic_ohlcv(n=120, seed=42)
    tgt = AtrBarrierFirstTarget(horizon=10, barrier_tau=0.5)
    y0 = tgt.compute(bars)
    t = 40
    bars2 = bars.copy()
    j = t + 10 + 5
    bars2.iloc[j, bars2.columns.get_loc("high")] += 50.0
    y1 = tgt.compute(bars2)
    assert y0["first_h10"].iloc[t] == y1["first_h10"].iloc[t]


def test_binary_predictor_and_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + rng.normal(scale=0.5, size=300) > 0).astype(int)
    model = HgbBinaryPredictor(max_iter=50, min_samples_leaf=10)
    model.fit(X, y)
    p = model.predict_proba(X[:20])
    assert p.shape == (20,)
    m = binary_metrics(y[:20], p)
    assert m["log_loss"] > 0


def test_run_train_first_synthetic(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg["state_ids"] = list(DEFAULT_STATE_IDS)
    cfg["data_path"] = None
    cfg["model"] = {**(cfg.get("model") or {}), "max_iter": 40, "min_samples_leaf": 10}
    bars = make_synthetic_ohlcv(n=2500, seed=43)
    art = tmp_path / "run"
    result = run_train(cfg, bars=bars, artifacts_dir=art)
    assert result["target_id"] == "atr_barrier_first"
    assert result["horizon"] == 30
    assert result["barrier_tau"] == 2.0
    assert result["head"]["n_train"] > 50
    assert (art / "model_first_h30.joblib").exists()
    assert "first_side_hit_rate" in result["head"]["metrics"]
