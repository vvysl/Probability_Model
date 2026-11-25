"""HGB ternary train + metrics on synthetic bars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pm.bars import make_synthetic_ohlcv
from pm.eval.metrics import ternary_metrics
from pm.eval.splits import time_holdout_indices
from pm.model.hgb import HgbTernaryPredictor
from pm.model.labels import align_proba, encode_ternary
from pm.pipeline import load_config
from pm.state.catalog import V1_STATE_IDS
from pm.train import run_train, train_horizon


def test_encode_align_proba():
    y = np.array([-1, 0, 1, -1])
    enc = encode_ternary(y)
    assert list(enc) == [0, 1, 2, 0]
    # Fake clf output with only classes 0 and 2 present
    raw = np.array([[0.7, 0.3], [0.2, 0.8]])
    aligned = align_proba(raw, np.array([0, 2]))
    assert aligned.shape == (2, 3)
    assert aligned[0, 0] == pytest.approx(0.7)
    assert aligned[0, 1] == pytest.approx(0.0)
    assert aligned[0, 2] == pytest.approx(0.3)


def test_hgb_fit_predict_proba_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = rng.choice([-1, 0, 1], size=200, p=[0.3, 0.4, 0.3])
    model = HgbTernaryPredictor(max_iter=50, min_samples_leaf=10)
    model.fit(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = model.predict(X[:10])
    assert set(pred).issubset({-1, 0, 1})


def test_ternary_metrics_smoke():
    from pm.eval.metrics import flat_micro_direction_metrics

    y = np.array([-1, 0, 1, 1, 0, -1])
    proba = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.2, 0.6],
            [0.3, 0.5, 0.2],
            [0.5, 0.3, 0.2],
        ]
    )
    m = ternary_metrics(y, proba)
    assert m["n"] == 6
    assert m["log_loss"] > 0
    assert "brier_mean" in m

    # Flat rows: index 1 z>0 & favor flat-ish but up>down → +; index 4 z<0 & down>up → +
    z = np.array([-1.0, 0.2, 1.0, 1.0, -0.1, -1.0])
    proba_tilt = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.3, 0.5],  # favor up, z>0 → +
            [0.1, 0.2, 0.7],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],  # favor down, z<0 → +
            [0.5, 0.3, 0.2],
        ]
    )
    micro = flat_micro_direction_metrics(y, z, proba_tilt)
    assert micro["n_flat_scored"] == 2
    assert micro["hit_rate"] == pytest.approx(1.0)


def test_purge_gap():
    train_idx, hold_idx, cut = time_holdout_indices(100, holdout_frac=0.2, purge=30)
    assert cut == 80
    assert train_idx[-1] == 49  # 80 - 30 - 1
    assert hold_idx[0] == 80
    assert set(train_idx).isdisjoint(set(hold_idx))


def test_run_train_synthetic(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg["state_ids"] = list(V1_STATE_IDS)
    cfg["data_path"] = None
    bars = make_synthetic_ohlcv(n=600, seed=21)
    art = tmp_path / "run"
    result = run_train(cfg, bars=bars, artifacts_dir=art)
    assert result["backend"] == "hgb"
    assert set(result["horizons"]) == {"3", "10", "30"}
    for h, pack in result["horizons"].items():
        assert pack["n_train"] > 50
        assert pack["metrics"]["log_loss"] > 0
        assert (art / f"model_h{h}.joblib").exists()
    assert (art / "metrics.json").exists()


def test_rejects_lgbm_backend():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg["model"] = {"backend": "lgbm"}
    bars = make_synthetic_ohlcv(n=400, seed=22)
    with pytest.raises(ValueError, match="lgbm deferred"):
        run_train(cfg, bars=bars)
