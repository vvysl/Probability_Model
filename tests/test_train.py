"""HGB train helpers (barrier primary + legacy ternary + rejects)."""

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
from pm.state.catalog import DEFAULT_STATE_IDS
from pm.train import run_train


def test_encode_align_proba():
    y = np.array([-1, 0, 1, -1])
    enc = encode_ternary(y)
    assert list(enc) == [0, 1, 2, 0]
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


def test_ternary_metrics_smoke():
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


def test_purge_gap():
    train_idx, hold_idx, cut = time_holdout_indices(100, holdout_frac=0.2, purge=30)
    assert cut == 80
    assert train_idx[-1] == 49
    assert hold_idx[0] == 80


def test_rejects_lgbm_backend():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg["model"] = {"backend": "lgbm"}
    bars = make_synthetic_ohlcv(n=400, seed=22)
    with pytest.raises(ValueError, match="lgbm deferred"):
        run_train(cfg, bars=bars)


def test_legacy_ternary_still_trainable(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg["target_id"] = "ternary_displacement"
    cfg["horizons"] = [3, 10, 30]
    cfg["target"] = {"atr_period": 14, "flat_tau": {3: 0.75, 10: 1.25, 30: 2.0}}
    cfg["state_ids"] = list(DEFAULT_STATE_IDS)
    cfg["model"] = {**(cfg.get("model") or {}), "max_iter": 30, "min_samples_leaf": 10}
    bars = make_synthetic_ohlcv(n=2500, seed=21)
    result = run_train(cfg, bars=bars, artifacts_dir=tmp_path / "ternary")
    assert result["target_id"] == "ternary_displacement"
    assert set(result["horizons"]) == {"3", "10", "30"}
