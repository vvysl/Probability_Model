"""
Train one HGB ternary model per horizon (v1).

LGBM backend deferred to a server environment (libomp / heavier deps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pm.bars import load_ohlcv, normalize_ohlcv
from pm.eval.metrics import ternary_metrics
from pm.eval.splits import time_holdout_indices
from pm.model.hgb import DEFAULT_HGB_PARAMS, HgbTernaryPredictor
from pm.pipeline import describe_readiness, load_config
from pm.state.registry import build_state_matrix
from pm.targets.registry import get_target
from pm.targets.ternary_displacement import TernaryDisplacementTarget


def _resolve_target(cfg: dict[str, Any]) -> TernaryDisplacementTarget:
    target_id = cfg.get("target_id") or "ternary_displacement"
    base = get_target(target_id)
    horizons = tuple(cfg.get("horizons") or list(base.horizons))
    tcfg = cfg.get("target") or {}
    return TernaryDisplacementTarget(
        horizons=horizons,
        atr_period=int(tcfg.get("atr_period", 14)),
        flat_tau=float(tcfg.get("flat_tau", 0.15)),
        target_id=str(target_id),
    )


def build_xy(
    bars: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_ids = list(cfg.get("state_ids") or [])
    X = build_state_matrix(bars, state_ids)
    y = _resolve_target(cfg).compute(bars)
    return X, y


def _aligned_mask(X: pd.DataFrame, y_col: pd.Series) -> np.ndarray:
    return X.notna().all(axis=1).to_numpy() & y_col.notna().to_numpy()


def train_horizon(
    X: pd.DataFrame,
    y_col: pd.Series,
    *,
    holdout_frac: float,
    purge: int,
    hgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one horizon; return model + holdout metrics + split info."""
    mask = _aligned_mask(X, y_col)
    idx = np.flatnonzero(mask)
    if len(idx) < 50:
        raise ValueError(f"too few aligned rows: {len(idx)}")

    Xv = X.to_numpy(dtype=float)[idx]
    yv = y_col.to_numpy(dtype=float)[idx].astype(int)

    train_rel, hold_rel, cut_rel = time_holdout_indices(
        len(idx), holdout_frac=holdout_frac, purge=purge
    )
    # Map relative positions back to original bar indices for reporting
    train_idx = idx[train_rel]
    hold_idx = idx[hold_rel]

    model = HgbTernaryPredictor(**(hgb_params or {}))
    model.fit(Xv[train_rel], yv[train_rel])
    proba = model.predict_proba(Xv[hold_rel])
    metrics = ternary_metrics(yv[hold_rel], proba)

    return {
        "model": model,
        "metrics": metrics,
        "n_aligned": int(len(idx)),
        "n_train": int(len(train_rel)),
        "n_holdout": int(len(hold_rel)),
        "purge": int(purge),
        "holdout_cut_aligned": int(cut_rel),
        "train_bar_idx_range": [int(train_idx[0]), int(train_idx[-1])],
        "holdout_bar_idx_range": [int(hold_idx[0]), int(hold_idx[-1])],
    }


def run_train(
    cfg: dict[str, Any],
    bars: pd.DataFrame | None = None,
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Train one HGB model per horizon; optionally write artifacts.

    ``artifacts_dir`` gets ``metrics.json`` and ``model_h{H}.joblib``.
    """
    status = describe_readiness(cfg)
    if not status["ready_for_train"]:
        raise RuntimeError(f"config not ready: {status}")

    if bars is None:
        path = cfg.get("data_path")
        if not path:
            raise ValueError("bars not provided and config.data_path is empty")
        bars = load_ohlcv(path)
    else:
        bars = normalize_ohlcv(bars)

    X, y = build_xy(bars, cfg)
    horizons = [int(h) for h in (cfg.get("horizons") or [])]
    if not horizons:
        raise ValueError("config.horizons is empty")
    purge = int(cfg.get("purge", max(horizons)))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))
    hgb_params = {**DEFAULT_HGB_PARAMS, **(cfg.get("model") or {})}
    # Force balanced for v1 unless explicitly overridden
    hgb_params.setdefault("class_weight", "balanced")
    # Strip non-sklearn keys if any
    backend = hgb_params.pop("backend", "hgb")
    if backend != "hgb":
        raise ValueError(
            f"backend={backend!r} not supported yet; use 'hgb' "
            "(lgbm deferred to server)"
        )

    results: dict[str, Any] = {
        "status": status,
        "backend": "hgb",
        "symbol": cfg.get("symbol"),
        "purge": purge,
        "holdout_frac": holdout_frac,
        "horizons": {},
    }

    art = Path(artifacts_dir) if artifacts_dir else None
    if art:
        art.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        col = f"y_h{h}"
        if col not in y.columns:
            raise KeyError(f"missing target column {col}")
        packed = train_horizon(
            X,
            y[col],
            holdout_frac=holdout_frac,
            purge=purge,
            hgb_params=hgb_params,
        )
        model = packed.pop("model")
        results["horizons"][str(h)] = packed
        if art:
            joblib.dump(model, art / f"model_h{h}.joblib")

    if art:
        meta = {k: v for k, v in results.items()}
        (art / "metrics.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    return results


def train_from_config(
    config_path: Path | str,
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if artifacts_dir is None:
        artifacts_dir = cfg.get("artifacts_dir") or "artifacts/run"
    return run_train(cfg, artifacts_dir=artifacts_dir)
