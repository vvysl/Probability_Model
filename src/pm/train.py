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
from pm.eval.metrics import flat_micro_direction_metrics, ternary_metrics
from pm.eval.splits import time_holdout_indices
from pm.model.hgb import DEFAULT_HGB_PARAMS, HgbTernaryPredictor
from pm.pipeline import describe_readiness, load_config
from pm.state.registry import build_state_matrix
from pm.targets.registry import get_target
from pm.targets.ternary_displacement import (
    DEFAULT_FLAT_TAU_BY_H,
    TernaryDisplacementTarget,
)


def _resolve_target(cfg: dict[str, Any]) -> TernaryDisplacementTarget:
    target_id = cfg.get("target_id") or "ternary_displacement"
    base = get_target(target_id)
    horizons = tuple(cfg.get("horizons") or list(base.horizons))
    tcfg = cfg.get("target") or {}
    flat_tau = tcfg.get("flat_tau", DEFAULT_FLAT_TAU_BY_H)
    return TernaryDisplacementTarget(
        horizons=horizons,
        atr_period=int(tcfg.get("atr_period", 14)),
        flat_tau=flat_tau,
        target_id=str(target_id),
    )


def build_xy(
    bars: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TernaryDisplacementTarget]:
    state_ids = list(cfg.get("state_ids") or [])
    X = build_state_matrix(bars, state_ids)
    target = _resolve_target(cfg)
    y = target.compute(bars)
    z = target.compute_z(bars)
    return X, y, z, target


def _aligned_mask(X: pd.DataFrame, y_col: pd.Series) -> np.ndarray:
    return X.notna().all(axis=1).to_numpy() & y_col.notna().to_numpy()


def train_horizon(
    X: pd.DataFrame,
    y_col: pd.Series,
    z_col: pd.Series,
    *,
    holdout_frac: float,
    purge: int,
    hgb_params: dict[str, Any] | None = None,
    tau: float | None = None,
) -> dict[str, Any]:
    """Fit one horizon; return model + holdout metrics + split info."""
    mask = _aligned_mask(X, y_col)
    idx = np.flatnonzero(mask)
    if len(idx) < 50:
        raise ValueError(f"too few aligned rows: {len(idx)}")

    Xv = X.to_numpy(dtype=float)[idx]
    yv = y_col.to_numpy(dtype=float)[idx].astype(int)
    zv = z_col.to_numpy(dtype=float)[idx]

    train_rel, hold_rel, cut_rel = time_holdout_indices(
        len(idx), holdout_frac=holdout_frac, purge=purge
    )
    train_idx = idx[train_rel]
    hold_idx = idx[hold_rel]

    model = HgbTernaryPredictor(**(hgb_params or {}))
    model.fit(Xv[train_rel], yv[train_rel])
    proba = model.predict_proba(Xv[hold_rel])
    metrics = ternary_metrics(yv[hold_rel], proba)
    metrics["flat_micro_direction"] = flat_micro_direction_metrics(
        yv[hold_rel], zv[hold_rel], proba
    )
    if tau is not None:
        metrics["flat_tau"] = float(tau)

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

    X, y, z, target = build_xy(bars, cfg)
    horizons = [int(h) for h in (cfg.get("horizons") or [])]
    if not horizons:
        raise ValueError("config.horizons is empty")
    purge = int(cfg.get("purge", max(horizons)))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))
    hgb_params = {**DEFAULT_HGB_PARAMS, **(cfg.get("model") or {})}
    hgb_params.setdefault("class_weight", "balanced")
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
        "flat_tau_by_h": {str(h): target.tau(h) for h in horizons},
        "horizons": {},
    }

    art = Path(artifacts_dir) if artifacts_dir else None
    if art:
        art.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        col = f"y_h{h}"
        zcol = f"z_h{h}"
        if col not in y.columns:
            raise KeyError(f"missing target column {col}")
        if zcol not in z.columns:
            raise KeyError(f"missing z column {zcol}")
        packed = train_horizon(
            X,
            y[col],
            z[zcol],
            holdout_frac=holdout_frac,
            purge=purge,
            hgb_params=hgb_params,
            tau=target.tau(h),
        )
        model = packed.pop("model")
        results["horizons"][str(h)] = packed
        if art:
            joblib.dump(model, art / f"model_h{h}.joblib")

    if art:
        (art / "metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
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
