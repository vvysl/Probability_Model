"""
Train HGB probability models.

Primary: ATR barrier first — P(up first), P(down first), P(none) within n.
Legacy: independent touch heads; ternary displacement.
LGBM deferred to server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pm.bars import load_ohlcv, normalize_ohlcv
from pm.eval.binary_metrics import binary_metrics
from pm.eval.metrics import flat_micro_direction_metrics, ternary_metrics
from pm.eval.splits import time_holdout_indices
from pm.model.binary import HgbBinaryPredictor
from pm.model.hgb import DEFAULT_HGB_PARAMS, HgbTernaryPredictor
from pm.pipeline import describe_readiness, load_config
from pm.state.registry import build_state_matrix
from pm.targets.atr_barrier import AtrBarrierFirstTarget, AtrBarrierTouchTarget
from pm.targets.registry import get_target
from pm.targets.ternary_displacement import (
    DEFAULT_FLAT_TAU_BY_H,
    TernaryDisplacementTarget,
)


def _aligned_mask(X: pd.DataFrame, y_col: pd.Series) -> np.ndarray:
    return X.notna().all(axis=1).to_numpy() & y_col.notna().to_numpy()


def _hgb_params(cfg: dict[str, Any]) -> dict[str, Any]:
    hgb_params = {**DEFAULT_HGB_PARAMS, **(cfg.get("model") or {})}
    hgb_params.setdefault("class_weight", "balanced")
    backend = hgb_params.pop("backend", "hgb")
    if backend != "hgb":
        raise ValueError(
            f"backend={backend!r} not supported yet; use 'hgb' "
            "(lgbm deferred to server)"
        )
    return hgb_params


def _barrier_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    tcfg = cfg.get("target") or {}
    horizons = cfg.get("horizons") or [30]
    return {
        "horizon": int(tcfg.get("horizon") or horizons[0]),
        "atr_period": int(tcfg.get("atr_period", 14)),
        "barrier_tau": float(tcfg.get("barrier_tau", 2.0)),
    }


def _resolve_barrier_touch(cfg: dict[str, Any]) -> AtrBarrierTouchTarget:
    return AtrBarrierTouchTarget(
        **_barrier_kwargs(cfg), target_id="atr_barrier_touch"
    )


def _resolve_barrier_first(cfg: dict[str, Any]) -> AtrBarrierFirstTarget:
    return AtrBarrierFirstTarget(
        **_barrier_kwargs(cfg), target_id="atr_barrier_first"
    )


def _resolve_ternary(cfg: dict[str, Any]) -> TernaryDisplacementTarget:
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


def train_binary_head(
    X: pd.DataFrame,
    y_col: pd.Series,
    *,
    holdout_frac: float,
    purge: int,
    hgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mask = _aligned_mask(X, y_col)
    idx = np.flatnonzero(mask)
    if len(idx) < 50:
        raise ValueError(f"too few aligned rows: {len(idx)}")

    Xv = X.to_numpy(dtype=float)[idx]
    yv = y_col.to_numpy(dtype=float)[idx].astype(int)

    train_rel, hold_rel, cut_rel = time_holdout_indices(
        len(idx), holdout_frac=holdout_frac, purge=purge
    )
    train_idx = idx[train_rel]
    hold_idx = idx[hold_rel]

    model = HgbBinaryPredictor(**(hgb_params or {}))
    model.fit(Xv[train_rel], yv[train_rel])
    p = model.predict_proba(Xv[hold_rel])
    metrics = binary_metrics(yv[hold_rel], p)

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


def train_multiclass_head(
    X: pd.DataFrame,
    y_col: pd.Series,
    *,
    holdout_frac: float,
    purge: int,
    hgb_params: dict[str, Any] | None = None,
    z_col: pd.Series | None = None,
    tau: float | None = None,
    class_names: tuple[str, str, str] = ("down", "none", "up"),
) -> dict[str, Any]:
    """Ternary-style labels in {-1,0,1}; proba columns [down, none/flat, up]."""
    mask = _aligned_mask(X, y_col)
    idx = np.flatnonzero(mask)
    if len(idx) < 50:
        raise ValueError(f"too few aligned rows: {len(idx)}")

    Xv = X.to_numpy(dtype=float)[idx]
    yv = y_col.to_numpy(dtype=float)[idx].astype(int)

    train_rel, hold_rel, cut_rel = time_holdout_indices(
        len(idx), holdout_frac=holdout_frac, purge=purge
    )
    train_idx = idx[train_rel]
    hold_idx = idx[hold_rel]

    model = HgbTernaryPredictor(**(hgb_params or {}))
    model.fit(Xv[train_rel], yv[train_rel])
    proba = model.predict_proba(Xv[hold_rel])
    metrics = ternary_metrics(yv[hold_rel], proba)
    # Rename flat → configured middle name in report copy
    metrics["class_names"] = list(class_names)
    if "class_frac_true" in metrics:
        metrics["class_frac_true"] = {
            class_names[0]: metrics["class_frac_true"]["down"],
            class_names[1]: metrics["class_frac_true"]["flat"],
            class_names[2]: metrics["class_frac_true"]["up"],
        }
        metrics["class_frac_pred"] = {
            class_names[0]: metrics["class_frac_pred"]["down"],
            class_names[1]: metrics["class_frac_pred"]["flat"],
            class_names[2]: metrics["class_frac_pred"]["up"],
        }
    if z_col is not None:
        zv = z_col.to_numpy(dtype=float)[idx]
        metrics["flat_micro_direction"] = flat_micro_direction_metrics(
            yv[hold_rel], zv[hold_rel], proba
        )
    if tau is not None:
        metrics["barrier_tau"] = float(tau)

    # P(up first) > P(down first) agreement among decisive outcomes
    decisive = yv[hold_rel] != 0
    if decisive.any():
        favor_up = proba[decisive, 2] > proba[decisive, 0]
        true_up = yv[hold_rel][decisive] == 1
        metrics["first_side_hit_rate"] = float(np.mean(favor_up == true_up))
        metrics["n_decisive"] = int(decisive.sum())
    else:
        metrics["first_side_hit_rate"] = None
        metrics["n_decisive"] = 0

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


def run_train_barrier_first(
    cfg: dict[str, Any],
    bars: pd.DataFrame,
    artifacts_dir: Path | None,
) -> dict[str, Any]:
    target = _resolve_barrier_first(cfg)
    X = build_state_matrix(bars, list(cfg.get("state_ids") or []))
    y = target.compute(bars)
    purge = int(cfg.get("purge", target.horizon))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))
    hgb_params = _hgb_params(cfg)
    col = target.column_name()

    packed = train_multiclass_head(
        X,
        y[col],
        holdout_frac=holdout_frac,
        purge=purge,
        hgb_params=hgb_params,
        tau=target.barrier_tau,
        class_names=("down_first", "none", "up_first"),
    )
    model = packed.pop("model")

    results: dict[str, Any] = {
        "status": describe_readiness(cfg),
        "backend": "hgb",
        "target_id": target.id,
        "symbol": cfg.get("symbol"),
        "horizon": target.horizon,
        "barrier_tau": target.barrier_tau,
        "atr_period": target.atr_period,
        "purge": purge,
        "holdout_frac": holdout_frac,
        "column": col,
        "head": packed,
    }

    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifacts_dir / f"model_{col}.joblib")
        (artifacts_dir / "metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def run_train_barrier(
    cfg: dict[str, Any],
    bars: pd.DataFrame,
    artifacts_dir: Path | None,
) -> dict[str, Any]:
    target = _resolve_barrier_touch(cfg)
    X = build_state_matrix(bars, list(cfg.get("state_ids") or []))
    y = target.compute(bars)
    purge = int(cfg.get("purge", target.horizon))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))
    hgb_params = _hgb_params(cfg)

    results: dict[str, Any] = {
        "status": describe_readiness(cfg),
        "backend": "hgb",
        "target_id": target.id,
        "symbol": cfg.get("symbol"),
        "horizon": target.horizon,
        "barrier_tau": target.barrier_tau,
        "atr_period": target.atr_period,
        "purge": purge,
        "holdout_frac": holdout_frac,
        "heads": {},
    }

    art = artifacts_dir
    if art:
        art.mkdir(parents=True, exist_ok=True)

    for col in target.column_names():
        packed = train_binary_head(
            X,
            y[col],
            holdout_frac=holdout_frac,
            purge=purge,
            hgb_params=hgb_params,
        )
        model = packed.pop("model")
        results["heads"][col] = packed
        if art:
            joblib.dump(model, art / f"model_{col}.joblib")

    if art:
        (art / "metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def run_train_ternary(
    cfg: dict[str, Any],
    bars: pd.DataFrame,
    artifacts_dir: Path | None,
) -> dict[str, Any]:
    target = _resolve_ternary(cfg)
    X = build_state_matrix(bars, list(cfg.get("state_ids") or []))
    y = target.compute(bars)
    z = target.compute_z(bars)
    horizons = [int(h) for h in target.horizons]
    purge = int(cfg.get("purge", max(horizons)))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))
    hgb_params = _hgb_params(cfg)

    results: dict[str, Any] = {
        "status": describe_readiness(cfg),
        "backend": "hgb",
        "target_id": target.id,
        "symbol": cfg.get("symbol"),
        "purge": purge,
        "holdout_frac": holdout_frac,
        "flat_tau_by_h": {str(h): target.tau(h) for h in horizons},
        "horizons": {},
    }

    art = artifacts_dir
    if art:
        art.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        col = f"y_h{h}"
        zcol = f"z_h{h}"
        packed = train_multiclass_head(
            X,
            y[col],
            holdout_frac=holdout_frac,
            purge=purge,
            hgb_params=hgb_params,
            z_col=z[zcol],
            tau=target.tau(h),
            class_names=("down", "flat", "up"),
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


def run_train(
    cfg: dict[str, Any],
    bars: pd.DataFrame | None = None,
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any]:
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

    art = Path(artifacts_dir) if artifacts_dir else None
    target_id = cfg.get("target_id") or "atr_barrier_first"
    if target_id == "atr_barrier_first":
        return run_train_barrier_first(cfg, bars, art)
    if target_id == "atr_barrier_touch":
        return run_train_barrier(cfg, bars, art)
    if target_id == "ternary_displacement":
        return run_train_ternary(cfg, bars, art)
    raise ValueError(f"unsupported target_id={target_id!r}")


def train_from_config(
    config_path: Path | str,
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if artifacts_dir is None:
        artifacts_dir = cfg.get("artifacts_dir") or "artifacts/run"
    return run_train(cfg, artifacts_dir=artifacts_dir)
