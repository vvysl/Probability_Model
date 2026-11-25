"""Assemble bars → state → targets → model from config (stubs until steps 2–3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pm.bars import load_ohlcv, normalize_ohlcv
from pm.eval.splits import time_holdout_cut
from pm.state.registry import build_state_matrix, list_state_ids
from pm.targets.registry import get_target, list_target_ids


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def describe_readiness(cfg: dict[str, Any]) -> dict[str, Any]:
    """Report what is configured vs still empty (state catalog = step 3)."""
    horizons = cfg.get("horizons") or []
    state_ids = cfg.get("state_ids") or []
    target_id = cfg.get("target_id")
    return {
        "horizons": list(horizons),
        "state_ids": list(state_ids),
        "target_id": target_id,
        "registered_states": list_state_ids(),
        "registered_targets": list_target_ids(),
        "ready_for_train": bool(horizons) and bool(state_ids) and bool(target_id),
        "note": "v1 ready: state + targets + HGB train (lgbm later/server).",
    }


def run_scaffold(cfg: dict[str, Any], bars=None):
    """
    Wire layers without training.

    Raises clearly if registries/config still empty (expected in step 1).
    """
    status = describe_readiness(cfg)
    if bars is None and cfg.get("data_path"):
        bars = load_ohlcv(cfg["data_path"])
    elif bars is not None:
        bars = normalize_ohlcv(bars)

    state_ids = list(cfg.get("state_ids") or [])
    X = build_state_matrix(bars, state_ids) if bars is not None else None

    target = None
    y = None
    target_id = cfg.get("target_id")
    if target_id:
        target = get_target(target_id)
        if bars is not None:
            y = target.compute(bars)

    cut = None
    if bars is not None:
        cut = time_holdout_cut(len(bars), float(cfg.get("holdout_frac", 0.2)))

    return {
        "status": status,
        "bars": bars,
        "X": X,
        "y": y,
        "holdout_cut": cut,
    }
