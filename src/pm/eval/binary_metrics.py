"""Metrics for binary probability forecasts P(y=1)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def binary_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    if len(y_true) != len(p):
        raise ValueError("length mismatch")

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "mean_p": float(np.mean(p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy@0.5": float(np.mean((p >= 0.5).astype(int) == y_true)),
    }
    # AUC needs both classes present
    if y_true.min() != y_true.max():
        out["auc"] = float(roc_auc_score(y_true, p))
    else:
        out["auc"] = None
    return out
