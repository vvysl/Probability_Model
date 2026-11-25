"""Holdout metrics for ternary probability forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from pm.model.labels import PROBA_NAMES, TERNARY_CLASSES, encode_ternary


def ternary_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """
    ``y_true`` in {{-1,0,1}}; ``proba`` shape (n, 3) = [down, flat, up].
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError("proba must have shape (n, 3)")
    if len(y_true) != len(proba):
        raise ValueError("y_true and proba length mismatch")

    y_enc = encode_ternary(y_true)
    # Clip for numerical stability in log_loss
    p = np.clip(proba, 1e-7, 1.0 - 1e-7)
    p = p / p.sum(axis=1, keepdims=True)

    labels = [0, 1, 2]
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_enc, p, labels=labels)),
        "class_frac_true": {
            name: float(np.mean(y_true == cls))
            for name, cls in zip(PROBA_NAMES, TERNARY_CLASSES)
        },
        "class_frac_pred": {
            name: float(np.mean(np.argmax(proba, axis=1) == i))
            for i, name in enumerate(PROBA_NAMES)
        },
    }

    # One-vs-rest Brier per class
    briers = {}
    for i, name in enumerate(PROBA_NAMES):
        y_bin = (y_enc == i).astype(float)
        briers[name] = float(brier_score_loss(y_bin, p[:, i]))
    out["brier_ovr"] = briers
    out["brier_mean"] = float(np.mean(list(briers.values())))

    pred = np.array(TERNARY_CLASSES)[np.argmax(proba, axis=1)]
    out["accuracy"] = float(np.mean(pred == y_true))
    return out
