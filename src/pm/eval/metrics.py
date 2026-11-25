"""Holdout metrics for ternary probability forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from pm.model.labels import CLASS_FLAT, PROBA_NAMES, TERNARY_CLASSES, encode_ternary


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


def flat_micro_direction_metrics(
    y_true: np.ndarray,
    z: np.ndarray,
    proba: np.ndarray,
) -> dict[str, Any]:
    """
    When true label is flat (|z| < tau), score residual sign vs model tilt.

    Model favors up if P(up) > P(down). Skip rows with z == 0.
    + if sign(z) matches favor; − otherwise.
    """
    y_true = np.asarray(y_true)
    z = np.asarray(z, dtype=float)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) != len(z) or len(y_true) != len(proba):
        raise ValueError("y_true, z, proba length mismatch")

    flat = y_true == CLASS_FLAT
    # proba columns: [down, flat, up] → indices 0 and 2
    favor_up = proba[:, 2] > proba[:, 0]
    favor_down = proba[:, 0] > proba[:, 2]
    # tie P(up)==P(down) → skip
    decisive = favor_up | favor_down
    nonzero = z != 0
    mask = flat & decisive & nonzero & np.isfinite(z)

    n = int(mask.sum())
    if n == 0:
        return {
            "n_flat_scored": 0,
            "n_flat_total": int(flat.sum()),
            "hit_rate": None,
            "n_plus": 0,
            "n_minus": 0,
        }

    sign_z = np.sign(z[mask])
    favor_sign = np.where(favor_up[mask], 1.0, -1.0)
    hit = sign_z == favor_sign
    n_plus = int(hit.sum())
    n_minus = int((~hit).sum())
    return {
        "n_flat_scored": n,
        "n_flat_total": int(flat.sum()),
        "hit_rate": float(n_plus / n),
        "n_plus": n_plus,
        "n_minus": n_minus,
    }
