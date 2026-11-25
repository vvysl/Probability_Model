"""Ternary label encoding for classifiers."""

from __future__ import annotations

import numpy as np

# Canonical probability column order.
CLASS_DOWN = -1
CLASS_FLAT = 0
CLASS_UP = 1
TERNARY_CLASSES: tuple[int, ...] = (CLASS_DOWN, CLASS_FLAT, CLASS_UP)
PROBA_NAMES: tuple[str, ...] = ("down", "flat", "up")

# sklearn / HGB integer codes
_ENC = {CLASS_DOWN: 0, CLASS_FLAT: 1, CLASS_UP: 2}
_DEC = {v: k for k, v in _ENC.items()}


def encode_ternary(y: np.ndarray) -> np.ndarray:
    """Map {-1,0,1} → {0,1,2}."""
    y = np.asarray(y, dtype=int)
    out = np.empty(y.shape, dtype=int)
    for raw, enc in _ENC.items():
        out[y == raw] = enc
    unknown = ~np.isin(y, list(_ENC.keys()))
    if unknown.any():
        raise ValueError(f"labels must be in {-1, 0, 1}; got {np.unique(y[unknown])}")
    return out


def decode_ternary(y_enc: np.ndarray) -> np.ndarray:
    """Map {0,1,2} → {-1,0,1}."""
    y_enc = np.asarray(y_enc, dtype=int)
    out = np.empty(y_enc.shape, dtype=int)
    for enc, raw in _DEC.items():
        out[y_enc == enc] = raw
    return out


def align_proba(proba: np.ndarray, classes_enc: np.ndarray) -> np.ndarray:
    """
    Reorder predict_proba columns to [down, flat, up].

    ``classes_enc`` is ``estimator.classes_`` in {0,1,2}.
    Missing classes get a zero column.
    """
    proba = np.asarray(proba, dtype=float)
    n = proba.shape[0]
    out = np.zeros((n, 3), dtype=float)
    for col, enc in enumerate(classes_enc):
        enc_i = int(enc)
        if enc_i not in _DEC:
            raise ValueError(f"unexpected class code {enc_i}")
        out[:, enc_i] = proba[:, col]
    return out
