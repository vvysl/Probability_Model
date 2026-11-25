"""Binary HGB predictor — P(y=1 | x) for barrier touch heads."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pm.model.hgb import DEFAULT_HGB_PARAMS


class HgbBinaryPredictor:
    """Labels in {0,1}; predict_proba → shape (n,) = P(y=1)."""

    def __init__(self, **hgb_params: Any) -> None:
        params = {**DEFAULT_HGB_PARAMS, **hgb_params}
        self.params = params
        self._clf = HistGradientBoostingClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> HgbBinaryPredictor:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("binary labels must be in {0,1}")
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1) as shape (n,)."""
        X = np.asarray(X, dtype=float)
        raw = self._clf.predict_proba(X)
        classes = list(self._clf.classes_)
        if 1 not in classes:
            return np.zeros(len(X), dtype=float)
        col = classes.index(1)
        return raw[:, col]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
