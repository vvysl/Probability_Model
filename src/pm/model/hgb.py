"""HistGradientBoosting ternary predictor (v1 backend). LGBM deferred to server."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pm.model.labels import align_proba, decode_ternary, encode_ternary


DEFAULT_HGB_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "max_iter": 200,
    "l2_regularization": 1.0,
    "min_samples_leaf": 40,
    "early_stopping": False,
    "random_state": 0,
    "class_weight": "balanced",
}


class HgbTernaryPredictor:
    """
    One ternary classifier: P(down), P(flat), P(up).

    Labels in {{-1, 0, 1}}; ``predict_proba`` → shape (n, 3) in that order.
    """

    def __init__(self, **hgb_params: Any) -> None:
        params = {**DEFAULT_HGB_PARAMS, **hgb_params}
        self.params = params
        self._clf = HistGradientBoostingClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> HgbTernaryPredictor:
        X = np.asarray(X, dtype=float)
        y_enc = encode_ternary(y)
        self._clf.fit(X, y_enc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        raw = self._clf.predict_proba(X)
        return align_proba(raw, self._clf.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax class in {{-1, 0, 1}}."""
        proba = self.predict_proba(X)
        enc = np.argmax(proba, axis=1)
        return decode_ternary(enc)
