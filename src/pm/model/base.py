"""Predictor protocol — P(y | x); concrete backends later."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Predictor(Protocol):
    """Fits on tabular state; returns probabilities matching TargetSpec."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit on training rows. ``y`` shape depends on active TargetSpec."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probabilities.

        Shape must be consistent with the TargetSpec used at fit time
        (e.g. (n, 2) for binary, or multi-horizon layout declared later).
        """
