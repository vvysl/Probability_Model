"""
Ternary displacement target on fixed 1m horizons.

Movement := close-to-close displacement in causal ATR units, not path length.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pm.targets.scale import atr

# Encoding for classifiers / predict_proba layouts later.
LABEL_DOWN = -1
LABEL_FLAT = 0
LABEL_UP = 1

DEFAULT_HORIZONS: tuple[int, ...] = (3, 10, 30)
DEFAULT_ATR_PERIOD = 14
DEFAULT_FLAT_TAU = 0.15


class TernaryDisplacementTarget:
    """
    For each horizon h (bars on 1m):

        z_h = (close_{t+h} - close_t) / ATR_t
        y   = +1 if z_h > tau, -1 if z_h < -tau, else 0

    Tail of length h is NaN (no future close). ATR warmup is NaN → label NaN.
    """

    def __init__(
        self,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        atr_period: int = DEFAULT_ATR_PERIOD,
        flat_tau: float = DEFAULT_FLAT_TAU,
        target_id: str = "ternary_displacement",
    ) -> None:
        hs = tuple(int(h) for h in horizons)
        if not hs or any(h < 1 for h in hs):
            raise ValueError("horizons must be positive ints")
        if flat_tau < 0:
            raise ValueError("flat_tau must be >= 0")
        self._id = target_id
        self._horizons = hs
        self.atr_period = int(atr_period)
        self.flat_tau = float(flat_tau)

    @property
    def id(self) -> str:
        return self._id

    @property
    def horizons(self) -> Sequence[int]:
        return self._horizons

    def column_name(self, h: int) -> str:
        return f"y_h{h}"

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        if "close" not in bars.columns:
            raise ValueError("bars must include 'close'")
        close = bars["close"].astype(float)
        scale = atr(bars, period=self.atr_period)
        # Avoid div-by-zero; treat non-positive ATR as missing scale.
        scale = scale.where(scale > 0)

        cols: dict[str, pd.Series] = {}
        for h in self._horizons:
            future = close.shift(-h)
            z = (future - close) / scale
            y = pd.Series(np.nan, index=bars.index, dtype=float)
            valid = z.notna()
            y.loc[valid & (z > self.flat_tau)] = LABEL_UP
            y.loc[valid & (z < -self.flat_tau)] = LABEL_DOWN
            y.loc[valid & (z >= -self.flat_tau) & (z <= self.flat_tau)] = LABEL_FLAT
            cols[self.column_name(h)] = y

        return pd.DataFrame(cols, index=bars.index)
