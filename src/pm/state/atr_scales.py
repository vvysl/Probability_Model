"""
Multi-scale ATR market constants.

ATR(14)  — short-term volatility
ATR(60)  — local intraday regime
ATR(240) — slower volatility regime

Used as continuous scales for measuring price movement (denominators),
not as discrete signals.
"""

from __future__ import annotations

import pandas as pd

from pm.vol import ATR_PERIODS, atr


class AtrScale:
    """Causal Wilder ATR at a fixed period; id = ``atr_{period}``."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = int(period)
        self._id = f"atr_{self.period}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return atr(bars, period=self.period).rename(self._id)


def default_atr_scales() -> list[AtrScale]:
    return [AtrScale(p) for p in ATR_PERIODS]
