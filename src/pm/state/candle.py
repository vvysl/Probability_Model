"""C — candle structure in ATR units + directional consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pm.state._util import atr14, safe_div


class BodyAtr:
    id = "body_atr"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        body = bars["close"].astype(float) - bars["open"].astype(float)
        return safe_div(body, atr14(bars)).rename(self.id)


class UpperWickAtr:
    id = "upper_wick_atr"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        o = bars["open"].astype(float)
        c = bars["close"].astype(float)
        h = bars["high"].astype(float)
        wick = h - np.maximum(o, c)
        return safe_div(wick, atr14(bars)).rename(self.id)


class LowerWickAtr:
    id = "lower_wick_atr"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        o = bars["open"].astype(float)
        c = bars["close"].astype(float)
        low = bars["low"].astype(float)
        wick = np.minimum(o, c) - low
        return safe_div(wick, atr14(bars)).rename(self.id)


class RangeAtr:
    id = "range_atr"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        rng = bars["high"].astype(float) - bars["low"].astype(float)
        return safe_div(rng, atr14(bars)).rename(self.id)


class ClosePos:
    """(close - low) / (high - low); 0.5 when range is 0."""

    id = "close_pos"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        h = bars["high"].astype(float)
        low = bars["low"].astype(float)
        c = bars["close"].astype(float)
        rng = h - low
        pos = (c - low) / rng.where(rng > 0)
        return pos.fillna(0.5).rename(self.id)


class DirConsist:
    """Fraction of bullish candles (close > open) over trailing window."""

    def __init__(self, window: int = 15) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._id = f"dir_consist_{self.window}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        up = (bars["close"].astype(float) > bars["open"].astype(float)).astype(float)
        return up.rolling(self.window, min_periods=self.window).mean().rename(self._id)


def default_candle_vars() -> list:
    return [
        BodyAtr(),
        UpperWickAtr(),
        LowerWickAtr(),
        RangeAtr(),
        ClosePos(),
        DirConsist(15),
    ]
