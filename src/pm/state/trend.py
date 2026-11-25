"""D — EMA distances / slopes (ATR-normalized) + price efficiency."""

from __future__ import annotations

import pandas as pd

from pm.state._util import atr14, safe_div


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


class DistEma:
    """(close - EMA_span) / ATR14."""

    def __init__(self, span: int) -> None:
        if span < 1:
            raise ValueError("span must be >= 1")
        self.span = int(span)
        self._id = f"dist_ema_{self.span}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        return safe_div(close - _ema(close, self.span), atr14(bars)).rename(self._id)


class EmaSlope:
    """(EMA_t - EMA_{t-lag}) / ATR14."""

    def __init__(self, span: int, lag: int) -> None:
        if span < 1 or lag < 1:
            raise ValueError("span and lag must be >= 1")
        self.span = int(span)
        self.lag = int(lag)
        self._id = f"ema_slope_{self.span}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        e = _ema(close, self.span)
        return safe_div(e - e.shift(self.lag), atr14(bars)).rename(self._id)


class Efficiency:
    """
    |C_t - C_{t-N}| / sum_{i=0..N-1} |C_{t-i} - C_{t-i-1}|
    in [0, 1]; nearer 1 = more directional.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._id = f"eff_{self.window}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        net = (close - close.shift(self.window)).abs()
        step = close.diff().abs()
        path = step.rolling(self.window, min_periods=self.window).sum()
        return safe_div(net, path).rename(self._id)


def default_trend_vars() -> list:
    return [
        DistEma(10),
        DistEma(30),
        DistEma(60),
        EmaSlope(span=10, lag=5),
        EmaSlope(span=60, lag=15),
        Efficiency(30),
    ]
