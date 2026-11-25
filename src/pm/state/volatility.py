"""B — multi-scale ATR, ATR ratios, realized vol."""

from __future__ import annotations

import pandas as pd

from pm.state.atr_scales import default_atr_scales
from pm.state._util import safe_div, simple_return
from pm.vol import atr


class AtrRatio:
    """atr_fast / atr_slow — vol expansion vs slower regime."""

    def __init__(self, fast: int, slow: int) -> None:
        if fast < 1 or slow < 1 or fast >= slow:
            raise ValueError("need 1 <= fast < slow")
        self.fast = int(fast)
        self.slow = int(slow)
        self._id = f"atr_ratio_{self.fast}_{self.slow}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return safe_div(atr(bars, self.fast), atr(bars, self.slow)).rename(self._id)


class RealizedVol:
    """Std of 1m simple returns over a trailing window (causal)."""

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self._id = f"rv_{self.window}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        r1 = simple_return(bars["close"].astype(float), 1)
        return r1.rolling(self.window, min_periods=self.window).std(ddof=0).rename(
            self._id
        )


def default_volatility_vars() -> list:
    vars_: list = []
    vars_.extend(default_atr_scales())
    vars_.append(AtrRatio(14, 60))
    vars_.append(AtrRatio(60, 240))
    vars_.append(RealizedVol(14))
    vars_.append(RealizedVol(60))
    return vars_
