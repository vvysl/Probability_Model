"""A — ATR-normalized returns over 1m lags."""

from __future__ import annotations

import pandas as pd

from pm.state._util import atr14, safe_div, simple_return

RET_LAGS: tuple[int, ...] = (1, 3, 5, 15, 30, 60)


class RetAtr:
    """(close_t / close_{t-lag} - 1) / ATR14."""

    def __init__(self, lag: int) -> None:
        if lag < 1:
            raise ValueError("lag must be >= 1")
        self.lag = int(lag)
        self._id = f"ret_atr_{self.lag}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        r = simple_return(bars["close"].astype(float), self.lag)
        return safe_div(r, atr14(bars)).rename(self._id)


def default_return_vars() -> list[RetAtr]:
    return [RetAtr(lag) for lag in RET_LAGS]
