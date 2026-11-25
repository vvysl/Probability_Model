"""E — relative volume features (no raw volume in the model matrix)."""

from __future__ import annotations

import pandas as pd

from pm.state._util import safe_div, simple_return


class RelVolume:
    """volume / mean(volume, window)."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._id = f"rvol_{self.window}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        v = bars["volume"].astype(float)
        mean = v.rolling(self.window, min_periods=self.window).mean()
        return safe_div(v, mean).rename(self._id)


class VolZScore:
    """(volume - mean_w) / std_w."""

    def __init__(self, window: int = 60) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self._id = f"vol_z_{self.window}"

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        v = bars["volume"].astype(float)
        mean = v.rolling(self.window, min_periods=self.window).mean()
        std = v.rolling(self.window, min_periods=self.window).std(ddof=0)
        return safe_div(v - mean, std).rename(self._id)


class VolChange:
    """volume_t / volume_{t-1} - 1."""

    id = "vol_chg"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        v = bars["volume"].astype(float)
        return (v / v.shift(1) - 1.0).rename(self.id)


class VolXRet:
    """(volume / mean_60) * 1m return — signed volume intensity."""

    id = "vol_x_ret"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        v = bars["volume"].astype(float)
        mean60 = v.rolling(60, min_periods=60).mean()
        r1 = simple_return(bars["close"].astype(float), 1)
        return (safe_div(v, mean60) * r1).rename(self.id)


def default_volume_vars() -> list:
    return [
        RelVolume(5),
        RelVolume(30),
        VolZScore(60),
        VolChange(),
        VolXRet(),
    ]
