"""
ATR barrier outcomes on a fixed horizon (probabilistic system, not a strategy).

1) Independent touch (legacy diagnostic):
     P(touch +tau*ATR), P(touch -tau*ATR) within n

2) First barrier (primary):
     who is touched first — up / down / none within n
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pm.vol import atr

DEFAULT_HORIZON = 30
DEFAULT_ATR_PERIOD = 14
DEFAULT_BARRIER_TAU = 2.0

# First-barrier class encoding (reuse ternary predictor layout).
LABEL_DOWN_FIRST = -1
LABEL_NONE = 0
LABEL_UP_FIRST = 1


def _future_windows(
    high: np.ndarray, low: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (win_h, win_l) for t=0..N-n-1 as windows of high/low[t+1:t+1+n]."""
    N = len(high)
    t_end = N - n
    win_h = sliding_window_view(high, n)[1 : t_end + 1]
    win_l = sliding_window_view(low, n)[1 : t_end + 1]
    return win_h, win_l, t_end


class AtrBarrierTouchTarget:
    """Independent hit_up / hit_down within n (both may be 1)."""

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        atr_period: int = DEFAULT_ATR_PERIOD,
        barrier_tau: float = DEFAULT_BARRIER_TAU,
        target_id: str = "atr_barrier_touch",
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if barrier_tau <= 0:
            raise ValueError("barrier_tau must be > 0")
        self._id = target_id
        self.horizon = int(horizon)
        self.atr_period = int(atr_period)
        self.barrier_tau = float(barrier_tau)

    @property
    def id(self) -> str:
        return self._id

    @property
    def horizons(self) -> Sequence[int]:
        return (self.horizon,)

    def column_names(self) -> tuple[str, str]:
        n = self.horizon
        return (f"hit_up_h{n}", f"hit_down_h{n}")

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        if not {"high", "low", "close"}.issubset(bars.columns):
            raise ValueError("bars must include high, low, close")

        n = self.horizon
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        scale = atr(bars, period=self.atr_period).to_numpy(dtype=float)
        scale = np.where(scale > 0, scale, np.nan)

        N = len(bars)
        hit_up = np.full(N, np.nan)
        hit_down = np.full(N, np.nan)
        up_name, down_name = self.column_names()

        if N <= n + 1:
            return pd.DataFrame(
                {up_name: hit_up, down_name: hit_down}, index=bars.index
            )

        win_h, win_l, t_end = _future_windows(high, low, n)
        atr_t = scale[:t_end]
        c_t = close[:t_end]
        up_thr = c_t + self.barrier_tau * atr_t
        down_thr = c_t - self.barrier_tau * atr_t
        valid = np.isfinite(atr_t) & np.isfinite(c_t)

        fut_max = win_h.max(axis=1)
        fut_min = win_l.min(axis=1)
        hit_up[:t_end] = np.where(valid, (fut_max >= up_thr).astype(float), np.nan)
        hit_down[:t_end] = np.where(valid, (fut_min <= down_thr).astype(float), np.nan)

        return pd.DataFrame(
            {up_name: hit_up, down_name: hit_down}, index=bars.index
        )


class AtrBarrierFirstTarget:
    """
    First barrier within n:

        +1  up barrier touched first
        -1  down barrier touched first
        0   neither, or both on the same first bar (ambiguous)

    Path scanned bar-by-bar on (t+1 .. t+n).
    """

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        atr_period: int = DEFAULT_ATR_PERIOD,
        barrier_tau: float = DEFAULT_BARRIER_TAU,
        target_id: str = "atr_barrier_first",
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if barrier_tau <= 0:
            raise ValueError("barrier_tau must be > 0")
        self._id = target_id
        self.horizon = int(horizon)
        self.atr_period = int(atr_period)
        self.barrier_tau = float(barrier_tau)

    @property
    def id(self) -> str:
        return self._id

    @property
    def horizons(self) -> Sequence[int]:
        return (self.horizon,)

    def column_name(self) -> str:
        return f"first_h{self.horizon}"

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        if not {"high", "low", "close"}.issubset(bars.columns):
            raise ValueError("bars must include high, low, close")

        n = self.horizon
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        scale = atr(bars, period=self.atr_period).to_numpy(dtype=float)
        scale = np.where(scale > 0, scale, np.nan)

        N = len(bars)
        y = np.full(N, np.nan)
        col = self.column_name()

        if N <= n + 1:
            return pd.DataFrame({col: y}, index=bars.index)

        win_h, win_l, t_end = _future_windows(high, low, n)
        atr_t = scale[:t_end]
        c_t = close[:t_end]
        up_thr = c_t + self.barrier_tau * atr_t
        down_thr = c_t - self.barrier_tau * atr_t
        valid = np.isfinite(atr_t) & np.isfinite(c_t)

        up_hit = win_h >= up_thr[:, None]
        down_hit = win_l <= down_thr[:, None]
        has_up = up_hit.any(axis=1)
        has_down = down_hit.any(axis=1)
        # argmax on bool → index of first True; meaningless if none
        fu = np.where(has_up, np.argmax(up_hit, axis=1), n + 1)
        fd = np.where(has_down, np.argmax(down_hit, axis=1), n + 1)

        labels = np.full(t_end, float(LABEL_NONE))
        labels[(fu < fd) & has_up] = float(LABEL_UP_FIRST)
        labels[(fd < fu) & has_down] = float(LABEL_DOWN_FIRST)
        # fu == fd with both → NONE (same-bar ambiguity)
        labels = np.where(valid, labels, np.nan)
        y[:t_end] = labels

        return pd.DataFrame({col: y}, index=bars.index)
