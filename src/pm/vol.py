"""Causal volatility scales (asset-agnostic). Shared by state and targets."""

from __future__ import annotations

import pandas as pd

# Multi-scale ATR periods for regime / movement measurement (1m bars).
ATR_PERIODS: tuple[int, ...] = (14, 60, 240)


def true_range(bars: pd.DataFrame) -> pd.Series:
    """True range at each bar using prior close (causal)."""
    prev_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR at bar t using only bars <= t.

    Warmup: first ``period - 1`` values may be NaN (``min_periods=period``).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(bars)
    # Wilder: ATR_t = ATR_{t-1}*(period-1)/period + TR_t/period
    # ≈ ewm(alpha=1/period, adjust=False).
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
