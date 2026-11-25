"""Causal volatility scale for displacement labels (asset-agnostic)."""

from __future__ import annotations

import pandas as pd


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

    First ``period`` rows (0-indexed: period-1 warmup after TR) are NaN
    until enough history exists; pandas ewm with adjust=False matches Wilder
    after a simple-mean seed via expanding min_periods.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(bars)
    # Wilder: ATR_t = ATR_{t-1} * (period-1)/period + TR_t / period
    # Equivalent to ewm(alpha=1/period, adjust=False) after seeding.
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
