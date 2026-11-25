"""Shared helpers for causal state features."""

from __future__ import annotations

import pandas as pd

from pm.vol import atr


def atr14(bars: pd.DataFrame) -> pd.Series:
    return atr(bars, period=14)


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide; non-positive / non-finite denominator → NaN."""
    den_ok = den.where(den > 0)
    return num / den_ok


def simple_return(close: pd.Series, lag: int) -> pd.Series:
    return close / close.shift(lag) - 1.0
