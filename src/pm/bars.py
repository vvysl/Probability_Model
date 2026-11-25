"""OHLCV load and schema normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a frame with time + OHLCV (time as column or index).
    Return UTC DatetimeIndex + float OHLCV, sorted, monotonic.
    """
    out = df.copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True)
        out = out.set_index("time")
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("OHLCV needs a 'time' column or DatetimeIndex")
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out.index.name = "time"

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    out = out.loc[:, list(REQUIRED_COLUMNS)].astype(float)
    out = out.sort_index()
    if not out.index.is_monotonic_increasing:
        raise ValueError("time index must be monotonic increasing after sort")
    if out.index.has_duplicates:
        raise ValueError("duplicate timestamps in OHLCV")
    return out


def load_ohlcv(path: Path | str) -> pd.DataFrame:
    """Load CSV with columns time,open,high,low,close,volume."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    return normalize_ohlcv(raw)


def make_synthetic_ohlcv(
    n: int = 64,
    start: str = "2024-01-01 00:00:00+00:00",
    freq: str = "1min",
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic synthetic bars for tests (not for research)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.05, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.0, 0.05, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.0, 0.05, size=n)
    volume = rng.uniform(100.0, 1000.0, size=n)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    df.index.name = "time"
    return normalize_ohlcv(df)
