"""
Session clock for alphabet-style features.

Default calendar matches Quanqora research canon: Europe/Warsaw wall-clock
segments asia|london|overlap|us|off_session. Override via SessionSpec later
if needed — package stays free of Quanqora imports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SESSION_TZ_NAME = "Europe/Warsaw"
SESSION_TZ = ZoneInfo(SESSION_TZ_NAME)

# Minutes from local midnight: [start, end)
M_ASIA = 0
M_LONDON = 9 * 60
M_OVERLAP = 15 * 60 + 30
M_US = 18 * 60
M_OFF = 22 * 60 + 30

SESSION_BOUNDARIES: tuple[tuple[str, int], ...] = (
    ("asia", M_ASIA),
    ("london", M_LONDON),
    ("overlap", M_OVERLAP),
    ("us", M_US),
    ("off_session", M_OFF),
)


def to_session_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.astimezone(SESSION_TZ)


def mins_in_session_local(dt: datetime) -> int:
    loc = to_session_local(dt)
    return loc.hour * 60 + loc.minute


def session_id_from_mins(mins: int) -> str:
    m = int(mins) % (24 * 60)
    if m >= M_OFF:
        return "off_session"
    if m < M_LONDON:
        return "asia"
    if m < M_OVERLAP:
        return "london"
    if m < M_US:
        return "overlap"
    return "us"


def session_id(dt: datetime) -> str:
    return session_id_from_mins(mins_in_session_local(dt))


def mins_to_session_end(dt: datetime) -> float:
    """Minutes of Warsaw local time until next canon session boundary."""
    m = mins_in_session_local(dt)
    if m >= M_OFF:
        return float(24 * 60 - m)
    if m < M_ASIA:
        return float(M_ASIA - m)
    if m < M_LONDON:
        return float(M_LONDON - m)
    if m < M_OVERLAP:
        return float(M_OVERLAP - m)
    if m < M_US:
        return float(M_US - m)
    return float(M_OFF - m)


def session_ids_for_index(index: Sequence) -> np.ndarray:
    """Vector of session_id strings aligned to a DatetimeIndex."""
    out = np.empty(len(index), dtype=object)
    for i, ts in enumerate(index):
        t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        out[i] = session_id(t)
    return out


def annotate_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """Return frame with session_id, mins_in_session_clock, mins_to_session_end."""
    idx = bars.index
    sids = session_ids_for_index(idx)
    mte = np.empty(len(idx), dtype=float)
    for i, ts in enumerate(idx):
        t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        mte[i] = mins_to_session_end(t)
    return pd.DataFrame(
        {"session_id": sids, "mins_to_session_end": mte},
        index=idx,
    )
