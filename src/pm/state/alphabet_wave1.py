"""
Alphabet wave-1 continuous state (Quanqora Core A subset).

Reimplemented in-repo — no Quanqora import. Windows from alphabet_params
(a priori bar counts). Session calendar: Europe/Warsaw (research default).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from pm.state import alphabet_params as P
from pm.state.session_clock import (
    M_LONDON,
    M_OFF,
    M_OVERLAP,
    M_US,
    SESSION_TZ_NAME,
)
from pm.vol import atr

WAVE1_STATE_IDS: tuple[str, ...] = (
    "vwap_dist",
    "session_range_pos",
    "dist_session_extreme_atr",
    "gap_atr",
    "mins_in_session",
    "mins_to_session_end",
    "session_vol_frac",
    "gk_rv_ratio",
    "range_compression",
    "vol_pct",
    "ret_skew",
    "ret_accel",
    "sma_open_approach",
)


def _local_mins(index: pd.DatetimeIndex) -> np.ndarray:
    if index.tz is None:
        local = index.tz_localize("UTC").tz_convert(SESSION_TZ_NAME)
    else:
        local = index.tz_convert(SESSION_TZ_NAME)
    return (local.hour * 60 + local.minute).to_numpy(dtype=int)


def _session_ids_from_mins(mins: np.ndarray) -> np.ndarray:
    out = np.empty(len(mins), dtype=object)
    m = mins % (24 * 60)
    out[:] = "us"
    out[m >= M_OFF] = "off_session"
    out[m < M_LONDON] = "asia"
    out[(m >= M_LONDON) & (m < M_OVERLAP)] = "london"
    out[(m >= M_OVERLAP) & (m < M_US)] = "overlap"
    return out


def _mins_to_end_from_mins(mins: np.ndarray) -> np.ndarray:
    m = mins % (24 * 60)
    out = np.empty(len(m), dtype=float)
    out[m >= M_OFF] = (24 * 60 - m[m >= M_OFF]).astype(float)
    mask = m < M_LONDON
    out[mask] = (M_LONDON - m[mask]).astype(float)
    mask = (m >= M_LONDON) & (m < M_OVERLAP)
    out[mask] = (M_OVERLAP - m[mask]).astype(float)
    mask = (m >= M_OVERLAP) & (m < M_US)
    out[mask] = (M_US - m[mask]).astype(float)
    mask = (m >= M_US) & (m < M_OFF)
    out[mask] = (M_OFF - m[mask]).astype(float)
    return out


def _vol_pct_causal(atr14: np.ndarray, w: int) -> np.ndarray:
    """
    Empirical CDF of ATR vs past window.

    Uses a stride-trick window view in row-chunks so peak memory stays O(chunk*w)
    instead of O(n*w).
    """
    n = len(atr14)
    vpct = np.full(n, np.nan)
    if n <= w:
        return vpct
    # Keep chunk*w*8 roughly under ~32MB
    chunk = max(64, min(2048, (4_000_000 // max(w, 1))))
    from numpy.lib.stride_tricks import sliding_window_view

    # win[j] = atr[j:j+w]; past for time i is atr[i-w:i] == win[i-w]
    win = sliding_window_view(atr14, window_shape=w)  # (n-w+1, w) view
    for start in range(w, n, chunk):
        end = min(n, start + chunk)
        past = win[start - w : end - w]  # (end-start, w)
        cur = atr14[start:end][:, None]
        valid = np.isfinite(past)
        dens = valid.sum(axis=1)
        counts = ((past <= cur) & valid).sum(axis=1)
        vals = np.where(dens >= 8, counts / np.maximum(dens, 1), np.nan)
        vals = np.where(np.isfinite(atr14[start:end]), vals, np.nan)
        vpct[start:end] = vals
    return vpct


def compute_alphabet_wave1(bars: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with WAVE1_STATE_IDS columns, causal, aligned to bars."""
    n = len(bars)
    idx = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.DatetimeIndex(bars.index)
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    v = bars["volume"].to_numpy(dtype=float)

    atr14 = atr(bars, period=P.ATR_N).to_numpy(dtype=float)
    out = {fid: np.full(n, np.nan, dtype=float) for fid in WAVE1_STATE_IDS}

    mins = _local_mins(idx)
    sids = _session_ids_from_mins(mins)
    out["mins_to_session_end"] = _mins_to_end_from_mins(mins)

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sids[1:] != sids[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)

    # int64 ns for fast minutes-from-open
    ts_ns = np.asarray(idx.asi8, dtype=np.int64)
    hist_vol: dict[str, list[float]] = defaultdict(list)

    for a, b in zip(starts.tolist(), ends.tolist()):
        sid = str(sids[a])
        g = np.nan
        if a > 0 and np.isfinite(atr14[a]) and atr14[a] > 0:
            g = (o[a] - c[a - 1]) / atr14[a]

        past = hist_vol[sid]
        e_vol = float(np.median(past[-P.W_SESSION_VOL_HIST :])) if past else np.nan

        open_ns = ts_ns[a]
        seg_o = o[a:b]
        seg_h = h[a:b]
        seg_l = low[a:b]
        seg_c = c[a:b]
        seg_v = np.maximum(v[a:b], 0.0)
        seg_atr = atr14[a:b]
        seg_ts = ts_ns[a:b]

        out["mins_in_session"][a:b] = (seg_ts - open_ns) / 60e9
        out["gap_atr"][a:b] = g

        cum_v = np.cumsum(seg_v)
        if np.isfinite(e_vol) and e_vol > 0:
            out["session_vol_frac"][a:b] = cum_v / e_vol

        tp = (seg_h + seg_l + seg_c) / 3.0
        cum_pv = np.cumsum(tp * seg_v)
        cum_vv = np.cumsum(seg_v)
        sess_hi = np.maximum.accumulate(seg_h)
        sess_lo = np.minimum.accumulate(seg_l)
        span = sess_hi - sess_lo
        with np.errstate(divide="ignore", invalid="ignore"):
            srp = np.where(span > 0, (seg_c - sess_lo) / span, 0.5)
        out["session_range_pos"][a:b] = srp

        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(cum_vv > 0, cum_pv / cum_vv, np.nan)
            ok = (seg_atr > 0) & np.isfinite(seg_atr)
            out["vwap_dist"][a:b] = np.where(ok, (seg_c - vwap) / seg_atr, np.nan)
            dist_edge = np.minimum(seg_c - sess_lo, sess_hi - seg_c)
            out["dist_session_extreme_atr"][a:b] = np.where(ok, dist_edge / seg_atr, np.nan)

        # unused but keeps lint calm if seg_o referenced for future
        _ = seg_o
        hist_vol[sid].append(float(np.sum(seg_v)))

    # --- gk_rv_ratio ---
    log_hl = np.log(np.maximum(h / np.maximum(low, 1e-18), 1e-18))
    log_co = np.log(np.maximum(c / np.maximum(o, 1e-18), 1e-18))
    gk_bar = 0.5 * log_hl**2 - (2.0 * math.log(2.0) - 1.0) * log_co**2
    gk_mean = pd.Series(gk_bar).rolling(P.W_YZ, min_periods=P.W_YZ).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        sig = np.sqrt(np.maximum(gk_mean, 0.0))
        px = np.where(c > 0, c, np.nan)
        out["gk_rv_ratio"] = np.where(
            (atr14 > 0) & np.isfinite(atr14) & np.isfinite(sig) & np.isfinite(px),
            (sig * px) / atr14,
            np.nan,
        )

    # --- range_compression ---
    bar_range = h - low
    med_s = pd.Series(bar_range).rolling(P.K_SHORT, min_periods=P.K_SHORT).median().to_numpy()
    med_l = pd.Series(bar_range).rolling(P.K_LONG, min_periods=P.K_LONG).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["range_compression"] = np.where(med_l > 0, med_s / med_l, np.nan)

    out["vol_pct"] = _vol_pct_causal(atr14, P.W_VOL_PCT)

    # --- ret_skew ---
    log_c = np.log(np.maximum(c, 1e-18))
    rets = np.diff(log_c, prepend=np.nan)
    out["ret_skew"] = pd.Series(rets).rolling(P.W_SKEW, min_periods=P.W_SKEW).skew().to_numpy()

    # --- ret_accel ---
    sra = np.full(n, np.nan)
    hr = P.H_RET
    sra[hr:] = (c[hr:] - c[:-hr]) / atr14[hr:]
    sra[hr:] = np.where((atr14[hr:] > 0) & np.isfinite(atr14[hr:]), sra[hr:], np.nan)
    acc = np.full(n, np.nan)
    acc[2 * hr :] = sra[2 * hr :] - sra[hr : n - hr]
    out["ret_accel"] = acc

    # --- sma_open_approach ---
    sma = pd.Series(c).rolling(P.N_MA, min_periods=P.N_MA).mean().to_numpy()
    open_off = o - sma
    body = c - o
    with np.errstate(divide="ignore", invalid="ignore"):
        appr = -np.sign(open_off) * body / atr14
    appr = np.where(
        (atr14 > 0) & np.isfinite(atr14) & np.isfinite(sma),
        appr,
        np.nan,
    )
    appr = np.where(np.abs(open_off) <= 1e-18, 0.0, appr)
    appr[: P.N_MA - 1] = np.nan
    out["sma_open_approach"] = appr

    return pd.DataFrame(out, index=idx)[list(WAVE1_STATE_IDS)]


class _Wave1Var:
    """Thin StateVar over shared wave-1 frame (one column)."""

    def __init__(self, feature_id: str) -> None:
        self._id = feature_id

    @property
    def id(self) -> str:
        return self._id

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return compute_alphabet_wave1(bars)[self._id].rename(self._id)


def default_alphabet_wave1_vars() -> list[_Wave1Var]:
    return [_Wave1Var(fid) for fid in WAVE1_STATE_IDS]
