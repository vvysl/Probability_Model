"""v1 continuous state catalog (~30 features, blocks A–E)."""

from __future__ import annotations

# Ordered default feature set for configs / training.
V1_STATE_IDS: tuple[str, ...] = (
    # A — price movement (return / ATR14)
    "ret_atr_1",
    "ret_atr_3",
    "ret_atr_5",
    "ret_atr_15",
    "ret_atr_30",
    "ret_atr_60",
    # B — volatility
    "atr_14",
    "atr_60",
    "atr_240",
    "atr_ratio_14_60",
    "atr_ratio_60_240",
    "rv_14",
    "rv_60",
    # C — candle structure
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "range_atr",
    "close_pos",
    "dir_consist_15",
    # D — trend
    "dist_ema_10",
    "dist_ema_30",
    "dist_ema_60",
    "ema_slope_10",
    "ema_slope_60",
    "eff_30",
    # E — volume (relative only; no raw volume)
    "rvol_5",
    "rvol_30",
    "vol_z_60",
    "vol_chg",
    "vol_x_ret",
)

assert len(V1_STATE_IDS) == 30
