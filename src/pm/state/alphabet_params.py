"""A priori alphabet windows (Quanqora alphabet/params.py) — bar counts, not tuned."""

from __future__ import annotations

ATR_N = 14
W_YZ = 48
K_SHORT = 12
K_LONG = 48
H_RET = 6
W_SKEW = 48
# Alphabet used 1440 on 5m (~5 sessions). On 1m that is only ~1 day; keep 1440
# as a priori day-scale window (exact CDF is chunked in compute).
W_VOL_PCT = 1440
N_MA = 20
W_SESSION_VOL_HIST = 40
