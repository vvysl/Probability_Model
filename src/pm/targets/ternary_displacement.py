"""
Ternary displacement target on fixed 1m horizons.

Movement := close-to-close displacement in causal ATR units, not path length.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from pm.targets.scale import atr

# Encoding for classifiers / predict_proba layouts later.
LABEL_DOWN = -1
LABEL_FLAT = 0
LABEL_UP = 1

DEFAULT_HORIZONS: tuple[int, ...] = (3, 10, 30)
DEFAULT_ATR_PERIOD = 14
# Realizable displacement thresholds (ATR14 units), per horizon.
DEFAULT_FLAT_TAU_BY_H: dict[int, float] = {3: 0.75, 10: 1.25, 30: 2.0}
# Backward-compatible scalar default if a lone float is passed.
DEFAULT_FLAT_TAU = 0.75

FlatTauSpec = Union[float, Mapping[int, float], Mapping[str, float]]


def _normalize_tau_map(
    horizons: Sequence[int], flat_tau: FlatTauSpec
) -> dict[int, float]:
    if isinstance(flat_tau, (int, float)):
        tau = float(flat_tau)
        if tau < 0:
            raise ValueError("flat_tau must be >= 0")
        return {int(h): tau for h in horizons}

    out: dict[int, float] = {}
    for h in horizons:
        if h in flat_tau:
            raw = flat_tau[h]  # type: ignore[index]
        elif str(h) in flat_tau:
            raw = flat_tau[str(h)]  # type: ignore[index]
        else:
            raise KeyError(f"flat_tau missing horizon {h}; got keys {list(flat_tau)}")
        t = float(raw)
        if t < 0:
            raise ValueError(f"flat_tau[{h}] must be >= 0")
        out[int(h)] = t
    return out


class TernaryDisplacementTarget:
    """
    For each horizon h (bars on 1m):

        z_h = (close_{t+h} - close_t) / ATR_t
        y   = +1 if z_h > tau_h, -1 if z_h < -tau_h, else 0

    ``tau_h`` may differ by horizon (default 0.75 / 1.25 / 2.0 for 3/10/30).
    Tail of length h is NaN; ATR warmup → label NaN.
    """

    def __init__(
        self,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        atr_period: int = DEFAULT_ATR_PERIOD,
        flat_tau: FlatTauSpec | None = None,
        target_id: str = "ternary_displacement",
    ) -> None:
        hs = tuple(int(h) for h in horizons)
        if not hs or any(h < 1 for h in hs):
            raise ValueError("horizons must be positive ints")
        self._id = target_id
        self._horizons = hs
        self.atr_period = int(atr_period)
        if flat_tau is None:
            # Prefer per-h defaults when horizons match; else scalar fallback.
            if set(hs) <= set(DEFAULT_FLAT_TAU_BY_H):
                flat_tau = {h: DEFAULT_FLAT_TAU_BY_H[h] for h in hs}
            else:
                flat_tau = DEFAULT_FLAT_TAU
        self.flat_tau_by_h = _normalize_tau_map(hs, flat_tau)

    @property
    def id(self) -> str:
        return self._id

    @property
    def horizons(self) -> Sequence[int]:
        return self._horizons

    def column_name(self, h: int) -> str:
        return f"y_h{h}"

    def z_column_name(self, h: int) -> str:
        return f"z_h{h}"

    def tau(self, h: int) -> float:
        return self.flat_tau_by_h[int(h)]

    def compute_z(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Continuous displacement z_h in ATR units (for eval / diagnostics)."""
        if "close" not in bars.columns:
            raise ValueError("bars must include 'close'")
        close = bars["close"].astype(float)
        scale = atr(bars, period=self.atr_period).where(lambda s: s > 0)
        cols: dict[str, pd.Series] = {}
        for h in self._horizons:
            z = (close.shift(-h) - close) / scale
            cols[self.z_column_name(h)] = z
        return pd.DataFrame(cols, index=bars.index)

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        z_frame = self.compute_z(bars)
        cols: dict[str, pd.Series] = {}
        for h in self._horizons:
            z = z_frame[self.z_column_name(h)]
            tau = self.tau(h)
            y = pd.Series(np.nan, index=bars.index, dtype=float)
            valid = z.notna()
            y.loc[valid & (z > tau)] = LABEL_UP
            y.loc[valid & (z < -tau)] = LABEL_DOWN
            y.loc[valid & (z >= -tau) & (z <= tau)] = LABEL_FLAT
            cols[self.column_name(h)] = y
        return pd.DataFrame(cols, index=bars.index)
