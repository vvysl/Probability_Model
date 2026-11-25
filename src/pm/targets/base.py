"""TargetSpec contract — horizons + outcome mapping."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import pandas as pd


@runtime_checkable
class TargetSpec(Protocol):
    """Defines y_{t,h} for one or more horizons."""

    @property
    def id(self) -> str:
        """Stable registry id."""

    @property
    def horizons(self) -> Sequence[int]:
        """Horizon lengths in bars (step 2 will set concrete values)."""

    def compute(self, bars: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Labels aligned to bars index; tail of each horizon is NaN.

        Must not leak future into features — labels *are* future by design;
        callers must drop NaN and apply purge/embargo in eval.
        """
