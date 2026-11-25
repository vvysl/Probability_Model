"""StateVar contract — continuous, causal features."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StateVar(Protocol):
    """One continuous market constant (or a small aligned group)."""

    @property
    def id(self) -> str:
        """Stable registry id."""

    def compute(self, bars: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Compute state aligned to ``bars.index``.

        Must use only information available at or before each t
        (warmup may yield NaN at the head).
        """
