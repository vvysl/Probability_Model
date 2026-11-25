"""Time-ordered splits. Purge/embargo added when overlapping labels exist."""

from __future__ import annotations


def time_holdout_cut(n: int, holdout_frac: float = 0.2) -> int:
    """
    Return train end index for a chronological holdout on sorted rows.

    Train = [0, cut), holdout = [cut, n).
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    cut = int(n * (1.0 - holdout_frac))
    if cut < 1 or cut >= n:
        raise ValueError(f"holdout_frac={holdout_frac} leaves empty train or holdout for n={n}")
    return cut
