"""Time-ordered splits with purge for overlapping horizon labels."""

from __future__ import annotations

import numpy as np


def time_holdout_cut(n: int, holdout_frac: float = 0.2) -> int:
    """
    Return holdout start index for a chronological split on sorted rows.

    Train candidates = [0, cut), holdout = [cut, n).
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    cut = int(n * (1.0 - holdout_frac))
    if cut < 1 or cut >= n:
        raise ValueError(
            f"holdout_frac={holdout_frac} leaves empty train or holdout for n={n}"
        )
    return cut


def time_holdout_indices(
    n: int,
    holdout_frac: float = 0.2,
    purge: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Chronological train/holdout index arrays with purge gap.

    Rows in ``[cut - purge, cut)`` are dropped so train labels with horizon
    up to ``purge`` do not overlap the holdout window.
    """
    cut = time_holdout_cut(n, holdout_frac)
    purge = int(purge)
    if purge < 0:
        raise ValueError("purge must be >= 0")
    train_end = cut - purge
    if train_end < 1:
        raise ValueError(
            f"purge={purge} leaves empty train for n={n}, cut={cut}"
        )
    train_idx = np.arange(0, train_end, dtype=int)
    holdout_idx = np.arange(cut, n, dtype=int)
    if len(holdout_idx) < 1:
        raise ValueError("empty holdout")
    return train_idx, holdout_idx, cut
