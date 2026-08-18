"""
Causal replacements for leaking group-count features found in XAUBot V2.

XAUBot used `.count().over(group)` which assigns the *future* group length
to every row. QuantGold uses expanding counts from the start of the run only.
"""

from __future__ import annotations

import pandas as pd


def bars_since_change(series: pd.Series) -> pd.Series:
    """
    Causal duration: number of bars since the value last changed (inclusive).

    At the first bar of a new regime/run this equals 1, and grows until change.
    Never equals the eventual full run length before the run ends.
    """
    changed = series != series.shift(1)
    changed.iloc[0] = True
    group_id = changed.cumsum()
    return series.groupby(group_id).cumcount() + 1


def causal_consecutive_direction(close: pd.Series) -> pd.Series:
    """Causal up/down streak length with sign (positive=up, negative=down)."""
    direction = (close.diff() > 0).astype(int) - (close.diff() < 0).astype(int)
    direction = direction.fillna(0).astype(int)
    # Restart groups when direction changes or flat
    restart = (direction != direction.shift(1)) | (direction == 0)
    restart.iloc[0] = True
    gid = restart.cumsum()
    length = direction.groupby(gid).cumcount() + 1
    return (length * direction).astype(int)
