"""Purging and embargo helpers for overlapping label horizons."""

from __future__ import annotations

import pandas as pd


def purge_embargo_mask(
    timestamps: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series,
    *,
    label_horizon_bars: int,
    embargo_bars: int,
) -> pd.Series:
    """
    Remove from the training set any rows whose label horizon overlaps the test set,
    plus an embargo buffer before test start.

    Returns a cleaned train_mask.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    cleaned = train_mask.copy()
    if not test_mask.any():
        return cleaned

    test_start = ts[test_mask].min()
    # Approximate bar spacing from median diff
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return cleaned
    bar_delta = diffs.median()
    purge_delta = bar_delta * int(label_horizon_bars)
    embargo_delta = bar_delta * int(embargo_bars)
    cutoff = test_start - purge_delta - embargo_delta
    cleaned = cleaned & (ts <= cutoff)
    return cleaned
