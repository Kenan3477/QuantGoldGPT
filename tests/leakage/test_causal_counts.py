"""Causal replacements must not equal future full-run length mid-run."""

import pandas as pd

from quantgold.features.causal_counts import bars_since_change, causal_consecutive_direction


def test_bars_since_change_is_prefix_not_full_length():
    s = pd.Series([0, 0, 0, 1, 1, 1, 1])
    dur = bars_since_change(s)
    # First regime length eventual = 3, but first bar must be 1 not 3
    assert list(dur) == [1, 2, 3, 1, 2, 3, 4]
    assert dur.iloc[0] == 1
    assert dur.iloc[3] == 1


def test_leaky_group_count_would_differ():
    """Document the XAUBot-style leak for contrast."""
    s = pd.Series([0, 0, 0, 1, 1])
    changed = s != s.shift(1)
    changed.iloc[0] = True
    gid = changed.cumsum()
    leaky = s.groupby(gid).transform("count")
    causal = bars_since_change(s)
    assert list(leaky) == [3, 3, 3, 2, 2]
    assert list(causal) == [1, 2, 3, 1, 2]
    assert leaky.iloc[0] != causal.iloc[0]


def test_causal_consecutive_direction():
    close = pd.Series([10.0, 10.5, 11.0, 10.8, 10.6, 10.6])
    streak = causal_consecutive_direction(close)
    # up, up, down, down, flat
    assert streak.iloc[1] == 1
    assert streak.iloc[2] == 2
    assert streak.iloc[3] == -1
    assert streak.iloc[4] == -2
