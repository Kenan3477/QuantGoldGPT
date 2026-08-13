import numpy as np
import pandas as pd

from quantgold.config.settings import TripleBarrierConfig
from quantgold.labels.triple_barrier import (
    LABEL_AMBIGUOUS,
    LABEL_DOWN,
    LABEL_TIMEOUT,
    LABEL_UP,
    TripleBarrierLabeler,
)


def _ohlc_from_close(closes):
    c = pd.Series(closes, dtype=float)
    # Wide range so barriers can be touched when we want
    return pd.DataFrame(
        {
            "open": c,
            "high": c + 2.0,
            "low": c - 2.0,
            "close": c,
        }
    )


def test_same_bar_policy_ambiguous_default():
    # Construct path where a future bar touches both sides relative to tight ATR barriers
    closes = [100.0] * 30 + [100.0, 100.0]
    df = _ohlc_from_close(closes)
    # Force ATR ~ 1 by crafting TR; simpler: monkeypatch via config large multipliers? 
    # Instead set high/low on bar 31 extreme both ways already (+2/-2).
    cfg = TripleBarrierConfig(
        upper_atr_mult=0.5,
        lower_atr_mult=0.5,
        max_holding_bars=5,
        atr_period=5,
        same_bar_policy="ambiguous",
    )
    # Make ATR small: flatten ranges then one wide bar
    df.loc[:, "high"] = df["close"] + 0.1
    df.loc[:, "low"] = df["close"] - 0.1
    df.loc[31, "high"] = 110.0
    df.loc[31, "low"] = 90.0
    result = TripleBarrierLabeler(cfg).label(df)
    # Index 30 decides using path from 31
    assert result.labels.iloc[30] == LABEL_AMBIGUOUS


def test_timeout_when_no_touch():
    closes = np.linspace(100, 100.2, 40)
    df = _ohlc_from_close(closes)
    df["high"] = df["close"] + 0.01
    df["low"] = df["close"] - 0.01
    cfg = TripleBarrierConfig(
        upper_atr_mult=5.0,
        lower_atr_mult=5.0,
        max_holding_bars=3,
        atr_period=5,
    )
    result = TripleBarrierLabeler(cfg).label(df)
    # Mid-series should time out
    assert result.labels.iloc[20] == LABEL_TIMEOUT


def test_up_barrier_first():
    closes = [100.0] * 20 + [100.0, 101.0, 102.0]
    df = _ohlc_from_close(closes)
    df["high"] = df["close"] + 0.05
    df["low"] = df["close"] - 0.05
    df.loc[21, "high"] = 103.0  # strong upside touch
    cfg = TripleBarrierConfig(
        upper_atr_mult=1.0,
        lower_atr_mult=1.0,
        max_holding_bars=5,
        atr_period=5,
    )
    # ATR roughly 0.1 → upper ~ 100.1 from close 100
    result = TripleBarrierLabeler(cfg).label(df)
    assert result.labels.iloc[20] == LABEL_UP
