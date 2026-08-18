"""
Causal market-structure features.

SMC terminology is NOT assumed predictive. Features are confirmation-time safe:
swing points are only marked after the right-side confirmation bar closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantgold.features.registry import FeatureRegistry

STRUCTURE_FEATURE_NAMES = [
    "swing_high_dist",
    "swing_low_dist",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "bos_bull_recent",
    "bos_bear_recent",
    "prev_day_high_dist",
    "prev_day_low_dist",
    "rolling_high_20_dist",
    "rolling_low_20_dist",
]


class StructureFeatureBuilder:
    FEATURE_NAMES = STRUCTURE_FEATURE_NAMES

    def __init__(self, swing_left: int = 3, swing_right: int = 3):
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.registry = FeatureRegistry(self.FEATURE_NAMES)

    def transform(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        out = df.copy()
        high = out["high"].astype(float).to_numpy()
        low = out["low"].astype(float).to_numpy()
        close = out["close"].astype(float)
        n = len(out)

        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)
        L, R = self.swing_left, self.swing_right
        # Confirm swing at bar i+R (no lookahead into unconfirmed future beyond R,
        # and we assign at confirmation index i+R, not at center i).
        conf_sh = np.zeros(n)
        conf_sl = np.zeros(n)
        for i in range(L, n - R):
            window_h = high[i - L : i + R + 1]
            window_l = low[i - L : i + R + 1]
            if high[i] == np.max(window_h):
                conf_sh[i + R] = 1.0
                swing_high[i + R] = high[i]
            if low[i] == np.min(window_l):
                conf_sl[i + R] = 1.0
                swing_low[i + R] = low[i]

        last_sh = pd.Series(swing_high).ffill()
        last_sl = pd.Series(swing_low).ffill()
        out["swing_high_dist"] = (last_sh.to_numpy() - close.to_numpy()) / close.replace(0, np.nan)
        out["swing_low_dist"] = (close.to_numpy() - last_sl.to_numpy()) / close.replace(0, np.nan)

        out["bars_since_swing_high"] = self._bars_since(conf_sh)
        out["bars_since_swing_low"] = self._bars_since(conf_sl)

        # BOS: close beyond last confirmed swing (uses only past confirmed swings)
        bos_bull = (close.to_numpy() > last_sh.to_numpy()).astype(float)
        bos_bear = (close.to_numpy() < last_sl.to_numpy()).astype(float)
        # recent = within last 5 bars
        out["bos_bull_recent"] = pd.Series(bos_bull).rolling(5, min_periods=1).max()
        out["bos_bear_recent"] = pd.Series(bos_bear).rolling(5, min_periods=1).max()

        ts = pd.to_datetime(out[time_col], utc=True)
        day = ts.dt.floor("D")
        day_high = out.groupby(day)["high"].transform("max")
        day_low = out.groupby(day)["low"].transform("min")
        # previous day extremes: shift daily aggregates
        daily = out.groupby(day).agg(h=("high", "max"), l=("low", "min"))
        prev_h = day.map(daily["h"].shift(1))
        prev_l = day.map(daily["l"].shift(1))
        out["prev_day_high_dist"] = (prev_h.to_numpy() - close.to_numpy()) / close.replace(0, np.nan)
        out["prev_day_low_dist"] = (close.to_numpy() - prev_l.to_numpy()) / close.replace(0, np.nan)

        roll_h = out["high"].rolling(20, min_periods=5).max()
        roll_l = out["low"].rolling(20, min_periods=5).min()
        out["rolling_high_20_dist"] = (roll_h - close) / close.replace(0, np.nan)
        out["rolling_low_20_dist"] = (close - roll_l) / close.replace(0, np.nan)

        FeatureRegistry.assert_no_label_leakage(self.FEATURE_NAMES)
        return out

    @staticmethod
    def _bars_since(flags: np.ndarray) -> np.ndarray:
        out = np.zeros(len(flags), dtype=float)
        last = -10_000
        for i, f in enumerate(flags):
            if f > 0:
                last = i
            out[i] = i - last if last >= 0 else np.nan
        return out
