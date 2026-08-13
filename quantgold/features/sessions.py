"""Session feature family (UTC-based; research parameters)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from quantgold.features.registry import FeatureRegistry


SESSION_FEATURE_NAMES = [
    "hour_utc",
    "weekday",
    "session_asia",
    "session_london",
    "session_ny",
    "session_london_ny_overlap",
    "session_london_open",
    "session_range_pct",
    "dist_session_high",
    "dist_session_low",
    "prev_session_return",
]


class SessionFeatureBuilder:
    """
    Explicit session modelling.

    Sessions (UTC approximations):
      Asia 00:00–07:00, London 07:00–16:00, NY 13:00–21:00, overlap 13:00–16:00
    """

    FEATURE_NAMES = SESSION_FEATURE_NAMES

    def __init__(self):
        self.registry = FeatureRegistry(self.FEATURE_NAMES)

    def transform(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        out = df.copy()
        ts = pd.to_datetime(out[time_col], utc=True)
        hour = ts.dt.hour
        out["hour_utc"] = hour.astype(float)
        out["weekday"] = ts.dt.weekday.astype(float)
        out["session_asia"] = ((hour >= 0) & (hour < 7)).astype(float)
        out["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)
        out["session_ny"] = ((hour >= 13) & (hour < 21)).astype(float)
        out["session_london_ny_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)
        out["session_london_open"] = ((hour >= 7) & (hour < 9)).astype(float)

        # Session date key: London day boundary at 00:00 UTC for simplicity
        session_id = ts.dt.floor("D")
        grp = out.groupby(session_id)
        sess_high = grp["high"].transform("max")
        sess_low = grp["low"].transform("min")
        sess_open = grp["open"].transform("first")
        close = out["close"].astype(float)
        out["session_range_pct"] = (sess_high - sess_low) / close.replace(0, np.nan)
        out["dist_session_high"] = (sess_high - close) / close.replace(0, np.nan)
        out["dist_session_low"] = (close - sess_low) / close.replace(0, np.nan)

        # Previous session return = prior day close/open - 1 (causal: use shift of daily)
        daily_ret = (grp["close"].transform("last") / sess_open) - 1.0
        # Map previous day's return onto today
        day_ret = daily_ret.groupby(session_id).transform("last")
        prev = day_ret.groupby(session_id).first().shift(1)
        out["prev_session_return"] = session_id.map(prev)

        FeatureRegistry.assert_no_label_leakage(self.FEATURE_NAMES)
        return out
