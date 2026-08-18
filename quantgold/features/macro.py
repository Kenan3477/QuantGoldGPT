"""
Macro-event proximity features.

Event calendar is injected by the caller (CSV/API). Until a verified calendar
is provided, features remain NaN / blocked-state optional.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantgold.features.registry import FeatureRegistry

MACRO_FEATURE_NAMES = [
    "minutes_to_event",
    "minutes_since_event",
    "event_importance",
    "pre_event_window",
    "post_event_window",
    "event_block",
]


class MacroEventFeatureBuilder:
    FEATURE_NAMES = MACRO_FEATURE_NAMES

    def __init__(self, block_before_min: int = 30, block_after_min: int = 15):
        self.block_before_min = block_before_min
        self.block_after_min = block_after_min
        self.registry = FeatureRegistry(self.FEATURE_NAMES)

    def transform(
        self,
        df: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        *,
        time_col: str = "available_timestamp",
    ) -> pd.DataFrame:
        """
        events columns: event_time (UTC), importance (1-3), event_type (str)
        """
        out = df.copy()
        t = pd.to_datetime(out[time_col], utc=True)
        n = len(out)
        if events is None or events.empty:
            for c in self.FEATURE_NAMES:
                out[c] = np.nan if c != "event_block" else 0.0
            out["event_block"] = 0.0
            return out

        ev = events.copy()
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        ev = ev.sort_values("event_time")
        minutes_to = np.full(n, np.nan)
        minutes_since = np.full(n, np.nan)
        importance = np.full(n, np.nan)
        pre = np.zeros(n)
        post = np.zeros(n)
        block = np.zeros(n)

        ev_times = ev["event_time"].to_numpy()
        ev_imp = ev.get("importance", pd.Series(np.ones(len(ev)))).to_numpy()

        for i, ti in enumerate(t):
            # next event
            future = ev_times[ev_times >= ti.to_datetime64()]
            past = ev_times[ev_times <= ti.to_datetime64()]
            if len(future):
                delta_min = (pd.Timestamp(future[0], tz="UTC") - ti).total_seconds() / 60.0
                minutes_to[i] = delta_min
                idx = int(np.where(ev_times == future[0])[0][0])
                importance[i] = ev_imp[idx]
                if 0 <= delta_min <= self.block_before_min:
                    pre[i] = 1.0
                    if ev_imp[idx] >= 3:
                        block[i] = 1.0
            if len(past):
                delta_min = (ti - pd.Timestamp(past[-1], tz="UTC")).total_seconds() / 60.0
                minutes_since[i] = delta_min
                if 0 <= delta_min <= self.block_after_min:
                    post[i] = 1.0
                    idx = int(np.where(ev_times == past[-1])[0][0])
                    if ev_imp[idx] >= 3:
                        block[i] = 1.0

        out["minutes_to_event"] = minutes_to
        out["minutes_since_event"] = minutes_since
        out["event_importance"] = importance
        out["pre_event_window"] = pre
        out["post_event_window"] = post
        out["event_block"] = block
        FeatureRegistry.assert_no_label_leakage(self.FEATURE_NAMES)
        return out
