"""Causal base price/return/volatility features (M0/M1 baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from quantgold.features.registry import FeatureRegistry


@dataclass
class FeatureMatrix:
    frame: pd.DataFrame
    feature_columns: List[str]
    prediction_timestamp_col: str = "available_timestamp"


class BaseFeatureBuilder:
    """
    Leakage-safe baseline features.

    All rolling stats use only past/current completed bar information.
    Higher-timeframe joins must go through quantgold.data.timestamps.align_higher_timeframe.
    """

    FEATURE_NAMES = [
        "log_return_1",
        "log_return_5",
        "log_return_20",
        "realized_vol_20",
        "atr_14",
        "atr_pct_14",
        "range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "dist_from_sma_20",
        "momentum_10",
        "vol_adjusted_momentum_10",
    ]

    def __init__(self):
        self.registry = FeatureRegistry(self.FEATURE_NAMES)

    def transform(self, df: pd.DataFrame) -> FeatureMatrix:
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        open_ = out["open"].astype(float)

        log_close = np.log(close.replace(0, np.nan))
        out["log_return_1"] = log_close.diff(1)
        out["log_return_5"] = log_close.diff(5)
        out["log_return_20"] = log_close.diff(20)
        out["realized_vol_20"] = out["log_return_1"].rolling(20, min_periods=20).std()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr_14"] = tr.rolling(14, min_periods=14).mean()
        out["atr_pct_14"] = out["atr_14"] / close

        bar_range = (high - low).replace(0, np.nan)
        out["range_pct"] = bar_range / close
        body = (close - open_).abs()
        out["body_pct"] = body / bar_range
        out["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
        out["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range

        sma20 = close.rolling(20, min_periods=20).mean()
        out["dist_from_sma_20"] = (close - sma20) / close
        out["momentum_10"] = close.pct_change(10)
        out["vol_adjusted_momentum_10"] = out["momentum_10"] / out["realized_vol_20"].replace(0, np.nan)

        FeatureRegistry.assert_no_label_leakage(self.FEATURE_NAMES)
        return FeatureMatrix(frame=out, feature_columns=list(self.FEATURE_NAMES))
