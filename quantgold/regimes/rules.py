"""
Simple volatility/trend rule regimes.

HMM/GMM adapters come later; rules provide a leakage-safe baseline that can be
fit (parameterised) inside each walk-forward training fold only.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"


class RuleRegimeDetector:
    def __init__(
        self,
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        high_vol_quantile: float = 0.8,
        low_vol_quantile: float = 0.2,
    ):
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.high_vol_quantile = high_vol_quantile
        self.low_vol_quantile = low_vol_quantile
        self._high_vol_threshold: Optional[float] = None
        self._low_vol_threshold: Optional[float] = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "RuleRegimeDetector":
        """Fit thresholds on TRAINING fold only."""
        vol = self._realized_vol(df)
        self._high_vol_threshold = float(vol.quantile(self.high_vol_quantile))
        self._low_vol_threshold = float(vol.quantile(self.low_vol_quantile))
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("RuleRegimeDetector.fit() must be called on the training fold first")
        vol = self._realized_vol(df)
        ret = df["close"].pct_change(self.trend_lookback)
        regimes = []
        for v, r in zip(vol, ret):
            if np.isnan(v) or np.isnan(r):
                regimes.append(Regime.TRANSITION.value)
            elif v >= self._high_vol_threshold:
                regimes.append(Regime.HIGH_VOLATILITY.value)
            elif v <= self._low_vol_threshold:
                regimes.append(Regime.LOW_VOLATILITY.value)
            elif r > 0.01:
                regimes.append(Regime.TRENDING_UP.value)
            elif r < -0.01:
                regimes.append(Regime.TRENDING_DOWN.value)
            else:
                regimes.append(Regime.RANGING.value)
        return pd.Series(regimes, index=df.index, name="regime")

    def _realized_vol(self, df: pd.DataFrame) -> pd.Series:
        lr = np.log(df["close"].astype(float)).diff()
        return lr.rolling(self.vol_lookback, min_periods=self.vol_lookback).std()
