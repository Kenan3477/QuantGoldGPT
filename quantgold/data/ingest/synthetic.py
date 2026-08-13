"""Deterministic synthetic OHLCV for offline unit/integration tests."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantgold.data.ingest.base import MarketDataSource

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


class SyntheticSource(MarketDataSource):
    name = "synthetic"

    def __init__(self, seed: int = 42, start_price: float = 2000.0):
        self.seed = seed
        self.start_price = start_price

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        n = limit or 2000
        minutes = TF_MINUTES[timeframe.upper()]
        start_ts = pd.Timestamp(start or "2019-01-01", tz="UTC")
        idx = pd.date_range(start_ts, periods=n, freq=f"{minutes}min")
        rng = np.random.RandomState(self.seed + sum(ord(c) for c in symbol))
        # Slight symbol-specific drift
        drift = 0.00002 if "XAU" in symbol.upper() else 0.00003
        rets = rng.normal(drift, 0.0015, size=n)
        close = self.start_price * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0001, 0.001, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.001, n))
        volume = rng.randint(100, 1000, size=n).astype(float)
        return pd.DataFrame(
            {
                "timestamp": idx,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "spread": np.full(n, 0.2 if "XAU" in symbol.upper() else 0.02),
            }
        )
