"""Abstract market data source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class MarketDataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return DataFrame with columns:
        timestamp, open, high, low, close, volume [, spread]
        Timestamps are bar open times in UTC.
        """
        raise NotImplementedError
