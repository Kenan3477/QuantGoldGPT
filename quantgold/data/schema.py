"""Canonical bar schema for QuantGold datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


OHLCV_COLUMNS = (
    "timestamp",          # bar open time (UTC)
    "available_timestamp",  # earliest time feature/bar may be used (typically bar close)
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spread",
    "symbol",
    "timeframe",
)


@dataclass(frozen=True)
class CanonicalBar:
    timestamp: datetime
    available_timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str
    spread: Optional[float] = None
