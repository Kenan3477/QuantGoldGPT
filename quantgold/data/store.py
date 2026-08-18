"""Parquet-backed canonical dataset store (M1 scaffold)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from quantgold.data.schema import OHLCV_COLUMNS
from quantgold.data.timestamps import bar_available_timestamp


class CanonicalDataStore:
    """Read/write canonical OHLCV datasets with availability timestamps."""

    def __init__(self, root: str | Path = "artifacts/datasets"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol.upper() / f"{timeframe.upper()}.parquet"

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        timeframe_minutes: int,
    ) -> Path:
        frame = df.copy()
        if "timestamp" not in frame.columns:
            raise KeyError("DataFrame must include 'timestamp'")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if "available_timestamp" not in frame.columns:
            frame["available_timestamp"] = frame["timestamp"].map(
                lambda ts: bar_available_timestamp(ts, timeframe_minutes)
            )
        frame["symbol"] = symbol.upper()
        frame["timeframe"] = timeframe.upper()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in frame.columns:
                raise KeyError(f"Missing required column '{col}'")
        if "spread" not in frame.columns:
            frame["spread"] = pd.NA

        ordered = [c for c in OHLCV_COLUMNS if c in frame.columns]
        frame = frame[ordered].sort_values("timestamp").reset_index(drop=True)

        out = self.path_for(symbol, timeframe)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        return out

    def load_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["available_timestamp"] = pd.to_datetime(df["available_timestamp"], utc=True)
        return df

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self.path_for(symbol, timeframe).exists()
