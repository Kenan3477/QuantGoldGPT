"""
MT5 historical adapter.

Preserves XAUBot's broker-connectivity idea behind a QuantGold interface.
Requires MetaTrader5 package and a running terminal — optional dependency.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from quantgold.data.ingest.base import MarketDataSource

TF_TO_MT5 = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 16385,  # TIMEFRAME_H1
    "H4": 16388,
    "D1": 16408,
}


class MT5Source(MarketDataSource):
    name = "mt5"

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.login = login
        self.password = password
        self.server = server
        self.path = path

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise ImportError(
                "MetaTrader5 is not installed. Use YFinanceSource for research "
                "or pip install quantgold[mt5] on a Windows host with MT5."
            ) from exc

        tf = TF_TO_MT5.get(timeframe.upper())
        if tf is None:
            raise ValueError(timeframe)

        initialized = mt5.initialize(path=self.path) if self.path else mt5.initialize()
        if not initialized:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        try:
            if self.login and self.password and self.server:
                if not mt5.login(self.login, password=self.password, server=self.server):
                    raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

            bars = limit or 5000
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            if rates is None:
                raise RuntimeError(f"MT5 copy_rates failed: {mt5.last_error()}")
            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume"})
            out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            if "spread" in df.columns:
                out["spread"] = df["spread"].astype(float)
            return out.sort_values("timestamp").reset_index(drop=True)
        finally:
            mt5.shutdown()
