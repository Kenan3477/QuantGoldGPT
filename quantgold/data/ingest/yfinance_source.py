"""
YFinance ingest adapter for research datasets.

Maps QuantGold symbols to Yahoo tickers. This is a research fallback when MT5
history is unavailable — spreads are approximate and must not be treated as
broker-accurate execution data.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from quantgold.data.ingest.base import MarketDataSource

SYMBOL_MAP: Dict[str, str] = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "US2Y": "^IRX",
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "NDX": "^IXIC",
    "WTI": "CL=F",
    "COPPER": "HG=F",
}

TF_MAP: Dict[str, str] = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "H1": "1h",
    "H4": "1h",  # resampled
    "D1": "1d",
}


class YFinanceSource(MarketDataSource):
    name = "yfinance"

    def __init__(self, symbol_map: Optional[Dict[str, str]] = None):
        self.symbol_map = symbol_map or SYMBOL_MAP

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        ticker = self.symbol_map.get(symbol.upper(), symbol)
        tf = timeframe.upper()
        interval = TF_MAP.get(tf)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Yahoo limits intraday history; choose sensible defaults.
        if start or end:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval if tf != "H4" else "1h",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        else:
            period = {
                "M1": "7d",
                "M5": "60d",
                "M15": "60d",
                "H1": "730d",
                "H4": "730d",
                "D1": "max",
            }[tf]
            raw = yf.download(
                ticker,
                period=period,
                interval=interval if tf != "H4" else "1h",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

        if raw is None or raw.empty:
            raise RuntimeError(f"No yfinance data for {symbol} ({ticker}) {timeframe}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]

        df = raw.reset_index()
        # datetime column name varies
        time_col = "datetime" if "datetime" in df.columns else "date" if "date" in df.columns else df.columns[0]
        df = df.rename(columns={time_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise KeyError(f"Missing {col} in yfinance frame for {symbol}")

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
        if tf == "H4":
            df = self._resample_h4(df)

        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        if limit:
            df = df.tail(limit)
        # Approximate research spread placeholders (not broker quotes)
        mid = df["close"].astype(float)
        df["spread"] = (mid * 0.00005).clip(lower=0.01)
        return df.reset_index(drop=True)

    @staticmethod
    def _resample_h4(df: pd.DataFrame) -> pd.DataFrame:
        x = df.set_index("timestamp")
        ohlc = x.resample("4h").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        return ohlc.reset_index()
