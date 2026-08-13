"""
Free data source: Alpha Vantage API.

Alpha Vantage provides free forex, crypto, and stock data.
Website: https://www.alphavantage.co/

Free tier:
- 500 API requests per day
- 5 API requests per minute
- Intraday data (1min, 5min, 15min, 30min, 60min)
- Daily, weekly, monthly data

Sign up for free API key: https://www.alphavantage.co/support/#api-key

Set environment variable:
    export ALPHAVANTAGE_API_KEY="your_key_here"

Limitations:
- Intraday data: Last 1-2 months only (free tier)
- Extended history requires premium
- Rate limits: 5 calls/min, 500/day
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal
import requests
import polars as pl
import pandas as pd

from quantgold.data.ingest.base import MarketDataSource


class AlphaVantageSource(MarketDataSource):
    """
    Fetch data from Alpha Vantage free API.
    
    Example:
        source = AlphaVantageSource(api_key="your_key")
        df = source.fetch("XAUUSD", "M15", start, end)
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Symbol mapping: our symbols → Alpha Vantage symbols
    SYMBOL_MAP = {
        "XAUUSD": "XAU/USD",  # Gold
        "XAGUSD": "XAG/USD",  # Silver
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
        "USDJPY": "USD/JPY",
    }
    
    # Timeframe mapping: our TF → Alpha Vantage interval
    TIMEFRAME_MAP = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "60min",
        "D1": "daily",
    }
    
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        """
        Initialize Alpha Vantage source.
        
        Args:
            api_key: Alpha Vantage API key (or set ALPHAVANTAGE_API_KEY env var)
            cache_dir: Directory to cache API responses (avoid re-fetching)
        """
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. "
                "Get free key: https://www.alphavantage.co/support/#api-key\n"
                "Set via ALPHAVANTAGE_API_KEY env var or pass to constructor."
            )
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/alphavantage")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting: 5 calls/min
        self._last_call_time = 0
        self._min_call_interval = 12.0  # seconds between calls (5 calls/min = 12s interval)
    
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data from Alpha Vantage.
        
        Args:
            symbol: Instrument symbol (e.g., "XAUUSD")
            timeframe: Timeframe (e.g., "M1", "M5", "M15", "H1", "D1")
            start: Start date
            end: End date
            
        Returns:
            Polars DataFrame in canonical format
            
        Note:
            Free tier only provides last 1-2 months of intraday data.
            For longer history, use daily data or premium tier.
        """
        # Map symbol and timeframe
        av_symbol = self.SYMBOL_MAP.get(symbol)
        if not av_symbol:
            raise ValueError(f"Symbol {symbol} not supported by Alpha Vantage")
        
        av_interval = self.TIMEFRAME_MAP.get(timeframe)
        if not av_interval:
            raise ValueError(f"Timeframe {timeframe} not supported")
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_{timeframe}_{start.date()}_{end.date()}.parquet"
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            return pl.read_parquet(cache_file)
        
        # Determine function based on timeframe
        if timeframe in ["M1", "M5", "M15", "M30", "H1"]:
            function = "FX_INTRADAY"
            outputsize = "full"  # Get all available data (limited to ~1-2 months on free tier)
        else:
            function = "FX_DAILY"
            outputsize = "full"
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        # API request
        params = {
            "function": function,
            "from_symbol": av_symbol.split("/")[0],
            "to_symbol": av_symbol.split("/")[1],
            "interval": av_interval if function == "FX_INTRADAY" else None,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json",
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        print(f"Fetching {symbol} {timeframe} from Alpha Vantage...")
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
        
        # Parse response
        if function == "FX_INTRADAY":
            time_series_key = f"Time Series FX (Intraday)"
        else:
            time_series_key = "Time Series FX (Daily)"
        
        time_series = data.get(time_series_key, {})
        if not time_series:
            raise ValueError(f"No data returned for {symbol} {timeframe}")
        
        # Convert to DataFrame
        records = []
        for timestamp_str, values in time_series.items():
            records.append({
                "timestamp": pd.to_datetime(timestamp_str).tz_localize("UTC"),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": 0,  # Forex doesn't have volume
            })
        
        df_pd = pd.DataFrame(records)
        df_pd = df_pd.sort_values("timestamp")
        
        # Filter date range
        df_pd = df_pd[(df_pd["timestamp"] >= start) & (df_pd["timestamp"] < end)]
        
        # Convert to Polars
        df = pl.from_pandas(df_pd)
        
        # Add available_timestamp
        timeframe_seconds = self._parse_timeframe_seconds(timeframe)
        df = df.with_columns(
            (pl.col("timestamp") + pl.duration(seconds=timeframe_seconds))
            .alias("available_timestamp")
        )
        
        # Cache result
        df.write_parquet(cache_file)
        print(f"Cached to: {cache_file}")
        
        return df
    
    def _wait_for_rate_limit(self):
        """Enforce 5 calls/min rate limit."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_call_interval:
            wait_time = self._min_call_interval - elapsed
            print(f"Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self._last_call_time = time.time()
    
    def _parse_timeframe_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        mapping = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "D1": 86400,
        }
        return mapping.get(timeframe, 86400)
