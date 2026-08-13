"""
Free data source: FRED (Federal Reserve Economic Data).

FRED provides free U.S. macroeconomic and financial data.
Website: https://fred.stlouisfed.org/

Free API:
- Unlimited requests (with rate limiting)
- Historical economic data (CPI, GDP, unemployment, etc.)
- Financial data (yields, rates, indices)

Get free API key: https://fred.stlouisfed.org/docs/api/api_key.html

Set environment variable:
    export FRED_API_KEY="your_key_here"

Useful series for gold trading:
- DGS10: 10-Year Treasury Constant Maturity Rate
- DGS2: 2-Year Treasury Constant Maturity Rate
- DEXCHUS: China / U.S. Foreign Exchange Rate
- VIXCLS: CBOE Volatility Index
- DCOILWTICO: Crude Oil Prices (WTI)
- GOLDAMGBD228NLBM: Gold Fixing Price (London, PM)
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Literal
import requests
import polars as pl
import pandas as pd

from quantgold.data.ingest.base import MarketDataSource


class FredSource(MarketDataSource):
    """
    Fetch macroeconomic data from FRED API.
    
    Example:
        source = FredSource(api_key="your_key")
        df = source.fetch_series("DGS10", start, end)  # 10-year yield
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # Common series for gold trading
    SERIES_MAP = {
        "US10Y": "DGS10",  # 10-Year Treasury Yield
        "US2Y": "DGS2",    # 2-Year Treasury Yield
        "VIX": "VIXCLS",   # VIX (CBOE Volatility Index)
        "OIL": "DCOILWTICO",  # WTI Crude Oil
        "GOLD": "GOLDAMGBD228NLBM",  # Gold London PM Fix
        "CPI": "CPIAUCSL",  # Consumer Price Index
        "UNEMPLOYMENT": "UNRATE",  # Unemployment Rate
        "GDP": "GDP",  # Gross Domestic Product
    }
    
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        """
        Initialize FRED source.
        
        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
            cache_dir: Directory to cache API responses
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key required. "
                "Get free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Set via FRED_API_KEY env var or pass to constructor."
            )
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/fred")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch data for a symbol (mapped to FRED series).
        
        Args:
            symbol: Symbol like "US10Y", "VIX", etc.
            timeframe: Ignored (FRED data is daily or lower frequency)
            start: Start date
            end: End date
            
        Returns:
            Polars DataFrame with timestamp and close columns
        """
        series_id = self.SERIES_MAP.get(symbol, symbol)
        return self.fetch_series(series_id, start, end)
    
    def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch a FRED series by ID.
        
        Args:
            series_id: FRED series ID (e.g., "DGS10", "VIXCLS")
            start: Start date
            end: End date
            
        Returns:
            Polars DataFrame with timestamp, close, and available_timestamp
        """
        # Check cache
        cache_file = self.cache_dir / f"{series_id}_{start.date()}_{end.date()}.parquet"
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            return pl.read_parquet(cache_file)
        
        # API request
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        
        print(f"Fetching FRED series {series_id}...")
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for errors
        if "error_code" in data:
            raise ValueError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        observations = data.get("observations", [])
        if not observations:
            raise ValueError(f"No data returned for series {series_id}")
        
        # Convert to DataFrame
        records = []
        for obs in observations:
            # Skip missing values
            if obs["value"] == ".":
                continue
            
            records.append({
                "timestamp": pd.to_datetime(obs["date"]).tz_localize("UTC"),
                "close": float(obs["value"]),
            })
        
        if not records:
            raise ValueError(f"No valid data for series {series_id}")
        
        df_pd = pd.DataFrame(records)
        df_pd = df_pd.sort_values("timestamp")
        
        # Convert to Polars
        df = pl.from_pandas(df_pd)
        
        # Add available_timestamp (next day for daily data)
        df = df.with_columns(
            (pl.col("timestamp") + pl.duration(days=1))
            .alias("available_timestamp")
        )
        
        # Cache result
        df.write_parquet(cache_file)
        print(f"Cached to: {cache_file}")
        
        return df
    
    def fetch_multiple_series(
        self,
        series_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch multiple FRED series.
        
        Args:
            series_ids: List of FRED series IDs
            start: Start date
            end: End date
            
        Returns:
            Dict mapping series_id → DataFrame
        """
        result = {}
        for series_id in series_ids:
            try:
                result[series_id] = self.fetch_series(series_id, start, end)
            except Exception as e:
                print(f"Warning: Failed to fetch {series_id}: {e}")
        return result
