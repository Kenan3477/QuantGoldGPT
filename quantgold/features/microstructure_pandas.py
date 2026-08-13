"""
Pandas-compatible microstructure features for intraday gold/silver trading.

Integrates with the existing FeatureBundle (pandas-based) system.
All features are CAUSAL (no lookahead).
"""

import pandas as pd
import numpy as np
from typing import List


class MicrostructureFeatureBuilder:
    """
    Build microstructure features for M5/M15 intraday data (pandas version).
    
    Features:
    - Spread proxy (high-low / close)
    - Intraday range percentile
    - Opening range breakout (first 30min of session)
    - Distance from day open
    - Bar body/wick ratios
    - Volume ratios (if available)
    """
    
    FEATURE_NAMES: List[str] = [
        "spread",
        "spread_vs_ma",
        "range_vs_ma",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_vs_ma",
        "dist_from_day_open",
        "dist_from_day_open_pct",
        "above_opening_range_high",
        "below_opening_range_low",
        "dist_to_opening_range_high",
        "dist_to_opening_range_low",
        "continues_prev_direction",
    ]
    
    def transform(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Add microstructure features to OHLCV dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV + timestamp
            lookback: Lookback period for moving averages/percentiles
            
        Returns:
            DataFrame with additional feature columns
        """
        # Work on a copy
        df = df.copy()
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Spread proxy (no bid/ask available in free data)
        # Use (high - low) / close as spread proxy
        df['spread'] = (df['high'] - df['low']) / df['close']
        
        # Spread percentile vs. recent history
        df['spread_vs_ma'] = (
            df['spread'] / df['spread'].rolling(window=lookback, min_periods=1).mean()
        ).fillna(1.0)
        
        # 2. Intraday range features
        bar_range = df['high'] - df['low']
        
        # Range percentile (current range / avg range)
        df['range_vs_ma'] = (
            bar_range / bar_range.rolling(window=lookback, min_periods=1).mean()
        ).fillna(1.0)
        
        # 3. Bar body and wick features
        bar_body = df['close'] - df['open']
        
        # Upper wick (depends on bar direction)
        upper_wick = np.where(
            df['close'] >= df['open'],
            df['high'] - df['close'],  # Bullish bar
            df['high'] - df['open']     # Bearish bar
        )
        
        # Lower wick (depends on bar direction)
        lower_wick = np.where(
            df['close'] >= df['open'],
            df['open'] - df['low'],     # Bullish bar
            df['close'] - df['low']     # Bearish bar
        )
        
        # Body size as % of total range
        df['body_pct'] = (
            np.abs(bar_body) / bar_range
        ).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # Wick ratios
        df['upper_wick_pct'] = (
            upper_wick / bar_range
        ).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        df['lower_wick_pct'] = (
            lower_wick / bar_range
        ).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # 4. Volume features (if available, else skip)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_vs_ma'] = (
                df['volume'] / df['volume'].rolling(window=lookback, min_periods=1).mean()
            ).fillna(1.0)
        else:
            df['volume_vs_ma'] = 1.0
        
        # 5. Opening range features (session-based)
        # Detect start of trading day (00:00 UTC for 24h markets)
        df['trade_date'] = df['timestamp'].dt.date
        
        # Get daily open (first bar of each day)
        daily_open = df.groupby('trade_date').agg({
            'timestamp': 'min',
            'open': 'first'
        }).rename(columns={'timestamp': 'day_start_time', 'open': 'day_open'})
        
        # Merge back
        df = df.merge(daily_open, on='trade_date', how='left')
        
        # Distance from day open
        df['dist_from_day_open'] = df['close'] - df['day_open']
        df['dist_from_day_open_pct'] = (
            (df['close'] - df['day_open']) / df['day_open']
        )
        
        # 6. Opening range breakout (first 30 minutes)
        # Minutes since day start
        df['minutes_since_day_start'] = (
            (df['timestamp'] - df['day_start_time']).dt.total_seconds() / 60
        )
        
        # Mark opening range period (first 30 minutes)
        df['is_opening_range'] = df['minutes_since_day_start'] <= 30
        
        # Calculate opening range high/low per day
        opening_range = df[df['is_opening_range']].groupby('trade_date').agg({
            'high': 'max',
            'low': 'min'
        }).rename(columns={'high': 'opening_range_high', 'low': 'opening_range_low'})
        
        df = df.merge(opening_range, on='trade_date', how='left')
        
        # Opening range breakout features
        df['above_opening_range_high'] = (
            df['close'] > df['opening_range_high']
        ).astype(int)
        
        df['below_opening_range_low'] = (
            df['close'] < df['opening_range_low']
        ).astype(int)
        
        df['dist_to_opening_range_high'] = df['close'] - df['opening_range_high']
        df['dist_to_opening_range_low'] = df['close'] - df['opening_range_low']
        
        # 7. Consecutive direction (causal)
        # Track if current bar continues the trend of previous bar
        is_bullish_bar = (df['close'] > df['open']).astype(int)
        
        df['continues_prev_direction'] = (
            is_bullish_bar == is_bullish_bar.shift(1)
        ).fillna(False).astype(int)
        
        # Clean up temporary columns
        df = df.drop(columns=[
            'trade_date', 'day_start_time', 'day_open', 
            'minutes_since_day_start', 'is_opening_range',
            'opening_range_high', 'opening_range_low'
        ], errors='ignore')
        
        return df
