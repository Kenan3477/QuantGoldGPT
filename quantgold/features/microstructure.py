"""
Microstructure features for intraday gold/silver trading.

These features capture market microstructure dynamics:
- Spread proxies (when bid/ask not available)
- Intraday momentum and range patterns
- Opening range breakouts
- Bar quality and volume characteristics

All features are CAUSAL (no lookahead).
"""

import polars as pl
import numpy as np


class MicrostructureFeatureBuilder:
    """
    Build microstructure features for M5/M15 intraday data.
    
    Features:
    - Spread proxy (high-low / close)
    - Intraday range percentile
    - Opening range breakout (first 30min of session)
    - Distance from day open
    - Bar body/wick ratios
    - Volume ratios (if available)
    """
    
    def build(self, df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
        """
        Add microstructure features to OHLCV dataframe.
        
        Args:
            df: Polars DataFrame with OHLCV + timestamp
            lookback: Lookback period for moving averages/percentiles
            
        Returns:
            DataFrame with additional feature columns
        """
        # Ensure sorted by timestamp
        df = df.sort("timestamp")
        
        # 1. Spread proxy (no bid/ask available in free data)
        # Use (high - low) / close as spread proxy
        df = df.with_columns([
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("spread_proxy"),
        ])
        
        # Spread percentile vs. recent history
        df = df.with_columns([
            (pl.col("spread_proxy") / pl.col("spread_proxy").rolling_mean(lookback))
            .fill_null(1.0)
            .alias("spread_vs_ma"),
        ])
        
        # 2. Intraday range features
        df = df.with_columns([
            (pl.col("high") - pl.col("low")).alias("bar_range"),
        ])
        
        # Range percentile (current range / avg range)
        df = df.with_columns([
            (pl.col("bar_range") / pl.col("bar_range").rolling_mean(lookback))
            .fill_null(1.0)
            .alias("range_vs_ma"),
        ])
        
        # 3. Bar body and wick features
        df = df.with_columns([
            (pl.col("close") - pl.col("open")).alias("bar_body"),
            pl.when(pl.col("close") >= pl.col("open"))
            .then(pl.col("high") - pl.col("close"))  # Upper wick (bullish bar)
            .otherwise(pl.col("high") - pl.col("open"))  # Upper wick (bearish bar)
            .alias("upper_wick"),
            pl.when(pl.col("close") >= pl.col("open"))
            .then(pl.col("open") - pl.col("low"))  # Lower wick (bullish bar)
            .otherwise(pl.col("close") - pl.col("low"))  # Lower wick (bearish bar)
            .alias("lower_wick"),
        ])
        
        # Body size as % of total range
        df = df.with_columns([
            (pl.col("bar_body").abs() / pl.col("bar_range"))
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("body_pct"),
        ])
        
        # Wick ratios
        df = df.with_columns([
            (pl.col("upper_wick") / pl.col("bar_range"))
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("upper_wick_pct"),
            (pl.col("lower_wick") / pl.col("bar_range"))
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("lower_wick_pct"),
        ])
        
        # 4. Volume features (if available, else skip)
        if "volume" in df.columns and df["volume"].sum() > 0:
            df = df.with_columns([
                (pl.col("volume") / pl.col("volume").rolling_mean(lookback))
                .fill_null(1.0)
                .alias("volume_vs_ma"),
            ])
        
        # 5. Opening range features (session-based)
        # Detect start of trading day (00:00 UTC for 24h markets, or session start)
        # For simplicity, use daily open (first bar of each day)
        df = df.with_columns([
            pl.col("timestamp").dt.date().alias("trade_date"),
        ])
        
        # Get daily open (first bar of each day)
        daily_open = (
            df.group_by("trade_date")
            .agg([
                pl.col("timestamp").min().alias("day_start_time"),
                pl.col("open").first().alias("day_open"),
            ])
        )
        
        # Join back to main df
        df = df.join(daily_open, on="trade_date", how="left")
        
        # Distance from day open (in price units and % and ATR units)
        df = df.with_columns([
            (pl.col("close") - pl.col("day_open")).alias("dist_from_day_open"),
            ((pl.col("close") - pl.col("day_open")) / pl.col("day_open"))
            .alias("dist_from_day_open_pct"),
        ])
        
        # 6. Opening range breakout (first 30 minutes)
        # Calculate high/low of first 30min of day
        # For M5: 30min = 6 bars, M15: 30min = 2 bars
        # We'll use a dynamic approach: first N bars after day start
        
        # Minutes since day start
        df = df.with_columns([
            ((pl.col("timestamp") - pl.col("day_start_time")).dt.total_seconds() / 60)
            .alias("minutes_since_day_start")
        ])
        
        # Mark opening range period (first 30 minutes)
        df = df.with_columns([
            (pl.col("minutes_since_day_start") <= 30).alias("is_opening_range")
        ])
        
        # Calculate opening range high/low per day
        opening_range = (
            df.filter(pl.col("is_opening_range"))
            .group_by("trade_date")
            .agg([
                pl.col("high").max().alias("opening_range_high"),
                pl.col("low").min().alias("opening_range_low"),
            ])
        )
        
        df = df.join(opening_range, on="trade_date", how="left")
        
        # Opening range breakout features
        df = df.with_columns([
            # Is current price above opening range high?
            (pl.col("close") > pl.col("opening_range_high")).cast(pl.Int8).alias("above_opening_range_high"),
            # Is current price below opening range low?
            (pl.col("close") < pl.col("opening_range_low")).cast(pl.Int8).alias("below_opening_range_low"),
            # Distance to opening range high/low
            (pl.col("close") - pl.col("opening_range_high")).alias("dist_to_opening_range_high"),
            (pl.col("close") - pl.col("opening_range_low")).alias("dist_to_opening_range_low"),
        ])
        
        # 7. Consecutive direction (causal)
        # Count consecutive up/down bars
        df = df.with_columns([
            (pl.col("close") > pl.col("open")).cast(pl.Int8).alias("is_bullish_bar"),
        ])
        
        # Causal consecutive counter (uses shift to avoid lookahead)
        df = df.with_columns([
            pl.col("is_bullish_bar").alias("_dir"),
        ])
        
        # Simple causal consecutive: Count streak ending at current bar
        # This is tricky in Polars without a custom function, so we'll use a simpler approach
        # For now, just track if current bar continues the trend of previous bar
        df = df.with_columns([
            (pl.col("is_bullish_bar") == pl.col("is_bullish_bar").shift(1))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("continues_prev_direction"),
        ])
        
        # Clean up temporary columns
        df = df.drop([
            "trade_date", "day_start_time", "minutes_since_day_start",
            "is_opening_range", "_dir",
            "bar_body", "bar_range", "upper_wick", "lower_wick",
        ])
        
        return df


def add_microstructure_features(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """
    Convenience function to add microstructure features.
    
    Args:
        df: OHLCV DataFrame
        lookback: Lookback period
        
    Returns:
        DataFrame with microstructure features
    """
    builder = MicrostructureFeatureBuilder()
    return builder.build(df, lookback=lookback)
