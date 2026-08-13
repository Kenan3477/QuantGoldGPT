"""
Multi-timeframe (MTF) features for trend alignment and volatility cascade.

Key concepts:
- Trend alignment: Are multiple timeframes bullish/bearish together?
- Volatility cascade: How does volatility compare across timeframes?
- Support/Resistance from higher timeframes

All features use `align_higher_timeframe` to ensure no lookahead bias.
"""

import polars as pl
from quantgold.data.timestamps import align_higher_timeframe


class MultitimeframeFeatureBuilder:
    """
    Build multi-timeframe features.
    
    Requires:
    - Base timeframe data (e.g., M5)
    - Higher timeframe data (e.g., H1, H4, D1)
    
    Features:
    - SMA trend alignment across timeframes
    - Count of bullish/bearish timeframes
    - ATR cascade (volatility comparison)
    - Distance to higher TF swing highs/lows
    """
    
    def __init__(self, higher_tf_data: dict[str, pl.DataFrame]):
        """
        Initialize with higher timeframe data.
        
        Args:
            higher_tf_data: Dict mapping timeframe (e.g., "H1", "H4", "D1") → DataFrame
        """
        self.higher_tf_data = higher_tf_data
    
    def build(
        self,
        df: pl.DataFrame,
        base_tf: str = "M5",
        sma_period: int = 20,
        atr_period: int = 14,
    ) -> pl.DataFrame:
        """
        Add multi-timeframe features to base timeframe data.
        
        Args:
            df: Base timeframe OHLCV data (e.g., M5)
            base_tf: Base timeframe name
            sma_period: SMA period for trend detection
            atr_period: ATR period for volatility
            
        Returns:
            DataFrame with MTF features
        """
        # Ensure base df is sorted
        df = df.sort("timestamp")
        
        # Calculate base TF indicators first
        df = self._add_base_indicators(df, sma_period, atr_period)
        
        # Track trend alignment across timeframes
        bullish_count = pl.lit(0).alias("mtf_bullish_count")
        bearish_count = pl.lit(0).alias("mtf_bearish_count")
        
        # For each higher timeframe, join and add features
        for tf_name, tf_df in self.higher_tf_data.items():
            # Calculate indicators on higher TF
            tf_df = self._add_base_indicators(tf_df, sma_period, atr_period)
            
            # Align higher TF to base TF (no lookahead)
            df = align_higher_timeframe(
                base_df=df,
                higher_df=tf_df,
                join_cols=["close", "high", "low", f"sma_{sma_period}", f"atr_{atr_period}", "is_above_sma"],
                suffix=f"_{tf_name}",
            )
            
            # Count bullish/bearish TFs
            bullish_count = bullish_count + pl.col(f"is_above_sma_{tf_name}").fill_null(0).cast(pl.Int32)
            bearish_count = bearish_count + (1 - pl.col(f"is_above_sma_{tf_name}").fill_null(0).cast(pl.Int32))
        
        # Add trend alignment counts
        df = df.with_columns([
            bullish_count,
            bearish_count,
        ])
        
        # Trend alignment score (-N to +N, where N = number of higher TFs)
        num_tfs = len(self.higher_tf_data)
        df = df.with_columns([
            (pl.col("mtf_bullish_count") - pl.col("mtf_bearish_count")).alias("mtf_trend_alignment")
        ])
        
        # Calculate volatility cascade (current TF ATR vs. higher TF ATRs)
        # This shows if volatility is expanding or contracting across TFs
        for tf_name in self.higher_tf_data.keys():
            if f"atr_{atr_period}_{tf_name}" in df.columns:
                df = df.with_columns([
                    (pl.col(f"atr_{atr_period}") / pl.col(f"atr_{atr_period}_{tf_name}").fill_null(1.0))
                    .fill_null(1.0)
                    .fill_nan(1.0)
                    .alias(f"atr_ratio_{tf_name}")
                ])
        
        return df
    
    def _add_base_indicators(self, df: pl.DataFrame, sma_period: int, atr_period: int) -> pl.DataFrame:
        """Add SMA and ATR to a dataframe."""
        # SMA
        df = df.with_columns([
            pl.col("close").rolling_mean(sma_period).alias(f"sma_{sma_period}"),
        ])
        
        # Is price above SMA?
        df = df.with_columns([
            (pl.col("close") > pl.col(f"sma_{sma_period}"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_above_sma"),
        ])
        
        # ATR (simplified: average of high-low range)
        df = df.with_columns([
            (pl.col("high") - pl.col("low")).rolling_mean(atr_period).alias(f"atr_{atr_period}"),
        ])
        
        return df


def add_multitimeframe_features(
    base_df: pl.DataFrame,
    base_tf: str,
    higher_tf_data: dict[str, pl.DataFrame],
    sma_period: int = 20,
    atr_period: int = 14,
) -> pl.DataFrame:
    """
    Convenience function to add multi-timeframe features.
    
    Args:
        base_df: Base timeframe data (e.g., M5)
        base_tf: Base timeframe name
        higher_tf_data: Dict of higher TF data (e.g., {"H1": h1_df, "H4": h4_df, "D1": d1_df})
        sma_period: SMA period
        atr_period: ATR period
        
    Returns:
        DataFrame with MTF features
        
    Example:
        ```python
        m5_df = store.load_ohlcv("XAUUSD", "M5")
        h1_df = store.load_ohlcv("XAUUSD", "H1")
        h4_df = store.load_ohlcv("XAUUSD", "H4")
        d1_df = store.load_ohlcv("XAUUSD", "D1")
        
        m5_with_mtf = add_multitimeframe_features(
            m5_df,
            base_tf="M5",
            higher_tf_data={"H1": h1_df, "H4": h4_df, "D1": d1_df},
        )
        ```
    """
    builder = MultitimeframeFeatureBuilder(higher_tf_data)
    return builder.build(base_df, base_tf=base_tf, sma_period=sma_period, atr_period=atr_period)
