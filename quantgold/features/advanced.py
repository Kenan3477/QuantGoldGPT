"""
Advanced features for maximum predictive power.

These features capture market microstructure, regime changes, and momentum quality.
"""

import pandas as pd
import numpy as np
from typing import List


class AdvancedFeatureBuilder:
    """
    High-value features based on quantitative research:
    
    1. Volatility Regime (percentile-based)
    2. Order Flow Proxy (volume-weighted momentum)
    3. Momentum Quality (trend acceleration/deceleration)
    4. Price Action Quality (clean trends vs choppy)
    5. Multi-timeframe Alignment
    """
    
    FEATURE_NAMES: List[str] = [
        # Volatility regime
        "vol_regime",
        "vol_percentile_20",
        "vol_percentile_60",
        "vol_expanding",
        
        # Order flow proxy
        "volume_momentum",
        "volume_divergence",
        "buy_sell_pressure",
        
        # Momentum quality
        "momentum_acceleration",
        "trend_consistency",
        "bars_in_trend",
        
        # Price action quality
        "price_efficiency",
        "noise_ratio",
        "higher_highs_lows",
        
        # Time-based
        "hour_volatility_ratio",
        "session_momentum",
        
        # Multi-timeframe
        "htf_ltf_alignment",
        "htf_trend_strength",
    ]
    
    def transform(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Add advanced features to dataframe."""
        df = df.copy()
        
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns for various metrics
        returns = df['close'].pct_change()
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # 1. VOLATILITY REGIME
        rolling_vol = returns.rolling(window=lookback).std()
        df['vol_percentile_20'] = rolling_vol.rolling(window=60).apply(
            lambda x: (x.iloc[-1] <= np.percentile(x, 20)).astype(float) if len(x) > 0 else 0,
            raw=False
        ).fillna(0)
        df['vol_percentile_60'] = rolling_vol.rolling(window=60).apply(
            lambda x: (x.iloc[-1] >= np.percentile(x, 60)).astype(float) if len(x) > 0 else 0,
            raw=False
        ).fillna(0)
        
        # Volatility regime: 0=low, 1=medium, 2=high
        vol_low_threshold = rolling_vol.rolling(window=100).quantile(0.33)
        vol_high_threshold = rolling_vol.rolling(window=100).quantile(0.67)
        df['vol_regime'] = np.where(
            rolling_vol <= vol_low_threshold, 0,
            np.where(rolling_vol >= vol_high_threshold, 2, 1)
        )
        
        # Is volatility expanding?
        vol_ma_short = rolling_vol.rolling(window=5).mean()
        vol_ma_long = rolling_vol.rolling(window=20).mean()
        df['vol_expanding'] = (vol_ma_short > vol_ma_long).astype(int)
        
        # 2. ORDER FLOW PROXY (volume-weighted momentum)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            # Volume-weighted price change
            price_change = df['close'] - df['open']
            volume_norm = df['volume'] / df['volume'].rolling(window=lookback).mean().replace(0, 1)
            df['volume_momentum'] = (price_change * volume_norm).rolling(window=5).mean().fillna(0)
            
            # Volume divergence (price up but volume down = weak)
            price_direction = np.sign(returns)
            volume_direction = np.sign(df['volume'].pct_change())
            df['volume_divergence'] = (price_direction != volume_direction).astype(int)
            
            # Buy/sell pressure (up days vs down days, volume-weighted)
            up_volume = np.where(returns > 0, df['volume'], 0)
            down_volume = np.where(returns < 0, df['volume'], 0)
            up_vol_sum = pd.Series(up_volume).rolling(window=lookback).sum()
            down_vol_sum = pd.Series(down_volume).rolling(window=lookback).sum()
            df['buy_sell_pressure'] = (up_vol_sum - down_vol_sum) / (up_vol_sum + down_vol_sum + 1e-9)
            df['buy_sell_pressure'] = df['buy_sell_pressure'].fillna(0)
        else:
            df['volume_momentum'] = 0.0
            df['volume_divergence'] = 0
            df['buy_sell_pressure'] = 0.0
        
        # 3. MOMENTUM QUALITY
        # Trend acceleration (is momentum increasing?)
        momentum = returns.rolling(window=10).mean()
        momentum_change = momentum.diff()
        df['momentum_acceleration'] = momentum_change.rolling(window=5).mean().fillna(0)
        
        # Trend consistency (what % of recent bars moved in trend direction?)
        trend_direction = np.sign(momentum)
        bar_direction = np.sign(returns)
        consistent = (trend_direction == bar_direction).astype(float)
        df['trend_consistency'] = consistent.rolling(window=lookback).mean().fillna(0.5)
        
        # Bars in current trend (consecutive bars in same direction)
        df['bars_in_trend'] = self._count_consecutive_direction(returns)
        
        # 4. PRICE ACTION QUALITY
        # Price efficiency (straight line distance / path distance)
        price_move = (df['close'] - df['close'].shift(lookback)).abs()
        path_distance = returns.abs().rolling(window=lookback).sum()
        df['price_efficiency'] = (price_move / (path_distance + 1e-9)).fillna(0)
        
        # Noise ratio (intrabar range / close-to-close move)
        intrabar_range = df['high'] - df['low']
        close_to_close = (df['close'] - df['close'].shift(1)).abs()
        df['noise_ratio'] = (intrabar_range / (close_to_close + 1e-9)).rolling(window=10).mean().fillna(1)
        
        # Higher highs and higher lows (trend quality)
        high_ma = df['high'].rolling(window=5).max()
        low_ma = df['low'].rolling(window=5).min()
        higher_highs = (high_ma > high_ma.shift(5)).astype(int)
        higher_lows = (low_ma > low_ma.shift(5)).astype(int)
        df['higher_highs_lows'] = higher_highs + higher_lows  # 0=downtrend, 1=mixed, 2=uptrend
        
        # 5. TIME-BASED FEATURES
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            
            # Hour-specific volatility (current hour vs average hour volatility)
            hour_vol = df.groupby('hour')['close'].transform(
                lambda x: x.pct_change().rolling(window=20).std().mean()
            )
            overall_vol = returns.rolling(window=20).std()
            df['hour_volatility_ratio'] = (overall_vol / (hour_vol + 1e-9)).fillna(1.0)
            
            # Session momentum (momentum within current trading session)
            df['trade_date'] = df['timestamp'].dt.date
            session_start = df.groupby('trade_date')['close'].transform('first')
            df['session_momentum'] = (df['close'] - session_start) / session_start
            
            df = df.drop(columns=['hour', 'trade_date'], errors='ignore')
        else:
            df['hour_volatility_ratio'] = 1.0
            df['session_momentum'] = 0.0
        
        # 6. MULTI-TIMEFRAME FEATURES
        # Higher timeframe trend (simulated via longer moving average)
        htf_ma_short = df['close'].rolling(window=20).mean()
        htf_ma_long = df['close'].rolling(window=60).mean()
        htf_trend = np.sign(htf_ma_short - htf_ma_long)
        
        # Lower timeframe trend
        ltf_trend = np.sign(df['close'].rolling(window=5).mean() - df['close'].rolling(window=20).mean())
        
        # Alignment (1 if both agree, 0 if disagree)
        df['htf_ltf_alignment'] = (htf_trend == ltf_trend).astype(int)
        
        # HTF trend strength
        htf_distance = ((htf_ma_short - htf_ma_long) / htf_ma_long * 100).fillna(0)
        df['htf_trend_strength'] = htf_distance.abs()
        
        return df
    
    def _count_consecutive_direction(self, returns: pd.Series) -> pd.Series:
        """Count consecutive bars in same direction (causal)."""
        direction = np.sign(returns)
        
        # Initialize count
        count = pd.Series(0, index=returns.index)
        current_count = 0
        current_dir = 0
        
        for i in range(len(direction)):
            if pd.isna(direction.iloc[i]) or direction.iloc[i] == 0:
                # Neutral bar, reset
                current_count = 0
                current_dir = 0
            elif direction.iloc[i] == current_dir:
                # Same direction, increment
                current_count += 1
            else:
                # Direction change, reset
                current_count = 1
                current_dir = direction.iloc[i]
            
            count.iloc[i] = current_count
        
        return count
