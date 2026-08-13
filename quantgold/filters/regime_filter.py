"""
Regime-aware trading filter to avoid low-quality market conditions.

This module implements a regime filter that prevents trading in choppy,
low-conviction, or unfavorable market conditions.

Key principles:
- Only trade when market conditions favor our strategy
- Avoid consolidation/choppy periods (high failure rate)
- Avoid extreme volatility (unpredictable outcomes)
- Prefer trending/momentum regimes
"""

from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np


RegimeType = Literal["trending_bullish", "trending_bearish", "volatile", "choppy", "normal"]


@dataclass
class RegimeConfig:
    """Configuration for regime filtering."""
    
    # ATR-based volatility thresholds
    vol_low_percentile: float = 0.20  # Below this = choppy
    vol_high_percentile: float = 0.90  # Above this = too volatile
    
    # Trend strength (ADX)
    adx_trending_threshold: float = 25.0  # ADX > 25 = trending
    adx_choppy_threshold: float = 15.0    # ADX < 15 = choppy
    
    # Enable/disable specific regimes
    allow_trending: bool = True
    allow_volatile: bool = False  # Disable by default (too risky)
    allow_choppy: bool = False    # Disable by default (low win rate)
    allow_normal: bool = True
    
    # Lookback for regime calculation
    lookback_periods: int = 100


class RegimeFilter:
    """
    Filter trades based on current market regime.
    
    Usage:
        filter = RegimeFilter(config)
        regime = filter.detect_regime(df)
        should_trade = filter.should_trade(regime)
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each bar.
        
        Args:
            df: DataFrame with OHLCV + indicators (must have atr, volatility)
        
        Returns:
            Series of regime labels
        """
        # Ensure we have required columns
        required = ['atr_14', 'realized_vol_20']
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Fallback: always return "normal" if indicators missing
            return pd.Series(['normal'] * len(df), index=df.index)
        
        # Calculate ADX if not present
        if 'adx_14' not in df.columns:
            df = self._add_adx(df)
        
        # Get volatility percentiles
        vol_percentiles = df['atr_14'].rolling(
            window=self.config.lookback_periods,
            min_periods=20
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Get ADX values
        adx = df.get('adx_14', 20.0)  # Default to neutral if missing
        
        # Classify regime
        regimes = []
        for i in range(len(df)):
            vol_pct = vol_percentiles.iloc[i] if not pd.isna(vol_percentiles.iloc[i]) else 0.5
            adx_val = adx.iloc[i] if isinstance(adx, pd.Series) else adx
            
            # Check for choppy (low vol + low ADX)
            if vol_pct < self.config.vol_low_percentile and adx_val < self.config.adx_choppy_threshold:
                regimes.append("choppy")
            
            # Check for volatile (high vol)
            elif vol_pct > self.config.vol_high_percentile:
                regimes.append("volatile")
            
            # Check for trending (high ADX)
            elif adx_val > self.config.adx_trending_threshold:
                # Determine trend direction from price action
                if i >= 20:
                    recent_return = (df['close'].iloc[i] / df['close'].iloc[i-20]) - 1
                    if recent_return > 0.01:
                        regimes.append("trending_bullish")
                    elif recent_return < -0.01:
                        regimes.append("trending_bearish")
                    else:
                        regimes.append("normal")
                else:
                    regimes.append("normal")
            
            # Default: normal
            else:
                regimes.append("normal")
        
        return pd.Series(regimes, index=df.index)
    
    def should_trade(self, regime: str | pd.Series) -> bool | pd.Series:
        """
        Determine if trading should be allowed in given regime.
        
        Args:
            regime: Single regime label or Series of regime labels
        
        Returns:
            Boolean or Series of booleans indicating if trading is allowed
        """
        if isinstance(regime, pd.Series):
            return regime.apply(self._should_trade_single)
        else:
            return self._should_trade_single(regime)
    
    def _should_trade_single(self, regime: str) -> bool:
        """Check if a single regime allows trading."""
        if regime.startswith("trending"):
            return self.config.allow_trending
        elif regime == "volatile":
            return self.config.allow_volatile
        elif regime == "choppy":
            return self.config.allow_choppy
        elif regime == "normal":
            return self.config.allow_normal
        else:
            # Unknown regime - be conservative
            return False
    
    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index) for trend strength.
        
        ADX measures trend strength (0-100):
        - ADX < 20: Weak trend or ranging
        - ADX 20-40: Trend developing
        - ADX > 40: Strong trend
        """
        df = df.copy()
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        df['adx_14'] = adx
        df['plus_di_14'] = plus_di
        df['minus_di_14'] = minus_di
        
        return df
    
    def get_regime_stats(self, df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """
        Calculate performance statistics by regime.
        
        Useful for validating regime filter effectiveness.
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column for regime stats")
        
        df = df.copy()
        df['regime'] = regimes
        
        stats = []
        for regime_type in df['regime'].unique():
            regime_data = df[df['regime'] == regime_type]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate win rate if we have success information
            if 'success' in df.columns:
                success_rate = (regime_data['success'] == True).sum() / len(regime_data)
            else:
                success_rate = None
            
            stats.append({
                'regime': regime_type,
                'count': len(regime_data),
                'pct_total': len(regime_data) / len(df),
                'success_rate': success_rate,
            })
        
        return pd.DataFrame(stats)


def apply_regime_filter_to_predictions(
    predictions: pd.DataFrame,
    ohlcv: pd.DataFrame,
    config: RegimeConfig = None,
) -> pd.DataFrame:
    """
    Apply regime filter to existing predictions.
    
    This is a post-processing step that converts predictions to NO_TRADE
    if the regime filter rejects trading.
    
    Args:
        predictions: DataFrame with predictions (must have 'side', 'timestamp')
        ohlcv: DataFrame with OHLCV data and indicators
        config: Regime configuration
    
    Returns:
        DataFrame with filtered predictions
    """
    filter = RegimeFilter(config)
    
    # Detect regimes for entire OHLCV dataset
    regimes = filter.detect_regime(ohlcv)
    should_trade = filter.should_trade(regimes)
    
    # Create regime lookup by timestamp
    regime_lookup = pd.DataFrame({
        'timestamp': ohlcv['timestamp'],
        'regime': regimes,
        'should_trade': should_trade,
    }).set_index('timestamp')
    
    # Apply filter to predictions
    predictions = predictions.copy()
    predictions = predictions.join(regime_lookup, on='timestamp', how='left')
    
    # Override predictions in bad regimes
    predictions.loc[predictions['should_trade'] == False, 'side'] = "NO_TRADE"
    predictions.loc[predictions['should_trade'] == False, 'reason'] = 'regime_filter'
    
    return predictions


if __name__ == "__main__":
    # Example usage
    from quantgold.data.store import CanonicalDataStore
    from quantgold.pipeline.dataset import prepare_research_dataset
    from quantgold.config.settings import load_settings
    
    print("Testing regime filter...")
    
    # Load data
    cfg = load_settings()
    prep_ds = prepare_research_dataset(
        symbol="XAUUSD",
        timeframe="H4",
        settings=cfg,
    )
    df = prep_ds.frame
    
    # Detect regimes
    filter = RegimeFilter()
    regimes = filter.detect_regime(df)
    
    print(f"\nRegime distribution:")
    print(regimes.value_counts(normalize=True))
    
    print(f"\nTrading allowed:")
    should_trade = filter.should_trade(regimes)
    print(f"  {should_trade.sum() / len(should_trade):.1%} of bars allow trading")
    
    # Calculate stats by regime
    if 'success' in df.columns:
        stats = filter.get_regime_stats(df, regimes)
        print(f"\nPerformance by regime:")
        print(stats)
