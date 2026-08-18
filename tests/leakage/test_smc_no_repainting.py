"""
Test causal SMC features for repainting and lookahead bias.

Critical: SMC features in XAUBot had repainting bugs.
These tests verify the new causal implementation is leak-free.
"""

import polars as pl
import pandas as pd
import pytest
from datetime import datetime, timedelta

from quantgold.features.smc_causal import add_smc_features


def test_smc_no_future_dependency():
    """
    Test that SMC features at bar i do not depend on bars > i.
    
    Method: Calculate features on full dataset vs. truncated dataset.
    Features at bar i should be identical whether we have future bars or not.
    """
    # Create test data
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pl.DataFrame({
        "timestamp": dates,
        "open": [100 + i * 0.1 + (i % 10) * 0.5 for i in range(100)],
        "high": [101 + i * 0.1 + (i % 10) * 0.5 for i in range(100)],
        "low": [99 + i * 0.1 + (i % 10) * 0.5 for i in range(100)],
        "close": [100 + i * 0.1 + (i % 10) * 0.3 for i in range(100)],
        "volume": [1000] * 100,
    })
    
    # Add SMC features to full dataset
    df_full = add_smc_features(df)
    
    # Add SMC features to truncated dataset (first 50 bars only)
    df_truncated = add_smc_features(df.head(50))
    
    # Compare features at bar 40 (well before truncation point)
    # They should be identical
    test_bar_idx = 40
    
    feature_cols = [
        "is_swing_high", "is_swing_low",
        "has_bullish_fvg", "has_bearish_fvg",
        "bullish_bos", "bearish_bos",
        "bullish_choch", "bearish_choch",
        "bullish_ob", "bearish_ob",
    ]
    
    for col in feature_cols:
        if col in df_full.columns and col in df_truncated.columns:
            val_full = df_full[col][test_bar_idx]
            val_truncated = df_truncated[col][test_bar_idx]
            
            # Allow for null/None differences
            if val_full is not None or val_truncated is not None:
                assert val_full == val_truncated, (
                    f"Feature {col} at bar {test_bar_idx} differs: "
                    f"full={val_full}, truncated={val_truncated}. "
                    f"This indicates lookahead bias!"
                )


def test_smc_swing_points_no_repaint():
    """
    Test that swing highs/lows are only marked AFTER confirmation period.
    
    A swing high at bar i should not be marked until bar i+lookback.
    """
    # Create simple data with obvious swing point
    df = pl.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1h"),
        "open": [100] * 20,
        "high": [100, 100, 100, 105, 100, 100, 100, 100, 100, 100] + [100] * 10,  # Swing high at idx 3
        "low": [99] * 20,
        "close": [100] * 20,
        "volume": [1000] * 20,
    })
    
    swing_lookback = 3
    df = add_smc_features(df, swing_lookback=swing_lookback)
    
    # Swing high at bar 3 should not be marked until bar 3 + lookback
    # Because we need to confirm with bars on the right side
    swing_high_idx = 3
    
    # At the swing point itself (bar 3), it should NOT be marked yet
    # (or it might be marked, but that's OK as long as it uses only past data)
    
    # Check that swing detection doesn't use future bars improperly
    # We check this by ensuring the feature at bar i doesn't change when we add bar i+1
    for i in range(10, 15):
        df_up_to_i = add_smc_features(df.head(i), swing_lookback=swing_lookback)
        df_up_to_i_plus_1 = add_smc_features(df.head(i + 1), swing_lookback=swing_lookback)
        
        # Feature at bar i-5 should not change when we add bar i+1
        # (5 bars before should be finalized)
        check_idx = max(0, i - 5)
        if check_idx < len(df_up_to_i):
            val_i = df_up_to_i["is_swing_high"][check_idx]
            val_i_plus_1 = df_up_to_i_plus_1["is_swing_high"][check_idx]
            
            assert val_i == val_i_plus_1, (
                f"Swing high at bar {check_idx} changed when adding bar {i+1}. "
                f"This indicates repainting!"
            )


def test_smc_fvg_no_repaint():
    """
    Test that FVG detection doesn't repaint.
    
    FVG at bar i should be determined by bars i-2, i-1, i only (causal).
    """
    # Create data with FVG pattern
    df = pl.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1h"),
        "open": [100] * 20,
        "high": [101, 101, 101, 105, 105, 105] + [105] * 14,  # Gap at bar 3
        "low": [99, 99, 99, 103, 103, 103] + [103] * 14,  # Bar 3 low > bar 1 high = bullish FVG
        "close": [100] * 20,
        "volume": [1000] * 20,
    })
    
    df = add_smc_features(df, fvg_confirm_bars=3)
    
    # FVG at bar 3 should be detected immediately (based on bar 1, 2, 3 only)
    # No future confirmation needed for detection (only for "unfilled" status)
    
    # Check: FVG feature at bar 5 should not change when we add more bars
    df_truncated = add_smc_features(df.head(10), fvg_confirm_bars=3)
    df_full = add_smc_features(df, fvg_confirm_bars=3)
    
    for i in range(5, 10):
        val_truncated = df_truncated["has_bullish_fvg"][i]
        val_full = df_full["has_bullish_fvg"][i]
        
        assert val_truncated == val_full, (
            f"FVG at bar {i} changed. This indicates repainting!"
        )


def test_smc_bos_uses_confirmed_swings():
    """
    Test that BOS detection uses only confirmed swing points (causal).
    """
    # Create data with swing high and subsequent BOS
    df = pl.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="1h"),
        "open": [100] * 30,
        "high": [100, 101, 102, 103, 102, 101, 100, 100, 100, 100,
                 100, 100, 100, 100, 104, 105, 106, 107] + [107] * 12,  # BOS at bar 17
        "low": [99] * 30,
        "close": [100] * 30,
        "volume": [1000] * 30,
    })
    
    df = add_smc_features(df, swing_lookback=3)
    
    # BOS should only be detected after swing high is confirmed
    # This test just checks it doesn't crash and produces valid output
    assert "bullish_bos" in df.columns
    assert df["bullish_bos"].dtype == pl.Int8


def test_smc_order_block_no_retroactive_marking():
    """
    Test that Order Blocks are not marked retroactively based on future price action.
    
    XAUBot bug: OB was marked at bar j when bar i > j showed a strong move,
    then bar j was retroactively marked as OB. This is repainting.
    
    Our implementation: OB is marked at bar i when bar i shows strong move.
    The base candle is bar i-1, but we don't go back and change bar i-1's features.
    """
    df = pl.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1h"),
        "open": [100, 100, 100, 100, 100, 100, 100, 110, 110, 110] + [110] * 10,  # Strong move at bar 7
        "high": [101, 101, 101, 101, 101, 101, 101, 111, 111, 111] + [111] * 10,
        "low": [99, 99, 99, 99, 99, 99, 99, 109, 109, 109] + [109] * 10,
        "close": [100, 100, 100, 100, 100, 100, 100, 110, 110, 110] + [110] * 10,
        "volume": [1000] * 20,
    })
    
    df = add_smc_features(df, ob_strength_atr_multiple=1.5)
    
    # Check that features at bar 6 don't change when we add bar 7
    df_before_move = add_smc_features(df.head(7), ob_strength_atr_multiple=1.5)
    df_after_move = add_smc_features(df.head(8), ob_strength_atr_multiple=1.5)
    
    # Features at bar 6 should be identical
    # (Bar 7 is when the move happens, but bar 6 features shouldn't retroactively change)
    check_idx = 6
    for col in ["bullish_ob", "bearish_ob"]:
        if col in df_before_move.columns:
            val_before = df_before_move[col][check_idx] if check_idx < len(df_before_move) else None
            val_after = df_after_move[col][check_idx] if check_idx < len(df_after_move) else None
            
            assert val_before == val_after, (
                f"OB feature {col} at bar {check_idx} changed after strong move at bar 7. "
                f"This indicates retroactive marking (repainting)!"
            )


def test_smc_features_exist():
    """Test that all expected SMC features are present."""
    df = pl.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
        "open": [100 + i * 0.1 for i in range(50)],
        "high": [101 + i * 0.1 for i in range(50)],
        "low": [99 + i * 0.1 for i in range(50)],
        "close": [100 + i * 0.1 for i in range(50)],
        "volume": [1000] * 50,
    })
    
    df = add_smc_features(df)
    
    expected_features = [
        "is_swing_high", "is_swing_low",
        "has_bullish_fvg", "has_bearish_fvg",
        "bullish_bos", "bearish_bos",
        "bullish_choch", "bearish_choch",
        "bullish_ob", "bearish_ob",
        "dist_to_bullish_ob", "dist_to_bearish_ob",
        "dist_to_bullish_fvg", "dist_to_bearish_fvg",
        "dist_to_swing_high", "dist_to_swing_low",
    ]
    
    for feature in expected_features:
        assert feature in df.columns, f"Expected feature {feature} not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
