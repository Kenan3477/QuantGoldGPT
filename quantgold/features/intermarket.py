"""
Intermarket features with timestamp-safe asof joins.

External series must expose available_timestamp. We never forward-fill beyond
availability.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from quantgold.data.timestamps import align_higher_timeframe
from quantgold.features.registry import FeatureRegistry

INTERMARKET_FEATURE_NAMES = [
    # Existing features
    "dxy_log_return_1",
    "vix_level",
    "vix_change_1",
    "us10y_change_1",
    "spx_log_return_1",
    "xau_xag_ratio",
    "xau_xag_ratio_z_20",
    # Enhanced features (Sprint 1 Bootstrap)
    "dxy_log_return_5",
    "dxy_log_return_20",
    "dxy_rsi_14",
    "dxy_above_sma_50",
    "real_yield_proxy",
    "vix_roc_5",
    "vix_percentile_100",
    "spx_log_return_5",
    "spx_log_return_20",
    "spx_drawdown_from_high",
    "xau_xag_ratio_ma_50",
    "xau_xag_ratio_vs_ma",
]


class IntermarketFeatureBuilder:
    FEATURE_NAMES = INTERMARKET_FEATURE_NAMES

    def __init__(self):
        self.registry = FeatureRegistry(self.FEATURE_NAMES)

    def transform(
        self,
        df: pd.DataFrame,
        externals: Optional[Dict[str, pd.DataFrame]] = None,
        *,
        peer_metal: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        externals keys e.g. DXY, VIX, US10Y, SPX with columns:
        timestamp, available_timestamp, close
        
        Enhanced features:
        - DXY: Multi-period returns, RSI, SMA distance
        - VIX: Rate of change, percentile
        - US10Y: Real yield proxy (10Y - 2% inflation assumption)
        - SPX: Multi-period returns, drawdown from high
        - XAU/XAG: Ratio MA and distance
        """
        out = df.copy()
        externals = externals or {}

        def _add_return(name: str, col_out: str) -> None:
            if name not in externals:
                out[col_out] = np.nan
                return
            ext = externals[name][["available_timestamp", "close"]].copy()
            ext = ext.rename(columns={"close": f"{name}_close"})
            ext[col_out] = np.log(ext[f"{name}_close"].astype(float)).diff()
            merged = align_higher_timeframe(
                out,
                ext[["available_timestamp", col_out]],
                higher_available_col="available_timestamp",
            )
            out[col_out] = merged[col_out].values

        # Basic returns (existing)
        _add_return("DXY", "dxy_log_return_1")
        _add_return("SPX", "spx_log_return_1")

        # Enhanced DXY features
        if "DXY" in externals:
            ext = externals["DXY"][["available_timestamp", "close"]].copy()
            ext = ext.rename(columns={"close": "dxy_close"})
            
            # Multi-period returns
            ext["dxy_log_return_5"] = np.log(ext["dxy_close"].astype(float)).diff(5)
            ext["dxy_log_return_20"] = np.log(ext["dxy_close"].astype(float)).diff(20)
            
            # RSI (Relative Strength Index)
            delta = ext["dxy_close"].astype(float).diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            ext["dxy_rsi_14"] = 100 - (100 / (1 + rs))
            
            # Above SMA(50)?
            sma_50 = ext["dxy_close"].astype(float).rolling(50, min_periods=10).mean()
            ext["dxy_above_sma_50"] = (ext["dxy_close"].astype(float) > sma_50).astype(float)
            
            # Merge enhanced features
            for col in ["dxy_log_return_5", "dxy_log_return_20", "dxy_rsi_14", "dxy_above_sma_50"]:
                merged = align_higher_timeframe(
                    out, ext[["available_timestamp", col]], higher_available_col="available_timestamp"
                )
                out[col] = merged[col].values
        else:
            for col in ["dxy_log_return_5", "dxy_log_return_20", "dxy_rsi_14", "dxy_above_sma_50"]:
                out[col] = np.nan

        # Enhanced VIX features
        if "VIX" in externals:
            ext = externals["VIX"][["available_timestamp", "close"]].rename(columns={"close": "vix_level"})
            ext["vix_change_1"] = ext["vix_level"].astype(float).diff()
            
            # Rate of change (5-period)
            ext["vix_roc_5"] = ext["vix_level"].astype(float).pct_change(5)
            
            # Percentile (vs. last 100 bars)
            ext["vix_percentile_100"] = ext["vix_level"].astype(float).rolling(100, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan,
                raw=False
            )
            
            merged = align_higher_timeframe(out, ext, higher_available_col="available_timestamp")
            out["vix_level"] = merged["vix_level"].values
            out["vix_change_1"] = merged["vix_change_1"].values
            out["vix_roc_5"] = merged["vix_roc_5"].values
            out["vix_percentile_100"] = merged["vix_percentile_100"].values
        else:
            out["vix_level"] = np.nan
            out["vix_change_1"] = np.nan
            out["vix_roc_5"] = np.nan
            out["vix_percentile_100"] = np.nan

        # Enhanced US10Y features (real yield proxy)
        if "US10Y" in externals:
            ext = externals["US10Y"][["available_timestamp", "close"]].rename(columns={"close": "us10y"})
            ext["us10y_change_1"] = ext["us10y"].astype(float).diff()
            
            # Real yield proxy: 10Y yield - 2% (assumed inflation)
            ext["real_yield_proxy"] = ext["us10y"].astype(float) - 2.0
            
            merged = align_higher_timeframe(
                out, ext[["available_timestamp", "us10y_change_1", "real_yield_proxy"]], 
                higher_available_col="available_timestamp"
            )
            out["us10y_change_1"] = merged["us10y_change_1"].values
            out["real_yield_proxy"] = merged["real_yield_proxy"].values
        else:
            out["us10y_change_1"] = np.nan
            out["real_yield_proxy"] = np.nan

        # Enhanced SPX features
        if "SPX" in externals:
            ext = externals["SPX"][["available_timestamp", "close"]].copy()
            ext = ext.rename(columns={"close": "spx_close"})
            
            # Multi-period returns
            ext["spx_log_return_5"] = np.log(ext["spx_close"].astype(float)).diff(5)
            ext["spx_log_return_20"] = np.log(ext["spx_close"].astype(float)).diff(20)
            
            # Drawdown from high
            rolling_max = ext["spx_close"].astype(float).rolling(252, min_periods=20).max()  # 1-year high
            ext["spx_drawdown_from_high"] = (ext["spx_close"].astype(float) / rolling_max - 1) * 100
            
            for col in ["spx_log_return_5", "spx_log_return_20", "spx_drawdown_from_high"]:
                merged = align_higher_timeframe(
                    out, ext[["available_timestamp", col]], higher_available_col="available_timestamp"
                )
                out[col] = merged[col].values
        else:
            for col in ["spx_log_return_5", "spx_log_return_20", "spx_drawdown_from_high"]:
                out[col] = np.nan

        # Enhanced XAU/XAG ratio features
        if peer_metal is not None and not peer_metal.empty:
            peer = peer_metal[["available_timestamp", "close"]].rename(columns={"close": "peer_close"})
            merged = align_higher_timeframe(out, peer, higher_available_col="available_timestamp")
            ratio = out["close"].astype(float).values / np.where(
                merged["peer_close"].astype(float).values == 0,
                np.nan,
                merged["peer_close"].astype(float).values,
            )
            out["xau_xag_ratio"] = ratio
            
            # Z-score (existing)
            out["xau_xag_ratio_z_20"] = (
                (pd.Series(ratio) - pd.Series(ratio).rolling(20, min_periods=5).mean())
                / pd.Series(ratio).rolling(20, min_periods=5).std()
            ).values
            
            # MA(50) and distance
            ratio_ma_50 = pd.Series(ratio).rolling(50, min_periods=10).mean()
            out["xau_xag_ratio_ma_50"] = ratio_ma_50.values
            out["xau_xag_ratio_vs_ma"] = ((pd.Series(ratio) / ratio_ma_50) - 1) * 100  # % distance
        else:
            out["xau_xag_ratio"] = np.nan
            out["xau_xag_ratio_z_20"] = np.nan
            out["xau_xag_ratio_ma_50"] = np.nan
            out["xau_xag_ratio_vs_ma"] = np.nan

        FeatureRegistry.assert_no_label_leakage(self.FEATURE_NAMES)
        return out
