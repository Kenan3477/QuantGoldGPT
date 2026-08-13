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
    "dxy_log_return_1",
    "vix_level",
    "vix_change_1",
    "us10y_change_1",
    "spx_log_return_1",
    "xau_xag_ratio",
    "xau_xag_ratio_z_20",
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

        _add_return("DXY", "dxy_log_return_1")
        _add_return("SPX", "spx_log_return_1")

        if "VIX" in externals:
            ext = externals["VIX"][["available_timestamp", "close"]].rename(columns={"close": "vix_level"})
            ext["vix_change_1"] = ext["vix_level"].astype(float).diff()
            merged = align_higher_timeframe(out, ext, higher_available_col="available_timestamp")
            out["vix_level"] = merged["vix_level"].values
            out["vix_change_1"] = merged["vix_change_1"].values
        else:
            out["vix_level"] = np.nan
            out["vix_change_1"] = np.nan

        if "US10Y" in externals:
            ext = externals["US10Y"][["available_timestamp", "close"]].rename(columns={"close": "us10y"})
            ext["us10y_change_1"] = ext["us10y"].astype(float).diff()
            merged = align_higher_timeframe(
                out, ext[["available_timestamp", "us10y_change_1"]], higher_available_col="available_timestamp"
            )
            out["us10y_change_1"] = merged["us10y_change_1"].values
        else:
            out["us10y_change_1"] = np.nan

        if peer_metal is not None and not peer_metal.empty:
            peer = peer_metal[["available_timestamp", "close"]].rename(columns={"close": "peer_close"})
            merged = align_higher_timeframe(out, peer, higher_available_col="available_timestamp")
            ratio = out["close"].astype(float).values / np.where(
                merged["peer_close"].astype(float).values == 0,
                np.nan,
                merged["peer_close"].astype(float).values,
            )
            out["xau_xag_ratio"] = ratio
            out["xau_xag_ratio_z_20"] = (
                (pd.Series(ratio) - pd.Series(ratio).rolling(20, min_periods=5).mean())
                / pd.Series(ratio).rolling(20, min_periods=5).std()
            ).values
        else:
            out["xau_xag_ratio"] = np.nan
            out["xau_xag_ratio_z_20"] = np.nan

        FeatureRegistry.assert_no_label_leakage(self.FEATURE_NAMES)
        return out
