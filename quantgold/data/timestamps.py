"""
Timestamp contracts to prevent lookahead bias.

Every feature used at prediction time T must satisfy:
    available_timestamp <= prediction_timestamp
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd


TimestampLike = Union[datetime, pd.Timestamp]


@dataclass(frozen=True)
class FeatureTimestampContract:
    feature_name: str
    feature_timestamp: TimestampLike
    available_timestamp: TimestampLike
    prediction_timestamp: TimestampLike

    def is_valid(self) -> bool:
        return pd.Timestamp(self.available_timestamp) <= pd.Timestamp(self.prediction_timestamp)

    def validate(self) -> None:
        if not self.is_valid():
            raise LookaheadError(
                f"Feature '{self.feature_name}' available at {self.available_timestamp} "
                f"but prediction_timestamp={self.prediction_timestamp}"
            )


class LookaheadError(ValueError):
    """Raised when a feature would use information not yet available."""


def assert_no_lookahead(
    contracts: Iterable[FeatureTimestampContract],
) -> None:
    """Validate a batch of feature timestamp contracts."""
    for contract in contracts:
        contract.validate()


def bar_available_timestamp(
    bar_open: TimestampLike,
    timeframe_minutes: int,
) -> pd.Timestamp:
    """
    Earliest safe use time for a completed bar.

    Convention: MT5/broker bars are labelled at open; QuantGold only uses a bar
    after it has closed (open + timeframe).
    """
    return pd.Timestamp(bar_open) + timedelta(minutes=timeframe_minutes)


def align_higher_timeframe(
    base: pd.DataFrame,
    higher: pd.DataFrame,
    *,
    base_time_col: str = "timestamp",
    higher_time_col: str = "timestamp",
    higher_available_col: str = "available_timestamp",
    prediction_time_col: Optional[str] = None,
    suffixes: Sequence[str] = ("", "_htf"),
) -> pd.DataFrame:
    """
    As-of join higher-timeframe bars onto base bars using availability timestamps.

    Only HTF rows with available_timestamp <= base prediction time are joined.
    This avoids using a still-forming higher-timeframe candle.
    """
    if base.empty:
        return base.copy()
    if higher.empty:
        out = base.copy()
        return out

    pred_col = prediction_time_col or "available_timestamp"
    if pred_col not in base.columns:
        raise KeyError(f"Base frame missing prediction time column '{pred_col}'")
    if higher_available_col not in higher.columns:
        raise KeyError(f"Higher frame missing '{higher_available_col}'")

    left = base.sort_values(pred_col).copy()
    right = higher.sort_values(higher_available_col).copy()

    merged = pd.merge_asof(
        left,
        right,
        left_on=pred_col,
        right_on=higher_available_col,
        direction="backward",
        suffixes=suffixes,
    )
    return merged


def validate_feature_frame(
    df: pd.DataFrame,
    *,
    prediction_col: str = "available_timestamp",
    feature_available_cols: Optional[List[str]] = None,
) -> None:
    """
    Ensure optional per-feature availability columns do not exceed prediction time.
    """
    if prediction_col not in df.columns:
        raise KeyError(prediction_col)
    pred = pd.to_datetime(df[prediction_col])
    for col in feature_available_cols or []:
        if col not in df.columns:
            continue
        avail = pd.to_datetime(df[col])
        bad = avail > pred
        if bad.any():
            n = int(bad.sum())
            raise LookaheadError(
                f"{n} rows have {col} > {prediction_col} (lookahead)"
            )
