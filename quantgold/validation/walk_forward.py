"""Expanding / rolling chronological walk-forward splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional

import pandas as pd

from quantgold.config.settings import ValidationConfig


@dataclass(frozen=True)
class WalkForwardSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """
    Chronological walk-forward splitter.

    Example (years):
      TRAIN 2018–2021 / VALIDATE 2022 / TEST 2023
      then roll forward by step_years.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def split(
        self,
        timestamps: pd.Series,
        *,
        final_holdout_start: Optional[str] = None,
    ) -> List[WalkForwardSplit]:
        ts = pd.to_datetime(timestamps, utc=True).sort_values().reset_index(drop=True)
        if ts.empty:
            return []

        holdout = final_holdout_start or self.config.final_holdout_start
        holdout_ts = pd.Timestamp(holdout, tz="UTC") if holdout else None

        start = ts.iloc[0]
        end = ts.iloc[-1]
        if holdout_ts is not None:
            end = min(end, holdout_ts - pd.Timedelta(seconds=1))

        cfg = self.config
        folds: List[WalkForwardSplit] = []
        fold_id = 0

        # Use year anchors for clarity; bar-level masks applied by callers.
        cursor = pd.Timestamp(year=start.year, month=1, day=1, tz="UTC")
        while True:
            train_start = cursor
            train_end = train_start + pd.DateOffset(years=cfg.train_years) - pd.Timedelta(seconds=1)
            val_start = train_end + pd.Timedelta(seconds=1)
            val_end = val_start + pd.DateOffset(years=cfg.validation_years) - pd.Timedelta(seconds=1)
            test_start = val_end + pd.Timedelta(seconds=1)
            test_end = test_start + pd.DateOffset(years=cfg.test_years) - pd.Timedelta(seconds=1)

            if test_end > end:
                break

            folds.append(
                WalkForwardSplit(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=val_start,
                    validation_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_id += 1
            cursor = cursor + pd.DateOffset(years=cfg.step_years)

        return folds

    def iter_masks(
        self,
        df: pd.DataFrame,
        time_col: str = "available_timestamp",
        **kwargs,
    ) -> Iterator[tuple[WalkForwardSplit, pd.Series, pd.Series, pd.Series]]:
        splits = self.split(df[time_col], **kwargs)
        t = pd.to_datetime(df[time_col], utc=True)
        for sp in splits:
            train = (t >= sp.train_start) & (t <= sp.train_end)
            val = (t >= sp.validation_start) & (t <= sp.validation_end)
            test = (t >= sp.test_start) & (t <= sp.test_end)
            yield sp, train, val, test
