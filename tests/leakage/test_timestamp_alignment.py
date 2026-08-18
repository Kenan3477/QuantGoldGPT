"""Timestamp contract and HTF alignment leakage tests."""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from quantgold.data.timestamps import (
    FeatureTimestampContract,
    LookaheadError,
    align_higher_timeframe,
    assert_no_lookahead,
    bar_available_timestamp,
)


def test_feature_contract_rejects_lookahead():
    pred = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    avail = pred + timedelta(minutes=5)
    contract = FeatureTimestampContract("h1_close", pred, avail, pred)
    with pytest.raises(LookaheadError):
        assert_no_lookahead([contract])


def test_bar_available_timestamp_is_close_time():
    open_ts = pd.Timestamp("2024-01-01T10:00:00Z")
    assert bar_available_timestamp(open_ts, 15) == pd.Timestamp("2024-01-01T10:15:00Z")


def test_align_higher_timeframe_uses_availability_not_open():
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-01T10:15:00Z", "2024-01-01T10:30:00Z"]
            ),
            "available_timestamp": pd.to_datetime(
                ["2024-01-01T10:15:00Z", "2024-01-01T10:30:00Z", "2024-01-01T10:45:00Z"]
            ),
            "close": [1.0, 2.0, 3.0],
        }
    )
    # H1 bar opens 10:00, available only at 11:00
    higher = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T10:00:00Z"]),
            "available_timestamp": pd.to_datetime(["2024-01-01T11:00:00Z"]),
            "h1_close": [99.0],
        }
    )
    merged = align_higher_timeframe(base, higher)
    # Before 11:00 availability, h1_close must be NaN
    assert pd.isna(merged.loc[0, "h1_close"])
    assert pd.isna(merged.loc[1, "h1_close"])
    assert pd.isna(merged.loc[2, "h1_close"])

    # Add a row after HTF close
    base2 = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01T11:00:00Z")],
                    "available_timestamp": [pd.Timestamp("2024-01-01T11:15:00Z")],
                    "close": [4.0],
                }
            ),
        ],
        ignore_index=True,
    )
    merged2 = align_higher_timeframe(base2, higher)
    assert merged2.loc[3, "h1_close"] == 99.0
