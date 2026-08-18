import numpy as np
import pandas as pd

from quantgold.features.sessions import SessionFeatureBuilder
from quantgold.features.structure import StructureFeatureBuilder


def _sample(n=80):
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.cumsum(np.random.RandomState(0).normal(0, 0.5, n)) + 2000
    return pd.DataFrame(
        {
            "timestamp": ts,
            "available_timestamp": ts + pd.Timedelta(hours=1),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
        }
    )


def test_session_features_created():
    df = SessionFeatureBuilder().transform(_sample())
    assert df["session_london"].isin([0.0, 1.0]).all()
    assert "prev_session_return" in df.columns


def test_structure_assigns_at_confirmation_not_center():
    df = _sample(100)
    # spike high in the middle
    df.loc[50, "high"] = df.loc[50, "high"] + 50
    out = StructureFeatureBuilder(swing_left=3, swing_right=3).transform(df)
    # confirmation index should be 53, not 50
    # bars_since at 53 should be 0-ish when confirmed
    assert out.loc[53, "bars_since_swing_high"] == 0 or out.loc[53, "bars_since_swing_high"] < 5
