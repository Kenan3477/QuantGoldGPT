import numpy as np
import pandas as pd

from quantgold.features.bundle import FeatureBundle
from quantgold.features.registry import FORBIDDEN_LABEL_COLUMNS


def test_feature_bundle_has_no_forbidden_columns():
    n = 100
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(1900, 2000, n)
    df = pd.DataFrame(
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
    built = FeatureBundle().transform(df)
    assert not (set(built.feature_columns) & FORBIDDEN_LABEL_COLUMNS)
    assert "target_return" not in built.frame.columns
