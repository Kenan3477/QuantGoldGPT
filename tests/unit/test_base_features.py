import numpy as np
import pandas as pd

from quantgold.features.base import BaseFeatureBuilder


def test_base_features_do_not_use_future_close():
    n = 60
    close = np.cumsum(np.random.RandomState(0).normal(0, 1, n)) + 2000
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n),
        }
    )
    fm = BaseFeatureBuilder().transform(df)
    # Mutate a future close; early feature rows must remain unchanged
    baseline = fm.frame.loc[10, fm.feature_columns].copy()
    df2 = df.copy()
    df2.loc[50, "close"] = df2.loc[50, "close"] + 1000
    fm2 = BaseFeatureBuilder().transform(df2)
    # Features at index 10 should be identical (no future leakage from bar 50)
    assert np.allclose(
        baseline.astype(float).fillna(0),
        fm2.frame.loc[10, fm.feature_columns].astype(float).fillna(0),
        equal_nan=True,
    )
