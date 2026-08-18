import pandas as pd

from quantgold.config.settings import ValidationConfig
from quantgold.validation.purged import purge_embargo_mask
from quantgold.validation.walk_forward import WalkForwardSplitter


def test_walk_forward_chronological_and_disjoint():
    idx = pd.date_range("2018-01-01", "2025-12-31", freq="D", tz="UTC")
    df = pd.DataFrame({"available_timestamp": idx})
    splitter = WalkForwardSplitter(
        ValidationConfig(train_years=3, validation_years=1, test_years=1, step_years=1)
    )
    folds = list(splitter.iter_masks(df))
    assert len(folds) >= 1
    for sp, train, val, test in folds:
        assert sp.train_end < sp.validation_start
        assert sp.validation_end < sp.test_start
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val & test).any()


def test_purge_removes_label_overlap():
    ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"))
    train = pd.Series([True] * 80 + [False] * 20)
    test = pd.Series([False] * 80 + [True] * 20)
    cleaned = purge_embargo_mask(
        ts, train, test, label_horizon_bars=12, embargo_bars=5
    )
    # Some trailing train rows near test must be removed
    assert cleaned.sum() < train.sum()
    assert cleaned.iloc[-1] is False or cleaned.iloc[79] == False
