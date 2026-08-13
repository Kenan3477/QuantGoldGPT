"""Ensure label / future-return columns can never enter feature matrices."""

import pandas as pd
import pytest

from quantgold.features.registry import FeatureRegistry, FORBIDDEN_LABEL_COLUMNS
from quantgold.labels.triple_barrier import TripleBarrierLabeler


def test_forbidden_includes_xaubot_target_return():
    assert "target_return" in FORBIDDEN_LABEL_COLUMNS
    assert "target" in FORBIDDEN_LABEL_COLUMNS


def test_registry_rejects_label_registration():
    reg = FeatureRegistry()
    with pytest.raises(ValueError, match="label/leakage"):
        reg.register(["rsi", "target_return"])


def test_assert_no_label_leakage():
    with pytest.raises(ValueError, match="Label/leakage"):
        FeatureRegistry.assert_no_label_leakage(["log_return_1", "target_return", "atr_14"])


def test_select_strips_forbidden_even_if_present_in_frame_columns():
    reg = FeatureRegistry(["log_return_1", "atr_14"])
    cols = ["log_return_1", "atr_14", "target", "target_return", "tb_label"]
    assert reg.select(cols) == ["log_return_1", "atr_14"]


def test_triple_barrier_label_columns_are_denylisted():
    for col in TripleBarrierLabeler.label_columns():
        assert col in FORBIDDEN_LABEL_COLUMNS
