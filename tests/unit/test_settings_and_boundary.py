from quantgold.config.settings import load_settings
from quantgold.execution.guard import assert_no_research_imports
import pytest


def test_load_default_settings():
    settings = load_settings()
    assert settings.project_name == "QuantGold"
    assert "XAUUSD" in settings.instrument_symbols()
    assert "XAGUSD" in settings.instrument_symbols()
    assert settings.triple_barrier.same_bar_policy == "ambiguous"
    assert settings.decision.allow_no_trade is True


def test_execution_guard_detects_research_import():
    with pytest.raises(RuntimeError, match="research modules"):
        assert_no_research_imports(["quantgold.research.ablation", "numpy"])


def test_execution_guard_allows_clean_modules():
    assert_no_research_imports(["quantgold.execution.paper", "quantgold.risk.engine"])
