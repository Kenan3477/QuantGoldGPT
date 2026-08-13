"""
QuantGold — modular XAUUSD / XAGUSD quantitative research & selective signal platform.

XAUBot AI is used only as an engineering scaffold. Trading logic, labels, thresholds,
and performance claims are independently redesigned and validated.
"""

__version__ = "0.1.0"
__system__ = "QuantGold"

# Public surface kept intentionally narrow.
from quantgold.config.settings import QuantGoldSettings, load_settings

__all__ = [
    "__version__",
    "__system__",
    "QuantGoldSettings",
    "load_settings",
]
