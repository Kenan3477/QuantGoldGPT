"""Market data ingest adapters."""

from quantgold.data.ingest.base import MarketDataSource
from quantgold.data.ingest.yfinance_source import YFinanceSource
from quantgold.data.ingest.synthetic import SyntheticSource

__all__ = ["MarketDataSource", "YFinanceSource", "SyntheticSource"]
