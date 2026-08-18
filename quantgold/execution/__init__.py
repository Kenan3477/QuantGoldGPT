"""Live / paper execution adapters (isolated from research)."""

from quantgold.execution.base import BrokerAdapter, OrderRequest, OrderResult
from quantgold.execution.paper import PaperBroker

__all__ = ["BrokerAdapter", "OrderRequest", "OrderResult", "PaperBroker"]
