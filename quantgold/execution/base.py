"""Broker adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from quantgold.models.base import Side


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    lots: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "QuantGold"


@dataclass
class OrderResult:
    accepted: bool
    order_id: Optional[str]
    fill_price: Optional[float]
    reason: str


class BrokerAdapter(ABC):
    @abstractmethod
    def submit(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError
