"""Paper broker for forward evaluation before production."""

from __future__ import annotations

import itertools
from typing import List

from quantgold.execution.base import BrokerAdapter, OrderRequest, OrderResult
from quantgold.models.base import Side


class PaperBroker(BrokerAdapter):
    def __init__(self, fill_price: float = 0.0):
        self.fill_price = fill_price
        self._ids = itertools.count(1)
        self.orders: List[OrderRequest] = []

    def submit(self, order: OrderRequest) -> OrderResult:
        if order.side == Side.NO_TRADE:
            return OrderResult(False, None, None, "no_trade")
        if order.lots <= 0:
            return OrderResult(False, None, None, "non_positive_lots")
        oid = f"PAPER-{next(self._ids)}"
        self.orders.append(order)
        return OrderResult(True, oid, self.fill_price, "paper_fill")
