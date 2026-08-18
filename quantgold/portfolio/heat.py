"""Portfolio heat tracking for XAU + XAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PortfolioHeat:
    max_heat_pct: float = 3.0
    _open_risk_pct: Dict[str, float] = field(default_factory=dict)

    def set_position_risk(self, symbol: str, risk_pct: float) -> None:
        self._open_risk_pct[symbol.upper()] = float(risk_pct)

    def clear(self, symbol: str) -> None:
        self._open_risk_pct.pop(symbol.upper(), None)

    @property
    def total_heat_pct(self) -> float:
        return float(sum(self._open_risk_pct.values()))

    def can_add(self, additional_risk_pct: float) -> bool:
        return (self.total_heat_pct + additional_risk_pct) <= self.max_heat_pct
