"""
Measurable risk engine.

Confidence may gate entries elsewhere; position size is computed here with
strict upper bounds. ML confidence cannot unbounded-leverage size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantgold.config.settings import RiskConfig


@dataclass
class RiskDecision:
    approved: bool
    lots: float
    risk_amount: float
    reason: str


class RiskEngine:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self._daily_loss_pct = 0.0
        self._weekly_loss_pct = 0.0
        self._open_positions = 0
        self._portfolio_heat_pct = 0.0

    def update_state(
        self,
        *,
        daily_loss_pct: float = 0.0,
        weekly_loss_pct: float = 0.0,
        open_positions: int = 0,
        portfolio_heat_pct: float = 0.0,
    ) -> None:
        self._daily_loss_pct = daily_loss_pct
        self._weekly_loss_pct = weekly_loss_pct
        self._open_positions = open_positions
        self._portfolio_heat_pct = portfolio_heat_pct

    def size_order(
        self,
        *,
        equity: float,
        stop_distance_price: float,
        point_value: float = 1.0,
        confidence: float = 0.5,
    ) -> RiskDecision:
        cfg = self.config
        if self._daily_loss_pct >= cfg.max_daily_loss_pct:
            return RiskDecision(False, 0.0, 0.0, "max_daily_loss")
        if self._weekly_loss_pct >= cfg.max_weekly_loss_pct:
            return RiskDecision(False, 0.0, 0.0, "max_weekly_loss")
        if self._open_positions >= cfg.max_positions:
            return RiskDecision(False, 0.0, 0.0, "max_positions")
        if self._portfolio_heat_pct >= cfg.max_portfolio_heat_pct:
            return RiskDecision(False, 0.0, 0.0, "portfolio_heat")
        if stop_distance_price <= 0 or equity <= 0:
            return RiskDecision(False, 0.0, 0.0, "invalid_inputs")

        # Confidence multiplier capped
        conf = max(0.0, min(1.0, confidence))
        mult = 1.0 + (cfg.max_confidence_size_multiplier - 1.0) * conf
        mult = min(mult, cfg.max_confidence_size_multiplier)

        risk_amount = equity * (cfg.risk_per_trade_pct / 100.0) * mult
        lots = risk_amount / (stop_distance_price * point_value)
        lots = max(0.0, lots)
        return RiskDecision(True, float(lots), float(risk_amount), "approved")
