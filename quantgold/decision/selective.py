"""NO_TRADE-aware selective decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantgold.models.base import Side


@dataclass
class SelectiveDecision:
    side: Side
    calibrated_probability: float
    meta_probability: float
    reason: str
    threshold: float


@dataclass
class SelectivePolicy:
    min_calibrated_probability: float = 0.65
    min_meta_probability: float = 0.60
    max_disagreement: float = 0.20
    require_meta: bool = True
    event_block: bool = True

    def decide(
        self,
        *,
        candidate_side: Side,
        calibrated_probability: float,
        meta_probability: Optional[float] = None,
        disagreement: float = 0.0,
        event_blocked: bool = False,
    ) -> SelectiveDecision:
        thr = self.min_calibrated_probability
        if event_blocked and self.event_block:
            return SelectiveDecision(Side.NO_TRADE, calibrated_probability, meta_probability or 0.0, "event_block", thr)
        if candidate_side == Side.NO_TRADE:
            return SelectiveDecision(Side.NO_TRADE, calibrated_probability, meta_probability or 0.0, "base_no_trade", thr)
        if disagreement > self.max_disagreement:
            return SelectiveDecision(Side.NO_TRADE, calibrated_probability, meta_probability or 0.0, "model_disagreement", thr)
        if calibrated_probability < self.min_calibrated_probability:
            return SelectiveDecision(Side.NO_TRADE, calibrated_probability, meta_probability or 0.0, "low_calibrated_prob", thr)
        meta_p = meta_probability if meta_probability is not None else calibrated_probability
        if self.require_meta and meta_p < self.min_meta_probability:
            return SelectiveDecision(Side.NO_TRADE, calibrated_probability, meta_p, "meta_reject", thr)
        return SelectiveDecision(candidate_side, calibrated_probability, meta_p, "accepted", thr)
