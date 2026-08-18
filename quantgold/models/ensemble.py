"""Ensemble agreement / disagreement filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from quantgold.models.base import ModelPrediction, Side


@dataclass
class EnsembleAgreementFilter:
    """
    Require model agreement before allowing a trade.

    Example:
      strong agreement → keep side
      disagreement → NO_TRADE
    """

    max_disagreement: float = 0.20
    min_mean_probability: float = 0.65

    def combine(self, predictions: Sequence[ModelPrediction]) -> ModelPrediction:
        if not predictions:
            return ModelPrediction(
                side=Side.NO_TRADE,
                probability_buy=0.5,
                probability_sell=0.5,
                raw_confidence=0.0,
                model_name="ensemble_empty",
            )

        p_buys = [p.probability_buy for p in predictions]
        mean_buy = sum(p_buys) / len(p_buys)
        disagreement = max(p_buys) - min(p_buys)
        mean_sell = 1.0 - mean_buy

        if disagreement > self.max_disagreement:
            side = Side.NO_TRADE
            conf = max(mean_buy, mean_sell)
        elif mean_buy >= self.min_mean_probability:
            side = Side.BUY
            conf = mean_buy
        elif mean_sell >= self.min_mean_probability:
            side = Side.SELL
            conf = mean_sell
        else:
            side = Side.NO_TRADE
            conf = max(mean_buy, mean_sell)

        return ModelPrediction(
            side=side,
            probability_buy=mean_buy,
            probability_sell=mean_sell,
            raw_confidence=conf,
            model_name="ensemble_mean",
            extras={
                "disagreement": disagreement,
                "members": [p.model_name for p in predictions],
            },
        )
