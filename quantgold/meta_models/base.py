"""Meta-label interface: should we trust this candidate trade?"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantgold.models.base import Side


@dataclass
class MetaLabelDecision:
    take_trade: bool
    success_probability: float
    reason: str


class MetaLabelModel(ABC):
    """
    Second-stage model.

    Inputs may include base probability, disagreement, regime, session, ATR, etc.
    Output is P(candidate trade succeeds) and a take/skip decision.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y_success: pd.Series) -> "MetaLabelModel":
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        X: pd.DataFrame,
        *,
        min_success_probability: float = 0.6,
    ) -> list[MetaLabelDecision]:
        raise NotImplementedError


class ThresholdMetaLabel(MetaLabelModel):
    """
    Placeholder meta-labeler using calibrated base probability only.

    Replaced by a trained model once baseline walk-forward exists.
    """

    def __init__(self, probability_col: str = "probability_buy"):
        self.probability_col = probability_col
        self._fitted = False

    def fit(self, X: pd.DataFrame, y_success: pd.Series) -> "ThresholdMetaLabel":
        self._fitted = True
        return self

    def decide(
        self,
        X: pd.DataFrame,
        *,
        min_success_probability: float = 0.6,
    ) -> list[MetaLabelDecision]:
        out = []
        for _, row in X.iterrows():
            # Use max directional probability if side-specific cols exist
            p = float(row.get("raw_confidence", row.get(self.probability_col, 0.5)))
            take = p >= min_success_probability
            out.append(
                MetaLabelDecision(
                    take_trade=take,
                    success_probability=p,
                    reason="threshold_meta_placeholder",
                )
            )
        return out
