"""Common probabilistic model interface for fair comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


@dataclass
class ModelPrediction:
    side: Side
    probability_buy: float
    probability_sell: float
    raw_confidence: float
    model_name: str
    extras: Optional[Dict[str, Any]] = None


class ProbabilisticModel(ABC):
    """Tabular classifier interface (XGBoost / LightGBM / CatBoost adapters)."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ProbabilisticModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return array shape (n, 2) as [p_sell_or_down, p_buy_or_up] or binary p_up in col1."""
        raise NotImplementedError

    def predict(
        self,
        X: pd.DataFrame,
        *,
        min_probability: float = 0.65,
        allow_no_trade: bool = True,
    ) -> List[ModelPrediction]:
        proba = self.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return shape (n, 2+)")
        # Convention: column -1 is P(up/BUY)
        p_buy = proba[:, -1]
        p_sell = 1.0 - p_buy
        out: List[ModelPrediction] = []
        for pb, ps in zip(p_buy, p_sell):
            if pb >= min_probability and pb >= ps:
                side = Side.BUY
                conf = float(pb)
            elif ps >= min_probability and ps > pb:
                side = Side.SELL
                conf = float(ps)
            else:
                side = Side.NO_TRADE if allow_no_trade else (Side.BUY if pb >= ps else Side.SELL)
                conf = float(max(pb, ps))
            out.append(
                ModelPrediction(
                    side=side,
                    probability_buy=float(pb),
                    probability_sell=float(ps),
                    raw_confidence=conf,
                    model_name=self.name,
                )
            )
        return out
