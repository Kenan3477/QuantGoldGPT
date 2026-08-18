"""Trained meta-label model: P(candidate trade succeeds)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from quantgold.meta_models.base import MetaLabelDecision, MetaLabelModel


class SklearnMetaLabelModel(MetaLabelModel):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._model = GradientBoostingClassifier(random_state=random_state)
        self._fitted = False
        self.feature_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y_success: pd.Series) -> "SklearnMetaLabelModel":
        self.feature_columns = list(X.columns)
        if y_success.nunique() < 2 or len(X) < 20:
            # Not enough diversity — mark unfitted; decide() falls back to passthrough
            self._fitted = False
            return self
        self._model.fit(X.values, y_success.astype(int).values)
        self._fitted = True
        return self

    def decide(
        self,
        X: pd.DataFrame,
        *,
        min_success_probability: float = 0.6,
    ) -> list[MetaLabelDecision]:
        if not self._fitted:
            # Passthrough using raw_confidence if present
            out = []
            for _, row in X.iterrows():
                p = float(row.get("raw_confidence", row.get("probability_buy", 0.5)))
                out.append(
                    MetaLabelDecision(
                        take_trade=p >= min_success_probability,
                        success_probability=p,
                        reason="meta_unfitted_passthrough",
                    )
                )
            return out
        proba = self._model.predict_proba(X[self.feature_columns].values)
        # positive class column
        classes = list(self._model.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        p_success = proba[:, pos_idx]
        return [
            MetaLabelDecision(
                take_trade=bool(p >= min_success_probability),
                success_probability=float(p),
                reason="meta_model",
            )
            for p in p_success
        ]
