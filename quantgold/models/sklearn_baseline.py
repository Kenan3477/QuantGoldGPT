"""
Sklearn gradient-boosting baseline.

Used as a dependency-light baseline before XGBoost/LightGBM/CatBoost adapters.
Not claimed to be optimal — exists for walk-forward plumbing tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from quantgold.models.base import ProbabilisticModel


class SklearnGBMBaseline(ProbabilisticModel):
    name = "sklearn_gbm_baseline"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._model = GradientBoostingClassifier(random_state=random_state)
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SklearnGBMBaseline":
        # Map triple-barrier {-1,0,1,2} → binary up vs not-up for smoke baseline.
        y_bin = (y.astype(float) == 1).astype(int)
        self._model.fit(X.values, y_bin.values, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X.values)
