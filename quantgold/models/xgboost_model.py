"""XGBoost adapter behind ProbabilisticModel (optional dependency)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantgold.models.base import ProbabilisticModel


class XGBoostModel(ProbabilisticModel):
    name = "xgboost"

    def __init__(self, random_state: int = 42, n_estimators: int = 100, max_depth: int = 4):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        from xgboost import XGBClassifier

        y_bin = (y.astype(float) == 1).astype(int)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=2,
        )
        self._model.fit(X.values, y_bin.values, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X.values)


def available_model_backends() -> list[str]:
    backends = ["sklearn_gbm_baseline"]
    try:
        import xgboost  # noqa: F401

        backends.append("xgboost")
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401

        backends.append("lightgbm")
    except ImportError:
        pass
    return backends


def make_model(name: str, random_state: int = 42) -> ProbabilisticModel:
    if name == "sklearn_gbm_baseline":
        from quantgold.models.sklearn_baseline import SklearnGBMBaseline

        return SklearnGBMBaseline(random_state=random_state)
    if name == "xgboost":
        return XGBoostModel(random_state=random_state)
    if name == "lightgbm":
        from quantgold.models.lightgbm_model import LightGBMModel

        return LightGBMModel(random_state=random_state)
    raise ValueError(f"Unknown model: {name}")
