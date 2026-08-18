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
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        from xgboost import XGBClassifier

        # Convert to numpy if needed
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        y_bin = (y_arr.astype(float) == 1).astype(int)
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
        self._model.fit(X_arr, y_bin, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict_proba(X_arr)


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
    
    # Add ensemble if all required models are available
    has_xgb = "xgboost" in backends
    has_lgbm = "lightgbm" in backends
    try:
        import catboost  # noqa: F401
        has_catboost = True
    except ImportError:
        has_catboost = False
    
    # Ensemble requires at least XGB, LGBM, CatBoost, RF, ExtraTrees
    # RF and ExtraTrees are always available (sklearn)
    if has_xgb and has_lgbm and has_catboost:
        backends.append("ensemble")
    
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
    if name == "ensemble":
        from quantgold.models.ensemble_multi import MultiModelEnsemble, EnsembleMember
        from quantgold.models.lightgbm_model import LightGBMModel
        from quantgold.models.catboost_model import CatBoostModel
        from quantgold.models.sklearn_ensemble import RandomForestModel, ExtraTreesModel
        
        # Create 5-model ensemble with equal weights (simple average for now)
        models = [
            EnsembleMember("xgb", XGBoostModel(random_state=random_state, n_estimators=100)),
            EnsembleMember("lgbm", LightGBMModel(random_state=random_state, n_estimators=100)),
            EnsembleMember("cat", CatBoostModel(random_seed=random_state, iterations=100, verbose=False)),  # CatBoost uses random_seed
            EnsembleMember("rf", RandomForestModel(random_state=random_state, n_estimators=100)),
            EnsembleMember("et", ExtraTreesModel(random_state=random_state, n_estimators=100)),
        ]
        
        return MultiModelEnsemble(
            models=models,
            strategy="simple_average",  # Equal weights
        )
    raise ValueError(f"Unknown model: {name}")
