"""
CatBoost model adapter for QuantGold.

CatBoost is a gradient boosting library with native support for:
- Categorical features (no need for one-hot encoding)
- Missing values
- GPU training
- Ordered boosting (reduces overfitting)

Install: pip install catboost
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd

from quantgold.models.base import ProbabilisticModel, ModelPrediction, Side


class CatBoostModel(ProbabilisticModel):
    """
    CatBoost classifier adapter for QuantGold.
    
    Example:
        model = CatBoostModel(iterations=500, depth=6)
        model.fit(X_train, y_train, categorical_features=['session', 'regime'])
        predictions = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        *,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize CatBoost model.
        
        Args:
            iterations: Number of boosting iterations
            depth: Tree depth
            learning_rate: Learning rate
            l2_leaf_reg: L2 regularization
            random_seed: Random seed for reproducibility
            verbose: Print training progress
            **kwargs: Additional CatBoost parameters
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError(
                "CatBoost not installed. Install with: pip install catboost"
            )
        
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=verbose,
            loss_function='Logloss',
            eval_metric='AUC',
            **kwargs,
        )
        self.categorical_features_ = None
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        categorical_features: Optional[list[str]] = None,
    ) -> CatBoostModel:
        """
        Train CatBoost model.
        
        Args:
            X: Training features
            y: Training labels (0/1)
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            categorical_features: List of categorical feature names
            
        Returns:
            self
        """
        # Store categorical features for later
        self.categorical_features_ = categorical_features
        
        # Prepare eval set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        # Fit model
        self.model.fit(
            X,
            y,
            cat_features=categorical_features,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            verbose=False,
        )
        
        return self
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [class_0, class_1]
        """
        return self.model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame | np.ndarray, *, min_probability: float = 0.5) -> List[ModelPrediction]:
        """
        Predict with full metadata.
        
        Args:
            X: Features
            min_probability: Minimum probability threshold for BUY/SELL
            
        Returns:
            List of ModelPrediction objects
        """
        proba = self.predict_proba(X)
        
        results = []
        for i in range(len(X)):
            p_buy = float(proba[i, 1])
            p_sell = float(proba[i, 0])
            
            if p_buy >= min_probability and p_buy >= p_sell:
                side = Side.BUY
                conf = p_buy
            elif p_sell >= min_probability and p_sell > p_buy:
                side = Side.SELL
                conf = p_sell
            else:
                side = Side.NO_TRADE
                conf = max(p_buy, p_sell)
            
            results.append(ModelPrediction(
                side=side,
                probability_buy=p_buy,
                probability_sell=p_sell,
                raw_confidence=conf,
                model_name="CatBoost",
            ))
        
        return results
    
    def get_feature_importance(
        self,
        importance_type: str = "PredictionValuesChange",
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance
                - "PredictionValuesChange": Default, prediction value change
                - "LossFunctionChange": Loss function change
                - "Interaction": Feature interaction strength
                
        Returns:
            DataFrame with feature names and importances
        """
        feature_names = self.model.feature_names_
        importances = self.model.get_feature_importance(type=importance_type)
        
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
