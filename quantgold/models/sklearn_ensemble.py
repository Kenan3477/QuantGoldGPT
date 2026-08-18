"""
Sklearn ensemble models for QuantGold.

RandomForest and ExtraTrees (Extremely Randomized Trees) are both
ensemble methods that build multiple decision trees.

Key differences:
- RandomForest: Finds best split for each node using bootstrap samples
- ExtraTrees: Uses random splits (more randomization, faster training)

Both are decorrelated from gradient boosting methods, making them
good ensemble members.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from quantgold.models.base import ProbabilisticModel, ModelPrediction, Side


class RandomForestModel(ProbabilisticModel):
    """
    RandomForest classifier adapter for QuantGold.
    
    RandomForest builds multiple decision trees on bootstrap samples
    and averages their predictions.
    
    Advantages:
    - Robust to overfitting (with enough trees)
    - Handles missing values and outliers well
    - Provides feature importance
    - Decorrelated from gradient boosting
    
    Example:
        model = RandomForestModel(n_estimators=200, max_depth=10)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Initialize RandomForest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Max tree depth (None = unlimited)
            min_samples_split: Min samples to split a node
            min_samples_leaf: Min samples in a leaf
            max_features: Number of features per split ("sqrt", "log2", or int)
            random_state: Random seed
            n_jobs: Parallel jobs (-1 = all cores)
            **kwargs: Additional sklearn parameters
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> RandomForestModel:
        """
        Train RandomForest model.
        
        Args:
            X: Training features
            y: Training labels (-1/0/1, will be converted to binary 0/1)
            **kwargs: Ignored (for API compatibility)
            
        Returns:
            self
        """
        # Convert labels to binary (same as XGBoost)
        # -1 (DOWN) and 0 (TIMEOUT) → 0, 1 (UP) → 1
        y_arr = y.values if isinstance(y, pd.Series) else y
        y_bin = (y_arr.astype(float) == 1).astype(int)
        
        self.model.fit(X, y_bin)
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
                model_name="RandomForest",
            ))
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (mean decrease in impurity).
        
        Returns:
            DataFrame with feature names and importances
        """
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(self.model.n_features_in_)]
        
        importances = self.model.feature_importances_
        
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)


class ExtraTreesModel(ProbabilisticModel):
    """
    ExtraTrees (Extremely Randomized Trees) classifier adapter.
    
    ExtraTrees is similar to RandomForest but uses random splits
    instead of optimal splits, making it:
    - Faster to train (no split optimization)
    - More randomized (better decorrelation)
    - Less prone to overfitting
    
    Good for ensembles because it's highly decorrelated from
    both RandomForest and gradient boosting methods.
    
    Example:
        model = ExtraTreesModel(n_estimators=200, max_depth=10)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Initialize ExtraTrees model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Max tree depth (None = unlimited)
            min_samples_split: Min samples to split a node
            min_samples_leaf: Min samples in a leaf
            max_features: Number of features per split ("sqrt", "log2", or int)
            random_state: Random seed
            n_jobs: Parallel jobs (-1 = all cores)
            **kwargs: Additional sklearn parameters
        """
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> ExtraTreesModel:
        """
        Train ExtraTrees model.
        
        Args:
            X: Training features
            y: Training labels (-1/0/1, will be converted to binary 0/1)
            **kwargs: Ignored (for API compatibility)
            
        Returns:
            self
        """
        # Convert labels to binary (same as XGBoost)
        # -1 (DOWN) and 0 (TIMEOUT) → 0, 1 (UP) → 1
        y_arr = y.values if isinstance(y, pd.Series) else y
        y_bin = (y_arr.astype(float) == 1).astype(int)
        
        self.model.fit(X, y_bin)
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
                model_name="ExtraTrees",
            ))
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (mean decrease in impurity).
        
        Returns:
            DataFrame with feature names and importances
        """
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(self.model.n_features_in_)]
        
        importances = self.model.feature_importances_
        
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
