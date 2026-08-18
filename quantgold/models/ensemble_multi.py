"""
Enhanced ensemble system for QuantGold.

Supports multiple base models and combination strategies:
- Simple averaging
- Weighted averaging (by validation performance)
- Stacking (meta-model on base predictions)
- Disagreement filtering
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from quantgold.models.base import ProbabilisticModel, ModelPrediction, Side


@dataclass
class EnsembleMember:
    """A single model in the ensemble."""
    name: str
    model: ProbabilisticModel
    weight: float = 1.0  # For weighted averaging


class MultiModelEnsemble(ProbabilisticModel):
    """
    Ensemble of multiple models with various combination strategies.
    
    Example:
        # Create ensemble with 5 models
        ensemble = MultiModelEnsemble(
            models=[
                EnsembleMember("xgboost", xgb_model, weight=1.2),
                EnsembleMember("lightgbm", lgbm_model, weight=1.0),
                EnsembleMember("catboost", cat_model, weight=1.1),
                EnsembleMember("randomforest", rf_model, weight=0.9),
                EnsembleMember("extratrees", et_model, weight=0.8),
            ],
            strategy="weighted_average",
            disagreement_threshold=0.15,
        )
        
        # Fit all models
        ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Predict
        predictions = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        models: List[EnsembleMember],
        strategy: str = "weighted_average",
        disagreement_threshold: float = 0.15,
        min_agreement_count: int = 3,
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of EnsembleMember objects
            strategy: Combination strategy:
                - "simple_average": Equal weight averaging
                - "weighted_average": Use member weights
                - "majority_vote": Majority class, fallback to NO_TRADE
                - "disagreement_filter": Only trade if models agree
            disagreement_threshold: Max allowed disagreement (std dev of probabilities)
            min_agreement_count: Min models agreeing on direction for trade
        """
        self.models = models
        self.strategy = strategy
        self.disagreement_threshold = disagreement_threshold
        self.min_agreement_count = min_agreement_count
        
        # Normalize weights
        total_weight = sum(m.weight for m in models)
        for m in self.models:
            m.weight = m.weight / total_weight
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        **kwargs,
    ) -> MultiModelEnsemble:
        """
        Train all models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            **kwargs: Additional model-specific parameters
            
        Returns:
            self
        """
        for member in self.models:
            print(f"Training {member.name}...")
            try:
                # Try to pass validation data if model supports it
                member.model.fit(X, y, X_val=X_val, y_val=y_val, **kwargs)
            except TypeError:
                # Model doesn't support X_val/y_val
                member.model.fit(X, y, **kwargs)
        
        # Auto-weight by validation performance if available
        if X_val is not None and y_val is not None:
            self._auto_weight_by_performance(X_val, y_val)
        
        return self
    
    def _auto_weight_by_performance(
        self,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ):
        """
        Automatically adjust weights based on validation performance.
        
        Uses log loss as the metric (lower is better).
        """
        from sklearn.metrics import log_loss
        
        performances = []
        for member in self.models:
            try:
                proba = member.model.predict_proba(X_val)
                loss = log_loss(y_val, proba)
                performances.append(loss)
            except Exception as e:
                print(f"Warning: Could not evaluate {member.name}: {e}")
                performances.append(1.0)  # Default/neutral loss
        
        # Convert losses to weights (inverse: lower loss = higher weight)
        # Use exp(-loss) to emphasize differences
        weights = np.exp(-np.array(performances))
        weights = weights / weights.sum()  # Normalize
        
        for i, member in enumerate(self.models):
            member.weight = weights[i]
            print(f"  {member.name}: weight={member.weight:.3f}, val_loss={performances[i]:.3f}")
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble strategy.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 2) with ensemble probabilities
        """
        # Get predictions from all models
        all_proba = []
        for member in self.models:
            proba = member.model.predict_proba(X)
            all_proba.append(proba)
        
        # Stack into 3D array: (n_models, n_samples, 2)
        all_proba = np.array(all_proba)
        
        # Apply strategy
        if self.strategy == "simple_average":
            return self._simple_average(all_proba)
        elif self.strategy == "weighted_average":
            return self._weighted_average(all_proba)
        elif self.strategy == "majority_vote":
            return self._majority_vote(all_proba)
        elif self.strategy == "disagreement_filter":
            return self._disagreement_filter(all_proba)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _simple_average(self, all_proba: np.ndarray) -> np.ndarray:
        """Simple arithmetic mean of probabilities."""
        return np.mean(all_proba, axis=0)
    
    def _weighted_average(self, all_proba: np.ndarray) -> np.ndarray:
        """Weighted average using member weights."""
        weights = np.array([m.weight for m in self.models])
        weights = weights.reshape(-1, 1, 1)  # Broadcast to (n_models, 1, 1)
        return np.sum(all_proba * weights, axis=0)
    
    def _majority_vote(self, all_proba: np.ndarray) -> np.ndarray:
        """Majority vote with probability fallback."""
        # Get predicted class for each model (0 or 1)
        predictions = (all_proba[:, :, 1] > 0.5).astype(int)  # (n_models, n_samples)
        
        # Count votes for class 1
        votes_for_1 = np.sum(predictions, axis=0)
        
        # Majority class
        ensemble_class = (votes_for_1 > len(self.models) / 2).astype(int)
        
        # Convert back to probabilities
        # If strong majority, use high confidence; else use vote proportion
        confidence = votes_for_1 / len(self.models)
        
        proba = np.zeros((len(ensemble_class), 2))
        proba[:, 1] = confidence
        proba[:, 0] = 1 - confidence
        
        return proba
    
    def _disagreement_filter(self, all_proba: np.ndarray) -> np.ndarray:
        """
        Filter predictions where models disagree.
        
        Sets probability to 0.5 (neutral) if disagreement is high.
        """
        # Calculate disagreement (std dev of positive class probability)
        positive_proba = all_proba[:, :, 1]  # (n_models, n_samples)
        disagreement = np.std(positive_proba, axis=0)  # (n_samples,)
        
        # Get weighted average
        ensemble_proba = self._weighted_average(all_proba)
        
        # Where disagreement is high, set to neutral (0.5)
        high_disagreement = disagreement > self.disagreement_threshold
        ensemble_proba[high_disagreement, :] = 0.5
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> List[ModelPrediction]:
        """
        Predict with full metadata.
        
        Args:
            X: Features
            
        Returns:
            List of ModelPrediction objects (one per sample)
        """
        proba = self.predict_proba(X)
        
        # Calculate disagreement stats
        all_proba = np.array([m.model.predict_proba(X) for m in self.models])
        positive_proba = all_proba[:, :, 1]
        disagreement = np.std(positive_proba, axis=0)
        
        results = []
        for i in range(len(X)):
            p_buy = float(proba[i, 1])
            p_sell = float(proba[i, 0])
            
            # Determine side
            if p_buy > 0.5:
                side = Side.BUY
                conf = p_buy
            elif p_sell > 0.5:
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
                model_name=f"Ensemble_{self.strategy}_{len(self.models)}models",
                extras={
                    "member_count": len(self.models),
                    "member_names": [m.name for m in self.models],
                    "disagreement": float(disagreement[i]),
                    "disagreement_mean": float(np.mean(disagreement)),
                    "disagreement_max": float(np.max(disagreement)),
                },
            ))
        
        return results
    
    def get_member_predictions(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each ensemble member separately.
        
        Args:
            X: Features
            
        Returns:
            Dict mapping member name → probability array
        """
        results = {}
        for member in self.models:
            proba = member.model.predict_proba(X)
            results[member.name] = proba
        return results
    
    def get_disagreement_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Get disagreement score for each prediction.
        
        Args:
            X: Features
            
        Returns:
            Array of disagreement scores (std dev of probabilities)
        """
        all_proba = np.array([m.model.predict_proba(X) for m in self.models])
        positive_proba = all_proba[:, :, 1]
        return np.std(positive_proba, axis=0)
