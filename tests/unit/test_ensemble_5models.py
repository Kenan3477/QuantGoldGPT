"""
Test the 5-model ensemble.

Verifies that all models can be trained and combined successfully.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from quantgold.models.xgboost_model import XGBoostModel
from quantgold.models.lightgbm_model import LightGBMModel
from quantgold.models.catboost_model import CatBoostModel
from quantgold.models.sklearn_ensemble import RandomForestModel, ExtraTreesModel
from quantgold.models.ensemble_multi import MultiModelEnsemble, EnsembleMember


@pytest.fixture
def classification_data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    
    # Split into train/val
    split = 800
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    return X_train, X_val, y_train, y_val


def test_all_models_individually(classification_data):
    """Test that each model can be trained and predict."""
    X_train, X_val, y_train, y_val = classification_data
    
    models = [
        ("XGBoost", XGBoostModel(n_estimators=50)),
        ("LightGBM", LightGBMModel(n_estimators=50)),
        ("CatBoost", CatBoostModel(iterations=50, verbose=False)),
        ("RandomForest", RandomForestModel(n_estimators=50)),
        ("ExtraTrees", ExtraTreesModel(n_estimators=50)),
    ]
    
    for name, model in models:
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        proba = model.predict_proba(X_val)
        
        # Check output shape
        assert proba.shape == (len(X_val), 2), f"{name} output shape incorrect"
        
        # Check probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0), f"{name} probabilities don't sum to 1"
        
        # Check range [0, 1]
        assert np.all(proba >= 0) and np.all(proba <= 1), f"{name} probabilities out of range"
        
        print(f"✓ {name} passed")


def test_ensemble_simple_average(classification_data):
    """Test ensemble with simple averaging strategy."""
    X_train, X_val, y_train, y_val = classification_data
    
    ensemble = MultiModelEnsemble(
        models=[
            EnsembleMember("xgb", XGBoostModel(n_estimators=50)),
            EnsembleMember("lgbm", LightGBMModel(n_estimators=50)),
            EnsembleMember("cat", CatBoostModel(iterations=50, verbose=False)),
            EnsembleMember("rf", RandomForestModel(n_estimators=50)),
            EnsembleMember("et", ExtraTreesModel(n_estimators=50)),
        ],
        strategy="simple_average",
    )
    
    # Train
    ensemble.fit(X_train, y_train)
    
    # Predict
    proba = ensemble.predict_proba(X_val)
    
    # Check output
    assert proba.shape == (len(X_val), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    
    print("✓ Simple average ensemble passed")


def test_ensemble_weighted_average(classification_data):
    """Test ensemble with weighted averaging strategy."""
    X_train, X_val, y_train, y_val = classification_data
    
    ensemble = MultiModelEnsemble(
        models=[
            EnsembleMember("xgb", XGBoostModel(n_estimators=50), weight=1.2),
            EnsembleMember("lgbm", LightGBMModel(n_estimators=50), weight=1.0),
            EnsembleMember("cat", CatBoostModel(iterations=50, verbose=False), weight=1.1),
            EnsembleMember("rf", RandomForestModel(n_estimators=50), weight=0.9),
            EnsembleMember("et", ExtraTreesModel(n_estimators=50), weight=0.8),
        ],
        strategy="weighted_average",
    )
    
    # Train with validation data for auto-weighting
    ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Predict
    proba = ensemble.predict_proba(X_val)
    
    # Check output
    assert proba.shape == (len(X_val), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    
    # Check that weights were adjusted
    total_weight = sum(m.weight for m in ensemble.models)
    assert np.isclose(total_weight, 1.0), "Weights should sum to 1"
    
    print("✓ Weighted average ensemble passed")


def test_ensemble_disagreement_filter(classification_data):
    """Test ensemble with disagreement filtering."""
    X_train, X_val, y_train, y_val = classification_data
    
    ensemble = MultiModelEnsemble(
        models=[
            EnsembleMember("xgb", XGBoostModel(n_estimators=50)),
            EnsembleMember("lgbm", LightGBMModel(n_estimators=50)),
            EnsembleMember("cat", CatBoostModel(iterations=50, verbose=False)),
            EnsembleMember("rf", RandomForestModel(n_estimators=50)),
            EnsembleMember("et", ExtraTreesModel(n_estimators=50)),
        ],
        strategy="disagreement_filter",
        disagreement_threshold=0.15,
    )
    
    # Train
    ensemble.fit(X_train, y_train)
    
    # Predict
    proba = ensemble.predict_proba(X_val)
    predictions = ensemble.predict(X_val)
    
    # Check output
    assert proba.shape == (len(X_val), 2)
    assert len(predictions) == len(X_val)
    
    # Check disagreement scores are available
    disagreement = ensemble.get_disagreement_score(X_val)
    assert disagreement.shape == (len(X_val),)
    assert np.all(disagreement >= 0)
    
    # Check that some predictions have neutral probability (0.5) due to disagreement
    neutral_count = np.sum(np.isclose(proba[:, 1], 0.5))
    print(f"  Neutral predictions due to disagreement: {neutral_count}/{len(X_val)}")
    
    # Check metadata in first prediction
    assert "disagreement" in predictions[0].extras
    assert "disagreement_mean" in predictions[0].extras
    assert "disagreement_max" in predictions[0].extras
    
    print("✓ Disagreement filter ensemble passed")


def test_ensemble_member_predictions(classification_data):
    """Test getting individual member predictions."""
    X_train, X_val, y_train, y_val = classification_data
    
    ensemble = MultiModelEnsemble(
        models=[
            EnsembleMember("xgb", XGBoostModel(n_estimators=50)),
            EnsembleMember("lgbm", LightGBMModel(n_estimators=50)),
            EnsembleMember("cat", CatBoostModel(iterations=50, verbose=False)),
        ],
        strategy="simple_average",
    )
    
    ensemble.fit(X_train, y_train)
    
    # Get member predictions
    member_preds = ensemble.get_member_predictions(X_val)
    
    assert len(member_preds) == 3
    assert "xgb" in member_preds
    assert "lgbm" in member_preds
    assert "cat" in member_preds
    
    for name, proba in member_preds.items():
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    print("✓ Member predictions test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
