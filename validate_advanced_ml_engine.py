#!/usr/bin/env python3
"""
Quick validation of Advanced ML Prediction Engine
"""

def test_imports():
    """Test if all components can be imported"""
    try:
        from advanced_ml_prediction_engine import (
            PredictionResult, EnsemblePrediction,
            StrategyPerformanceTracker, PredictionValidator,
            TechnicalStrategy, SentimentStrategy, MacroStrategy,
            PatternStrategy, MomentumStrategy, AdvancedMLPredictionEngine
        )
        print("✅ All core classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_structures():
    """Test core data structures"""
    try:
        from advanced_ml_prediction_engine import PredictionResult
        from datetime import datetime, timezone
        
        prediction = PredictionResult(
            strategy_name="Test",
            timeframe="1H",
            current_price=2000.0,
            predicted_price=2010.0,
            price_change=10.0,
            price_change_percent=0.5,
            direction="bullish",
            confidence=0.7,
            support_level=1990.0,
            resistance_level=2020.0,
            stop_loss=1980.0,
            take_profit=2030.0,
            timestamp=datetime.now(timezone.utc),
            features_used=["test"],
            reasoning="test prediction"
        )
        
        assert prediction.current_price == 2000.0
        assert prediction.direction == "bullish"
        print("✅ PredictionResult works correctly")
        return True
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_performance_tracker():
    """Test performance tracking"""
    try:
        from advanced_ml_prediction_engine import StrategyPerformanceTracker
        
        tracker = StrategyPerformanceTracker()
        
        # Record performance
        tracker.record_performance("TestStrategy", 2000, 2005, 0.8)
        tracker.record_performance("TestStrategy", 2010, 2015, 0.7)
        
        # Get weights
        weights = tracker.get_all_weights(["TestStrategy"])
        
        assert "TestStrategy" in weights
        assert 0 < weights["TestStrategy"] <= 1
        print(f"✅ PerformanceTracker works: weight = {weights['TestStrategy']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Performance tracker test failed: {e}")
        return False

def test_validator():
    """Test prediction validator"""
    try:
        from advanced_ml_prediction_engine import PredictionValidator, PredictionResult
        from datetime import datetime, timezone
        
        validator = PredictionValidator()
        
        # Create valid prediction
        prediction = PredictionResult(
            strategy_name="Test",
            timeframe="1H",
            current_price=2000.0,
            predicted_price=2010.0,
            price_change=10.0,
            price_change_percent=0.5,
            direction="bullish",
            confidence=0.7,
            support_level=1990.0,
            resistance_level=2020.0,
            stop_loss=1980.0,
            take_profit=2030.0,
            timestamp=datetime.now(timezone.utc),
            features_used=["test"],
            reasoning="test"
        )
        
        is_valid, message, score = validator.validate_prediction(prediction)
        
        assert isinstance(is_valid, bool)
        assert isinstance(score, float)
        print(f"✅ Validator works: valid = {is_valid}, score = {score:.3f}")
        return True
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        return False

def main():
    print("🧪 Advanced ML Prediction Engine - Quick Validation")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structures", test_data_structures),
        ("Performance Tracker", test_performance_tracker),
        ("Validator", test_validator)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"📊 Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Advanced ML Engine is ready.")
    else:
        print("⚠️  Some tests failed. Check dependencies and implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
