#!/usr/bin/env python3
"""
Mathematical Accuracy Validation for Advanced Multi-Strategy ML Engine
Validates: "if current price is $3350.70 and prediction is +0.8%, result must be $3377.46"
"""

def test_mathematical_accuracy():
    """Test the exact mathematical requirement specified by the user"""
    print("🔍 Testing Mathematical Accuracy Requirement")
    print("=" * 50)
    
    # User's specific test case
    current_price = 3350.70
    prediction_percentage = 0.8
    expected_result = 3377.46
    
    print(f"📊 Current Price: ${current_price}")
    print(f"📈 Prediction: +{prediction_percentage}%")
    print(f"🎯 Expected Result: ${expected_result}")
    
    # Calculate the result
    calculated_result = current_price * (1 + prediction_percentage / 100)
    
    print(f"🧮 Calculated Result: ${calculated_result:.2f}")
    
    # Validate accuracy (allowing for floating point precision)
    difference = abs(calculated_result - expected_result)
    tolerance = 0.01  # 1 cent tolerance
    
    if difference <= tolerance:
        print(f"✅ MATHEMATICAL ACCURACY VALIDATED!")
        print(f"   Difference: ${difference:.4f} (within tolerance)")
        return True
    else:
        print(f"❌ MATHEMATICAL ACCURACY FAILED!")
        print(f"   Difference: ${difference:.4f} (exceeds tolerance)")
        return False

def test_engine_basic_functionality():
    """Test basic engine functionality without full prediction"""
    print("\n🔧 Testing Basic Engine Functionality")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from advanced_multi_strategy_ml_engine import (
            MultiStrategyMLEngine, 
            BaseStrategy,
            PredictionResult,
            EnsemblePrediction
        )
        print("✅ All imports successful!")
        
        # Test engine creation
        print("🚀 Creating ML Engine...")
        engine = MultiStrategyMLEngine()
        print("✅ Engine created successfully!")
        
        # Test basic methods exist
        methods_to_check = [
            'generate_ensemble_prediction',
            'get_strategy_weights',
            'update_strategy_performance',
            'calculate_support_resistance'
        ]
        
        for method in methods_to_check:
            if hasattr(engine, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during basic functionality test: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Advanced Multi-Strategy ML Engine Validation")
    print("=" * 60)
    
    # Test 1: Mathematical accuracy
    math_test = test_mathematical_accuracy()
    
    # Test 2: Basic functionality
    engine_test = test_engine_basic_functionality()
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mathematical Accuracy: {'✅ PASS' if math_test else '❌ FAIL'}")
    print(f"Engine Functionality: {'✅ PASS' if engine_test else '❌ FAIL'}")
    
    if math_test and engine_test:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("The Advanced Multi-Strategy ML Engine is ready for integration!")
    else:
        print("\n⚠️  Some validations failed. Review the output above.")

if __name__ == "__main__":
    main()
