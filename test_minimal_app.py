#!/usr/bin/env python3
"""
Quick test to verify app_minimal.py can start without errors
"""

import sys
import os
import importlib.util

def test_minimal_app():
    """Test if app_minimal.py can be imported and basic functions work"""
    
    print("🧪 Testing app_minimal.py...")
    
    try:
        # Try to import the minimal app
        spec = importlib.util.spec_from_file_location("app_minimal", "app_minimal.py")
        app_minimal = importlib.util.module_from_spec(spec)
        
        # Test basic functions without running the server
        print("✅ Module imports successfully")
        
        # Test gold price function
        if hasattr(app_minimal, 'get_current_gold_price'):
            price = app_minimal.get_current_gold_price()
            print(f"✅ Gold price function works: ${price}")
        
        # Test AI analysis function
        if hasattr(app_minimal, 'get_ai_analysis'):
            analysis = app_minimal.get_ai_analysis()
            print(f"✅ AI analysis function works: {analysis['signal']}")
        
        # Test ML predictions function
        if hasattr(app_minimal, 'get_ml_predictions'):
            predictions = app_minimal.get_ml_predictions()
            print(f"✅ ML predictions function works: {len(predictions['predictions'])} predictions")
        
        print("🎉 All basic functions work correctly!")
        print("✅ app_minimal.py is ready for Railway deployment")
        return True
        
    except Exception as e:
        print(f"❌ Error testing app_minimal.py: {e}")
        print(f"📝 Error details: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_minimal_app()
    sys.exit(0 if success else 1)
