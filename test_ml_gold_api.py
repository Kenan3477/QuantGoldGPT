#!/usr/bin/env python3
"""
Test ML Prediction System with new Gold API
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ml_gold_api():
    print('🧪 Testing ML Prediction System with new Gold API')
    print('=' * 50)
    
    try:
        from ml_prediction_api import get_real_time_gold_price
        
        print("✅ Successfully imported get_real_time_gold_price")
        
        # Test the function
        price = get_real_time_gold_price()
        print(f"✅ Current Gold Price from ML API: ${price:.2f}")
        
        if price > 3000:
            print("✅ Price looks realistic (above $3000)")
        else:
            print("❌ Price looks too low - might be using fallback")
            
    except Exception as e:
        print(f"❌ Error testing ML Gold API: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_gold_api()
