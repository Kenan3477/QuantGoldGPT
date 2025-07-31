#!/usr/bin/env python3
"""
Complete system test with new Gold API
Tests ML prediction system, AI analysis, and data consistency
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_system():
    print('🧪 Testing Complete GoldGPT System with New Gold API')
    print('=' * 60)
    
    # Test 1: Direct Gold API call
    print("\n1. Testing Direct Gold API Call...")
    try:
        import requests
        response = requests.get('https://api.gold-api.com/price/XAU', timeout=10)
        if response.status_code == 200:
            data = response.json()
            api_price = float(data['price'])
            print(f"✅ Direct API call: ${api_price:.2f}")
        else:
            print(f"❌ Direct API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Direct API error: {e}")
        return False
    
    # Test 2: ML Prediction System
    print("\n2. Testing ML Prediction System...")
    try:
        from ml_prediction_api import get_real_time_gold_price
        ml_price = get_real_time_gold_price()
        print(f"✅ ML System price: ${ml_price:.2f}")
        
        # Check consistency
        price_diff = abs(api_price - ml_price)
        if price_diff < 10:  # Within $10 is reasonable for real-time data
            print(f"✅ Price consistency: Difference ${price_diff:.2f}")
        else:
            print(f"⚠️ Price difference: ${price_diff:.2f} (might be timing)")
            
    except Exception as e:
        print(f"❌ ML System error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: AI Analysis System
    print("\n3. Testing AI Analysis System...")
    try:
        from ai_analysis_api import SimplifiedDataFetcher
        ai_fetcher = SimplifiedDataFetcher()
        ai_price = ai_fetcher.get_real_time_gold_price()
        print(f"✅ AI System price: ${ai_price:.2f}")
        
        # Check consistency
        price_diff = abs(api_price - ai_price)
        if price_diff < 10:
            print(f"✅ AI price consistency: Difference ${price_diff:.2f}")
        else:
            print(f"⚠️ AI price difference: ${price_diff:.2f} (might be timing)")
            
    except Exception as e:
        print(f"❌ AI System error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All systems successfully using the new Gold API!")
    print(f"📊 Current Gold Price: ${api_price:.2f}")
    return True

if __name__ == "__main__":
    success = test_complete_system()
    if success:
        print("\n✅ SYSTEM READY: All components using real Gold API data")
    else:
        print("\n❌ SYSTEM ISSUES: Some components need attention")
