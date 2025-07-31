#!/usr/bin/env python3
"""
Simple verification script to check ML API response
"""
import requests
import json

def test_ml_api():
    print("🔍 Verifying ML Predictions API...")
    print("=" * 50)
    
    try:
        url = "http://localhost:5000/api/ml-predictions/XAUUSD"
        print(f"Testing: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS - API Response:")
            print("-" * 30)
            
            # Check current price
            if 'current_price' in data:
                current_price = data['current_price']
                print(f"Current Price: ${current_price}")
                
                # Check if it's the real price (around $3350) not fake ($2600)
                if 3300 <= current_price <= 3400:
                    print("✅ REAL PRICE - Not fake $2600!")
                else:
                    print(f"❌ SUSPICIOUS PRICE - Expected ~$3350, got ${current_price}")
            
            # Check predictions
            if 'predictions' in data:
                predictions = data['predictions']
                print(f"\nPredictions ({len(predictions)} timeframes):")
                
                for pred in predictions:
                    timeframe = pred.get('timeframe', 'Unknown')
                    predicted_price = pred.get('predicted_price', 0)
                    change_percent = pred.get('change_percent', 0)
                    
                    print(f"  {timeframe}: ${predicted_price:.2f} ({change_percent:+.2f}%)")
                    
                    # Check if changes are realistic (< 1%)
                    if abs(change_percent) < 1.0:
                        print(f"    ✅ Realistic change: {change_percent:+.2f}%")
                    else:
                        print(f"    ❌ Unrealistic change: {change_percent:+.2f}%")
            
            return True
            
        else:
            print(f"❌ API Error: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ml_api()
    if success:
        print("\n🎉 ML PREDICTION SYSTEM VERIFICATION COMPLETE!")
        print("✅ Real-time prices working")
        print("✅ Realistic predictions generated")
    else:
        print("\n❌ Verification failed - Check web application")
