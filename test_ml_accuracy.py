#!/usr/bin/env python3
"""
Test ML Predictions API - Check what's being returned
"""
import requests
import json
from datetime import datetime

def test_ml_predictions():
    print("🔍 Testing ML Predictions API...")
    
    try:
        # Test the main ML predictions endpoint
        response = requests.get('http://localhost:5000/api/ml-predictions', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response received")
            print(f"Success: {data.get('success', False)}")
            print(f"Current Price: ${data.get('current_price', 0)}")
            print(f"Timestamp: {data.get('timestamp', 'N/A')}")
            
            if 'predictions' in data:
                print(f"\n📊 Found {len(data['predictions'])} predictions:")
                for i, pred in enumerate(data['predictions']):
                    current_price = data.get('current_price', 0)
                    predicted_price = pred.get('predicted_price', 0)
                    change_percent = pred.get('change_percent', 0)
                    
                    print(f"\n  Prediction {i+1}:")
                    print(f"    Timeframe: {pred.get('timeframe', 'N/A')}")
                    print(f"    Current Price: ${current_price:.2f}")
                    print(f"    Predicted Price: ${predicted_price:.2f}")
                    print(f"    Expected Change: {change_percent:.2f}%")
                    print(f"    Direction: {pred.get('direction', 'N/A')}")
                    print(f"    Confidence: {pred.get('confidence', 0)*100:.1f}%")
                    
                    # Check if math makes sense
                    if current_price > 0 and predicted_price > 0:
                        actual_change = ((predicted_price - current_price) / current_price) * 100
                        print(f"    ✅ Math Check - Calculated Change: {actual_change:.2f}%")
                        
                        if abs(actual_change - change_percent) > 0.1:
                            print(f"    ❌ MATH ERROR: Expected {change_percent:.2f}%, but calculated {actual_change:.2f}%")
                        else:
                            print(f"    ✅ Math correct")
            else:
                print("❌ No predictions found in response")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text[:500])
            
    except Exception as e:
        print(f"❌ Error testing ML predictions: {e}")

if __name__ == "__main__":
    test_ml_predictions()
