#!/usr/bin/env python3
"""
Test ML Prediction Accuracy Fix
"""
import requests
import json

def test_ml_prediction_accuracy():
    print("🔧 Testing ML Prediction Accuracy Fix...")
    
    try:
        response = requests.get('http://localhost:5000/api/ml-predictions', timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ API Response Status: {data.get('success', False)}")
            print(f"📊 Current Price: ${data.get('current_price', 0):.2f}")
            
            predictions = data.get('predictions', [])
            print(f"🔮 Found {len(predictions)} predictions")
            
            for i, pred in enumerate(predictions):
                current_price = data.get('current_price', 0)
                predicted_price = pred.get('predicted_price', 0)
                change_percent = pred.get('change_percent', 0)
                direction = pred.get('direction', 'unknown')
                
                print(f"\n  Prediction {i+1} ({pred.get('timeframe', 'N/A')}):")
                print(f"    Current: ${current_price:.2f}")
                print(f"    Predicted: ${predicted_price:.2f}")
                print(f"    Change: {change_percent:+.3f}%")
                print(f"    Direction: {direction}")
                
                # Verify math
                if current_price > 0 and predicted_price > 0:
                    expected_price_from_percent = current_price * (1 + change_percent / 100)
                    print(f"    Expected price from %: ${expected_price_from_percent:.2f}")
                    
                    if abs(predicted_price - expected_price_from_percent) < 0.01:
                        print("    ✅ Math is consistent!")
                    else:
                        print(f"    ❌ Math error: {abs(predicted_price - expected_price_from_percent):.2f} difference")
                    
                    # Check if direction matches the prediction
                    if change_percent > 0 and predicted_price > current_price:
                        print("    ✅ Positive change gives higher price")
                    elif change_percent < 0 and predicted_price < current_price:
                        print("    ✅ Negative change gives lower price")
                    elif change_percent == 0:
                        print("    ✅ No change prediction")
                    else:
                        print("    ❌ Direction/price mismatch!")
                        
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_ml_prediction_accuracy()
