#!/usr/bin/env python3
"""
DIRECT API TEST - Check current predictions
"""
import requests
import json

def test_current_predictions():
    print("🔍 CHECKING CURRENT PREDICTIONS:")
    
    try:
        # Test current gold price
        response = requests.get('http://localhost:5000/api/live-gold-price')
        if response.status_code == 200:
            price_data = response.json()
            current_price = price_data.get('price')
            print(f"💰 Current Gold Price: ${current_price}")
        else:
            current_price = 3327.5
            print(f"💰 Using fallback price: ${current_price}")
        
        # Test ML predictions
        print("\n🧠 ML PREDICTIONS:")
        response = requests.get('http://localhost:5000/api/ml-predictions')
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            
            for timeframe, pred in predictions.items():
                target = pred.get('target', 'N/A')
                signal = pred.get('signal', 'N/A') 
                change = pred.get('change_percent', 0)
                confidence = pred.get('confidence', 0)
                
                print(f"   {timeframe}: {signal} | Target: ${target} | Change: {change:.3f}% | Confidence: {confidence:.2f}")
                
                # Check if target is different from current price
                if isinstance(target, (int, float)) and isinstance(current_price, (int, float)):
                    price_diff = abs(target - current_price)
                    if price_diff < 0.1:
                        print(f"      ⚠️  WARNING: Target too close to current price!")
                    else:
                        print(f"      ✅ Good: ${price_diff:.2f} difference from current price")
        
        print("\n📊 TIMEFRAME PREDICTIONS:")
        response = requests.get('http://localhost:5000/api/timeframe-predictions')
        if response.status_code == 200:
            data = response.json()
            timeframes = data.get('timeframes', {})
            
            for tf, pred in timeframes.items():
                target = pred.get('target_price', 'N/A')
                signal = pred.get('signal', 'N/A')
                change = pred.get('change_percent', 0)
                
                print(f"   {tf}: {signal} | Target: ${target} | Change: {change:.3f}%")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_current_predictions()
