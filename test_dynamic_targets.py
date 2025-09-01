#!/usr/bin/env python3
"""
TEST DYNAMIC TARGET PRICES
Test the fixed ML prediction system to verify targets are truly dynamic
"""
import requests
import json
import time

def test_dynamic_targets():
    print("🎯 TESTING DYNAMIC TARGET PRICE GENERATION")
    print("=" * 60)
    
    # Test multiple times to verify diversity
    for test_round in range(3):
        print(f"\n📊 TEST ROUND {test_round + 1}:")
        
        try:
            # Test timeframe predictions
            response = requests.get('http://localhost:5000/api/timeframe-predictions')
            if response.status_code == 200:
                data = response.json()
                timeframes = data.get('timeframes', {})
                
                print(f"✅ Timeframe Predictions:")
                target_prices = []
                for tf, pred in timeframes.items():
                    target = pred.get('target_price', 'N/A')
                    change = pred.get('change_percent', 0)
                    signal = pred.get('signal', 'N/A')
                    target_prices.append(target)
                    print(f"   {tf}: {signal} | Target: ${target} | Change: {change}%")
                
                # Check for diversity
                unique_targets = len(set(target_prices))
                print(f"   🎯 Unique targets: {unique_targets}/{len(target_prices)}")
                
            else:
                print(f"❌ Timeframe API error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Timeframe test error: {e}")
        
        try:
            # Test ML predictions
            response = requests.get('http://localhost:5000/api/ml-predictions')
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                print(f"✅ ML Predictions:")
                ml_targets = []
                for tf, pred in predictions.items():
                    target = pred.get('target', 'N/A')
                    change = pred.get('change_percent', 0)
                    signal = pred.get('signal', 'N/A')
                    ml_targets.append(target)
                    print(f"   {tf}: {signal} | Target: ${target} | Change: {change}%")
                
                # Check for diversity
                unique_ml = len(set(ml_targets))
                print(f"   🎯 Unique ML targets: {unique_ml}/{len(ml_targets)}")
                
            else:
                print(f"❌ ML API error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ ML test error: {e}")
        
        # Wait between tests
        if test_round < 2:
            time.sleep(2)
    
    print(f"\n🎯 CONCLUSION:")
    print("If you see different target prices across rounds, the fix is working!")
    print("Dynamic targets should vary based on volatility, time factors, and market conditions!")

if __name__ == "__main__":
    test_dynamic_targets()
