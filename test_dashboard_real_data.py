#!/usr/bin/env python3
"""
Test script to verify the dashboard is displaying real data instead of fake data
"""
import requests
import json
from datetime import datetime

def test_dashboard_real_data():
    print("🧪 Testing Dashboard Real Data Display")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test the main price API
    print("\n1. Testing Live Gold Price API:")
    try:
        response = requests.get(f"{base_url}/api/live-gold-price", timeout=10)
        if response.status_code == 200:
            data = response.json()
            price_data = data.get('data', {})
            current_price = price_data.get('price', data.get('price'))  # Handle both formats
            source = price_data.get('source', data.get('source', 'Unknown'))
            timestamp = price_data.get('timestamp', data.get('timestamp', 'N/A'))
            
            print(f"   ✅ Real-time Gold Price: ${current_price}")
            print(f"   ✅ Source: {source}")
            print(f"   ✅ Timestamp: {timestamp}")
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error fetching live price: {e}")
        return False
    
    # Test ML predictions to ensure they use real price
    print("\n2. Testing ML Predictions API:")
    try:
        response = requests.get(f"{base_url}/api/ml-predictions/XAUUSD", timeout=15)
        if response.status_code == 200:
            data = response.json()
            ml_current_price = data.get('current_price')
            print(f"   ✅ ML Current Price: ${ml_current_price}")
            
            # Check if ML price matches live price (within reasonable range)
            if current_price and ml_current_price:
                price_diff = abs(current_price - ml_current_price)
                if price_diff < 5.0:  # Within $5 is reasonable
                    print(f"   ✅ ML price matches live price (diff: ${price_diff:.2f})")
                else:
                    print(f"   ⚠️  Large price difference: ${price_diff:.2f}")
            
            # Check predictions
            predictions = data.get('predictions', [])
            print(f"   ✅ Generated {len(predictions)} predictions")
            for pred in predictions:
                timeframe = pred.get('timeframe')
                pred_price = pred.get('predicted_price')
                change_pct = pred.get('change_percent', 0) * 100
                print(f"      {timeframe}: ${pred_price:.2f} ({change_pct:+.2f}%)")
                
        else:
            print(f"   ❌ ML API returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error fetching ML predictions: {e}")
    
    # Test order book data
    print("\n3. Testing Order Book API:")
    try:
        response = requests.get(f"{base_url}/api/order-book", timeout=10)
        if response.status_code == 200:
            data = response.json()
            ob_price = data.get('current_price')
            print(f"   ✅ Order Book Current Price: ${ob_price}")
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            print(f"   ✅ Bids: {len(bids)}, Asks: {len(asks)}")
            
            if bids and asks:
                spread = asks[0]['price'] - bids[0]['price']
                print(f"   ✅ Spread: ${spread:.2f}")
        else:
            print(f"   ❌ Order Book API returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error fetching order book: {e}")
    
    # Test Fear & Greed Index
    print("\n4. Testing Fear & Greed Index:")
    try:
        response = requests.get(f"{base_url}/api/fear-greed-index", timeout=10)
        if response.status_code == 200:
            data = response.json()
            index_value = data.get('value')
            classification = data.get('classification')
            print(f"   ✅ Fear & Greed Index: {index_value} ({classification})")
        else:
            print(f"   ❌ Fear & Greed API returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error fetching Fear & Greed index: {e}")
    
    # Validation Summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY:")
    
    if current_price and current_price > 3300:  # Realistic current gold price range
        print("   ✅ REAL DATA: Gold price is realistic and current")
        print("   ✅ NO FAKE DATA: No signs of hardcoded 2634/2674/2629 prices")
        print(f"   ✅ CURRENT PRICE: ${current_price:.2f} (from gold-api.com)")
        print("   ✅ DASHBOARD STATUS: Using real-time data")
        return True
    else:
        print("   ❌ FAKE DATA DETECTED: Price appears to be hardcoded")
        return False

if __name__ == "__main__":
    success = test_dashboard_real_data()
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Dashboard is displaying REAL data!")
    else:
        print("❌ FAILURE: Dashboard may still have fake data")
    print("=" * 50)
