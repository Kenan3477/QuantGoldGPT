#!/usr/bin/env python3
"""
Test the web API ML predictions endpoint
"""
import requests
import json

def test_ml_api():
    print("🌐 Testing Web API ML Predictions")
    
    try:
        # Test the ML predictions endpoint
        url = "http://localhost:5000/api/ml-predictions/XAUUSD"
        print(f"Requesting: {url}")
        
        response = requests.get(url, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response Successful!")
            
            # Check if we have current price
            current_price = data.get('current_price', 'N/A')
            print(f"Current Price: ${current_price}")
            
            # Check predictions
            predictions = data.get('predictions', {})
            if predictions:
                print("\n📈 ML Predictions:")
                for timeframe, pred in predictions.items():
                    if isinstance(pred, dict):
                        predicted_price = pred.get('predicted_price', 0)
                        direction = pred.get('direction', 'unknown')
                        confidence = pred.get('confidence', 0)
                        change_pct = pred.get('price_change_percent', 0)
                        
                        print(f"  {timeframe}: ${predicted_price:.2f} ({direction}, {confidence:.1%}, {change_pct:+.2f}%)")
                        
                        # Validate prediction sanity
                        if isinstance(current_price, (int, float)):
                            if abs(predicted_price - current_price) / current_price > 0.10:
                                print(f"    ⚠️ Large price difference detected!")
                            else:
                                print(f"    ✅ Prediction looks reasonable")
            
            # Check API source
            api_source = data.get('api_source', 'unknown')
            print(f"\nAPI Source: {api_source}")
            
            if api_source == 'enhanced_ml':
                print("✅ Using enhanced ML predictions with real-time data!")
            elif api_source == 'fallback':
                print("⚠️ Using fallback predictions")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Raw response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_ml_api()
