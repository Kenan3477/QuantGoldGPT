#!/usr/bin/env python3
"""
Simple test for ML predictions endpoint
"""
import requests
import json

def test_ml_predictions():
    try:
        print("🧪 Testing ML predictions endpoint...")
        
        # Test the main ML predictions endpoint
        response = requests.get('http://localhost:5000/api/ml-predictions', timeout=10)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ ML predictions endpoint working!")
            print(f"📊 Success: {data.get('success', False)}")
            print(f"💰 Current price: ${data.get('current_price', 'N/A')}")
            
            if 'predictions' in data:
                predictions = data['predictions']
                print(f"🔮 Available timeframes: {list(predictions.keys())}")
                
                # Show sample prediction
                if '15m' in predictions:
                    pred_15m = predictions['15m']
                    print(f"📈 15m Signal: {pred_15m.get('signal', 'N/A')}")
                    print(f"🎯 15m Target: ${pred_15m.get('target', 'N/A')}")
                    print(f"📊 15m Confidence: {pred_15m.get('confidence', 'N/A')}%")
            else:
                print("⚠️ No predictions data in response")
                print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ml_predictions()
