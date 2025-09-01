#!/usr/bin/env python3
"""
Test the fixed ML predictions and news APIs
"""
import requests
import json

def test_apis():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Fixed APIs")
    print("=" * 50)
    
    # Test ML Predictions
    print("1. Testing ML Predictions API...")
    try:
        response = requests.get(f"{base_url}/api/ml-predictions", timeout=10)
        print(f"   📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data.get('success', False)}")
            print(f"   💰 Current Price: ${data.get('current_price', 'N/A')}")
            
            if 'predictions' in data:
                predictions = data['predictions']
                print(f"   🔮 Available timeframes: {list(predictions.keys())}")
                
                # Show 15m prediction details
                if '15m' in predictions:
                    pred = predictions['15m']
                    print(f"   📈 15m: {pred.get('signal', 'N/A')} | Target: ${pred.get('target', 'N/A')} | Confidence: {pred.get('confidence', 'N/A')}")
            else:
                print("   ⚠️ No predictions data found")
        else:
            print(f"   ❌ Error: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print()
    
    # Test Market News
    print("2. Testing Market News API...")
    try:
        response = requests.get(f"{base_url}/api/market-news", timeout=10)
        print(f"   📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data.get('success', False)}")
            print(f"   📰 News count: {data.get('count', 0)}")
            
            if 'data' in data and data['data']:
                first_news = data['data'][0]
                print(f"   📝 Latest: {first_news.get('title', 'N/A')}")
                print(f"   🎯 Impact: {first_news.get('impact', 'N/A')}")
            else:
                print("   ⚠️ No news data found")
        else:
            print(f"   ❌ Error: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print()
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_apis()
