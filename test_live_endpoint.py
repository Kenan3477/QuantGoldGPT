#!/usr/bin/env python3

import requests
import json

def test_live_endpoint():
    """Test the live gold price endpoint to see what it returns"""
    try:
        url = "http://127.0.0.1:5000/api/live-gold-price"
        print(f"🔍 Testing endpoint: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ JSON Response:")
                print(json.dumps(data, indent=2))
                
                price = data.get('price', 'N/A')
                source = data.get('source', 'N/A')
                print(f"\n💰 Price: ${price}")
                print(f"📡 Source: {source}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"📄 Raw response: {response.text}")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Live Gold Price Endpoint")
    print("=" * 50)
    test_live_endpoint()
    print("✅ Test completed!")
