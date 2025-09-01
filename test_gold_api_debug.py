#!/usr/bin/env python3

import requests
import json
from datetime import datetime

def test_gold_api():
    """Test the Gold API to see what's happening"""
    print("🔍 Testing Gold API...")
    
    try:
        print("📡 Making request to https://api.gold-api.com/price/XAU")
        response = requests.get('https://api.gold-api.com/price/XAU', timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        print(f"📊 Response Text: {response.text}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ JSON Data: {json.dumps(data, indent=2)}")
                
                price = data.get('price', 0)
                print(f"💰 Extracted Price: ${price}")
                
                return {
                    'success': True,
                    'price': price,
                    'data': data
                }
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                return {'success': False, 'error': 'JSON decode failed'}
        else:
            print(f"❌ API request failed with status {response.status_code}")
            return {'success': False, 'error': f'HTTP {response.status_code}'}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request exception: {e}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {'success': False, 'error': str(e)}

def test_alternative_apis():
    """Test alternative gold price APIs"""
    print("\n🔍 Testing alternative APIs...")
    
    # Alternative 1: metals-api.com (free tier)
    try:
        print("📡 Testing metals-api.com...")
        # Note: This requires an API key for real use, but let's see what happens
        response = requests.get('https://api.metals.live/v1/spot/gold', timeout=10)
        print(f"metals-api status: {response.status_code}, response: {response.text[:200]}")
    except Exception as e:
        print(f"metals-api error: {e}")
    
    # Alternative 2: fcsapi.com (free tier)
    try:
        print("📡 Testing fcsapi.com...")
        # This also needs API key, but let's test
        response = requests.get('https://fcsapi.com/api-v3/forex/latest?symbol=XAU/USD&access_key=demo', timeout=10)
        print(f"fcsapi status: {response.status_code}, response: {response.text[:200]}")
    except Exception as e:
        print(f"fcsapi error: {e}")
    
    # Alternative 3: Try a simple financial data API
    try:
        print("📡 Testing financial modeling prep...")
        response = requests.get('https://financialmodelingprep.com/api/v3/quote/GLD', timeout=10)
        print(f"financialmodelingprep status: {response.status_code}, response: {response.text[:200]}")
    except Exception as e:
        print(f"financialmodelingprep error: {e}")

if __name__ == "__main__":
    print("🚀 Gold API Debug Test")
    print("=" * 50)
    
    result = test_gold_api()
    print(f"\n📊 Final Result: {result}")
    
    test_alternative_apis()
    
    print("\n✅ Test completed!")
