#!/usr/bin/env python3
"""
Test script to verify Gold API price fetching is working
"""
import requests
import json
from datetime import datetime

def test_gold_api():
    """Test Gold API endpoint"""
    url = 'https://api.gold-api.com/price/XAU'
    
    print("🧪 Testing Gold API endpoint...")
    print(f"📡 URL: {url}")
    
    try:
        response = requests.get(url, headers={
            'Accept': 'application/json',
            'User-Agent': 'GoldGPT-Pro/1.0',
            'Cache-Control': 'no-cache'
        }, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Gold API Response:")
            print(json.dumps(data, indent=2))
            
            # Try to extract price
            price = None
            if 'price' in data and isinstance(data['price'], (int, float)):
                price = float(data['price'])
            elif 'price_usd' in data:
                price = float(data['price_usd'])
            elif 'rates' in data and 'USD' in data['rates']:
                price = float(data['rates']['USD'])
            elif 'ask' in data and 'bid' in data:
                price = (float(data['ask']) + float(data['bid'])) / 2
            
            if price:
                print(f"💰 Extracted Gold Price: ${price:.2f}")
            else:
                print("❌ Could not extract price from response")
                
        else:
            print(f"❌ API Error: {response.status_code} {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_fallback_apis():
    """Test fallback API endpoints"""
    fallback_urls = [
        'https://api.metals.live/v1/spot/gold',
        'https://api.coinbase.com/v2/exchange-rates?currency=USD'
    ]
    
    print("\n🔄 Testing fallback APIs...")
    
    for url in fallback_urls:
        print(f"\n📡 Testing: {url}")
        try:
            response = requests.get(url, headers={'Accept': 'application/json'}, timeout=10)
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Response received (first 200 chars):")
                print(str(data)[:200] + "..." if len(str(data)) > 200 else str(data))
            else:
                print(f"❌ Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print(f"🚀 Gold Price API Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test main Gold API
    test_gold_api()
    
    # Test fallback APIs
    test_fallback_apis()
    
    print("\n" + "=" * 60)
    print("✅ Test completed")
