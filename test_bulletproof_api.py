#!/usr/bin/env python3
"""
Quick test of bulletproof gold price API
"""
import requests
import json

def test_bulletproof_api():
    print("🛡️ Testing Bulletproof Gold Price API...")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:5000/api/gold-price')
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response successful!")
            print(f"   Price: {data.get('price')}")
            print(f"   Formatted: {data.get('formatted')}")
            print(f"   Source: {data.get('source')}")
            print(f"   Live: {data.get('is_live')}")
            print(f"   Timestamp: {data.get('timestamp')}")
        else:
            print(f"❌ API returned status code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_bulletproof_api()
