#!/usr/bin/env python3
"""
Quick test to verify API endpoints are working
"""
import requests
import json

def test_endpoint(url, name):
    try:
        print(f"\n🔍 Testing {name}: {url}")
        response = requests.get(url, timeout=5)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ SUCCESS: {json.dumps(data, indent=2)}")
        else:
            print(f"  ❌ ERROR: {response.text}")
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")

if __name__ == "__main__":
    base_url = "http://localhost:5000"
    
    print("🚀 Testing GoldGPT API Endpoints")
    
    # Test the endpoints that are giving 404s
    test_endpoint(f"{base_url}/api/live-gold-price", "Live Gold Price")
    test_endpoint(f"{base_url}/api/signals/active", "Active Signals") 
    test_endpoint(f"{base_url}/api/gold-price", "Gold Price (existing)")
    test_endpoint(f"{base_url}/api/health", "Health Check")
