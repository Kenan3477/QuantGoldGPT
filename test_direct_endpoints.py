#!/usr/bin/env python3
"""
Direct test of the endpoints using requests
"""
import requests
import json
import time

def test_endpoint_direct(endpoint_name, url):
    print(f"\n🔍 Testing {endpoint_name}")
    print(f"   URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ SUCCESS - JSON Response:")
                print(f"   {json.dumps(data, indent=6)}")
            except:
                print(f"   ✅ SUCCESS - Text Response:")
                print(f"   {response.text[:200]}...")
        else:
            print(f"   ❌ ERROR Response:")
            print(f"   {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ CONNECTION ERROR: {e}")
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    print("🚀 Direct API Endpoint Testing")
    time.sleep(1)  # Wait a moment
    
    base_url = "http://127.0.0.1:5000"
    
    # Test the problematic endpoints
    test_endpoint_direct("Live Gold Price", f"{base_url}/api/live-gold-price")
    test_endpoint_direct("Active Signals", f"{base_url}/api/signals/active") 
    
    # Test known working endpoints for comparison
    test_endpoint_direct("Health Check", f"{base_url}/api/health")
    test_endpoint_direct("Regular Gold Price", f"{base_url}/api/gold-price")
