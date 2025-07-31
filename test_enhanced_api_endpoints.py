#!/usr/bin/env python3
"""
Test Enhanced Signals API Endpoints
"""
import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:5000"
    
    endpoints = [
        ("/api/enhanced-signals/active", "Active Signals"),
        ("/api/enhanced-signals/performance", "Performance Data"),
        ("/api/enhanced-signals/monitor", "Monitoring Status")
    ]
    
    print("🔍 Testing Enhanced Signals API Endpoints...")
    print("=" * 50)
    
    for endpoint, description in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"\n📡 Testing: {description}")
            print(f"URL: {url}")
            
            response = requests.get(url, timeout=5)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {json.dumps(data, indent=2)}")
            else:
                print(f"❌ Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection Error: Flask app may not be running on {base_url}")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 API endpoint testing complete")

if __name__ == "__main__":
    test_api_endpoints()
