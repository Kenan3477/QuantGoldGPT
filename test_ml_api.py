#!/usr/bin/env python3
"""
Test ML Dashboard API endpoints
"""

import requests
import json
import sys

def test_endpoint(endpoint, method='GET', data=None):
    """Test a single endpoint"""
    url = f"http://127.0.0.1:5000{endpoint}"
    print(f"\n🔍 Testing: {method} {endpoint}")
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=5)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=5)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"📊 Status Code: {response.status_code}")
        
        # Try to parse JSON response
        try:
            json_data = response.json()
            print(f"📋 Response: {json.dumps(json_data, indent=2)}")
            return response.status_code == 200
        except:
            print(f"📋 Response (text): {response.text[:200]}...")
            return response.status_code == 200
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server not running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🧠 ML Dashboard API Test")
    print("=" * 50)
    
    # Test basic server connectivity
    print("\n🔗 Testing basic connectivity...")
    basic_test = test_endpoint("/")
    
    if not basic_test:
        print("❌ Basic connectivity failed. Check if server is running.")
        return
    
    # Test ML Dashboard API endpoints
    endpoints_to_test = [
        ("/api/ml-health", "GET"),
        ("/api/ml-performance", "GET"),
        ("/api/ml-accuracy", "GET"),
        ("/api/ml-predictions", "POST", {"timeframes": ["15m", "1h"]}),
    ]
    
    results = []
    for endpoint_info in endpoints_to_test:
        if len(endpoint_info) == 2:
            endpoint, method = endpoint_info
            data = None
        else:
            endpoint, method, data = endpoint_info
            
        success = test_endpoint(endpoint, method, data)
        results.append((endpoint, success))
    
    # Test ML Dashboard Test page
    test_endpoint("/ml-dashboard-test", "GET")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    for endpoint, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {endpoint}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\n📈 Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
