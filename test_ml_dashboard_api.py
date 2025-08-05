#!/usr/bin/env python3
"""
Enhanced ML Dashboard API Test
Tests all enhanced ML dashboard endpoints to verify they work correctly
"""

import requests
import json
from datetime import datetime

def test_api_endpoint(url, method='GET', data=None):
    """Test a single API endpoint"""
    try:
        print(f"\n🧪 Testing: {method} {url}")
        
        if method == 'GET':
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
            
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ SUCCESS - Response has {len(data)} keys")
                
                # Show specific data for ML endpoints
                if 'predictions' in data:
                    print(f"   📊 Predictions: {len(data['predictions'])} timeframes")
                elif 'features' in data:
                    print(f"   🎯 Features: {len(data['features'])} importance factors")
                elif 'metrics' in data:
                    print(f"   📈 Metrics: {list(data['metrics'].keys())}")
                elif 'stats' in data:
                    print(f"   📊 Stats: {list(data['stats'].keys())}")
                elif 'context' in data:
                    print(f"   🌍 Context: {list(data['context'].keys())}")
                    
                return True
                
            except json.JSONDecodeError:
                print(f"   ⚠️ Invalid JSON response")
                return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:100]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION FAILED - Server not running")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def main():
    """Test all enhanced ML dashboard endpoints"""
    print("🚀 Enhanced ML Dashboard API Test Suite")
    print("=" * 60)
    print(f"🕒 Test started: {datetime.now().strftime('%H:%M:%S')}")
    
    base_url = "http://localhost:5000"
    
    # Core Enhanced ML Dashboard endpoints
    endpoints = [
        # Main endpoints
        (f"{base_url}/api/ml-dashboard/predictions", "GET"),
        (f"{base_url}/api/ml-dashboard/feature-importance", "GET"),
        (f"{base_url}/api/ml-dashboard/accuracy-metrics", "GET"),
        (f"{base_url}/api/ml-dashboard/model-stats", "GET"),
        (f"{base_url}/api/market-context", "GET"),
        (f"{base_url}/api/ml-dashboard/comprehensive-analysis", "GET"),
        
        # Legacy endpoints (these were causing 404s)
        (f"{base_url}/api/ml-predictions", "GET"),
        (f"{base_url}/api/ml-accuracy", "GET"),
        (f"{base_url}/api/ml-performance", "GET"),
        (f"{base_url}/api/ml-health", "GET"),
        (f"{base_url}/api/market-analysis", "GET"),
        
        # Strategy endpoints
        (f"{base_url}/strategy/api/signals/recent?limit=1", "GET"),
        (f"{base_url}/strategy/api/performance", "GET"),
        
        # Health check
        (f"{base_url}/api/health", "GET"),
    ]
    
    print(f"\n📋 Testing {len(endpoints)} endpoints...")
    
    results = []
    for url, method in endpoints:
        success = test_api_endpoint(url, method)
        endpoint_name = url.split('/')[-1].split('?')[0]
        results.append((endpoint_name, success))
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Enhanced ML Dashboard API is fully functional!")
        print("   All endpoints returning valid data!")
    else:
        failed = total - passed
        print(f"\n⚠️ {failed} endpoints failed")
        print("   Check the failed endpoints above for issues")
    
    print(f"\n🕒 Test completed: {datetime.now().strftime('%H:%M:%S')}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
