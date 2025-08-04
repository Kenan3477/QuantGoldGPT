#!/usr/bin/env python3
"""
Railway Deployment Diagnostic - Check API Endpoints
Run this to verify all endpoints are working after deployment
"""

import requests
import json
from datetime import datetime

def test_api_endpoint(base_url, endpoint, method='GET'):
    """Test a single API endpoint"""
    url = f"{base_url}{endpoint}"
    try:
        if method == 'POST':
            response = requests.post(url, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {endpoint} - Working (Status: 200)")
            return True
        else:
            print(f"❌ {endpoint} - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {endpoint} - Error: {str(e)[:50]}...")
        return False

def main():
    """Test all the endpoints that were causing 404 errors"""
    print("🔍 Railway Deployment API Endpoint Diagnostic")
    print("=" * 60)
    print(f"🕒 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Replace with your Railway domain
    base_url = "https://[YOUR-RAILWAY-DOMAIN].railway.app"
    print(f"🌐 Testing: {base_url}")
    print("\n📋 Testing previously failing endpoints:")
    
    # Test all the endpoints that were returning 404
    endpoints_to_test = [
        ("/api/health", "GET"),
        ("/api/ml-predictions", "POST"),
        ("/api/ml-accuracy?timeframe=7d", "GET"),
        ("/api/ml-performance", "GET"),
        ("/api/ml-health", "GET"),
        ("/api/market-regime/XAUUSD", "GET"),
        ("/api/news/sentiment-summary", "GET"),
        ("/strategy/api/signals/recent?limit=1", "GET"),
        ("/strategy/api/performance", "GET"),
        # Enhanced ML Dashboard endpoints
        ("/ml-dashboard/predictions", "GET"),
        ("/ml-dashboard/feature-importance", "GET"),
        ("/ml-dashboard/accuracy-metrics", "GET"),
        ("/ml-dashboard/model-stats", "GET"),
        ("/market-context", "GET"),
        ("/ml-dashboard/comprehensive-analysis", "GET"),
    ]
    
    passed = 0
    total = len(endpoints_to_test)
    
    for endpoint, method in endpoints_to_test:
        if test_api_endpoint(base_url, endpoint, method):
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} endpoints working")
    print(f"🎯 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL ENDPOINTS WORKING! Deployment successful!")
    else:
        print(f"\n⚠️ {total-passed} endpoints still failing - check Railway logs")
    
    print(f"\n💡 Instructions:")
    print(f"1. Replace '[YOUR-RAILWAY-DOMAIN]' with your actual Railway domain")
    print(f"2. Run this script after Railway deployment completes")
    print(f"3. All endpoints should return 200 OK status")

if __name__ == "__main__":
    main()
