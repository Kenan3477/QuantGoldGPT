#!/usr/bin/env python3
"""
GoldGPT Deployment Verification Script
Tests key functionality after Railway deployment
"""

import requests
import json
import time
import sys
from urllib.parse import urljoin

def test_deployment(base_url):
    """Test GoldGPT deployment endpoints"""
    
    print(f"🚀 Testing GoldGPT deployment at: {base_url}")
    print("=" * 60)
    
    tests = [
        ("Health Check", "/"),
        ("API Status", "/api/health"),
        ("Gold Price", "/api/gold-price"),
        ("AI Signals", "/api/ai-signals"),
        ("ML Predictions", "/api/ml-predictions/XAUUSD"),
        ("ML Dashboard", "/ml-predictions-dashboard")
    ]
    
    results = []
    
    for test_name, endpoint in tests:
        try:
            print(f"Testing {test_name}...")
            
            url = urljoin(base_url, endpoint)
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ {test_name}: SUCCESS (200)")
                
                # Try to parse JSON for API endpoints
                if endpoint.startswith('/api/'):
                    try:
                        data = response.json()
                        if isinstance(data, dict) and data.get('success'):
                            print(f"   📊 API Response: SUCCESS")
                        else:
                            print(f"   📊 API Response: {str(data)[:100]}...")
                    except:
                        print(f"   📊 Non-JSON response")
                        
                results.append((test_name, True, response.status_code))
            else:
                print(f"❌ {test_name}: FAILED ({response.status_code})")
                results.append((test_name, False, response.status_code))
                
        except requests.exceptions.Timeout:
            print(f"⏰ {test_name}: TIMEOUT")
            results.append((test_name, False, "TIMEOUT"))
        except requests.exceptions.ConnectionError:
            print(f"🔌 {test_name}: CONNECTION ERROR")
            results.append((test_name, False, "CONNECTION_ERROR"))
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False, str(e)))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEPLOYMENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, status in results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your GoldGPT deployment is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the Railway logs for more details.")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_deployment.py <RAILWAY_URL>")
        print("Example: python verify_deployment.py https://your-app.railway.app")
        sys.exit(1)
    
    base_url = sys.argv[1]
    
    # Ensure URL ends without trailing slash
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    success = test_deployment(base_url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
