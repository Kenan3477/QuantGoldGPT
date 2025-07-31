#!/usr/bin/env python3
"""
Quick test to validate Gold API and our fixes
"""
import requests
import json
from datetime import datetime

def test_gold_api_directly():
    """Test Gold API directly"""
    url = "https://api.gold-api.com/price/XAU"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = float(data.get('price', 0))
            
            print(f"✅ GOLD API WORKING")
            print(f"📡 URL: {url}")
            print(f"💰 Current Price: ${price:.2f}")
            print(f"📅 Response: {json.dumps(data, indent=2)}")
            print(f"🕒 Test Time: {datetime.now().strftime('%H:%M:%S')}")
            
            if 1000 < price < 5000:
                print(f"✅ Price validation PASSED (${price:.2f} is in valid range)")
                return True
            else:
                print(f"❌ Price validation FAILED (${price:.2f} outside 1000-5000 range)")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_backend_endpoint():
    """Test our Flask backend endpoint"""
    try:
        response = requests.get("http://localhost:5000/api/current-price/XAUUSD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ BACKEND ENDPOINT WORKING")
            print(f"📊 Backend Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"⚠️ Backend endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Backend endpoint not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🚀 XAU/USD PRICE VALIDATION TEST")
    print("=" * 50)
    
    # Test Gold API directly
    api_working = test_gold_api_directly()
    
    print("\n" + "-" * 50)
    
    # Test backend endpoint
    backend_working = test_backend_endpoint()
    
    print("\n" + "=" * 50)
    print(f"📋 TEST SUMMARY:")
    print(f"   Gold API: {'✅ WORKING' if api_working else '❌ FAILED'}")
    print(f"   Backend:  {'✅ WORKING' if backend_working else '⚠️ NEEDS RESTART'}")
    
    if api_working:
        print(f"\n🎯 SOLUTION STATUS: READY")
        print(f"   The Gold API is working perfectly.")
        print(f"   XAU/USD should update every 10 seconds with live data.")
        print(f"   Refresh your browser (Ctrl+F5) to see live prices.")
    else:
        print(f"\n❌ SOLUTION STATUS: API ISSUE")
        print(f"   Gold API is not responding. Check internet connection.")
