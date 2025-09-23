#!/usr/bin/env python3
"""
Railway Deployment Tester - Post-Deploy Verification
Tests signal generation after Railway deployment
"""

import requests
import json
from datetime import datetime

def test_railway_signals():
    """Test Railway signal generation after deployment"""
    
    print("🚂 Testing Railway Signal Generation")
    print("=" * 40)
    
    # Your actual Railway URL
    railway_url = "https://web-production-41882.up.railway.app"
    
    endpoints = [
        "/api/signals/generate",
        "/api/generate-signal", 
        "/api/emergency-signal"
    ]
    
    for endpoint in endpoints:
        url = f"{railway_url}{endpoint}"
        print(f"\n📡 Testing: {endpoint}")
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "signal" in data and "price" in data:
                    print(f"✅ SUCCESS: {data['signal']} at ${data['price']}")
                    print(f"   Method: {data.get('method', 'Unknown')}")
                    print(f"   Confidence: {data.get('confidence', 'N/A')}")
                else:
                    print(f"⚠️  Response missing signal data: {data}")
            else:
                print(f"❌ FAILED: HTTP {response.status_code}")
                print(f"   Response: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    print(f"\n🎯 If all tests pass, your Railway deployment is working!")
    print(f"   Signal generation should now work on Railway production.")

if __name__ == "__main__":
    test_railway_signals()
