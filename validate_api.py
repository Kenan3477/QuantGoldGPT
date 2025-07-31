#!/usr/bin/env python
"""
Quick validation of AI Signal Generator API endpoints
"""
import requests
import json
import time

def test_api_endpoints():
    base_url = "http://localhost:5000/api/signals"
    
    print("🧪 Testing AI Signal Generator API Endpoints...")
    print("=" * 50)
    
    # Test 1: Generate Signal
    print("\n1️⃣ Testing signal generation...")
    try:
        response = requests.post(f"{base_url}/generate", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Signal generated successfully!")
            print(f"   Signal Type: {data.get('signal_type', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}%")
            print(f"   Entry Price: ${data.get('entry_price', 'N/A')}")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Get Stats
    print("\n2️⃣ Testing stats endpoint...")
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats retrieved successfully!")
            print(f"   Total Signals: {data.get('total_signals', 0)}")
            print(f"   Win Rate: {data.get('win_rate', 0)}%")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Get Open Signals
    print("\n3️⃣ Testing open signals endpoint...")
    try:
        response = requests.get(f"{base_url}/open", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Open signals retrieved successfully!")
            print(f"   Number of open signals: {len(data.get('signals', []))}")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API endpoint testing complete!")

if __name__ == "__main__":
    test_api_endpoints()
