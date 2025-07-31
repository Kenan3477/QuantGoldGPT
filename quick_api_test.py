#!/usr/bin/env python
"""
Quick API test for trade signals
"""
import requests
import json

try:
    print("🔥 Testing AI Signal Generator API...")
    response = requests.get("http://localhost:5000/api/trade-signals", timeout=5)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API working successfully!")
        print(f"Current Signal: {data.get('current_signal', {}).get('signal_type', 'None')}")
        print(f"Total Open Signals: {len(data.get('open_signals', []))}")
        print(f"Statistics: {data.get('statistics', {}).get('total_signals', 0)} total signals")
    else:
        print(f"❌ API Error: {response.text}")

except Exception as e:
    print(f"❌ Connection Error: {e}")

print("Done!")
