#!/usr/bin/env python3
"""
Test signal generation on Railway deployment
"""

import requests
import json
import time

def test_railway_signals():
    """Test signal generation on Railway"""
    
    # Replace with your actual Railway URL
    base_url = "https://quantgoldgpt-production.up.railway.app"
    
    print("🚂 Testing Railway Signal Generation")
    print("="*50)
    
    # Test the generate signal endpoint
    endpoints = [
        "/api/generate-signal",
        "/api/signals/generate"
    ]
    
    for endpoint in endpoints:
        print(f"\n🧪 Testing {endpoint}...")
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ Response received")
                    print(f"Success: {data.get('success', False)}")
                    
                    if data.get('success'):
                        signal = data.get('signal', {})
                        print(f"Signal Type: {signal.get('signal_type', 'N/A')}")
                        print(f"Entry Price: ${signal.get('entry_price', 0):.2f}")
                        print(f"Take Profit: ${signal.get('take_profit', 0):.2f}")
                        print(f"Stop Loss: ${signal.get('stop_loss', 0):.2f}")
                        print(f"Confidence: {signal.get('confidence', 0):.1f}%")
                    else:
                        print(f"❌ Error: {data.get('error', 'Unknown error')}")
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON response: {response.text[:200]}")
            else:
                print(f"❌ HTTP Error {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            
        time.sleep(2)  # Brief pause between tests

if __name__ == "__main__":
    test_railway_signals()
