#!/usr/bin/env python3
"""
Test Trade Signals API Endpoint
"""
import requests
import json

def test_trade_signals_api():
    try:
        url = "http://localhost:5000/api/trade-signals"
        print(f"🔍 Testing: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response:")
            print(f"   Success: {data.get('success')}")
            print(f"   Current Signal: {data.get('current_signal')}")
            print(f"   Open Signals: {len(data.get('open_signals', []))}")
            print(f"   Statistics: {data.get('statistics')}")
            
            if data.get('open_signals'):
                print("\n📊 Open Signals Details:")
                for signal in data['open_signals']:
                    print(f"   - ID: {signal.get('id')}, Type: {signal.get('signal_type')}, Entry: ${signal.get('entry_price'):.2f}")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask app may not be running")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_trade_signals_api()
