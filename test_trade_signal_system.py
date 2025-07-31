#!/usr/bin/env python
"""
Comprehensive test of the Trade Signal Manager implementation
"""
import requests
import json

def test_trade_signal_system():
    """Test the complete trade signal system"""
    print("🧪 Testing Trade Signal Manager System")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Main dashboard loads
    print("\n1️⃣ Testing main dashboard...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard loads successfully")
            
            # Check if trade signal components are included
            content = response.text
            if 'trade-signal-manager.js' in content:
                print("✅ Trade Signal Manager JS included")
            if 'trade-signal-manager.css' in content:
                print("✅ Trade Signal Manager CSS included")
            if 'trade-signals-container' in content:
                print("✅ Trade Signals Container present")
        else:
            print(f"❌ Dashboard failed to load: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
    
    # Test 2: Trade signals API endpoint
    print("\n2️⃣ Testing trade signals API...")
    try:
        response = requests.get(f"{base_url}/api/trade-signals", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Trade signals API working!")
            
            if data.get('success'):
                print(f"   Current Signal: {data.get('current_signal', {}).get('signal_type', 'None')}")
                print(f"   Open Signals: {len(data.get('open_signals', []))}")
                print(f"   Total Signals: {data.get('statistics', {}).get('total_signals', 0)}")
                print(f"   Win Rate: {data.get('statistics', {}).get('win_rate', 0):.1f}%")
            else:
                print(f"⚠️ API returned error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}")
                
    except Exception as e:
        print(f"❌ API test error: {e}")
    
    # Test 3: Signal generation endpoint
    print("\n3️⃣ Testing signal generation...")
    try:
        response = requests.post(f"{base_url}/api/signals/generate", timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Signal generation successful!")
                signal = data.get('signal', {})
                print(f"   Signal Type: {signal.get('signal_type', 'N/A')}")
                print(f"   Confidence: {signal.get('confidence', 0):.1f}%")
                print(f"   Entry Price: ${signal.get('entry_price', 0):.2f}")
            else:
                print(f"⚠️ Signal generation returned error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Signal generation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Signal generation test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Trade Signal Manager System Test Complete!")
    print("\n📋 Next Steps:")
    print("   1. Open dashboard: http://localhost:5000")
    print("   2. Click 'Signals' in the left sidebar")
    print("   3. Test the 'Generate New Signal' button")
    print("   4. View signal statistics and history")

if __name__ == "__main__":
    test_trade_signal_system()
