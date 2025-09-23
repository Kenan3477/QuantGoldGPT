import requests
import time
import json

def test_railway_signals():
    BASE_URL = 'https://web-production-41882.up.railway.app'
    
    print("🚀 Testing Railway Signal System...")
    
    # Test 1: Generate a signal
    print("\n1. Generating signal...")
    try:
        response = requests.get(f'{BASE_URL}/api/signals/generate', timeout=20)
        if response.status_code == 200:
            data = response.json()
            if 'signal' in data:
                signal = data['signal']
                print(f"✅ Signal generated: {signal['signal_type']} @ ${signal['entry_price']}")
                print(f"   Signal ID: {signal.get('signal_id', 'MISSING')}")
                print(f"   Confidence: {signal.get('confidence', 'MISSING')}")
            else:
                print("❌ No signal in response")
                print(f"Response: {data}")
        else:
            print(f"❌ Generate failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Generate error: {e}")
    
    # Test 2: Check debug endpoint
    print("\n2. Checking debug info...")
    try:
        response = requests.get(f'{BASE_URL}/api/debug/signals', timeout=15)
        if response.status_code == 200:
            debug = response.json()
            print(f"✅ Active signals in memory: {debug.get('active_signals_count', 0)}")
            print(f"   Memory system exists: {debug.get('signal_memory_exists', False)}")
            if debug.get('memory_signals_count'):
                print(f"   Memory signals count: {debug['memory_signals_count']}")
        else:
            print(f"❌ Debug failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Debug error: {e}")
    
    # Test 3: Get tracked signals
    print("\n3. Getting tracked signals...")
    try:
        response = requests.get(f'{BASE_URL}/api/signals/tracked', timeout=15)
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            print(f"✅ Found {len(signals)} tracked signals")
            if signals:
                for i, signal in enumerate(signals[:2]):
                    print(f"   Signal {i+1}: {signal.get('signal_type')} @ ${signal.get('entry_price')} | P&L: {signal.get('pnl', 0)}")
            else:
                print("   ⚠️ No signals found - this is the issue!")
        else:
            print(f"❌ Tracked failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Tracked error: {e}")

if __name__ == "__main__":
    test_railway_signals()
