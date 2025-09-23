#!/usr/bin/env python3
"""
Debug Signal Storage and Retrieval
Test if signals are being stored and can be retrieved
"""

import requests
import json
import time

def debug_signal_storage():
    """Debug signal generation, storage, and retrieval"""
    
    base_url = "https://web-production-41882.up.railway.app"  # Railway deployment
    
    print("🔍 DEBUGGING SIGNAL STORAGE SYSTEM")
    print("=" * 50)
    
    # Step 1: Check current active signals BEFORE generation
    print("\n1. CHECKING ACTIVE SIGNALS BEFORE GENERATION")
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=10)
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            print(f"   📊 Active signals BEFORE: {len(signals)}")
            if signals:
                for i, signal in enumerate(signals):
                    print(f"   Signal {i+1}: {signal.get('signal_type', 'N/A')} @ ${signal.get('entry_price', 'N/A')}")
        else:
            print(f"   ❌ Failed to get active signals: {response.status_code}")
    except Exception as e:
        print(f"   💥 Error getting active signals: {e}")
    
    # Step 2: Generate a new signal
    print("\n2. GENERATING NEW SIGNAL")
    try:
        response = requests.get(f"{base_url}/api/signals/generate", timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and 'signal' in data:
                signal = data['signal']
                signal_id = signal.get('signal_id', 'N/A')
                signal_type = signal.get('signal_type', 'N/A')
                entry_price = signal.get('entry_price', 'N/A')
                
                print(f"   ✅ Signal generated successfully!")
                print(f"   📊 Signal ID: {signal_id}")
                print(f"   📈 Type: {signal_type}")
                print(f"   💰 Entry: ${entry_price}")
                print(f"   🎯 Confidence: {signal.get('confidence', 'N/A')}")
                print(f"   💾 Memory stored: {signal.get('memory_stored', 'N/A')}")
                
                return signal_id, signal_type, entry_price
            else:
                print(f"   ❌ Signal generation failed: {data}")
                return None, None, None
        else:
            print(f"   ❌ Signal request failed: {response.status_code}")
            return None, None, None
    except Exception as e:
        print(f"   💥 Error generating signal: {e}")
        return None, None, None
    
    
def test_after_generation(signal_id):
    """Test signal retrieval after generation"""
    base_url = "https://web-production-41882.up.railway.app"
    
    # Step 3: Wait a moment and check active signals AFTER generation
    print("\n3. CHECKING ACTIVE SIGNALS AFTER GENERATION")
    time.sleep(2)  # Wait for signal to be processed
    
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=10)
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            print(f"   📊 Active signals AFTER: {len(signals)}")
            
            if signals:
                found_new_signal = False
                for i, signal in enumerate(signals):
                    current_id = signal.get('signal_id', 'N/A')
                    signal_type = signal.get('signal_type', 'N/A')
                    entry_price = signal.get('entry_price', 'N/A')
                    pnl = signal.get('pnl', 'N/A')
                    
                    print(f"   Signal {i+1}: {signal_type} @ ${entry_price} | P&L: {pnl} | ID: {current_id}")
                    
                    if current_id == signal_id:
                        found_new_signal = True
                        print(f"   ✅ FOUND NEW SIGNAL in active list!")
                
                if not found_new_signal and signal_id:
                    print(f"   ⚠️ NEW SIGNAL NOT FOUND in active list!")
                    print(f"   🔍 Looking for ID: {signal_id}")
            else:
                print(f"   ⚠️ NO ACTIVE SIGNALS FOUND after generation!")
                
        else:
            print(f"   ❌ Failed to get active signals: {response.status_code}")
    except Exception as e:
        print(f"   💥 Error getting active signals: {e}")
    
    # Step 4: Check signal memory system directly
    print("\n4. CHECKING SIGNAL MEMORY SYSTEM")
    try:
        response = requests.get(f"{base_url}/api/signals/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Signal stats: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Failed to get signal stats: {response.status_code}")
    except Exception as e:
        print(f"   💥 Error getting signal stats: {e}")

def main():
    """Run complete signal storage debug"""
    signal_id, signal_type, entry_price = debug_signal_storage()
    
    if signal_id:
        test_after_generation(signal_id)
    
    print(f"\n🎯 DIAGNOSIS:")
    print("✅ If signals appear in AFTER but not in dashboard = Frontend issue")
    print("⚠️ If signals don't appear in AFTER = Backend storage issue") 
    print("❌ If signal generation fails = API issue")

if __name__ == "__main__":
    main()
