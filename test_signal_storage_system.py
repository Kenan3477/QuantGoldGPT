#!/usr/bin/env python3
"""
Signal Storage and Tracking Diagnostic Test
Tests the complete signal generation → storage → retrieval → tracking flow
"""

import requests
import json
import time
from datetime import datetime

def test_signal_storage_system():
    """Test the complete signal storage and tracking system"""
    
    print("🔍 SIGNAL STORAGE & TRACKING DIAGNOSTIC TEST")
    print("=" * 60)
    
    base_url = "https://web-production-41882.up.railway.app"
    
    # Step 1: Check initial signal count
    print("\n1️⃣ CHECKING INITIAL SIGNAL COUNT...")
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
        if response.status_code == 200:
            data = response.json()
            initial_count = len(data.get('signals', []))
            print(f"   📊 Initial signals in system: {initial_count}")
            
            # Show existing signals
            for i, signal in enumerate(data.get('signals', [])[:3]):  # Show first 3
                print(f"   Signal {i+1}: {signal.get('signal_id', 'NO_ID')} - {signal.get('signal_type', 'UNKNOWN')} - Entry: ${signal.get('entry_price', 0)}")
                
        else:
            print(f"   ❌ Failed to get tracked signals: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Error getting initial signals: {e}")
        return
    
    # Step 2: Generate a new signal
    print("\n2️⃣ GENERATING NEW SIGNAL...")
    try:
        response = requests.post(f"{base_url}/api/signals/generate", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                signal_data = data.get('signal', {})
                new_signal_id = signal_data.get('signal_id', 'NO_ID')
                signal_type = signal_data.get('signal_type', 'UNKNOWN')
                entry_price = signal_data.get('entry_price', 0)
                memory_stored = signal_data.get('memory_stored', False)
                
                print(f"   ✅ Generated signal: {new_signal_id}")
                print(f"   📊 Type: {signal_type} | Entry: ${entry_price}")
                print(f"   🧠 Memory stored: {'✅' if memory_stored else '❌'}")
                
            else:
                print(f"   ❌ Signal generation failed: {data.get('error', 'Unknown error')}")
                return
        else:
            print(f"   ❌ Signal generation request failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Error generating signal: {e}")
        return
    
    # Step 3: Wait briefly and check if signal appears in tracked signals
    print("\n3️⃣ CHECKING IF NEW SIGNAL APPEARS IN TRACKING...")
    time.sleep(2)  # Wait for database sync
    
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
        if response.status_code == 200:
            data = response.json()
            current_count = len(data.get('signals', []))
            signals = data.get('signals', [])
            
            print(f"   📊 Signals after generation: {current_count}")
            print(f"   📈 Signal count change: {current_count - initial_count}")
            
            # Look for our new signal
            found_new_signal = False
            for signal in signals:
                if signal.get('signal_id') == new_signal_id:
                    found_new_signal = True
                    print(f"   ✅ NEW SIGNAL FOUND in tracking system!")
                    print(f"      ID: {signal.get('signal_id')}")
                    print(f"      Type: {signal.get('signal_type')}")
                    print(f"      Entry: ${signal.get('entry_price')}")
                    print(f"      Status: {signal.get('status')}")
                    print(f"      P&L: ${signal.get('pnl', 0):.2f}")
                    break
            
            if not found_new_signal:
                print(f"   ❌ NEW SIGNAL NOT FOUND in tracking system!")
                print(f"   🔍 Looking for signal ID: {new_signal_id}")
                print("   📋 Current signals in system:")
                for i, signal in enumerate(signals):
                    print(f"      {i+1}. {signal.get('signal_id', 'NO_ID')} - {signal.get('signal_type', 'UNKNOWN')}")
            
        else:
            print(f"   ❌ Failed to get updated tracked signals: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking updated signals: {e}")
    
    # Step 4: Generate a second signal to test multi-signal tracking
    print("\n4️⃣ GENERATING SECOND SIGNAL TO TEST MULTI-SIGNAL TRACKING...")
    try:
        response = requests.post(f"{base_url}/api/signals/generate", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                signal_data = data.get('signal', {})
                second_signal_id = signal_data.get('signal_id', 'NO_ID')
                
                print(f"   ✅ Generated second signal: {second_signal_id}")
                
                # Check tracking again
                time.sleep(2)
                response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    final_count = len(data.get('signals', []))
                    
                    print(f"   📊 Final signal count: {final_count}")
                    print(f"   📈 Expected count: {initial_count + 2}")
                    
                    if final_count == initial_count + 2:
                        print("   ✅ BOTH SIGNALS SUCCESSFULLY TRACKED!")
                    elif final_count == initial_count + 1:
                        print("   ⚠️  ONLY ONE SIGNAL TRACKED (this is your reported issue)")
                    else:
                        print(f"   ❌ UNEXPECTED SIGNAL COUNT: {final_count}")
                    
                    # Show all current signals
                    print("\n📊 ALL CURRENT SIGNALS:")
                    for i, signal in enumerate(data.get('signals', [])):
                        signal_id = signal.get('signal_id', 'NO_ID')
                        signal_type = signal.get('signal_type', 'UNKNOWN')
                        entry_price = signal.get('entry_price', 0)
                        timestamp = signal.get('timestamp', 'NO_TIME')
                        memory_stored = signal.get('memory_stored', False)
                        
                        print(f"   {i+1}. ID: {signal_id}")
                        print(f"      Type: {signal_type} | Entry: ${entry_price}")
                        print(f"      Time: {timestamp}")
                        print(f"      Memory: {'✅' if memory_stored else '❌'}")
                        print(f"      Status: {signal.get('status', 'unknown')}")
                        print()
                
            else:
                print(f"   ❌ Second signal generation failed")
    except Exception as e:
        print(f"   ❌ Error with second signal: {e}")
    
    # Step 5: Test direct memory system access
    print("\n5️⃣ TESTING DIRECT MEMORY SYSTEM ACCESS...")
    try:
        response = requests.get(f"{base_url}/api/signal-status", timeout=30)
        if response.status_code == 200:
            data = response.json()
            memory_count = data.get('total_signals_in_memory', 0)
            active_signals_count = data.get('active_signals_count', 0)
            
            print(f"   🧠 Signals in memory system: {memory_count}")
            print(f"   🎯 Active signals in global list: {active_signals_count}")
            
            if memory_count != active_signals_count:
                print(f"   ⚠️  MISMATCH: Memory ({memory_count}) vs Active ({active_signals_count})")
                print("   This could be causing your tracking issues!")
            else:
                print(f"   ✅ Memory and active counts match")
            
        else:
            print(f"   ❌ Failed to get signal status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking memory system: {e}")
    
    print("\n" + "=" * 60)
    print("🔧 DIAGNOSIS COMPLETE")
    print("If signals are not staying in memory, the issue is likely:")
    print("1. Signals being stored in memory but not retrieved properly")
    print("2. Railway instance restarts clearing the global active_signals list")
    print("3. Duplicate detection preventing signals from being added")
    print("4. Database connectivity issues with the signal memory system")

if __name__ == "__main__":
    test_signal_storage_system()
