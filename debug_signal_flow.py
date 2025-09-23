#!/usr/bin/env python3
"""
Debug Complete Signal Flow - Railway Deployment
Find out exactly why signals aren't showing in active signals section
"""

import requests
import json
import time

def debug_signal_flow():
    """Debug complete signal flow step by step"""
    
    BASE_URL = 'https://web-production-41882.up.railway.app'
    
    print("🔍 DEBUGGING COMPLETE SIGNAL FLOW")
    print("=" * 60)
    
    # STEP 1: Check current state
    print("\n📊 STEP 1: CHECK CURRENT STATE")
    print("-" * 40)
    
    try:
        # Check debug endpoint
        response = requests.get(f'{BASE_URL}/api/debug/signals', timeout=15)
        if response.status_code == 200:
            debug_data = response.json()
            print(f"✅ Debug endpoint working")
            print(f"   Active signals count: {debug_data.get('active_signals_count', 'N/A')}")
            print(f"   Memory system exists: {debug_data.get('signal_memory_exists', False)}")
            print(f"   Advanced learning exists: {debug_data.get('advanced_learning_exists', False)}")
            if 'memory_signals_count' in debug_data:
                print(f"   Memory signals count: {debug_data['memory_signals_count']}")
        else:
            print(f"❌ Debug endpoint failed: {response.status_code}")
            
        # Check tracked signals BEFORE generation
        response = requests.get(f'{BASE_URL}/api/signals/tracked', timeout=15)
        if response.status_code == 200:
            tracked_data = response.json()
            signals_before = tracked_data.get('signals', [])
            print(f"   Tracked signals BEFORE: {len(signals_before)}")
        else:
            print(f"❌ Tracked signals failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Step 1 error: {e}")
    
    # STEP 2: Generate signal
    print("\n🎯 STEP 2: GENERATE SIGNAL")
    print("-" * 40)
    
    signal_id = None
    signal_data = None
    
    try:
        response = requests.get(f'{BASE_URL}/api/signals/generate', timeout=30)
        print(f"Signal generation status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Signal generation response received")
            
            # Print full response for debugging
            print(f"Response keys: {list(data.keys())}")
            
            if 'signal' in data:
                signal_data = data['signal']
                signal_id = signal_data.get('signal_id', 'NO_ID')
                signal_type = signal_data.get('signal_type', 'NO_TYPE')
                entry_price = signal_data.get('entry_price', 'NO_PRICE')
                
                print(f"✅ Signal created successfully")
                print(f"   ID: {signal_id}")
                print(f"   Type: {signal_type}")
                print(f"   Entry: ${entry_price}")
                print(f"   Memory stored: {signal_data.get('memory_stored', 'N/A')}")
                
                # Check if signal has all required fields
                required_fields = ['signal_id', 'signal_type', 'entry_price', 'take_profit', 'stop_loss']
                missing_fields = [field for field in required_fields if field not in signal_data]
                if missing_fields:
                    print(f"⚠️  Missing fields: {missing_fields}")
                else:
                    print(f"✅ All required fields present")
                    
            else:
                print(f"❌ No 'signal' in response")
                print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Signal generation failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Step 2 error: {e}")
        
    # STEP 3: Wait and check debug info again
    if signal_id:
        print("\n⏱️  STEP 3: WAIT AND CHECK STORAGE")
        print("-" * 40)
        
        time.sleep(3)  # Wait for processing
        
        try:
            response = requests.get(f'{BASE_URL}/api/debug/signals', timeout=15)
            if response.status_code == 200:
                debug_data = response.json()
                print(f"✅ Debug endpoint after generation")
                print(f"   Active signals count: {debug_data.get('active_signals_count', 'N/A')}")
                if 'memory_signals_count' in debug_data:
                    print(f"   Memory signals count: {debug_data['memory_signals_count']}")
                    
                # Check if our signal is in the debug list
                if 'active_signals_list' in debug_data:
                    active_list = debug_data['active_signals_list']
                    found_signal = any(s.get('id') == signal_id for s in active_list)
                    print(f"   Our signal in active list: {found_signal}")
                    
        except Exception as e:
            print(f"❌ Step 3 error: {e}")
            
    # STEP 4: Check tracked signals AFTER generation
    print("\n📈 STEP 4: CHECK TRACKED SIGNALS")
    print("-" * 40)
    
    try:
        response = requests.get(f'{BASE_URL}/api/signals/tracked', timeout=15)
        if response.status_code == 200:
            tracked_data = response.json()
            signals_after = tracked_data.get('signals', [])
            print(f"✅ Tracked signals AFTER: {len(signals_after)}")
            
            if signals_after:
                print("   Signal details:")
                for i, signal in enumerate(signals_after[:3]):  # Show first 3
                    sid = signal.get('signal_id', 'NO_ID')
                    stype = signal.get('signal_type', 'NO_TYPE')
                    entry = signal.get('entry_price', 'NO_PRICE')
                    pnl = signal.get('pnl', 'NO_PNL')
                    status = signal.get('status', 'NO_STATUS')
                    
                    print(f"     {i+1}. ID: {sid}")
                    print(f"        Type: {stype}, Entry: ${entry}")
                    print(f"        P&L: {pnl}, Status: {status}")
                    
                    if sid == signal_id:
                        print(f"        ✅ THIS IS OUR NEW SIGNAL!")
                        
                if signal_id and not any(s.get('signal_id') == signal_id for s in signals_after):
                    print(f"   ❌ OUR SIGNAL NOT FOUND in tracked signals!")
                    print(f"   🔍 Looking for ID: {signal_id}")
            else:
                print("   ❌ NO SIGNALS in tracked response")
                
            # Check response structure
            print(f"   Response success: {tracked_data.get('success', 'N/A')}")
            
        else:
            print(f"❌ Tracked signals failed: {response.status_code}")
            print(f"Response: {response.text[:300]}")
            
    except Exception as e:
        print(f"❌ Step 4 error: {e}")
        
    # STEP 5: Check what frontend would see
    print("\n🌐 STEP 5: FRONTEND PERSPECTIVE")
    print("-" * 40)
    
    try:
        # Simulate what JavaScript loadActiveSignals() does
        response = requests.get(f'{BASE_URL}/api/signals/tracked', timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            print("Frontend would receive:")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Signal count: {len(data.get('signals', []))}")
            
            if data.get('signals'):
                print("   Signals for display:")
                for signal in data['signals']:
                    print(f"     - {signal.get('signal_type')} @ ${signal.get('entry_price')} (P&L: {signal.get('pnl', 0)})")
            else:
                print("   ❌ NO SIGNALS TO DISPLAY")
                print("   This is why the Active Signals section is empty!")
        
    except Exception as e:
        print(f"❌ Step 5 error: {e}")
        
    # SUMMARY
    print("\n🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("If you see:")
    print("✅ Signal created BUT not in tracked signals = Storage/retrieval issue")
    print("✅ Signal in tracked BUT not displaying = Frontend issue")  
    print("❌ Signal not created = Generation issue")
    print("❌ No signals at all = System-wide problem")

if __name__ == "__main__":
    debug_signal_flow()
