#!/usr/bin/env python3
"""
COMPREHENSIVE SIGNAL SYSTEM TEST - POST FIXES
Test all signal generation, storage, and display functionality
"""

import requests
import json
import time

def test_complete_signal_system():
    """Test complete signal system after fixes"""
    
    BASE_URL = 'https://web-production-41882.up.railway.app'
    
    print("🔧 TESTING COMPLETE SIGNAL SYSTEM - POST FIXES")
    print("=" * 70)
    
    # STEP 1: Test signal generation with new fixes
    print("\n🎯 STEP 1: GENERATE SIGNAL WITH FIXES")
    print("-" * 50)
    
    try:
        response = requests.get(f'{BASE_URL}/api/signals/generate', timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response success: {data.get('success', False)}")
            
            if 'signal' in data:
                signal = data['signal']
                signal_id = signal.get('signal_id', 'NO_ID')
                frontend_id = signal.get('id', 'NO_FRONTEND_ID')
                signal_type = signal.get('signal_type', 'NO_TYPE')
                entry_price = signal.get('entry_price', 'NO_PRICE')
                
                print(f"✅ Signal generated:")
                print(f"   Backend ID: {signal_id}")
                print(f"   Frontend ID: {frontend_id}")
                print(f"   Type: {signal_type}")
                print(f"   Entry: ${entry_price}")
                print(f"   Memory stored: {signal.get('memory_stored', 'N/A')}")
                
                # Check if signal has ALL frontend-required fields
                required_fields = ['id', 'signal_id', 'signal_type', 'entry_price', 
                                 'take_profit', 'stop_loss', 'confidence', 'live_pnl', 
                                 'live_pnl_pct', 'current_price', 'pnl_status']
                
                missing = [field for field in required_fields if field not in signal]
                if missing:
                    print(f"⚠️  Missing fields: {missing}")
                else:
                    print(f"✅ All frontend fields present")
                    
                return signal_id, frontend_id
            else:
                print("❌ No signal in response")
                return None, None
        else:
            print(f"❌ Signal generation failed: {response.status_code}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None
    
def test_signal_retrieval(signal_id, frontend_id):
    """Test signal retrieval after generation"""
    
    BASE_URL = 'https://web-production-41882.up.railway.app'
    
    print(f"\n📊 STEP 2: TEST SIGNAL RETRIEVAL")
    print("-" * 50)
    
    time.sleep(3)  # Wait for processing
    
    try:
        response = requests.get(f'{BASE_URL}/api/signals/tracked', timeout=20)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response success: {data.get('success', False)}")
            
            signals = data.get('signals', [])
            print(f"✅ Retrieved {len(signals)} signals")
            
            if signals:
                print("\n📈 SIGNAL DETAILS:")
                signal_found = False
                
                for i, signal in enumerate(signals):
                    sid = signal.get('signal_id', 'NO_ID')
                    fid = signal.get('id', 'NO_FID')
                    stype = signal.get('signal_type', 'NO_TYPE')
                    entry = signal.get('entry_price', 'NO_ENTRY')
                    current = signal.get('current_price', 'NO_CURRENT')
                    live_pnl = signal.get('live_pnl', 'NO_PNL')
                    pnl_pct = signal.get('live_pnl_pct', 'NO_PCT')
                    status = signal.get('pnl_status', 'NO_STATUS')
                    
                    print(f"\n   Signal {i+1}:")
                    print(f"     Backend ID: {sid}")
                    print(f"     Frontend ID: {fid}")
                    print(f"     Type: {stype}")
                    print(f"     Entry: ${entry}")
                    print(f"     Current: ${current}")
                    print(f"     Live P&L: {live_pnl}")
                    print(f"     P&L %: {pnl_pct}%")
                    print(f"     Status: {status}")
                    
                    # Check if this is our generated signal
                    if sid == signal_id or fid == frontend_id:
                        signal_found = True
                        print(f"     ✅ THIS IS OUR GENERATED SIGNAL!")
                        
                        # Validate P&L calculation
                        try:
                            entry_val = float(entry)
                            current_val = float(current)
                            pnl_val = float(live_pnl)
                            
                            if stype == 'BUY':
                                expected_pnl = current_val - entry_val
                            else:
                                expected_pnl = entry_val - current_val
                            
                            if abs(pnl_val - expected_pnl) < 0.01:
                                print(f"     ✅ P&L calculation is CORRECT!")
                            else:
                                print(f"     ⚠️  P&L calculation issue: Expected {expected_pnl}, got {pnl_val}")
                                
                        except:
                            print(f"     ⚠️  Could not validate P&L calculation")
                
                if not signal_found:
                    print(f"\n❌ OUR SIGNAL NOT FOUND!")
                    print(f"   Looking for Backend ID: {signal_id}")
                    print(f"   Looking for Frontend ID: {frontend_id}")
                else:
                    print(f"\n✅ SIGNAL RETRIEVAL SUCCESS!")
                    
            else:
                print("❌ NO SIGNALS RETRIEVED")
                
        else:
            print(f"❌ Retrieval failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
def test_frontend_compatibility():
    """Test what frontend would actually see"""
    
    BASE_URL = 'https://web-production-41882.up.railway.app'
    
    print(f"\n🌐 STEP 3: FRONTEND COMPATIBILITY TEST")
    print("-" * 50)
    
    try:
        # Simulate exactly what loadActiveSignals() does
        response = requests.get(f'{BASE_URL}/api/signals/tracked', timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success') and data.get('signals'):
                signals = data['signals']
                print(f"✅ Frontend would receive {len(signals)} signals")
                
                # Test displayActiveSignals() requirements
                for signal in signals:
                    required_display_fields = [
                        'id', 'signal_type', 'timestamp', 'confidence',
                        'entry_price', 'take_profit', 'stop_loss',
                        'current_price', 'live_pnl', 'live_pnl_pct', 'pnl_status'
                    ]
                    
                    missing_fields = [field for field in required_display_fields if field not in signal]
                    
                    if missing_fields:
                        print(f"⚠️  Signal {signal.get('id', 'UNKNOWN')} missing: {missing_fields}")
                    else:
                        print(f"✅ Signal {signal.get('id', 'UNKNOWN')} has all display fields")
                        
            else:
                print("❌ Frontend would see NO SIGNALS")
                print("   This is why the Active Signals section is empty!")
        else:
            print(f"❌ Frontend API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Frontend test error: {e}")

def main():
    """Run complete test suite"""
    
    print("🚀 STARTING COMPREHENSIVE SIGNAL SYSTEM TEST")
    
    # Generate signal and get IDs
    signal_id, frontend_id = test_complete_signal_system()
    
    if signal_id:
        # Test retrieval
        test_signal_retrieval(signal_id, frontend_id)
    
    # Test frontend compatibility
    test_frontend_compatibility()
    
    print(f"\n🎯 FINAL DIAGNOSIS:")
    print("=" * 70)
    print("✅ If signal generation works AND retrieval shows the signal")
    print("   with correct P&L = SYSTEM IS FIXED!")
    print("❌ If signals don't appear in retrieval = Still broken")
    print("⚠️  If signals appear but missing fields = Partial fix")

if __name__ == "__main__":
    main()
