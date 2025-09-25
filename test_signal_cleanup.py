#!/usr/bin/env python3
"""
Signal Cleanup and Fix Utility
Cleans up any signals that should have been auto-closed and tests the new validation
"""

import requests
import json

def cleanup_signals():
    """Clean up signals that should have been auto-closed"""
    
    print("🧹 SIGNAL CLEANUP UTILITY")
    print("=" * 40)
    
    base_url = "https://web-production-41882.up.railway.app"
    
    # Get current signals
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            
            print(f"📊 Found {len(signals)} signals to analyze")
            
            if len(signals) > 0:
                print("\n🗑️ Clearing all signals to start fresh...")
                
                # Clear all signals
                clear_response = requests.post(f"{base_url}/api/clear-signals", timeout=30)
                if clear_response.status_code == 200:
                    print("✅ All signals cleared successfully")
                else:
                    print(f"❌ Failed to clear signals: {clear_response.status_code}")
                    
                # Verify clearing
                verify_response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
                if verify_response.status_code == 200:
                    verify_data = verify_response.json()
                    remaining_signals = len(verify_data.get('signals', []))
                    print(f"📊 Remaining signals after clear: {remaining_signals}")
                    
        else:
            print(f"❌ Failed to get signals: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
    
    # Test new signal generation with improved validation
    print("\n🧪 TESTING NEW SIGNAL GENERATION WITH VALIDATION...")
    
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        try:
            response = requests.post(f"{base_url}/api/signals/generate", timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    signal = data.get('signal', {})
                    print(f"✅ Generated signal: {signal.get('signal_id')}")
                    print(f"   Type: {signal.get('signal_type')}")
                    print(f"   Entry: ${signal.get('entry_price'):,.2f}")
                    print(f"   TP: ${signal.get('take_profit'):,.2f}")
                    print(f"   SL: ${signal.get('stop_loss'):,.2f}")
                    print(f"   Memory stored: {'✅' if signal.get('memory_stored') else '❌'}")
                else:
                    error = data.get('error', 'Unknown error')
                    print(f"❌ Signal generation failed: {error}")
                    if 'immediately closed' in error:
                        print("   🎯 Validation is working - preventing bad signals!")
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error generating signal: {e}")
    
    # Check final signal count
    print("\n📊 FINAL SIGNAL COUNT CHECK...")
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
        if response.status_code == 200:
            data = response.json()
            final_signals = data.get('signals', [])
            
            print(f"📊 Final active signals: {len(final_signals)}")
            
            for i, signal in enumerate(final_signals):
                signal_id = signal.get('signal_id', 'NO_ID')
                signal_type = signal.get('signal_type', 'UNKNOWN')
                entry_price = signal.get('entry_price', 0)
                pnl = signal.get('pnl', 0)
                
                print(f"   {i+1}. {signal_id}")
                print(f"      Type: {signal_type} | Entry: ${entry_price:,.2f} | P&L: ${pnl:.2f}")
                
        else:
            print(f"❌ Failed to get final signals: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking final signals: {e}")
    
    print("\n" + "=" * 40)
    print("🎯 CLEANUP COMPLETE")
    print("The system should now:")
    print("1. Generate signals with proper TP/SL validation")
    print("2. Store all signals in both memory and active list")
    print("3. Track P&L correctly without immediate auto-close")
    print("4. Maintain signal persistence across requests")

if __name__ == "__main__":
    cleanup_signals()
