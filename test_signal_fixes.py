#!/usr/bin/env python3
"""
Test Signal Generation with Current Gold Price
Verify that signals use exact current gold price as entry
"""

import requests
import json

def test_signal_fixes():
    """Test both local and Railway signal generation"""
    
    print("🧪 TESTING SIGNAL GENERATION FIXES")
    print("=" * 50)
    
    # Test local development server
    local_url = "http://127.0.0.1:5000"
    railway_url = "https://web-production-41882.up.railway.app"
    
    urls_to_test = [
        ("Local Development", local_url),
        ("Railway Production", railway_url)
    ]
    
    for name, base_url in urls_to_test:
        print(f"\n🔍 TESTING {name}")
        print("-" * 30)
        
        try:
            # Get current gold price first
            print("1. Getting current gold price...")
            price_response = requests.get(f"{base_url}/api/live-gold-price", timeout=10)
            if price_response.status_code == 200:
                price_data = price_response.json()
                current_price = price_data.get('price', 0)
                print(f"   ✅ Current gold price: ${current_price}")
            else:
                print(f"   ❌ Failed to get price: {price_response.status_code}")
                continue
            
            # Generate signal
            print("2. Generating signal...")
            signal_response = requests.get(f"{base_url}/api/signals/generate", timeout=15)
            if signal_response.status_code == 200:
                signal_data = signal_response.json()
                if 'signal' in signal_data:
                    signal = signal_data['signal']
                    entry_price = signal.get('entry_price', 0)
                    signal_type = signal.get('signal_type', 'N/A')
                    
                    print(f"   ✅ Signal generated: {signal_type}")
                    print(f"   📊 Entry price: ${entry_price}")
                    print(f"   💰 Current price: ${current_price}")
                    
                    # Check if entry price matches current price
                    price_diff = abs(entry_price - current_price)
                    if price_diff < 0.01:  # Allow tiny rounding differences
                        print(f"   ✅ PERFECT: Entry price matches current price!")
                    else:
                        print(f"   ⚠️ ISSUE: Entry price differs by ${price_diff:.2f}")
                        
                    # Show key signal details
                    print(f"   🎯 Take Profit: ${signal.get('take_profit', 'N/A')}")
                    print(f"   🛑 Stop Loss: ${signal.get('stop_loss', 'N/A')}")
                    print(f"   📈 Confidence: {signal.get('confidence', 'N/A')}")
                    
                    # Check for technical analysis
                    if 'technical_indicators' in signal or 'key_factors' in signal:
                        print(f"   ✅ Technical analysis present")
                    else:
                        print(f"   ⚠️ Limited technical analysis data")
                        
                else:
                    print(f"   ❌ No signal in response")
            else:
                print(f"   ❌ Signal generation failed: {signal_response.status_code}")
                
            # Test active signals
            print("3. Checking active signals...")
            active_response = requests.get(f"{base_url}/api/signals/tracked", timeout=10)
            if active_response.status_code == 200:
                active_data = active_response.json()
                signals = active_data.get('signals', [])
                print(f"   ✅ Found {len(signals)} active signals")
                
                if signals:
                    latest_signal = signals[-1]  # Get latest signal
                    print(f"   📊 Latest signal: {latest_signal.get('signal_type', 'N/A')} @ ${latest_signal.get('entry_price', 'N/A')}")
                    print(f"   💹 P&L: ${latest_signal.get('pnl', 'N/A')}")
                else:
                    print(f"   ℹ️ No active signals currently")
            else:
                print(f"   ❌ Active signals check failed: {active_response.status_code}")
                
        except Exception as e:
            print(f"   💥 Test failed for {name}: {e}")
    
    print(f"\n🎯 TEST SUMMARY")
    print("If you see 'PERFECT: Entry price matches current price!' above,")
    print("then the fix is working correctly!")
    print("Active signals should now show up in the dashboard.")

if __name__ == "__main__":
    test_signal_fixes()
