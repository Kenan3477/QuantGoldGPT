#!/usr/bin/env python3
"""
Signal Auto-Close Analysis Test
Checks if signals are being auto-closed too quickly due to tight stop losses
"""

import requests
import json
from datetime import datetime
import yfinance as yf

def test_signal_auto_close():
    """Test if signals are being auto-closed too quickly"""
    
    print("🔍 SIGNAL AUTO-CLOSE ANALYSIS")
    print("=" * 50)
    
    base_url = "https://web-production-41882.up.railway.app"
    
    # Get current gold price
    try:
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="1d")
        current_gold_price = float(hist["Close"].iloc[-1])
        print(f"📊 Current Gold Price: ${current_gold_price:,.2f}")
    except:
        current_gold_price = 3750.0
        print(f"📊 Using fallback Gold Price: ${current_gold_price:,.2f}")
    
    # Get tracked signals
    print("\n🎯 ANALYZING CURRENT SIGNALS...")
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=30)
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            
            print(f"📊 Found {len(signals)} active signals")
            
            for i, signal in enumerate(signals):
                signal_id = signal.get('signal_id', 'NO_ID')
                signal_type = signal.get('signal_type', 'UNKNOWN')
                entry_price = signal.get('entry_price', 0)
                take_profit = signal.get('take_profit', 0)
                stop_loss = signal.get('stop_loss', 0)
                timestamp = signal.get('timestamp', '')
                
                print(f"\n   Signal {i+1}: {signal_id}")
                print(f"   Type: {signal_type}")
                print(f"   Entry: ${entry_price:,.2f}")
                print(f"   Take Profit: ${take_profit:,.2f}")
                print(f"   Stop Loss: ${stop_loss:,.2f}")
                print(f"   Generated: {timestamp}")
                
                # Calculate distances from current price
                if signal_type == 'BUY':
                    distance_to_tp = take_profit - current_gold_price
                    distance_to_sl = current_gold_price - stop_loss
                    tp_hit = current_gold_price >= take_profit
                    sl_hit = current_gold_price <= stop_loss
                else:  # SELL
                    distance_to_tp = current_gold_price - take_profit
                    distance_to_sl = stop_loss - current_gold_price
                    tp_hit = current_gold_price <= take_profit
                    sl_hit = current_gold_price >= stop_loss
                
                print(f"   Distance to TP: ${distance_to_tp:+.2f}")
                print(f"   Distance to SL: ${distance_to_sl:+.2f}")
                
                # Check if should be auto-closed
                if tp_hit:
                    print("   🎯 TP HIT - Should be auto-closed!")
                elif sl_hit:
                    print("   🛑 SL HIT - Should be auto-closed!")
                else:
                    print("   ✅ Signal should remain active")
                
                # Calculate current P&L
                if signal_type == 'BUY':
                    pnl = current_gold_price - entry_price
                else:
                    pnl = entry_price - current_gold_price
                
                print(f"   P&L: ${pnl:+.2f}")
                
                # Check how long signal has been active
                try:
                    entry_time = datetime.fromisoformat(timestamp.replace('Z', ''))
                    time_elapsed = datetime.now() - entry_time
                    minutes_elapsed = int(time_elapsed.total_seconds() / 60)
                    print(f"   Age: {minutes_elapsed} minutes")
                except:
                    print("   Age: Unknown")
        
        else:
            print(f"❌ Failed to get signals: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error analyzing signals: {e}")
    
    # Test signal generation with wider stops
    print(f"\n🔧 TESTING SIGNAL GENERATION WITH ANALYSIS...")
    try:
        response = requests.post(f"{base_url}/api/signals/generate", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                signal = data.get('signal', {})
                signal_type = signal.get('signal_type', 'UNKNOWN')
                entry_price = signal.get('entry_price', 0)
                take_profit = signal.get('take_profit', 0)
                stop_loss = signal.get('stop_loss', 0)
                
                print(f"✅ Generated new signal:")
                print(f"   Type: {signal_type}")
                print(f"   Entry: ${entry_price:,.2f}")
                print(f"   Take Profit: ${take_profit:,.2f}")
                print(f"   Stop Loss: ${stop_loss:,.2f}")
                
                # Analyze the new signal
                tp_distance = abs(take_profit - entry_price)
                sl_distance = abs(stop_loss - entry_price)
                
                print(f"   TP Distance: ${tp_distance:.2f}")
                print(f"   SL Distance: ${sl_distance:.2f}")
                print(f"   Risk/Reward: 1:{tp_distance/sl_distance:.2f}")
                
                # Check if stops are too tight
                price_range_pct = (max(tp_distance, sl_distance) / entry_price) * 100
                if price_range_pct < 0.5:
                    print("   ⚠️  STOPS MAY BE TOO TIGHT! (< 0.5% range)")
                elif price_range_pct < 1.0:
                    print("   ⚠️  Stops are tight (< 1.0% range)")
                else:
                    print("   ✅ Stop distances look reasonable")
                
                # Check immediate auto-close risk
                if signal_type == 'BUY':
                    tp_hit = current_gold_price >= take_profit
                    sl_hit = current_gold_price <= stop_loss
                else:
                    tp_hit = current_gold_price <= take_profit
                    sl_hit = current_gold_price >= stop_loss
                
                if tp_hit or sl_hit:
                    print("   🚨 SIGNAL WOULD BE IMMEDIATELY AUTO-CLOSED!")
                    print("   This explains why signals disappear quickly!")
                else:
                    print("   ✅ Signal should remain active initially")
                    
            else:
                print(f"❌ Signal generation failed: {data.get('error')}")
        else:
            print(f"❌ Signal generation request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing signal generation: {e}")
    
    print("\n" + "=" * 50)
    print("🔧 ANALYSIS COMPLETE")
    print("Common causes of disappearing signals:")
    print("1. Stop losses too tight - signals auto-close immediately")
    print("2. Current price already past TP/SL when signal generated")
    print("3. Frequent Railway restarts clearing global signal list")
    print("4. Frontend not refreshing or showing closed signals")

if __name__ == "__main__":
    test_signal_auto_close()
