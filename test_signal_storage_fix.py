#!/usr/bin/env python3
"""
Test Signal Storage Fix
Verify signals are stored and displayed with live P&L
"""

import requests
import json
import time

def test_signal_storage_fix():
    """Test complete signal storage and display workflow"""
    
    base_url = "https://web-production-41882.up.railway.app"  # Railway deployment
    
    print("🔧 TESTING SIGNAL STORAGE FIXES")
    print("=" * 50)
    
    print("\n1. GENERATING NEW SIGNAL")
    try:
        response = requests.get(f"{base_url}/api/signals/generate", timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and 'signal' in data:
                signal = data['signal']
                signal_id = signal.get('signal_id', 'N/A')
                signal_type = signal.get('signal_type', 'N/A')
                entry_price = signal.get('entry_price', 'N/A')
                
                print(f"✅ Signal Generated Successfully!")
                print(f"   📊 ID: {signal_id}")
                print(f"   📈 Type: {signal_type}")
                print(f"   💰 Entry: ${entry_price}")
                print(f"   🎯 Confidence: {signal.get('confidence', 'N/A')}")
                print(f"   💾 Memory Stored: {signal.get('memory_stored', 'N/A')}")
                
                # Get current gold price for comparison
                price_response = requests.get(f"{base_url}/api/live-gold-price", timeout=10)
                if price_response.status_code == 200:
                    price_data = price_response.json()
                    current_price = price_data.get('price', 0)
                    print(f"   💎 Current Gold: ${current_price}")
                    
                    if abs(entry_price - current_price) < 0.01:
                        print("   ✅ PERFECT: Entry matches current price!")
                    else:
                        print(f"   ⚠️ Entry differs by ${abs(entry_price - current_price):.2f}")
                
                return signal_id
        else:
            print(f"❌ Signal generation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"💥 Error generating signal: {e}")
        return None
    
def test_signal_retrieval(signal_id):
    """Test signal retrieval and P&L calculation"""
    base_url = "https://web-production-41882.up.railway.app"
    
    print("\n2. TESTING SIGNAL RETRIEVAL & P&L")
    time.sleep(3)  # Wait for processing
    
    try:
        response = requests.get(f"{base_url}/api/signals/tracked", timeout=15)
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            
            print(f"✅ Retrieved {len(signals)} active signals")
            
            if signals:
                found_signal = False
                for i, signal in enumerate(signals):
                    current_id = signal.get('signal_id', 'N/A')
                    signal_type = signal.get('signal_type', 'N/A')
                    entry_price = signal.get('entry_price', 'N/A')
                    pnl = signal.get('pnl', 'N/A')
                    status = signal.get('status', 'N/A')
                    
                    print(f"   Signal {i+1}:")
                    print(f"     ID: {current_id}")
                    print(f"     Type: {signal_type}")
                    print(f"     Entry: ${entry_price}")
                    print(f"     P&L: {pnl}")
                    print(f"     Status: {status}")
                    
                    if current_id == signal_id:
                        found_signal = True
                        print(f"   ✅ FOUND NEW SIGNAL with live P&L!")
                        
                        # Validate P&L calculation
                        if isinstance(pnl, (int, float)) and pnl != 0:
                            print(f"   ✅ Live P&L calculation working!")
                        else:
                            print(f"   ⚠️ P&L calculation might not be working")
                
                if not found_signal and signal_id:
                    print(f"   ❌ Generated signal NOT FOUND in active list")
                    print(f"   🔍 Looking for ID: {signal_id}")
                    
            else:
                print(f"   ❌ NO ACTIVE SIGNALS FOUND")
                
        else:
            print(f"❌ Failed to retrieve signals: {response.status_code}")
    except Exception as e:
        print(f"💥 Error retrieving signals: {e}")

def test_signal_stats():
    """Test signal statistics"""
    base_url = "https://web-production-41882.up.railway.app"
    
    print("\n3. TESTING SIGNAL STATISTICS")
    try:
        response = requests.get(f"{base_url}/api/signals/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            
            print(f"✅ Signal Statistics:")
            print(f"   📊 Total Signals: {stats.get('total_signals', 0)}")
            print(f"   🎯 Win Rate: {stats.get('win_rate', 0)}%")
            print(f"   💰 Total P&L: ${stats.get('total_pnl', 0)}")
            print(f"   🔄 Active Signals: {stats.get('active_signals', 0)}")
            
            if stats.get('total_signals', 0) > 0:
                print(f"   ✅ Statistics are working!")
            else:
                print(f"   ⚠️ No signals counted in statistics")
                
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"💥 Error getting stats: {e}")

def main():
    """Run complete signal storage test"""
    signal_id = test_signal_storage_fix()
    
    if signal_id:
        test_signal_retrieval(signal_id)
    
    test_signal_stats()
    
    print(f"\n🎯 EXPECTED RESULTS:")
    print("✅ Generated signal should appear in active signals list")
    print("✅ Entry price should match current gold price exactly")
    print("✅ P&L should calculate based on current vs entry price")
    print("✅ Statistics should show non-zero values")
    print("✅ Signals should persist even after Railway restart")

if __name__ == "__main__":
    main()
