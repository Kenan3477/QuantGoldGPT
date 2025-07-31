#!/usr/bin/env python3
"""
Test Enhanced Signal API Endpoints
"""
import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Enhanced Signal API Endpoints")
    print("=" * 50)
    
    try:
        # Test 1: Generate Signal
        print("\n🎯 Testing Signal Generation...")
        response = requests.post(f"{base_url}/api/enhanced-signals/generate", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('signal'):
                signal = data['signal']
                print(f"✅ Signal Generated:")
                print(f"   Type: {signal['signal_type'].upper()}")
                print(f"   Entry: ${signal['entry_price']:.2f}")
                print(f"   TP: ${signal['target_price']:.2f}")
                print(f"   SL: ${signal['stop_loss']:.2f}")
                print(f"   Confidence: {signal['confidence']:.1f}%")
                print(f"   R:R: {signal['risk_reward_ratio']:.1f}:1")
            else:
                print(f"ℹ️ No signal: {data.get('message', 'Unknown')}")
        else:
            print(f"❌ Error: {response.status_code}")
            
        # Test 2: Monitor Signals
        print("\n🔄 Testing Signal Monitoring...")
        response = requests.get(f"{base_url}/api/enhanced-signals/monitor", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                monitoring = data['monitoring']
                print(f"✅ Monitoring Active:")
                print(f"   Current Price: ${monitoring.get('current_price', 0):.2f}")
                print(f"   Active Signals: {monitoring.get('active_signals', 0)}")
                print(f"   Updates: {len(monitoring.get('updates', []))}")
                print(f"   Closed: {len(monitoring.get('closed_signals', []))}")
            else:
                print(f"❌ Error: {data.get('error', 'Unknown')}")
        else:
            print(f"❌ Error: {response.status_code}")
            
        # Test 3: Performance
        print("\n📈 Testing Performance...")
        response = requests.get(f"{base_url}/api/enhanced-signals/performance", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                perf = data['performance']
                print(f"✅ Performance Data:")
                print(f"   Total Signals: {perf.get('total_signals', 0)}")
                print(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
                print(f"   Avg P&L: {perf.get('avg_profit_loss_pct', 0):.2f}%")
            else:
                print(f"❌ Error: {data.get('error', 'Unknown')}")
        else:
            print(f"❌ Error: {response.status_code}")
            
        # Test 4: Active Signals
        print("\n📊 Testing Active Signals...")
        response = requests.get(f"{base_url}/api/enhanced-signals/active", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                signals = data.get('active_signals', [])
                print(f"✅ Active Signals: {len(signals)}")
                for i, signal in enumerate(signals[:2]):
                    print(f"   Signal {i+1}: {signal.get('signal_type', '').upper()}")
                    print(f"              Entry: ${signal.get('entry_price', 0):.2f}")
                    print(f"              P&L: ${signal.get('unrealized_pnl', 0):.2f}")
            else:
                print(f"❌ Error: {data.get('error', 'Unknown')}")
        else:
            print(f"❌ Error: {response.status_code}")
            
        print("\n🎉 API Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_api_endpoints()
