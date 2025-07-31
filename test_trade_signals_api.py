#!/usr/bin/env python
"""
Test the new /api/trade-signals endpoint
"""
import requests
import json
from datetime import datetime

def test_trade_signals_api():
    """Test the comprehensive trade signals API endpoint"""
    url = "http://localhost:5000/api/trade-signals"
    
    print("🧪 Testing /api/trade-signals endpoint...")
    print(f"📡 Making request to: {url}")
    print("=" * 60)
    
    try:
        # Make GET request to the endpoint
        response = requests.get(url, timeout=10)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"⏰ Response Time: {response.elapsed.total_seconds():.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ SUCCESS! Trade signals retrieved successfully!")
            print("\n📈 API Response Structure:")
            print("-" * 40)
            
            # Display current signal
            current_signal = data.get('current_signal')
            if current_signal:
                print(f"🎯 Current Signal:")
                print(f"   Type: {current_signal.get('signal_type', 'N/A')}")
                print(f"   Confidence: {current_signal.get('confidence', 0):.1f}%")
                print(f"   Entry Price: ${current_signal.get('entry_price', 0):.2f}")
                print(f"   Target: ${current_signal.get('target_price', 0):.2f}")
                print(f"   Stop Loss: ${current_signal.get('stop_loss', 0):.2f}")
            else:
                print("🎯 Current Signal: No active signal")
            
            # Display statistics
            stats = data.get('statistics', {})
            print(f"\n📊 Statistics:")
            print(f"   Total Signals: {stats.get('total_signals', 0)}")
            print(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")
            print(f"   Avg Profit: {stats.get('avg_profit', 0):.2f}%")
            print(f"   Profit Factor: {stats.get('profit_factor', 0):.2f}")
            
            # Display open signals
            open_signals = data.get('open_signals', [])
            print(f"\n📋 Open Signals: {len(open_signals)} active")
            
            for i, signal in enumerate(open_signals[:3], 1):  # Show first 3
                print(f"   #{i}: {signal.get('signal_type', 'Unknown')} - {signal.get('confidence', 0):.1f}%")
            
            if len(open_signals) > 3:
                print(f"   ... and {len(open_signals) - 3} more")
            
            print(f"\n⏰ Timestamp: {data.get('timestamp', 'N/A')}")
            
        else:
            print(f"\n❌ FAILED! Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Error: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"\n🚫 Connection Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_trade_signals_api()
