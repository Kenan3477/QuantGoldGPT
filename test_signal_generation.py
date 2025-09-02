#!/usr/bin/env python3
"""
Test script to diagnose signal generation issues
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_generator():
    """Test the simple signal generator"""
    print("🧪 Testing Simple Signal Generator...")
    try:
        from simple_signal_generator import generate_signal_now
        result = generate_signal_now("GOLD", "1h")
        print(f"✅ Simple generator result: {result.get('success', False)}")
        if result.get('success'):
            print(f"   Signal: {result.get('signal_type')} at ${result.get('entry_price', 0):.2f}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Simple generator failed: {e}")
        return False

def test_advanced_generator():
    """Test the advanced signal generator"""
    print("\n🧪 Testing Advanced Signal Generator...")
    try:
        from advanced_trading_signal_manager import generate_trading_signal
        result = generate_trading_signal("GOLD", "1h")
        print(f"✅ Advanced generator result: {result.get('success', False)}")
        if result.get('success'):
            print(f"   Signal: {result.get('signal_type')} at ${result.get('entry_price', 0):.2f}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Advanced generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint():
    """Test the API endpoint directly"""
    print("\n🧪 Testing API endpoint...")
    try:
        import requests
        response = requests.get("http://localhost:5000/api/signals/generate", timeout=10)
        print(f"✅ API endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   API result: {data.get('success', False)}")
            if data.get('success'):
                signal = data.get('signal', {})
                print(f"   Signal: {signal.get('signal_type')} at ${signal.get('entry_price', 0):.2f}")
            else:
                print(f"   API Error: {data.get('error', 'Unknown')}")
        else:
            print(f"   HTTP Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 DIAGNOSING SIGNAL GENERATION ISSUES")
    print("="*50)
    
    # Test simple generator
    simple_works = test_simple_generator()
    
    # Test advanced generator
    advanced_works = test_advanced_generator()
    
    # Test API endpoint
    api_works = test_api_endpoint()
    
    print("\n📊 DIAGNOSIS RESULTS:")
    print(f"   Simple Generator: {'✅ WORKING' if simple_works else '❌ FAILED'}")
    print(f"   Advanced Generator: {'✅ WORKING' if advanced_works else '❌ FAILED'}")
    print(f"   API Endpoint: {'✅ WORKING' if api_works else '❌ FAILED'}")
    
    if not simple_works and not advanced_works:
        print("\n🚨 CRITICAL: Both generators failed!")
    elif simple_works and not advanced_works:
        print("\n⚠️ Advanced generator failed, but simple generator works")
    elif not api_works:
        print("\n⚠️ API endpoint failed")
    else:
        print("\n✅ All systems working")
