#!/usr/bin/env python3
"""
Debug Signal Generation Issues
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import yfinance as yf
        print("✅ yfinance imported")
    except Exception as e:
        print(f"❌ yfinance failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except Exception as e:
        print(f"❌ pandas failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported")
    except Exception as e:
        print(f"❌ numpy failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("✅ sklearn imported")
    except Exception as e:
        print(f"❌ sklearn failed: {e}")
        return False
    
    return True

def test_simple_signal():
    """Test simple signal generation"""
    print("\n🧪 Testing simple signal generation...")
    
    try:
        from simple_signal_generator import generate_signal_now
        result = generate_signal_now("GOLD", "1h")
        print(f"✅ Simple signal result: {result}")
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Simple signal failed: {e}")
        traceback.print_exc()
        return False

def test_advanced_signal():
    """Test advanced signal generation step by step"""
    print("\n🧪 Testing advanced signal generation...")
    
    try:
        # Import the class directly
        from advanced_trading_signal_manager import AdvancedSignalGenerator
        print("✅ AdvancedSignalGenerator imported")
        
        # Try to create instance
        generator = AdvancedSignalGenerator()
        print("✅ AdvancedSignalGenerator instance created")
        
        # Try to generate signal
        result = generator.generate_signal("GOLD", "1h")
        print(f"✅ Advanced signal result: {result}")
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Advanced signal failed: {e}")
        traceback.print_exc()
        return False

def test_gold_data_fetch():
    """Test gold data fetching"""
    print("\n🧪 Testing gold data fetching...")
    
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")  # Gold futures
        data = ticker.history(period="5d", interval="1h")
        
        if data.empty:
            print("⚠️ Gold futures data is empty, trying GOLD ticker...")
            ticker = yf.Ticker("GOLD")
            data = ticker.history(period="5d", interval="1h")
        
        if not data.empty:
            print(f"✅ Gold data fetched: {len(data)} rows")
            print(f"   Latest price: ${data['Close'][-1]:.2f}")
            return True
        else:
            print("❌ No gold data could be fetched")
            return False
            
    except Exception as e:
        print(f"❌ Gold data fetch failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 DEBUGGING SIGNAL GENERATION")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed - stopping")
        return
    
    # Test 2: Gold data
    if not test_gold_data_fetch():
        print("⚠️ Gold data fetch failed - this might cause issues")
    
    # Test 3: Simple signal
    simple_works = test_simple_signal()
    
    # Test 4: Advanced signal
    advanced_works = test_advanced_signal()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"   Simple Signal Generator: {'✅ WORKING' if simple_works else '❌ FAILED'}")
    print(f"   Advanced Signal Generator: {'✅ WORKING' if advanced_works else '❌ FAILED'}")
    
    if not simple_works and not advanced_works:
        print("\n💡 RECOMMENDATION: Both systems failed - check dependencies")
    elif simple_works and not advanced_works:
        print("\n💡 RECOMMENDATION: Use simple generator as fallback")
    else:
        print("\n✅ At least one signal system is working!")

if __name__ == "__main__":
    main()
