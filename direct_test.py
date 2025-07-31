#!/usr/bin/env python
"""
Direct test of AI Signal Generator functions
"""
import sys
import os

print("🔍 Testing AI Signal Generator functions directly...")

try:
    # Import the functions
    print("📦 Importing functions...")
    from ai_signal_generator import get_trade_signal, get_signal_stats, get_open_trade_signals
    print("✅ Imports successful!")
    
    # Test signal generation
    print("\n🎯 Testing signal generation...")
    signal = get_trade_signal()
    if signal:
        print(f"✅ Signal generated: {signal.get('signal_type', 'Unknown')} - {signal.get('confidence', 0):.1f}% confidence")
    else:
        print("❌ No signal generated")
    
    # Test stats
    print("\n📊 Testing signal stats...")
    stats = get_signal_stats()
    print(f"✅ Stats retrieved: {stats}")
    
    # Test open signals
    print("\n📋 Testing open signals...")
    open_signals = get_open_trade_signals()
    print(f"✅ Open signals retrieved: {len(open_signals)} signals")
    
    print("\n🎉 All functions working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
