#!/usr/bin/env python3
"""Test confidence values in pattern detection"""

from real_pattern_detection import get_real_candlestick_patterns, format_patterns_for_api

try:
    print("🔍 Testing pattern detection confidence values...")
    
    # Get raw patterns
    patterns = get_real_candlestick_patterns()
    print(f"📊 Found {len(patterns)} raw patterns")
    
    if patterns:
        for i, pattern in enumerate(patterns[:3]):
            name = pattern.get('name', pattern.get('pattern', 'Unknown'))
            confidence = pattern.get('confidence', 'Missing')
            signal = pattern.get('signal', 'Unknown')
            print(f"   Pattern {i+1}: {name}")
            print(f"   Confidence: {confidence}")
            print(f"   Signal: {signal}")
            print("   ---")
    
    # Test formatted patterns
    print("\n🎯 Testing formatted patterns...")
    formatted_patterns = format_patterns_for_api(patterns)
    print(f"📊 Formatted {len(formatted_patterns)} patterns")
    
    if formatted_patterns:
        for i, pattern in enumerate(formatted_patterns[:3]):
            name = pattern.get('pattern', 'Unknown')
            confidence = pattern.get('confidence', 'Missing')
            signal = pattern.get('signal', 'Unknown')
            print(f"   Formatted Pattern {i+1}: {name}")
            print(f"   Confidence: {confidence}")
            print(f"   Signal: {signal}")
            print("   ---")
    
    print("✅ Confidence test completed!")
    
except Exception as e:
    print(f"❌ Error testing confidence: {e}")
    import traceback
    traceback.print_exc()
