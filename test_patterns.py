#!/usr/bin/env python3
from real_pattern_detection import get_real_candlestick_patterns

print("Testing real pattern detection...")
try:
    patterns = get_real_candlestick_patterns()
    print(f"Found {len(patterns)} patterns:")
    for i, pattern in enumerate(patterns[:5]):  # Show first 5
        print(f"{i+1}. Pattern: {pattern}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
