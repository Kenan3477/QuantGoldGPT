#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
This simulates what happens in the Docker container
"""

import sys
import os

print("🔍 Testing Docker deployment imports...")
print(f"📍 Current working directory: {os.getcwd()}")
print(f"📍 Python path: {sys.path}")

# List files in current directory
print("\n📂 Files in current directory:")
for file in os.listdir('.'):
    if file.endswith('.py'):
        print(f"  ✅ {file}")

# Test required imports
try:
    print("\n🧪 Testing signal_memory_system import...")
    from signal_memory_system import SignalMemorySystem, SignalData, create_signal_data
    print("  ✅ signal_memory_system import successful")
except ImportError as e:
    print(f"  ❌ signal_memory_system import failed: {e}")
    sys.exit(1)

try:
    print("\n🧪 Testing real_pattern_detection import...")
    from real_pattern_detection import get_real_candlestick_patterns, format_patterns_for_api
    print("  ✅ real_pattern_detection import successful")
except ImportError as e:
    print(f"  ❌ real_pattern_detection import failed: {e}")
    sys.exit(1)

print("\n🎉 All imports successful! Ready for deployment!")
