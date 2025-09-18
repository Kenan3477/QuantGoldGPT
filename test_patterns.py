#!/usr/bin/env python3

import sys
import logging
from real_pattern_detection import RealCandlestickDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_pattern_detection():
    print("🔄 Testing Real-Time Pattern Detection...")
    
    try:
        # Create detector
        detector = RealCandlestickDetector()
        
        # Test pattern detection
        patterns = detector.detect_all_patterns()
        
        print(f"✅ Detection completed: Found {len(patterns)} patterns")
        
        if patterns:
            print("\n📊 Top 5 Patterns:")
            for i, pattern in enumerate(patterns[:5]):
                print(f"{i+1}. {pattern.get('name', 'Unknown')}: {pattern.get('confidence', 0):.1f}% confidence")
                print(f"   Signal: {pattern.get('signal', 'UNKNOWN')}")
                print(f"   Time: {pattern.get('timestamp', 'Unknown')}")
                print()
        else:
            print("❌ No patterns detected - this indicates an issue!")
            
        return patterns
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    patterns = test_pattern_detection()
