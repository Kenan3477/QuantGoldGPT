#!/usr/bin/env python3
"""
Test Flask app Gold API integration
"""
import sys
import os
sys.path.insert(0, '.')

def test_app_gold_api():
    try:
        from app import fetch_live_gold_price
        result = fetch_live_gold_price()
        if result:
            print(f'✅ App Gold Price: ${result["price"]:.2f} from {result["source"]}')
            return True
        else:
            print('❌ App failed to get price')
            return False
    except Exception as e:
        print(f'❌ App test error: {e}')
        return False

if __name__ == "__main__":
    print("🧪 Testing Flask App Gold API Integration")
    print("=" * 45)
    success = test_app_gold_api()
    if success:
        print("✅ Flask app successfully using new Gold API!")
    else:
        print("❌ Flask app needs attention")
