#!/usr/bin/env python3
"""
Quick test to see what the timeframe predictions API is actually returning
"""

import requests
import json
import time

def test_timeframe_predictions():
    """Test the timeframe predictions API multiple times"""
    print("🧪 Testing Timeframe Predictions API...")
    
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        try:
            response = requests.get('http://localhost:5000/api/timeframe-predictions')
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Success: {data.get('success', False)}")
                print(f"📊 Overall Bias: {data.get('market_overview', {}).get('overall_bias', 'N/A')}")
                
                timeframes = data.get('timeframes', {})
                print("\n🕒 Timeframe Signals:")
                
                signals_list = []
                for tf in ['5M', '15M', '30M', '1H', '4H', '1D', '1W']:
                    if tf in timeframes:
                        signal = timeframes[tf]['signal']
                        confidence = timeframes[tf]['confidence']
                        signals_list.append(signal)
                        print(f"  {tf}: {signal} ({confidence})")
                
                # Check for diversity
                unique_signals = set(signals_list)
                print(f"\n🎯 Signal Diversity: {len(unique_signals)} unique signals out of {len(signals_list)} timeframes")
                print(f"📈 Unique signals: {list(unique_signals)}")
                
                if len(unique_signals) == 1:
                    print("⚠️  WARNING: All timeframes have identical signals!")
                else:
                    print("✅ Good: Timeframes show varied signals")
                    
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            
        # Wait between tests
        if i < 2:
            time.sleep(2)

if __name__ == "__main__":
    test_timeframe_predictions()
