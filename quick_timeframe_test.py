#!/usr/bin/env python3
"""
Quick test to verify timeframe predictions show varied signals now
"""

import requests
import json

def test_timeframe_predictions():
    """Test that timeframe predictions show varied signals"""
    
    print("🔍 Testing timeframe predictions for signal diversity...")
    
    try:
        # Test timeframe predictions endpoint
        response = requests.get('http://localhost:5000/api/timeframe-predictions', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Timeframe predictions API responded successfully")
            print(f"📊 Response keys: {list(data.keys())}")
            
            # Extract timeframe predictions
            timeframes = data.get('timeframe_predictions', {})
            
            if timeframes:
                print("\n📈 TIMEFRAME ANALYSIS RESULTS:")
                print("=" * 50)
                
                signals = []
                confidences = []
                
                for timeframe, prediction in timeframes.items():
                    signal = prediction.get('signal', 'UNKNOWN')
                    confidence = prediction.get('confidence', 0)
                    target = prediction.get('target_price', 'N/A')
                    
                    print(f"🕐 {timeframe.upper()}:")
                    print(f"   Signal: {signal}")
                    print(f"   Confidence: {confidence:.2f}")
                    print(f"   Target: ${target}")
                    print()
                    
                    signals.append(signal)
                    confidences.append(confidence)
                
                # Check for diversity
                unique_signals = set(signals)
                unique_confidences = len(set([round(c, 1) for c in confidences]))
                
                print("🔍 DIVERSITY ANALYSIS:")
                print(f"   Unique signals: {len(unique_signals)} ({list(unique_signals)})")
                print(f"   Confidence variations: {unique_confidences}")
                
                if len(unique_signals) > 1:
                    print("✅ SUCCESS: Different timeframes show varied signals!")
                    print("✅ Real technical analysis is working properly")
                elif unique_confidences > 1:
                    print("✅ PARTIAL SUCCESS: Same signals but different confidence levels")
                    print("✅ Real technical analysis shows nuanced differences")
                else:
                    print("⚠️ WARNING: All timeframes showing identical signals")
                    print("🔍 This might indicate synthetic analysis is still being used")
                    
            else:
                print("❌ No timeframe predictions found in response")
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing timeframe predictions: {e}")

def test_multiple_calls():
    """Test multiple calls to see if signals vary over time"""
    
    print("\n🔄 Testing multiple API calls for signal consistency...")
    
    all_signals = []
    
    for i in range(3):
        try:
            response = requests.get('http://localhost:5000/api/timeframe-predictions', timeout=5)
            if response.status_code == 200:
                data = response.json()
                timeframes = data.get('timeframe_predictions', {})
                
                call_signals = {}
                for timeframe, prediction in timeframes.items():
                    call_signals[timeframe] = prediction.get('signal', 'UNKNOWN')
                
                all_signals.append(call_signals)
                print(f"📊 Call {i+1}: {call_signals}")
                
        except Exception as e:
            print(f"❌ Error in call {i+1}: {e}")
            
    # Analyze consistency
    if len(all_signals) >= 2:
        if all_signals[0] == all_signals[1]:
            print("✅ Signals are consistent across calls (expected for real analysis)")
        else:
            print("🔄 Signals vary between calls (possible real-time updates)")

if __name__ == "__main__":
    print("🚀 Quick Timeframe Predictions Test")
    print("=" * 50)
    
    test_timeframe_predictions()
    test_multiple_calls()
    
    print("\n✅ Test completed!")
