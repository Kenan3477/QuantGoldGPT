#!/usr/bin/env python3
"""
Quick API test to show diverse ML predictions
"""
import requests
import json

def test_timeframe_predictions():
    """Test the fixed timeframe predictions API"""
    print("🧪 TESTING FIXED TIMEFRAME PREDICTIONS API")
    print("=" * 60)
    
    try:
        # Test the API endpoint
        response = requests.get("http://localhost:5000/api/timeframe-predictions")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response Success!")
            print("📊 Timeframe Predictions:")
            print("-" * 40)
            
            # Show each prediction with details
            for prediction in data.get('predictions', []):
                timeframe = prediction.get('timeframe')
                signal = prediction.get('signal')
                percentage = prediction.get('percentage')
                target_price = prediction.get('target_price')
                confidence = prediction.get('confidence', 'N/A')
                
                print(f"🕒 {timeframe:>6}: {signal:>7} | {percentage:>+6}% | Target: ${target_price:>7} | Confidence: {confidence}")
            
            # Check for diversity
            print("\n" + "-" * 40)
            signals = [p.get('signal') for p in data.get('predictions', [])]
            percentages = [p.get('percentage') for p in data.get('predictions', [])]
            targets = [p.get('target_price') for p in data.get('predictions', [])]
            
            unique_signals = len(set(signals))
            unique_percentages = len(set(percentages))
            unique_targets = len(set(targets))
            
            print(f"📈 Signal Diversity: {unique_signals} unique signals from {signals}")
            print(f"📊 Percentage Diversity: {unique_percentages} unique percentages from {percentages}")
            print(f"🎯 Target Diversity: {unique_targets} unique targets from {targets}")
            
            if unique_signals > 1 and unique_percentages > 1 and unique_targets > 1:
                print("\n✅ SUCCESS: Predictions show REAL DIVERSITY!")
                print("🎉 The 'bullshit identical targets' issue is FIXED!")
            else:
                print("\n❌ PROBLEM: Still showing identical predictions")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_timeframe_predictions()
