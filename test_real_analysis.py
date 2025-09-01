#!/usr/bin/env python3

import requests
import json

def test_timeframe_predictions():
    """Test timeframe predictions API after populating database with real data"""
    
    print("🔍 Testing Timeframe Predictions API with Real Data")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:5000/api/timeframe-predictions', timeout=10)
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"🔍 Analysis Type: {data.get('analysis_type', 'Unknown')}")
            print(f"📊 Data Source: {data.get('data_source', 'Unknown')}")
            print(f"⏰ Generated At: {data.get('generated_at', 'Unknown')}")
            
            print("\n📈 TIMEFRAME PREDICTIONS:")
            print("-" * 40)
            
            predictions = data.get('predictions', {})
            signals = []
            
            for timeframe, pred in predictions.items():
                signal = pred.get('signal', 'UNKNOWN')
                confidence = pred.get('confidence', 0)
                signals.append(signal)
                
                print(f"⏰ {timeframe:12}: {signal:8} (Confidence: {confidence:.2f})")
                
                # Show technical indicators
                indicators = pred.get('technical_indicators', {})
                if indicators:
                    rsi = indicators.get('rsi', 'N/A')
                    macd = indicators.get('macd_signal', 'N/A')
                    bb = indicators.get('bollinger_position', 'N/A')
                    print(f"   📊 RSI: {rsi}, MACD: {macd}, BB: {bb}")
                
                print()
            
            # Check for signal diversity
            unique_signals = set(signals)
            print(f"🎯 Analysis Results:")
            print(f"   📊 Total timeframes: {len(signals)}")
            print(f"   🔄 Unique signals: {len(unique_signals)}")
            print(f"   📋 Signal types: {list(unique_signals)}")
            
            if len(unique_signals) > 1:
                print("   ✅ SUCCESS: Signals are VARIED (as expected)")
            else:
                print("   ❌ ISSUE: All signals are the same")
                
            # Check if using real analysis
            if 'synthetic' in data.get('analysis_type', '').lower():
                print("   ⚠️  WARNING: Still using synthetic analysis")
            else:
                print("   ✅ SUCCESS: Using REAL technical analysis")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_timeframe_predictions()
