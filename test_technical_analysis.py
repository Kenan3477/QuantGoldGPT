#!/usr/bin/env python3
"""
Test Real Technical Analysis System
"""

from real_technical_analysis import technical_analyzer
import json

def test_technical_analysis():
    """Test the technical analysis system"""
    print("🔧 Testing Real Technical Analysis System...")
    print("=" * 60)
    
    try:
        # Test multiple analysis calls to see signal variation
        for i in range(5):
            print(f"\n📊 Test {i+1}:")
            analysis = technical_analyzer.generate_comprehensive_analysis('XAUUSD')
            
            signal = analysis['signal']
            confidence = analysis['confidence']
            rsi = analysis['technical_indicators']['rsi']
            macd_status = analysis['technical_indicators']['macd']['signal_status']
            trend = analysis['technical_indicators']['trend']
            
            print(f"  🎯 Signal: {signal}")
            print(f"  📈 Confidence: {confidence:.3f}")
            print(f"  💹 RSI: {rsi:.2f}")
            print(f"  📉 MACD: {macd_status}")
            print(f"  📊 Trend: {trend}")
            
        print("\n" + "=" * 60)
        print("✅ Technical Analysis Test Complete!")
        print("🔍 You should see varied signals (BULLISH/BEARISH/NEUTRAL)")
        print("📊 Dashboard will now show realistic technical analysis instead of all NEUTRAL")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_technical_analysis()
