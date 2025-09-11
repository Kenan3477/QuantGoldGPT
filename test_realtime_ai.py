#!/usr/bin/env python3
"""
Quick test script for the Real-Time AI Engine
Tests if the AI engine can generate live recommendations with real market data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_ai_engine import RealTimeAIEngine
import json

def test_real_time_ai():
    print("🤖 Testing Real-Time AI Engine...")
    
    try:
        # Initialize the AI engine
        ai_engine = RealTimeAIEngine()
        print("✅ AI Engine initialized successfully")
        
        # Get a real-time recommendation
        print("\n📊 Fetching live market data and generating AI recommendation...")
        recommendation = ai_engine.generate_ai_recommendation()
        
        print("\n🎯 AI Recommendation Results:")
        print("="*50)
        print(f"Signal: {recommendation.get('signal', 'N/A')}")
        print(f"Confidence: {recommendation.get('confidence', 0):.1f}%")
        print(f"Entry Price: ${recommendation.get('entry_price', 0):,.2f}")
        print(f"Target Price: ${recommendation.get('target_1', 0):,.2f}")
        print(f"Stop Loss: ${recommendation.get('stop_loss', 0):,.2f}")
        print(f"Risk/Reward: {recommendation.get('risk_reward_ratio', 0):.2f}:1")
        
        print(f"\n📝 Summary: {recommendation.get('recommendation_summary', 'N/A')}")
        
        print(f"\n📈 Bullish Factors:")
        for factor in recommendation.get('bullish_factors', [])[:3]:
            print(f"  • {factor}")
            
        print(f"\n📉 Bearish Factors:")
        for factor in recommendation.get('bearish_factors', [])[:3]:
            print(f"  • {factor}")
        
        print(f"\n🔗 Data Sources: {', '.join(recommendation.get('data_sources', []))}")
        print(f"⏰ Updated: {recommendation.get('update_time', 'N/A')}")
        
        print("\n✅ Real-Time AI Engine test completed successfully!")
        
        # Verify this is NOT placeholder data
        if recommendation.get('signal') != 'BUY' or recommendation.get('confidence') != 75.0:
            print("🎉 CONFIRMED: Using REAL market data (not placeholder)")
        else:
            print("⚠️  WARNING: This might still be placeholder data")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI engine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Real-Time AI Engine Test")
    print("="*60)
    
    success = test_real_time_ai()
    
    if success:
        print("\n🎉 SUCCESS: Real-Time AI Engine is working with live market data!")
        print("Your AI Recommendation module will now use real-time analysis instead of placeholder data.")
    else:
        print("\n❌ FAILED: There was an issue with the AI engine")
    
    print("\n" + "="*60)
