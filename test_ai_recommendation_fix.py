#!/usr/bin/env python3
"""
Test Script for AI Recommendation Fix
Validates that price targets are logically consistent with signals
"""

import requests
import json
from real_time_ai_engine import get_real_time_ai_recommendation

def test_local_ai_engine():
    """Test the local AI engine directly"""
    print("🔍 Testing Local AI Recommendation Engine...")
    
    # Run multiple tests to check consistency
    for i in range(3):
        print(f"\n--- Test Run {i+1} ---")
        rec = get_real_time_ai_recommendation()
        
        signal = rec.get('signal', 'UNKNOWN')
        current_price = rec.get('current_price', 0)
        target_1 = rec.get('targets', {}).get('target_1', 0)
        target_2 = rec.get('targets', {}).get('target_2', 0)
        stop_loss = rec.get('stop_loss', 0)
        
        print(f"Signal: {signal}")
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Target 1: ${target_1:,.2f}")
        print(f"Target 2: ${target_2:,.2f}")
        print(f"Stop Loss: ${stop_loss:,.2f}")
        
        # Validate logic
        if signal == 'BULLISH':
            if target_1 > current_price and target_2 > current_price and stop_loss < current_price:
                print("✅ BULLISH signal logic is correct")
            else:
                print("❌ BULLISH signal logic is INCORRECT!")
                print(f"   Target 1 > Current: {target_1 > current_price}")
                print(f"   Target 2 > Current: {target_2 > current_price}")
                print(f"   Stop Loss < Current: {stop_loss < current_price}")
        
        elif signal == 'BEARISH':
            if target_1 < current_price and target_2 < current_price and stop_loss > current_price:
                print("✅ BEARISH signal logic is correct")
            else:
                print("❌ BEARISH signal logic is INCORRECT!")
                print(f"   Target 1 < Current: {target_1 < current_price}")
                print(f"   Target 2 < Current: {target_2 < current_price}")
                print(f"   Stop Loss > Current: {stop_loss > current_price}")
        
        elif signal == 'NEUTRAL':
            print("✅ NEUTRAL signal detected")
        
        print(f"Confidence: {rec.get('confidence', 0):.1f}%")

def test_railway_deployment():
    """Test the Railway deployment"""
    print("\n🌐 Testing Railway Deployment AI Recommendation...")
    
    try:
        response = requests.get(
            "https://web-production-41882.up.railway.app/api/ai-recommendation",
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                rec = data.get('recommendation', {})
                
                signal = rec.get('signal', 'UNKNOWN')
                current_price = rec.get('current_price', 0)
                target_1 = rec.get('target_1', 0)
                target_2 = rec.get('target_2', 0)
                stop_loss = rec.get('stop_loss', 0)
                
                print(f"Signal: {signal}")
                print(f"Current Price: ${current_price:,.2f}")
                print(f"Target 1: ${target_1:,.2f}")
                print(f"Target 2: ${target_2:,.2f}")
                print(f"Stop Loss: ${stop_loss:,.2f}")
                
                # Validate logic
                if signal == 'BULLISH':
                    if target_1 > current_price and target_2 > current_price and stop_loss < current_price:
                        print("✅ Railway BULLISH signal logic is correct")
                    else:
                        print("❌ Railway BULLISH signal logic is INCORRECT!")
                        print(f"   Target 1 > Current: {target_1 > current_price}")
                        print(f"   Target 2 > Current: {target_2 > current_price}")
                        print(f"   Stop Loss < Current: {stop_loss < current_price}")
                
                elif signal == 'BEARISH':
                    if target_1 < current_price and target_2 < current_price and stop_loss > current_price:
                        print("✅ Railway BEARISH signal logic is correct")
                    else:
                        print("❌ Railway BEARISH signal logic is INCORRECT!")
                        print(f"   Target 1 < Current: {target_1 < current_price}")
                        print(f"   Target 2 < Current: {target_2 < current_price}")
                        print(f"   Stop Loss > Current: {stop_loss > current_price}")
                
                elif signal == 'NEUTRAL':
                    print("✅ Railway NEUTRAL signal detected")
                
                print(f"Confidence: {rec.get('confidence', 0):.1f}%")
                print(f"Update Time: {rec.get('update_time', 'Unknown')}")
                
            else:
                print("❌ Railway API returned success=false")
                
        else:
            print(f"❌ Railway API request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing Railway deployment: {e}")

if __name__ == "__main__":
    print("🔧 AI Recommendation Logic Validation Test")
    print("=" * 50)
    
    # Test local engine
    test_local_ai_engine()
    
    # Test Railway deployment
    test_railway_deployment()
    
    print("\n" + "=" * 50)
    print("✅ AI Recommendation Testing Complete!")
    print("\nIf all tests show correct logic, the issue has been fixed!")
    print("The AI module should now show:")
    print("- BULLISH signals with targets ABOVE current price")
    print("- BEARISH signals with targets BELOW current price")
    print("- Stop losses on opposite side of current price from targets")
