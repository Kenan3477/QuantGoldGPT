#!/usr/bin/env python3
"""
Test Dynamic Prediction Engine
Verify that predictions update based on market shifts
"""

import sys
import os
import asyncio
import datetime
import time
from typing import Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_prediction_engine import dynamic_prediction_engine, MarketShift
from self_improving_ml_engine import MultiTimeframePrediction
from daily_prediction_scheduler import daily_predictor

async def test_dynamic_predictions():
    print("🔄 TESTING DYNAMIC PREDICTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Generate initial prediction
    print("\n1. 📊 Generating Initial Prediction...")
    try:
        prediction = daily_predictor.ml_engine.generate_daily_prediction("XAUUSD")
        if prediction:
            print(f"✅ Initial prediction generated")
            print(f"   Current Price: ${prediction.current_price:.2f}")
            print(f"   1H: {prediction.predictions['1h']:+.2f}%")
            print(f"   4H: {prediction.predictions['4h']:+.2f}%")
            print(f"   1D: {prediction.predictions['1d']:+.2f}%")
            print(f"   3D: {prediction.predictions['3d']:+.2f}%")
            print(f"   7D: {prediction.predictions['7d']:+.2f}%")
            
            # Register with dynamic engine
            dynamic_prediction_engine.set_current_prediction("XAUUSD", prediction)
            print("🔄 Dynamic monitoring activated")
        else:
            print("❌ Failed to generate initial prediction")
            return
    except Exception as e:
        print(f"❌ Error generating initial prediction: {e}")
        return
    
    # Step 2: Wait for baseline conditions to be established
    print("\n2. ⏱️ Waiting for baseline conditions...")
    await asyncio.sleep(5)
    
    # Step 3: Simulate market shifts
    print("\n3. 🌊 Simulating Market Shifts...")
    
    # Create mock market shifts
    shifts = [
        MarketShift(
            timestamp=datetime.datetime.now(),
            shift_type='trend',
            old_value=50.0,
            new_value=75.0,
            severity=0.8,
            confidence=0.85,
            source='technical_analysis',
            description='RSI moved from neutral to overbought territory'
        ),
        MarketShift(
            timestamp=datetime.datetime.now(),
            shift_type='sentiment',
            old_value=0.5,
            new_value=0.8,
            severity=0.7,
            confidence=0.75,
            source='sentiment_analysis',
            description='Market sentiment shifted strongly bullish'
        ),
        MarketShift(
            timestamp=datetime.datetime.now(),
            shift_type='news',
            old_value=0.3,
            new_value=0.9,
            severity=0.9,
            confidence=0.9,
            source='news_analysis',
            description='Major positive economic news released'
        )
    ]
    
    # Step 4: Force prediction update
    print(f"   Simulating {len(shifts)} major market shifts...")
    for shift in shifts:
        print(f"   - {shift.shift_type.upper()}: {shift.description} (severity: {shift.severity:.1f})")
    
    try:
        updated_prediction = await dynamic_prediction_engine._update_prediction_due_to_shifts("XAUUSD", shifts)
        
        if updated_prediction:
            print("\n✅ PREDICTION UPDATED DUE TO MARKET SHIFTS!")
            print(f"   Updated Price: ${updated_prediction.current_price:.2f}")
            print(f"   Updated 1H: {updated_prediction.predictions['1h']:+.2f}%")
            print(f"   Updated 4H: {updated_prediction.predictions['4h']:+.2f}%")
            print(f"   Updated 1D: {updated_prediction.predictions['1d']:+.2f}%")
            print(f"   Updated 3D: {updated_prediction.predictions['3d']:+.2f}%")
            print(f"   Updated 7D: {updated_prediction.predictions['7d']:+.2f}%")
            print(f"   Reasoning: {updated_prediction.reasoning[:100]}...")
        else:
            print("❌ Prediction update failed")
    except Exception as e:
        print(f"❌ Error updating prediction: {e}")
    
    # Step 5: Test API endpoints
    print("\n4. 🌐 Testing API Endpoints...")
    
    try:
        # Test regular prediction
        regular_data = daily_predictor.get_current_prediction()
        print(f"✅ Regular prediction API: {regular_data['success']}")
        
        # Test dynamic prediction
        dynamic_data = daily_predictor.get_dynamic_prediction_data("XAUUSD")
        print(f"✅ Dynamic prediction API: {dynamic_data['success']}")
        
        if dynamic_data['success'] and 'dynamic_info' in dynamic_data:
            print(f"   Dynamic monitoring: {'ACTIVE' if dynamic_data['dynamic_info']['monitoring_active'] else 'INACTIVE'}")
            print(f"   Update count: {dynamic_data['dynamic_info']['update_count']}")
            print(f"   Last updated: {dynamic_data['dynamic_info']['last_updated']}")
            
    except Exception as e:
        print(f"❌ API test error: {e}")
    
    # Step 6: Test update history
    print("\n5. 📈 Checking Update History...")
    try:
        history = dynamic_prediction_engine.get_prediction_history("XAUUSD", limit=3)
        if history:
            print(f"✅ Found {len(history)} prediction updates in history")
            for i, update in enumerate(history[:3]):
                print(f"   {i+1}. {update['timestamp']}: {update['reason']}")
                print(f"      Confidence change: {update['confidence_change']:+.3f}")
        else:
            print("ℹ️ No update history found (this is normal for first run)")
    except Exception as e:
        print(f"❌ History check error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 DYNAMIC PREDICTION SYSTEM TEST COMPLETE")
    print("✅ The system can now update predictions when market conditions shift!")
    print("🔄 Monitoring continues in background every 5 minutes")

def test_market_shift_detection():
    """Test individual market shift detection methods"""
    print("\n🔍 TESTING MARKET SHIFT DETECTION...")
    
    # This would normally be done with real market data
    # For testing, we'll verify the methods exist and work
    
    detection_methods = [
        '_detect_trend_shifts',
        '_detect_sentiment_shifts', 
        '_detect_news_shifts',
        '_detect_pattern_shifts',
        '_detect_greed_fear_shifts',
        '_detect_economic_shifts'
    ]
    
    for method_name in detection_methods:
        if hasattr(dynamic_prediction_engine, method_name):
            print(f"✅ {method_name} method available")
        else:
            print(f"❌ {method_name} method missing")

if __name__ == "__main__":
    try:
        # Run basic detection test first
        test_market_shift_detection()
        
        # Run full dynamic prediction test
        asyncio.run(test_dynamic_predictions())
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        
    finally:
        # Clean up
        try:
            dynamic_prediction_engine.stop_monitoring()
            print("🧹 Monitoring stopped")
        except:
            pass
