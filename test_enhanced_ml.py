#!/usr/bin/env python3
"""
Test Enhanced ML Prediction System with Real-Time Price Integration
This script tests the fixed ML system to ensure it uses real-time prices
"""

import asyncio
import json
from datetime import datetime
from ml_prediction_api import get_enhanced_ml_analysis, get_real_time_gold_price, ml_engine

async def test_enhanced_ml_system():
    """Test the enhanced ML prediction system"""
    print("🧪 Testing Enhanced ML Prediction System")
    print("=" * 60)
    
    # Test 1: Get real-time gold price
    print("📊 Test 1: Real-time Gold Price")
    current_price = get_real_time_gold_price()
    print(f"✅ Current Gold Price: ${current_price:.2f}")
    print()
    
    # Test 2: Get enhanced market features
    print("🧠 Test 2: Enhanced Market Features")
    try:
        market_features = await ml_engine.get_enhanced_market_features()
        print("✅ Enhanced Market Features:")
        for key, value in market_features.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Market features error: {e}")
    print()
    
    # Test 3: Get enhanced ML analysis
    print("🔮 Test 3: Enhanced ML Analysis")
    try:
        analysis = await get_enhanced_ml_analysis("GC=F")
        
        if analysis.get('success'):
            predictions = analysis.get('predictions', {})
            print(f"✅ Analysis successful! Current price in analysis: ${analysis.get('current_price', 'N/A')}")
            
            print("\n📈 Predictions by timeframe:")
            for timeframe, pred_data in predictions.items():
                if isinstance(pred_data, dict):
                    predicted_price = pred_data.get('predicted_price', 0)
                    direction = pred_data.get('direction', 'unknown')
                    confidence = pred_data.get('confidence', 0)
                    price_change_pct = pred_data.get('price_change_percent', 0)
                    
                    print(f"   {timeframe}:")
                    print(f"      Predicted Price: ${predicted_price:.2f}")
                    print(f"      Direction: {direction}")
                    print(f"      Confidence: {confidence:.2%}")
                    print(f"      Change: {price_change_pct:.2f}%")
                    
                    # Validate prediction is reasonable (within ±10% of current price)
                    if abs(predicted_price - current_price) / current_price > 0.10:
                        print(f"      ⚠️  WARNING: Prediction seems unrealistic!")
                    else:
                        print(f"      ✅ Prediction looks reasonable")
                    print()
            
            # Check market psychology integration
            market_psychology = analysis.get('market_psychology', {})
            if market_psychology:
                print("🧠 Market Psychology Integration:")
                sentiment_score = market_psychology.get('sentiment_score', 'N/A')
                fear_greed = market_psychology.get('fear_greed_index', 'N/A')
                news_impact = market_psychology.get('news_impact', 'N/A')
                print(f"   Sentiment Score: {sentiment_score}")
                print(f"   Fear/Greed Index: {fear_greed}")
                print(f"   News Impact: {news_impact}")
        else:
            print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Enhanced ML analysis error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 Enhanced ML Test Complete")

def validate_price_accuracy():
    """Validate that predictions are using current market prices"""
    print("\n🔍 Price Accuracy Validation")
    print("-" * 40)
    
    current_price = get_real_time_gold_price()
    print(f"Real-time API price: ${current_price:.2f}")
    
    # Expected price range for gold (as of 2025)
    expected_min = 3000
    expected_max = 4000
    
    if expected_min <= current_price <= expected_max:
        print("✅ Price is in expected range for 2025")
        return True
    else:
        print(f"⚠️  Price ${current_price:.2f} seems outside expected range (${expected_min}-${expected_max})")
        return False

if __name__ == "__main__":
    print("🚀 Starting Enhanced ML Prediction Test")
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate price accuracy first
    price_valid = validate_price_accuracy()
    
    if price_valid:
        # Run async test
        asyncio.run(test_enhanced_ml_system())
    else:
        print("❌ Price validation failed - check real-time price feed")
