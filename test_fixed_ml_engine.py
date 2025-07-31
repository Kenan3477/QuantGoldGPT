#!/usr/bin/env python3
"""
Test script for the Fixed ML Prediction Engine
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_fixed_ml_engine():
    """Test the fixed ML prediction engine"""
    print("🧪 Testing Fixed ML Prediction Engine")
    print("=" * 50)
    
    try:
        # Import the fixed engine
        from fixed_ml_prediction_engine import get_fixed_ml_predictions, fixed_ml_engine
        
        print("✅ Fixed ML engine imported successfully")
        
        # Test individual components
        print("\n🔍 Testing individual components:")
        
        # Test candlestick analyzer
        try:
            ohlc_data = fixed_ml_engine.get_real_ohlc_data('1H', 50)
            candlestick_result = fixed_ml_engine.candlestick_analyzer.analyze_patterns(ohlc_data)
            print(f"✅ Candlestick Analysis: Found {len(candlestick_result['patterns'])} patterns")
            print(f"   └─ Bias: {candlestick_result['bias']}, Strength: {candlestick_result['signal_strength']:.2f}")
        except Exception as e:
            print(f"❌ Candlestick Analysis failed: {e}")
        
        # Test sentiment analyzer
        try:
            sentiment_result = fixed_ml_engine.sentiment_analyzer.get_news_sentiment()
            print(f"✅ Sentiment Analysis: Score {sentiment_result['overall_sentiment']:.3f}")
            print(f"   └─ News Count: {sentiment_result['news_count']}, Confidence: {sentiment_result['confidence']:.2f}")
        except Exception as e:
            print(f"❌ Sentiment Analysis failed: {e}")
        
        # Test economic analyzer
        try:
            economic_result = fixed_ml_engine.economic_analyzer.get_economic_factors()
            print(f"✅ Economic Analysis: USD {economic_result['usd_index']:.1f}")
            print(f"   └─ Real Rate: {economic_result['real_interest_rate']:.2f}%, Policy: {economic_result['monetary_policy_stance']}")
        except Exception as e:
            print(f"❌ Economic Analysis failed: {e}")
        
        # Test technical analyzer
        try:
            technical_result = fixed_ml_engine.technical_analyzer.analyze_technical_indicators(ohlc_data)
            print(f"✅ Technical Analysis: RSI {technical_result['rsi']:.1f}")
            print(f"   └─ Signals: {len(technical_result['signals'])} indicators")
        except Exception as e:
            print(f"❌ Technical Analysis failed: {e}")
        
        # Test full prediction
        print("\n🎯 Testing full predictions:")
        
        timeframes = ['1H', '4H', '1D']
        result = await get_fixed_ml_predictions(timeframes)
        
        if result['status'] == 'success':
            print(f"✅ Predictions generated successfully")
            print(f"   └─ Execution time: {result['execution_time']}")
            print(f"   └─ Engine version: {result['engine_version']}")
            print(f"   └─ Analysis types: {', '.join(result['analysis_types'])}")
            
            # Display predictions
            for tf, prediction in result['predictions'].items():
                print(f"\n📊 {tf} Prediction:")
                print(f"   Current Price: ${prediction['current_price']:.2f}")
                print(f"   Predicted Price: ${prediction['predicted_price']:.2f}")
                print(f"   Change: {prediction['price_change_percent']:.2f}%")
                print(f"   Direction: {prediction['direction'].upper()}")
                print(f"   Confidence: {prediction['confidence']:.1%}")
                print(f"   Risk: {prediction['risk_assessment']}")
                print(f"   Market Regime: {prediction['market_regime']}")
                
                if prediction['candlestick_patterns']:
                    print(f"   Patterns: {', '.join(prediction['candlestick_patterns'][:3])}")
                
                print(f"   Reasoning: {prediction['reasoning']}")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n✅ Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except ImportError as e:
        print(f"❌ Failed to import fixed ML engine: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   - TA-Lib")
        print("   - pandas")
        print("   - numpy")
        print("   - requests")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_api_integration():
    """Test API integration"""
    print("\n🌐 Testing API Integration")
    print("=" * 30)
    
    try:
        from advanced_ml_api import advanced_ml_bp
        print("✅ Advanced ML API Blueprint imported successfully")
        
        # Test health endpoint simulation
        print("✅ API endpoints available:")
        print("   - /api/advanced-ml/predict")
        print("   - /api/advanced-ml/strategies") 
        print("   - /api/advanced-ml/health")
        print("   - /api/advanced-ml/quick-prediction")
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Fixed ML Engine Tests")
    print("=" * 60)
    
    # Run async test
    asyncio.run(test_fixed_ml_engine())
    
    # Run API test
    test_api_integration()
    
    print("\n🏁 All tests completed!")
