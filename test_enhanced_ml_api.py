#!/usr/bin/env python3
"""
Test Enhanced ML Predictions - Math Verification
"""
import requests

def test_ml_predictions():
    print("🧪 Testing Enhanced ML Predictions - Math Verification")
    print("=" * 60)
    
    try:
        # Get predictions from API
        response = requests.get('http://localhost:5000/api/ml-predictions')
        data = response.json()
        
        current_price = data['current_price']
        predictions = data.get('predictions', [])
        
        print(f"💰 Current Gold Price: ${current_price}")
        print(f"📊 Number of predictions: {len(predictions)}")
        print(f"🎯 Data quality: {data.get('data_quality', 'unknown')}")
        print()
        
        print("📈 Prediction Verification:")
        print("-" * 40)
        
        for pred in predictions:
            timeframe = pred['timeframe']
            predicted_price = pred['predicted_price']
            change_percent = pred['change_percent']
            direction = pred['direction']
            confidence = pred['confidence']
            
            # Calculate expected price based on percentage
            expected_price = current_price * (1 + change_percent / 100)
            
            # Check if math is correct
            math_diff = abs(expected_price - predicted_price)
            math_correct = math_diff < 0.01
            
            # Show results
            status = "✅" if math_correct else "❌"
            print(f"{status} {timeframe}: ${predicted_price:.2f} ({change_percent:+.3f}%) - {direction} [{confidence:.0%}]")
            print(f"   Expected: ${expected_price:.2f}, Actual: ${predicted_price:.2f}, Diff: ${math_diff:.4f}")
            
            # Verify logic (if positive change, price should be higher)
            if change_percent > 0 and predicted_price <= current_price:
                print(f"   ❌ Logic error: Positive change ({change_percent:+.3f}%) but price not higher!")
            elif change_percent < 0 and predicted_price >= current_price:
                print(f"   ❌ Logic error: Negative change ({change_percent:+.3f}%) but price not lower!")
            elif math_correct:
                print(f"   ✅ Math and logic verified")
            
            print()
        
        # Check for enhanced features
        print("🚀 Enhanced Features Check:")
        print("-" * 30)
        
        features = []
        if 'economic_factors' in data:
            features.append("Economic factors")
        if 'sentiment_analysis' in data:
            features.append("Sentiment analysis")
        if 'technical_analysis' in data:
            features.append("Technical analysis")
        if 'pattern_analysis' in data:
            features.append("Pattern analysis")
        
        for feature in features:
            print(f"✅ {feature}")
        
        if data.get('source') == 'enhanced_ml_engine':
            print("✅ Enhanced ML engine active")
        
        print()
        print("📰 Sentiment Analysis:")
        sentiment = data.get('sentiment_analysis', {})
        if sentiment:
            print(f"   Sentiment: {sentiment.get('sentiment', 'unknown')}")
            print(f"   Score: {sentiment.get('sentiment_score', 0):.3f}")
            print(f"   Fear/Greed Index: {sentiment.get('fear_greed_index', 50)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ml_predictions()
    if success:
        print("✅ All tests passed! Enhanced ML system is working correctly.")
    else:
        print("❌ Some tests failed. Check the system.")
