#!/usr/bin/env python3
"""
Test script for AI Analysis Center
Validates all components and data sources
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_analysis_api import AdvancedAIAnalyzer
import asyncio
import json

async def test_ai_analysis():
    """Test all AI analysis components"""
    print("🧪 Testing AI Analysis Center Components...\n")
    
    analyzer = AdvancedAIAnalyzer()
    
    # Test symbols
    test_symbols = ["XAUUSD", "EURUSD", "AAPL"]
    
    for symbol in test_symbols:
        print(f"📊 Testing Analysis for {symbol}")
        print("="*50)
        
        try:
            # Test comprehensive analysis
            result = await analyzer.get_comprehensive_analysis(symbol)
            
            print(f"✅ Overall Recommendation: {result['overall_recommendation']['signal']} "
                  f"(Confidence: {result['overall_recommendation']['confidence']:.1f}%)")
            
            # Technical Analysis
            if 'technical_analysis' in result:
                print(f"📈 Technical Indicators: {len(result['technical_analysis'])} loaded")
                
            # Sentiment Analysis  
            if 'sentiment_analysis' in result:
                sentiment = result['sentiment_analysis']
                print(f"💭 Market Sentiment: {sentiment['overall_sentiment']} "
                      f"(Score: {sentiment['sentiment_score']:.2f})")
                
            # ML Predictions
            if 'ml_predictions' in result:
                predictions = result['ml_predictions']
                print(f"🤖 ML Predictions: {len(predictions)} timeframes")
                for pred in predictions[:2]:  # Show first 2
                    print(f"   {pred['timeframe']}: {pred['predicted_price']:.2f} "
                          f"({pred['direction']} - {pred['confidence']:.1f}%)")
            
            # Market News
            if 'market_news' in result:
                news_count = len(result['market_news'])
                print(f"📰 Market News: {news_count} articles loaded")
                
            print()
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
            print()
    
    print("🎯 Testing Individual Components...")
    print("="*50)
    
    # Test technical analysis only
    try:
        tech_analysis = await analyzer.get_technical_analysis("XAUUSD", "1d")
        print(f"✅ Technical Analysis: {len(tech_analysis)} indicators")
    except Exception as e:
        print(f"❌ Technical Analysis Error: {str(e)}")
    
    # Test sentiment analysis only
    try:
        sentiment = await analyzer.get_sentiment_analysis("XAUUSD")
        print(f"✅ Sentiment Analysis: {sentiment['overall_sentiment']} "
              f"({sentiment['sentiment_score']:.2f})")
    except Exception as e:
        print(f"❌ Sentiment Analysis Error: {str(e)}")
    
    # Test ML predictions only
    try:
        ml_predictions = await analyzer.get_ml_predictions("XAUUSD")
        print(f"✅ ML Predictions: {len(ml_predictions)} timeframes")
    except Exception as e:
        print(f"❌ ML Predictions Error: {str(e)}")
    
    print("\n✅ AI Analysis Center Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_ai_analysis())
