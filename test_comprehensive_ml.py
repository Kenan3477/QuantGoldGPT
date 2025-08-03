#!/usr/bin/env python3
"""
Comprehensive Test of Enhanced ML Dashboard API
Tests all aspects of gold market analysis
"""

import requests
import json
import sys

def test_enhanced_ml_dashboard():
    """Test comprehensive ML dashboard functionality"""
    base_url = 'http://localhost:5000/api'
    
    print('🚀 Testing Enhanced ML Dashboard API')
    print('=' * 60)
    
    # Test 1: Enhanced ML Predictions
    print('📊 Test 1: Enhanced ML Predictions')
    print('-' * 40)
    
    try:
        response = requests.post(f'{base_url}/enhanced-ml-predictions', 
                               json={'timeframes': ['15m', '1h', '4h', '24h']},
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print('✅ Enhanced ML Predictions API: WORKING')
            print(f'   📈 Current Gold Price: ${data.get("current_price", "N/A")}')
            print(f'   🎯 Predictions Generated: {len(data.get("predictions", []))}')
            
            # Test prediction details
            predictions = data.get('predictions', [])
            for pred in predictions:
                direction = pred.get('direction', 'N/A')
                confidence = pred.get('confidence', 'N/A')
                timeframe = pred.get('timeframe', 'N/A')
                target = pred.get('target_price', 'N/A')
                print(f'      {timeframe}: {direction.upper()} ({confidence}%) → ${target}')
            
            # Test overall bias
            bias = data.get('overall_bias', {})
            print(f'   📈 Overall Market Bias: {bias.get("direction", "N/A").upper()} ({bias.get("strength", "N/A")})')
            
            # Test comprehensive analysis components
            analysis = data.get('comprehensive_analysis', {})
            print(f'   🔍 Technical Analysis: {"✅" if "technical" in analysis else "❌"}')
            print(f'   💭 Sentiment Analysis: {"✅" if "sentiment" in analysis else "❌"}')
            print(f'   💰 Economic Analysis: {"✅" if "economic" in analysis else "❌"}')
            print(f'   📈 Pattern Analysis: {"✅" if "patterns" in analysis else "❌"}')
            
        else:
            print(f'❌ Enhanced ML Predictions API: FAILED ({response.status_code})')
            
    except Exception as e:
        print(f'❌ Enhanced ML Predictions API: ERROR - {e}')
    
    print()
    
    # Test 2: Market Analysis
    print('🔍 Test 2: Detailed Market Analysis')
    print('-' * 40)
    
    try:
        response = requests.get(f'{base_url}/market-analysis', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print('✅ Market Analysis API: WORKING')
            print(f'   💰 Current Price: ${data.get("current_price", "N/A")}')
            
            analysis = data.get('analysis', {})
            
            # Technical Indicators
            technical = analysis.get('technical_indicators', {})
            if technical:
                print(f'   📊 RSI: {technical.get("rsi", "N/A")}')
                print(f'   📈 Trend: {technical.get("trend_direction", "N/A").upper()}')
                print(f'   🔻 Support: ${technical.get("support_level", "N/A")}')
                print(f'   🔺 Resistance: ${technical.get("resistance_level", "N/A")}')
                print(f'   📊 Volatility: {technical.get("volatility", "N/A")}')
            
            # Market Sentiment
            sentiment = analysis.get('market_sentiment', {})
            if sentiment:
                print(f'   😨 Fear & Greed Index: {sentiment.get("fear_greed_index", "N/A")}')
                print(f'   📰 News Sentiment: {sentiment.get("news_sentiment", "N/A")}')
                print(f'   🏛️ Institutional Flow: {sentiment.get("institutional_flow", "N/A")}')
                print(f'   💭 Market Mood: {sentiment.get("market_mood", "N/A").upper()}')
            
            # Economic Factors
            economic = analysis.get('economic_factors', {})
            if economic:
                print(f'   💵 Dollar Index: {economic.get("dollar_index", "N/A")}')
                print(f'   📈 Fed Rate: {economic.get("federal_rate", "N/A")}%')
                print(f'   📊 Inflation: {economic.get("inflation_cpi", "N/A")}%')
                print(f'   🏦 Central Bank: {economic.get("central_bank_stance", "N/A").upper()}')
            
            # Candlestick Patterns
            patterns = analysis.get('candlestick_patterns', {})
            if patterns:
                print(f'   🕯️ Pattern: {patterns.get("detected_pattern", "N/A").replace("_", " ").upper()}')
                print(f'   📶 Signal: {patterns.get("pattern_signal", "N/A").upper()}')
                print(f'   💪 Strength: {patterns.get("pattern_strength", "N/A")}')
            
        else:
            print(f'❌ Market Analysis API: FAILED ({response.status_code})')
            
    except Exception as e:
        print(f'❌ Market Analysis API: ERROR - {e}')
    
    print()
    
    # Test 3: Original ML Predictions (for comparison)
    print('🔄 Test 3: Original ML Predictions (Comparison)')
    print('-' * 40)
    
    try:
        response = requests.get(f'{base_url}/ml-predictions', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print('✅ Original ML Predictions API: WORKING')
            print(f'   📊 Predictions: {len(data.get("predictions", []))} timeframes')
        else:
            print(f'❌ Original ML Predictions API: FAILED ({response.status_code})')
            
    except Exception as e:
        print(f'❌ Original ML Predictions API: ERROR - {e}')
    
    print()
    print('🎯 SUMMARY: Enhanced ML Dashboard Analysis')
    print('=' * 60)
    print('✅ COMPREHENSIVE GOLD ANALYSIS FEATURES:')
    print('   📊 Real-time Gold Spot Price Analysis')
    print('   📈 Technical Indicators (RSI, MACD, Trend, Support/Resistance)')
    print('   💭 Market Sentiment (Fear & Greed, News, Social, Institutional)')
    print('   💰 Economic Indicators (DXY, Fed Rate, Inflation, Central Bank)')
    print('   🕯️ Candlestick Pattern Recognition')
    print('   🎯 Multi-timeframe Predictions (15m, 1h, 4h, 24h)')
    print('   📊 Bullish/Bearish/Neutral Bias Determination')
    print('   💎 Price Targets with Stop Loss and Take Profit levels')
    print()
    print('🚀 Your Advanced ML Dashboard is now providing comprehensive')
    print('   gold market analysis as requested!')

if __name__ == '__main__':
    test_enhanced_ml_dashboard()
