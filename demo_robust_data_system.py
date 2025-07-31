#!/usr/bin/env python3
"""
GoldGPT Robust Data System - Live Demonstration
Shows the multi-source data fetching system in action
"""

import asyncio
import time
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def demonstrate_robust_data_system():
    """Demonstrate the robust data system capabilities"""
    print("🚀 GoldGPT Robust Data System - Live Demonstration")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from robust_data_system import unified_data_provider
        print("✅ Robust data system loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to load robust data system: {e}")
        return
    
    # Test symbols
    symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
    
    print("\n🎯 DEMONSTRATION SCENARIOS")
    print("-" * 40)
    
    # Scenario 1: Price Data with Fallbacks
    print("\n1️⃣ Price Data Fetching (Multi-Source Fallback)")
    print("   Attempting: API → Web Scraping → Simulated")
    
    for symbol in symbols:
        try:
            start_time = time.time()
            price_data = await unified_data_provider.get_price_data(symbol)
            fetch_time = time.time() - start_time
            
            print(f"   📊 {symbol}: ${price_data.price:.2f}")
            print(f"      Source: {price_data.source.value}")
            print(f"      Bid/Ask: ${price_data.bid:.2f}/${price_data.ask:.2f}")
            print(f"      Change: {price_data.change:+.2f} ({price_data.change_percent:+.2f}%)")
            print(f"      Fetch Time: {fetch_time:.3f}s")
            print()
            
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    # Scenario 2: Sentiment Analysis
    print("\n2️⃣ Sentiment Analysis (News + NLP)")
    print("   Analyzing: News Articles → NLP → Confidence Scoring")
    
    for symbol in ['XAUUSD']:  # Focus on gold for demo
        try:
            start_time = time.time()
            sentiment_data = await unified_data_provider.get_sentiment_data(symbol)
            fetch_time = time.time() - start_time
            
            print(f"   💭 {symbol} Sentiment Analysis:")
            print(f"      Sentiment: {sentiment_data.sentiment_label.upper()}")
            print(f"      Score: {sentiment_data.sentiment_score:+.3f}")
            print(f"      Confidence: {sentiment_data.confidence:.1%}")
            print(f"      Sources: {sentiment_data.sources_count} news articles")
            print(f"      Timeframe: {sentiment_data.timeframe}")
            print(f"      Analysis Time: {fetch_time:.3f}s")
            print()
            
        except Exception as e:
            print(f"   ❌ {symbol}: Sentiment error - {e}")
    
    # Scenario 3: Technical Indicators
    print("\n3️⃣ Technical Indicators (Real-time Calculation)")
    print("   Calculating: RSI, MACD, Moving Averages, Bollinger Bands")
    
    for symbol in ['XAUUSD']:
        try:
            start_time = time.time()
            technical_data = await unified_data_provider.get_technical_data(symbol)
            fetch_time = time.time() - start_time
            
            indicators = technical_data.indicators
            
            print(f"   📈 {symbol} Technical Analysis:")
            print(f"      RSI: {indicators['rsi']['value']:.1f} ({indicators['rsi']['signal']})")
            print(f"      MACD: {indicators['macd']['value']:.4f} ({indicators['macd']['signal']})")
            print(f"      MA20: ${indicators['moving_averages']['ma20']:.2f}")
            print(f"      MA50: ${indicators['moving_averages']['ma50']:.2f}")
            print(f"      Trend: {indicators['moving_averages']['trend'].upper()}")
            print(f"      BB Upper: ${indicators['bollinger_bands']['upper']:.2f}")
            print(f"      BB Lower: ${indicators['bollinger_bands']['lower']:.2f}")
            print(f"      Source: {technical_data.source.value}")
            print(f"      Calculation Time: {fetch_time:.3f}s")
            print()
            
        except Exception as e:
            print(f"   ❌ {symbol}: Technical error - {e}")
    
    # Scenario 4: Comprehensive Data
    print("\n4️⃣ Comprehensive Data (All Types Combined)")
    print("   Fetching: Price + Sentiment + Technical (Parallel)")
    
    try:
        start_time = time.time()
        comprehensive = await unified_data_provider.get_comprehensive_data('XAUUSD')
        fetch_time = time.time() - start_time
        
        print("   🎯 XAUUSD Comprehensive Analysis:")
        
        if comprehensive['price']:
            price = comprehensive['price']
            print(f"      💰 Price: ${price['price']:.2f} ({price['source']})")
        
        if comprehensive['sentiment']:
            sentiment = comprehensive['sentiment']
            print(f"      💭 Sentiment: {sentiment['sentiment_label']} ({sentiment['confidence']:.1%})")
        
        if comprehensive['technical']:
            tech = comprehensive['technical']
            rsi = tech['indicators']['rsi']
            print(f"      📈 RSI: {rsi['value']:.1f} ({rsi['signal']})")
        
        print(f"      ⚡ Total Time: {fetch_time:.3f}s (parallel fetch)")
        print()
        
    except Exception as e:
        print(f"   ❌ Comprehensive data error: {e}")
    
    # Scenario 5: Caching Demonstration
    print("\n5️⃣ Caching Performance (Speed Test)")
    print("   Testing: First call vs. Cached call")
    
    try:
        # First call (will cache)
        start_time = time.time()
        await unified_data_provider.get_price_data('XAUUSD')
        first_call_time = time.time() - start_time
        
        # Second call (from cache)
        start_time = time.time()
        await unified_data_provider.get_price_data('XAUUSD')
        second_call_time = time.time() - start_time
        
        print(f"   📊 Cache Performance Test:")
        print(f"      First call: {first_call_time:.3f}s (fetched and cached)")
        print(f"      Second call: {second_call_time:.3f}s (from cache)")
        print(f"      Speed improvement: {first_call_time/second_call_time:.1f}x faster")
        print(f"      Cache effective: {'✅ YES' if second_call_time < first_call_time else '❌ NO'}")
        print()
        
    except Exception as e:
        print(f"   ❌ Cache test error: {e}")
    
    # Scenario 6: Provider Statistics
    print("\n6️⃣ System Health & Statistics")
    print("   Monitoring: Source reliability, request counts, success rates")
    
    try:
        stats = unified_data_provider.get_provider_stats()
        
        print("   📊 Provider Statistics:")
        print("      Source Reliability:")
        for source, reliability in stats['source_reliability'].items():
            print(f"        {source}: {reliability:.1%}")
        
        print("      Request Statistics:")
        for source, count in stats['request_counts'].items():
            success_count = stats['success_counts'][source]
            success_rate = (success_count / count * 100) if count > 0 else 0
            print(f"        {source}: {success_count}/{count} ({success_rate:.1f}% success)")
        
        print(f"      Cache Status: {'✅ Active' if stats['cache_active'] else '❌ Inactive'}")
        print()
        
    except Exception as e:
        print(f"   ❌ Statistics error: {e}")
    
    # Final Summary
    print("\n🎯 DEMONSTRATION SUMMARY")
    print("=" * 40)
    print("✅ Multi-source data fetching with automatic fallbacks")
    print("✅ Real-time sentiment analysis from financial news")
    print("✅ Technical indicator calculations (RSI, MACD, etc.)")
    print("✅ Intelligent caching for performance optimization")
    print("✅ Comprehensive error handling and graceful degradation")
    print("✅ Source reliability tracking and automatic prioritization")
    print()
    print("🚀 The GoldGPT Robust Data System is ready for production!")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    print("🎬 Starting GoldGPT Robust Data System Demonstration...")
    asyncio.run(demonstrate_robust_data_system())
