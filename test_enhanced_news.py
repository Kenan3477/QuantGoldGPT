"""
Test Enhanced News System
Quick verification that the enhanced news system works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_news_fetcher import real_time_news_fetcher
from enhanced_news_analyzer import enhanced_news_analyzer
from datetime import datetime
import json

def test_enhanced_news():
    print("🧪 Testing Enhanced News System...")
    print("=" * 50)
    
    # Test 1: Real-time news fetcher
    print("\n1. Testing Real-time News Fetcher...")
    try:
        news_articles = real_time_news_fetcher.get_enhanced_news(limit=3)
        print(f"✅ Fetched {len(news_articles)} enhanced news articles")
        
        if news_articles:
            print("\nSample Article:")
            article = news_articles[0]
            print(f"Title: {article['title'][:80]}...")
            print(f"Source: {article['source']}")
            print(f"Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.2f})")
            print(f"Price Impact 1H: {article.get('price_change_1h', 0):.2f}%")
            print(f"Confidence: {article['confidence_score']:.1%}")
        
    except Exception as e:
        print(f"❌ Error testing news fetcher: {e}")
    
    # Test 2: Enhanced analyzer
    print("\n2. Testing Enhanced News Analyzer...")
    try:
        # Test processing a sample article
        test_title = "Fed Signals Rate Cut as Gold Surges on Inflation Concerns"
        test_content = "Federal Reserve officials indicated potential monetary easing as gold prices climb amid rising inflation expectations and geopolitical tensions."
        
        processed = enhanced_news_analyzer.process_news_article(
            title=test_title,
            content=test_content,
            source="Test Source",
            published_at=datetime.now(),
            url="https://test.com"
        )
        
        if processed:
            print(f"✅ Successfully processed test article")
            print(f"Sentiment: {processed.sentiment_label} ({processed.sentiment_score:.2f})")
            print(f"Gold keywords found: {len([w for w in test_title.lower().split() if 'gold' in w or 'fed' in w])}")
            print(f"Confidence: {processed.confidence_score:.1%}")
        else:
            print("❌ Failed to process test article")
            
    except Exception as e:
        print(f"❌ Error testing analyzer: {e}")
    
    # Test 3: Database operations
    print("\n3. Testing Database Operations...")
    try:
        recent_articles = enhanced_news_analyzer.get_recent_articles(limit=2)
        print(f"✅ Retrieved {len(recent_articles)} articles from database")
        
        if recent_articles:
            for i, article in enumerate(recent_articles[:2], 1):
                print(f"\nArticle {i}:")
                print(f"  Title: {article.title[:60]}...")
                print(f"  Sentiment: {article.sentiment_label}")
                print(f"  Time ago: {enhanced_news_analyzer.calculate_time_ago(article.published_at)}")
        
    except Exception as e:
        print(f"❌ Error testing database: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Enhanced News System Test Complete!")

if __name__ == "__main__":
    test_enhanced_news()
