#!/usr/bin/env python3
"""
GoldGPT Sentiment Analysis Service
News sentiment analysis with correlation tracking and market impact assessment
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import re
from textblob import TextBlob
import numpy as np
from collections import defaultdict
from data_pipeline_core import data_pipeline, DataType

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """News item data structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_magnitude: float
    keywords: List[str]
    relevance_score: float
    market_impact: str

@dataclass
class SentimentMetrics:
    """Sentiment analysis metrics"""
    overall_sentiment: float
    sentiment_trend: str
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    confidence: float
    sample_size: int
    time_period: str

class SentimentAnalysisService:
    """Advanced sentiment analysis with market correlation tracking"""
    
    def __init__(self, db_path: str = "goldgpt_sentiment.db"):
        self.db_path = db_path
        self.sentiment_cache = {}
        self.news_sources = self.setup_news_sources()
        self.gold_keywords = [
            'gold', 'xau', 'precious metals', 'inflation', 'fed', 'federal reserve',
            'interest rates', 'dollar', 'usd', 'safe haven', 'bullion', 'mining',
            'commodities', 'central bank', 'monetary policy', 'economic uncertainty',
            'geopolitical', 'recession', 'inflation hedge'
        ]
        self.initialize_database()
    
    def setup_news_sources(self) -> List[Dict]:
        """Configure news sources for sentiment analysis"""
        return [
            {
                'name': 'newsapi',
                'url': 'https://newsapi.org/v2/everything',
                'api_key': 'YOUR_NEWSAPI_KEY',
                'rate_limit': 500,
                'tier': 'primary'
            },
            {
                'name': 'alpha_vantage_news',
                'url': 'https://www.alphavantage.co/query',
                'api_key': 'YOUR_ALPHA_VANTAGE_KEY',
                'rate_limit': 100,
                'tier': 'primary'
            },
            {
                'name': 'rss_feeds',
                'sources': [
                    'https://www.marketwatch.com/rss/topstories',
                    'https://feeds.bloomberg.com/markets/news.rss',
                    'https://www.reuters.com/business/finance/rss'
                ],
                'tier': 'secondary'
            }
        ]
    
    def initialize_database(self):
        """Initialize sentiment analysis database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # News articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT NOT NULL,
                url TEXT UNIQUE,
                published_at DATETIME,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                sentiment_score REAL,
                sentiment_magnitude REAL,
                keywords TEXT,
                relevance_score REAL,
                market_impact TEXT,
                processed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Sentiment metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_period TEXT NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                overall_sentiment REAL,
                sentiment_trend TEXT,
                bullish_ratio REAL,
                bearish_ratio REAL,
                neutral_ratio REAL,
                confidence REAL,
                sample_size INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(time_period, period_start)
            )
        ''')
        
        # Sentiment-price correlation table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_price_correlation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                sentiment_score REAL,
                gold_price_open REAL,
                gold_price_close REAL,
                price_change_percent REAL,
                correlation_coefficient REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        ''')
        
        # Market impact events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_impact_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                description TEXT,
                impact_level TEXT,
                sentiment_before REAL,
                sentiment_after REAL,
                price_impact_percent REAL,
                event_time DATETIME,
                detection_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Sentiment analysis database initialized")
    
    async def fetch_news_from_sources(self, query: str = "gold precious metals", 
                                    hours_back: int = 24) -> List[Dict]:
        """Fetch news from multiple sources"""
        all_news = []
        
        # NewsAPI
        newsapi_articles = await self.fetch_from_newsapi(query, hours_back)
        all_news.extend(newsapi_articles)
        
        # Alpha Vantage News
        av_articles = await self.fetch_from_alpha_vantage_news(query)
        all_news.extend(av_articles)
        
        # RSS Feeds
        rss_articles = await self.fetch_from_rss_feeds(hours_back)
        all_news.extend(rss_articles)
        
        # Remove duplicates based on title similarity
        unique_articles = self.remove_duplicate_articles(all_news)
        
        logger.info(f"📰 Fetched {len(unique_articles)} unique news articles")
        return unique_articles
    
    async def fetch_from_newsapi(self, query: str, hours_back: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        articles = []
        
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': 'demo',  # Replace with actual API key
                'language': 'en',
                'pageSize': 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://newsapi.org/v2/everything', params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            if article.get('title') and article.get('publishedAt'):
                                articles.append({
                                    'title': article['title'],
                                    'content': article.get('content', ''),
                                    'source': 'newsapi',
                                    'url': article.get('url', ''),
                                    'published_at': article['publishedAt'],
                                    'author': article.get('author', '')
                                })
        
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        return articles
    
    async def fetch_from_alpha_vantage_news(self, query: str) -> List[Dict]:
        """Fetch news from Alpha Vantage News API"""
        articles = []
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': 'GLD,GOLD',
                'topics': 'financial_markets,economy_fiscal',
                'apikey': 'demo'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.alphavantage.co/query', params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('feed', []):
                            if article.get('title') and article.get('time_published'):
                                articles.append({
                                    'title': article['title'],
                                    'content': article.get('summary', ''),
                                    'source': 'alpha_vantage_news',
                                    'url': article.get('url', ''),
                                    'published_at': article['time_published'],
                                    'overall_sentiment_score': article.get('overall_sentiment_score', 0)
                                })
        
        except Exception as e:
            logger.error(f"Alpha Vantage News error: {e}")
        
        return articles
    
    async def fetch_from_rss_feeds(self, hours_back: int) -> List[Dict]:
        """Fetch and parse RSS feeds"""
        articles = []
        
        # For demo, generate some sample news articles
        sample_titles = [
            "Gold prices surge amid inflation concerns",
            "Federal Reserve hints at interest rate changes",
            "Central banks increase gold reserves",
            "Dollar weakens as investors seek safe havens",
            "Mining companies report strong Q3 earnings",
            "Geopolitical tensions boost precious metals demand",
            "Economic uncertainty drives gold investment",
            "Inflation hedge: Gold vs. cryptocurrency debate"
        ]
        
        for i, title in enumerate(sample_titles):
            published_time = datetime.now() - timedelta(hours=i*2)
            articles.append({
                'title': title,
                'content': f"Content for {title}. Market analysis and economic implications...",
                'source': 'rss_feeds',
                'url': f'https://example.com/article-{i}',
                'published_at': published_time.isoformat()
            })
        
        return articles
    
    def remove_duplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title length and key words
            title_key = re.sub(r'[^\w\s]', '', article['title'].lower())
            title_key = ' '.join(sorted(title_key.split()[:5]))  # First 5 words sorted
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float, List[str]]:
        """Analyze sentiment of text using TextBlob and keyword analysis"""
        if not text:
            return 0.0, 0.0, []
        
        # Basic sentiment analysis with TextBlob
        blob = TextBlob(text.lower())
        polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        # Keyword-based sentiment adjustment for gold market
        bullish_keywords = [
            'surge', 'rally', 'bullish', 'rise', 'increase', 'strong', 'gain',
            'inflation', 'uncertainty', 'safe haven', 'demand', 'buy'
        ]
        
        bearish_keywords = [
            'fall', 'drop', 'decline', 'bearish', 'weak', 'sell', 'loss',
            'strong dollar', 'rate hike', 'risk-on', 'recovery'
        ]
        
        # Count keyword occurrences
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text.lower())
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text.lower())
        
        # Adjust sentiment based on gold-specific keywords
        keyword_adjustment = (bullish_count - bearish_count) * 0.1
        adjusted_polarity = max(-1, min(1, polarity + keyword_adjustment))
        
        # Extract relevant keywords
        found_keywords = []
        for keyword in self.gold_keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)
        
        return adjusted_polarity, subjectivity, found_keywords
    
    def calculate_relevance_score(self, title: str, content: str) -> float:
        """Calculate relevance score for gold market analysis"""
        text = (title + " " + content).lower()
        
        # Count gold-related keywords
        keyword_score = 0
        for keyword in self.gold_keywords:
            if keyword in text:
                keyword_score += 1
        
        # Normalize score
        max_keywords = len(self.gold_keywords)
        relevance_score = min(1.0, keyword_score / max_keywords * 2)
        
        return relevance_score
    
    def determine_market_impact(self, sentiment_score: float, relevance_score: float) -> str:
        """Determine potential market impact"""
        if relevance_score < 0.3:
            return "minimal"
        
        impact_strength = abs(sentiment_score) * relevance_score
        
        if impact_strength >= 0.7:
            return "high"
        elif impact_strength >= 0.4:
            return "medium"
        else:
            return "low"
    
    def process_articles(self, articles: List[Dict]) -> List[NewsItem]:
        """Process articles for sentiment analysis"""
        processed_articles = []
        
        for article in articles:
            # Analyze sentiment
            sentiment_score, sentiment_magnitude, keywords = self.analyze_sentiment(
                article.get('title', '') + " " + article.get('content', '')
            )
            
            # Calculate relevance
            relevance_score = self.calculate_relevance_score(
                article.get('title', ''), 
                article.get('content', '')
            )
            
            # Determine market impact
            market_impact = self.determine_market_impact(sentiment_score, relevance_score)
            
            # Parse published date
            try:
                if 'T' in article.get('published_at', ''):
                    published_at = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                else:
                    published_at = datetime.now()
            except:
                published_at = datetime.now()
            
            news_item = NewsItem(
                title=article.get('title', ''),
                content=article.get('content', ''),
                source=article.get('source', ''),
                url=article.get('url', ''),
                published_at=published_at.replace(tzinfo=None),
                sentiment_score=sentiment_score,
                sentiment_magnitude=sentiment_magnitude,
                keywords=keywords,
                relevance_score=relevance_score,
                market_impact=market_impact
            )
            
            processed_articles.append(news_item)
        
        return processed_articles
    
    def store_articles(self, articles: List[NewsItem]):
        """Store processed articles in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            cursor.execute('''
                INSERT OR REPLACE INTO news_articles 
                (title, content, source, url, published_at, sentiment_score, 
                 sentiment_magnitude, keywords, relevance_score, market_impact, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''', (
                article.title,
                article.content,
                article.source,
                article.url,
                article.published_at.isoformat(),
                article.sentiment_score,
                article.sentiment_magnitude,
                json.dumps(article.keywords),
                article.relevance_score,
                article.market_impact
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"📝 Stored {len(articles)} processed articles")
    
    def calculate_sentiment_metrics(self, period_hours: int = 24) -> SentimentMetrics:
        """Calculate sentiment metrics for specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = (datetime.now() - timedelta(hours=period_hours)).isoformat()
        
        cursor.execute('''
            SELECT sentiment_score, relevance_score, market_impact 
            FROM news_articles 
            WHERE published_at >= ? AND processed = 1
            ORDER BY published_at DESC
        ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return SentimentMetrics(
                overall_sentiment=0.0,
                sentiment_trend="neutral",
                bullish_ratio=0.33,
                bearish_ratio=0.33,
                neutral_ratio=0.34,
                confidence=0.0,
                sample_size=0,
                time_period=f"{period_hours}h"
            )
        
        # Weight sentiment by relevance
        weighted_sentiments = []
        for sentiment, relevance, impact in results:
            weight = relevance * (2 if impact == 'high' else 1.5 if impact == 'medium' else 1)
            weighted_sentiments.append(sentiment * weight)
        
        # Calculate metrics
        overall_sentiment = np.mean(weighted_sentiments)
        
        # Categorize sentiments
        bullish_count = sum(1 for s in weighted_sentiments if s > 0.1)
        bearish_count = sum(1 for s in weighted_sentiments if s < -0.1)
        neutral_count = len(weighted_sentiments) - bullish_count - bearish_count
        
        total = len(weighted_sentiments)
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        neutral_ratio = neutral_count / total
        
        # Determine trend
        if overall_sentiment > 0.2:
            trend = "bullish"
        elif overall_sentiment < -0.2:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Calculate confidence based on sample size and consistency
        consistency = 1 - np.std(weighted_sentiments) if len(weighted_sentiments) > 1 else 0.5
        sample_factor = min(1.0, total / 20)  # Full confidence at 20+ articles
        confidence = consistency * sample_factor
        
        return SentimentMetrics(
            overall_sentiment=round(overall_sentiment, 3),
            sentiment_trend=trend,
            bullish_ratio=round(bullish_ratio, 3),
            bearish_ratio=round(bearish_ratio, 3),
            neutral_ratio=round(neutral_ratio, 3),
            confidence=round(confidence, 3),
            sample_size=total,
            time_period=f"{period_hours}h"
        )
    
    async def get_real_time_sentiment(self, hours_back: int = 6) -> Dict:
        """Get real-time sentiment analysis"""
        logger.info(f"🔍 Fetching sentiment analysis for last {hours_back} hours")
        
        # Fetch fresh news
        articles = await self.fetch_news_from_sources(hours_back=hours_back)
        
        if articles:
            # Process articles
            processed = self.process_articles(articles)
            
            # Store in database
            self.store_articles(processed)
        
        # Calculate metrics
        metrics = self.calculate_sentiment_metrics(hours_back)
        
        # Get recent high-impact articles
        recent_articles = self.get_recent_high_impact_articles(limit=5)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'overall_sentiment': metrics.overall_sentiment,
                'sentiment_trend': metrics.sentiment_trend,
                'bullish_ratio': metrics.bullish_ratio,
                'bearish_ratio': metrics.bearish_ratio,
                'neutral_ratio': metrics.neutral_ratio,
                'confidence': metrics.confidence,
                'sample_size': metrics.sample_size
            },
            'recent_articles': recent_articles,
            'interpretation': self.interpret_sentiment(metrics),
            'market_signal': self.generate_market_signal(metrics)
        }
    
    def get_recent_high_impact_articles(self, limit: int = 5) -> List[Dict]:
        """Get recent high-impact articles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, source, sentiment_score, market_impact, published_at, url
            FROM news_articles 
            WHERE processed = 1 AND market_impact IN ('high', 'medium')
            ORDER BY published_at DESC, relevance_score DESC
            LIMIT ?
        ''', (limit,))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'title': row[0],
                'source': row[1],
                'sentiment_score': row[2],
                'market_impact': row[3],
                'published_at': row[4],
                'url': row[5]
            })
        
        conn.close()
        return articles
    
    def interpret_sentiment(self, metrics: SentimentMetrics) -> str:
        """Provide human-readable interpretation of sentiment"""
        if metrics.sample_size < 3:
            return "Limited data available for reliable sentiment analysis."
        
        sentiment = metrics.overall_sentiment
        trend = metrics.sentiment_trend
        confidence = metrics.confidence
        
        if confidence < 0.3:
            confidence_desc = "Low confidence"
        elif confidence < 0.7:
            confidence_desc = "Moderate confidence"
        else:
            confidence_desc = "High confidence"
        
        if trend == "bullish":
            return f"{confidence_desc} - Market sentiment is bullish for gold with {metrics.bullish_ratio:.0%} of news showing positive sentiment. This suggests potential upward price pressure."
        elif trend == "bearish":
            return f"{confidence_desc} - Market sentiment is bearish for gold with {metrics.bearish_ratio:.0%} of news showing negative sentiment. This suggests potential downward price pressure."
        else:
            return f"{confidence_desc} - Market sentiment is neutral with mixed signals. {metrics.bullish_ratio:.0%} bullish, {metrics.bearish_ratio:.0%} bearish news."
    
    def generate_market_signal(self, metrics: SentimentMetrics) -> Dict:
        """Generate trading signal based on sentiment"""
        sentiment = metrics.overall_sentiment
        confidence = metrics.confidence
        
        if confidence < 0.3:
            return {
                'signal': 'HOLD',
                'strength': 'WEAK',
                'reason': 'Insufficient data for reliable signal'
            }
        
        if sentiment > 0.3 and metrics.bullish_ratio > 0.6:
            return {
                'signal': 'BUY',
                'strength': 'STRONG' if confidence > 0.7 else 'MODERATE',
                'reason': f'Strong bullish sentiment ({metrics.bullish_ratio:.0%} positive news)'
            }
        elif sentiment < -0.3 and metrics.bearish_ratio > 0.6:
            return {
                'signal': 'SELL',
                'strength': 'STRONG' if confidence > 0.7 else 'MODERATE',
                'reason': f'Strong bearish sentiment ({metrics.bearish_ratio:.0%} negative news)'
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 'MODERATE',
                'reason': 'Mixed or neutral sentiment signals'
            }

# Global instance
sentiment_service = SentimentAnalysisService()

if __name__ == "__main__":
    # Test the sentiment service
    async def test_sentiment_service():
        print("🧪 Testing Sentiment Analysis Service...")
        
        # Test real-time sentiment
        sentiment_data = await sentiment_service.get_real_time_sentiment(hours_back=12)
        print(f"📊 Sentiment Analysis: {sentiment_data['metrics']}")
        print(f"🎯 Market Signal: {sentiment_data['market_signal']}")
        print(f"💡 Interpretation: {sentiment_data['interpretation']}")
        
        if sentiment_data['recent_articles']:
            print(f"📰 Recent high-impact articles: {len(sentiment_data['recent_articles'])}")
            for article in sentiment_data['recent_articles'][:2]:
                print(f"  - {article['title']} ({article['sentiment_score']:.2f})")
    
    asyncio.run(test_sentiment_service())
