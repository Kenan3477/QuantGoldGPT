#!/usr/bin/env python3
"""
GoldGPT Sentiment Analysis Service
Advanced news sentiment analysis with correlation tracking for market predictions
"""

import asyncio
import sqlite3
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import time
import numpy as np
from textblob import TextBlob
from collections import defaultdict
from data_pipeline_core import DataPipelineCore, DataType, DataSourceTier

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    relevance_score: float
    sentiment_score: float
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    keywords: List[str]
    market_impact: float  # -1.0 to 1.0

@dataclass
class SentimentSignal:
    """Market sentiment signal"""
    timestamp: datetime
    sentiment_score: float  # -1.0 to 1.0
    confidence: float
    news_count: int
    dominant_themes: List[str]
    market_correlation: float
    prediction_strength: str  # 'weak', 'moderate', 'strong'

class AdvancedSentimentAnalysisService:
    """Advanced sentiment analysis with market correlation tracking"""
    
    def __init__(self, pipeline: DataPipelineCore, db_path: str = "goldgpt_sentiment_analysis.db"):
        self.pipeline = pipeline
        self.db_path = db_path
        
        # Market-relevant keywords and their weights
        self.market_keywords = {
            # Bullish keywords for gold
            'bullish': {
                'inflation': 2.0, 'uncertainty': 1.8, 'safe haven': 2.5,
                'dollar weakness': 2.0, 'monetary policy': 1.5, 'recession': 1.8,
                'geopolitical tension': 2.2, 'supply shortage': 2.0, 'demand surge': 2.3,
                'fed dovish': 2.1, 'rate cut': 2.0, 'economic crisis': 2.5
            },
            # Bearish keywords for gold
            'bearish': {
                'dollar strength': -2.0, 'rate hike': -2.2, 'economic growth': -1.5,
                'fed hawkish': -2.1, 'tapering': -1.8, 'risk appetite': -1.6,
                'stock rally': -1.3, 'oversupply': -1.9, 'demand drop': -2.0,
                'recovery': -1.4, 'optimism': -1.2, 'stability': -1.1
            }
        }
        
        # News source reliability weights
        self.source_weights = {
            'reuters': 1.0, 'bloomberg': 1.0, 'wsj': 0.95, 'ft': 0.95,
            'cnbc': 0.85, 'marketwatch': 0.8, 'yahoo': 0.7, 'generic': 0.6
        }
        
        self.initialize_database()
        
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
                published_at DATETIME NOT NULL,
                relevance_score REAL NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                keywords TEXT,
                market_impact REAL NOT NULL,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sentiment signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                news_count INTEGER NOT NULL,
                dominant_themes TEXT,
                market_correlation REAL,
                prediction_strength TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sentiment-price correlation tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                sentiment_avg REAL NOT NULL,
                price_change_pct REAL NOT NULL,
                correlation_score REAL,
                accuracy_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        ''')
        
        # Keyword impact tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                impact_score REAL NOT NULL,
                frequency INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 0.5,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(keyword)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Sentiment analysis database initialized")
    
    async def fetch_news_articles(self, symbol: str = 'gold', limit: int = 50) -> List[NewsArticle]:
        """Fetch news articles from multiple sources"""
        articles = []
        
        try:
            # Simulate fetching from multiple news sources
            # In production, this would integrate with real news APIs
            
            news_sources = [
                {
                    'name': 'reuters',
                    'articles': await self.fetch_reuters_news(symbol, limit // 3)
                },
                {
                    'name': 'bloomberg',
                    'articles': await self.fetch_bloomberg_news(symbol, limit // 3)
                },
                {
                    'name': 'marketwatch',
                    'articles': await self.fetch_marketwatch_news(symbol, limit // 3)
                }
            ]
            
            for source in news_sources:
                articles.extend(source['articles'])
            
            # Remove duplicates and sort by relevance
            unique_articles = self.deduplicate_articles(articles)
            return sorted(unique_articles, key=lambda x: x.relevance_score, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []
    
    async def fetch_reuters_news(self, symbol: str, limit: int) -> List[NewsArticle]:
        """Simulate fetching Reuters news"""
        # This would integrate with Reuters API in production
        sample_articles = [
            {
                'title': 'Gold prices rise as dollar weakens on Fed dovish signals',
                'content': 'Gold prices climbed higher as the U.S. dollar weakened following dovish signals from Federal Reserve officials...',
                'source': 'reuters',
                'url': 'https://reuters.com/article1',
                'published_at': datetime.now() - timedelta(hours=2)
            },
            {
                'title': 'Inflation concerns boost safe-haven gold demand',
                'content': 'Rising inflation expectations are driving investors toward safe-haven assets like gold...',
                'source': 'reuters',
                'url': 'https://reuters.com/article2',
                'published_at': datetime.now() - timedelta(hours=4)
            }
        ]
        
        articles = []
        for article_data in sample_articles[:limit]:
            article = await self.process_article(article_data)
            if article:
                articles.append(article)
        
        return articles
    
    async def fetch_bloomberg_news(self, symbol: str, limit: int) -> List[NewsArticle]:
        """Simulate fetching Bloomberg news"""
        sample_articles = [
            {
                'title': 'Central bank gold reserves reach record highs amid uncertainty',
                'content': 'Central banks continue to increase their gold reserves as economic uncertainty persists...',
                'source': 'bloomberg',
                'url': 'https://bloomberg.com/article1',
                'published_at': datetime.now() - timedelta(hours=1)
            }
        ]
        
        articles = []
        for article_data in sample_articles[:limit]:
            article = await self.process_article(article_data)
            if article:
                articles.append(article)
        
        return articles
    
    async def fetch_marketwatch_news(self, symbol: str, limit: int) -> List[NewsArticle]:
        """Simulate fetching MarketWatch news"""
        sample_articles = [
            {
                'title': 'Gold ETFs see massive inflows as recession fears grow',
                'content': 'Gold exchange-traded funds are experiencing significant inflows as investors position for potential recession...',
                'source': 'marketwatch',
                'url': 'https://marketwatch.com/article1',
                'published_at': datetime.now() - timedelta(hours=3)
            }
        ]
        
        articles = []
        for article_data in sample_articles[:limit]:
            article = await self.process_article(article_data)
            if article:
                articles.append(article)
        
        return articles
    
    async def process_article(self, article_data: Dict) -> Optional[NewsArticle]:
        """Process and analyze individual news article"""
        try:
            title = article_data['title']
            content = article_data.get('content', '')
            source = article_data['source']
            
            # Calculate relevance score
            relevance = self.calculate_relevance_score(title + ' ' + content)
            
            # Skip if not relevant enough
            if relevance < 0.3:
                return None
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self.analyze_sentiment(title + ' ' + content)
            
            # Extract keywords
            keywords = self.extract_keywords(title + ' ' + content)
            
            # Calculate market impact
            market_impact = self.calculate_market_impact(title + ' ' + content, keywords, source)
            
            return NewsArticle(
                title=title,
                content=content,
                source=source,
                url=article_data['url'],
                published_at=article_data['published_at'],
                relevance_score=relevance,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                keywords=keywords,
                market_impact=market_impact
            )
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def calculate_relevance_score(self, text: str) -> float:
        """Calculate how relevant an article is to gold/precious metals"""
        gold_keywords = [
            'gold', 'precious metals', 'bullion', 'xau', 'mining',
            'federal reserve', 'inflation', 'dollar', 'safe haven',
            'monetary policy', 'central bank', 'interest rates'
        ]
        
        text_lower = text.lower()
        relevance_score = 0.0
        
        for keyword in gold_keywords:
            if keyword in text_lower:
                # Weight by keyword importance
                if keyword in ['gold', 'bullion', 'xau']:
                    relevance_score += 0.3
                elif keyword in ['federal reserve', 'inflation', 'dollar']:
                    relevance_score += 0.2
                else:
                    relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text using TextBlob and keyword analysis"""
        try:
            # Basic sentiment using TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Enhance with market-specific keyword analysis
            keyword_sentiment = self.analyze_market_keywords(text)
            
            # Combine both approaches (weighted average)
            combined_sentiment = (polarity * 0.4) + (keyword_sentiment * 0.6)
            
            # Determine label
            if combined_sentiment > 0.1:
                label = 'positive'
            elif combined_sentiment < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return combined_sentiment, label
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, 'neutral'
    
    def analyze_market_keywords(self, text: str) -> float:
        """Analyze text for market-specific keywords and their sentiment impact"""
        text_lower = text.lower()
        sentiment_score = 0.0
        keyword_count = 0
        
        # Check bullish keywords
        for keyword, weight in self.market_keywords['bullish'].items():
            if keyword in text_lower:
                sentiment_score += weight
                keyword_count += 1
        
        # Check bearish keywords
        for keyword, weight in self.market_keywords['bearish'].items():
            if keyword in text_lower:
                sentiment_score += weight  # weight is already negative
                keyword_count += 1
        
        # Normalize by keyword count to prevent over-weighting
        if keyword_count > 0:
            sentiment_score = sentiment_score / keyword_count
            # Scale to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score / 2.0))
        
        return sentiment_score
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction - in production, use more sophisticated NLP
        keywords = []
        text_lower = text.lower()
        
        # Extract market-relevant keywords
        all_keywords = list(self.market_keywords['bullish'].keys()) + list(self.market_keywords['bearish'].keys())
        
        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    def calculate_market_impact(self, text: str, keywords: List[str], source: str) -> float:
        """Calculate potential market impact of the news"""
        # Base impact from sentiment
        sentiment_score, _ = self.analyze_sentiment(text)
        
        # Weight by source reliability
        source_weight = self.source_weights.get(source, 0.6)
        
        # Weight by keyword importance
        keyword_impact = 0.0
        for keyword in keywords:
            if keyword in self.market_keywords['bullish']:
                keyword_impact += abs(self.market_keywords['bullish'][keyword]) * 0.1
            elif keyword in self.market_keywords['bearish']:
                keyword_impact += abs(self.market_keywords['bearish'][keyword]) * 0.1
        
        # Combine factors
        market_impact = sentiment_score * source_weight * (1 + keyword_impact)
        
        return max(-1.0, min(1.0, market_impact))
    
    def deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication - in production, use more sophisticated similarity matching
            title_words = set(article.title.lower().split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                # If more than 70% of words overlap, consider it a duplicate
                overlap = len(title_words.intersection(seen_words)) / len(title_words.union(seen_words))
                if overlap > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(article.title.lower())
        
        return unique_articles
    
    async def store_articles(self, articles: List[NewsArticle]):
        """Store news articles in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_articles
                    (title, content, source, url, published_at, relevance_score,
                     sentiment_score, sentiment_label, keywords, market_impact)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.title,
                    article.content,
                    article.source,
                    article.url,
                    article.published_at.isoformat(),
                    article.relevance_score,
                    article.sentiment_score,
                    article.sentiment_label,
                    json.dumps(article.keywords),
                    article.market_impact
                ))
                
                if cursor.rowcount > 0:
                    stored_count += 1
                    
            except Exception as e:
                logger.error(f"Error storing article: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Stored {stored_count} new articles")
    
    async def generate_sentiment_signal(self, hours_lookback: int = 24) -> SentimentSignal:
        """Generate aggregated sentiment signal from recent news"""
        try:
            # Get recent articles
            cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sentiment_score, market_impact, keywords, source, relevance_score
                FROM news_articles
                WHERE published_at >= ?
                AND relevance_score >= 0.3
                ORDER BY published_at DESC
            ''', (cutoff_time.isoformat(),))
            
            articles_data = cursor.fetchall()
            conn.close()
            
            if not articles_data:
                return self.generate_neutral_signal()
            
            # Calculate weighted sentiment
            weighted_sentiments = []
            all_keywords = []
            sources = []
            
            for sentiment, market_impact, keywords_json, source, relevance in articles_data:
                # Weight by relevance and source reliability
                source_weight = self.source_weights.get(source, 0.6)
                weight = relevance * source_weight
                
                weighted_sentiments.append(sentiment * weight)
                
                try:
                    keywords = json.loads(keywords_json)
                    all_keywords.extend(keywords)
                except:
                    pass
                
                sources.append(source)
            
            # Calculate overall sentiment
            if weighted_sentiments:
                avg_sentiment = np.mean(weighted_sentiments)
                sentiment_std = np.std(weighted_sentiments)
            else:
                avg_sentiment = 0.0
                sentiment_std = 0.0
            
            # Calculate confidence (lower std = higher confidence)
            confidence = max(0.1, min(1.0, 1.0 - sentiment_std))
            
            # Find dominant themes
            keyword_counts = defaultdict(int)
            for keyword in all_keywords:
                keyword_counts[keyword] += 1
            
            dominant_themes = [k for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
            
            # Determine prediction strength
            if abs(avg_sentiment) > 0.3 and confidence > 0.7:
                prediction_strength = 'strong'
            elif abs(avg_sentiment) > 0.15 and confidence > 0.5:
                prediction_strength = 'moderate'
            else:
                prediction_strength = 'weak'
            
            # Calculate market correlation (simplified)
            market_correlation = await self.calculate_historical_correlation()
            
            signal = SentimentSignal(
                timestamp=datetime.now(),
                sentiment_score=avg_sentiment,
                confidence=confidence,
                news_count=len(articles_data),
                dominant_themes=dominant_themes,
                market_correlation=market_correlation,
                prediction_strength=prediction_strength
            )
            
            # Store signal
            await self.store_sentiment_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return self.generate_neutral_signal()
    
    def generate_neutral_signal(self) -> SentimentSignal:
        """Generate neutral sentiment signal when no data is available"""
        return SentimentSignal(
            timestamp=datetime.now(),
            sentiment_score=0.0,
            confidence=0.1,
            news_count=0,
            dominant_themes=[],
            market_correlation=0.0,
            prediction_strength='weak'
        )
    
    async def store_sentiment_signal(self, signal: SentimentSignal):
        """Store sentiment signal in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_signals
            (timestamp, sentiment_score, confidence, news_count,
             dominant_themes, market_correlation, prediction_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.sentiment_score,
            signal.confidence,
            signal.news_count,
            json.dumps(signal.dominant_themes),
            signal.market_correlation,
            signal.prediction_strength
        ))
        
        conn.commit()
        conn.close()
    
    async def calculate_historical_correlation(self) -> float:
        """Calculate historical correlation between sentiment and price movements"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get historical sentiment-price correlations
            cursor.execute('''
                SELECT correlation_score FROM sentiment_correlations
                WHERE date >= ?
                ORDER BY date DESC
                LIMIT 30
            ''', ((datetime.now() - timedelta(days=30)).date(),))
            
            correlations = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if correlations:
                return np.mean(correlations)
            else:
                return 0.0  # No historical data
                
        except Exception as e:
            logger.error(f"Error calculating historical correlation: {e}")
            return 0.0
    
    async def update_news_and_generate_signal(self) -> SentimentSignal:
        """Full pipeline: fetch news, analyze, and generate signal"""
        try:
            logger.info("🔄 Starting news analysis pipeline...")
            
            # Fetch fresh news articles
            articles = await self.fetch_news_articles('gold', limit=50)
            logger.info(f"📰 Fetched {len(articles)} news articles")
            
            # Store articles
            if articles:
                await self.store_articles(articles)
            
            # Generate sentiment signal
            signal = await self.generate_sentiment_signal(hours_lookback=24)
            logger.info(f"📊 Generated sentiment signal: {signal.sentiment_score:.3f} "
                       f"(confidence: {signal.confidence:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in news analysis pipeline: {e}")
            return self.generate_neutral_signal()
    
    async def get_sentiment_history(self, days: int = 7) -> List[Dict]:
        """Get historical sentiment signals"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT timestamp, sentiment_score, confidence, news_count,
                   dominant_themes, prediction_strength
            FROM sentiment_signals
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (cutoff_date.isoformat(),))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'timestamp': row[0],
                'sentiment_score': row[1],
                'confidence': row[2],
                'news_count': row[3],
                'dominant_themes': json.loads(row[4]) if row[4] else [],
                'prediction_strength': row[5]
            })
        
        conn.close()
        return history

# Global instance
sentiment_service = AdvancedSentimentAnalysisService(DataPipelineCore())

if __name__ == "__main__":
    async def test_sentiment_service():
        print("🧪 Testing Advanced Sentiment Analysis Service...")
        
        # Test news fetching and analysis
        signal = await sentiment_service.update_news_and_generate_signal()
        print(f"📊 Sentiment Signal: {signal}")
        
        # Test sentiment history
        history = await sentiment_service.get_sentiment_history(days=7)
        print(f"📈 Sentiment History: {len(history)} records")
    
    asyncio.run(test_sentiment_service())
