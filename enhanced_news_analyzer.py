"""
Enhanced News Analysis System for GoldGPT
Tracks news sentiment and correlates with gold price movements for predictive analysis
"""

import sqlite3
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re
from textblob import TextBlob
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str    # bullish, bearish, neutral
    gold_price_at_publish: float
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    confidence_score: float

class EnhancedNewsAnalyzer:
    def __init__(self, db_path='goldgpt_news_analysis.db'):
        self.db_path = db_path
        self.init_database()
        
        # Keywords for gold market sentiment
        self.bullish_keywords = [
            'inflation', 'uncertainty', 'crisis', 'recession', 'fed dovish',
            'lower rates', 'stimulus', 'quantitative easing', 'dollar weakness',
            'geopolitical tension', 'war', 'safe haven', 'hedge', 'store of value',
            'gold rally', 'precious metals rise', 'mining companies surge'
        ]
        
        self.bearish_keywords = [
            'fed hawkish', 'rate hike', 'strong dollar', 'economic growth',
            'risk on', 'equity rally', 'crypto surge', 'deflation',
            'gold falls', 'precious metals decline', 'dollar strength',
            'yield rise', 'taper', 'normalization'
        ]
        
    def init_database(self):
        """Initialize SQLite database for news analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT,
                    published_at TIMESTAMP,
                    url TEXT UNIQUE,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    gold_price_at_publish REAL,
                    price_change_1h REAL,
                    price_change_4h REAL,
                    price_change_24h REAL,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_published_at ON news_analysis(published_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment ON news_analysis(sentiment_label)')
            
            conn.commit()
            conn.close()
            logger.info("News analysis database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing news database: {e}")
    
    def analyze_sentiment(self, title: str, content: str) -> Tuple[float, str, float]:
        """Analyze sentiment of news article and return score, label, and confidence"""
        try:
            # Combine title and content for analysis
            full_text = f"{title} {content}".lower()
            
            # Use TextBlob for basic sentiment
            blob = TextBlob(full_text)
            base_sentiment = blob.sentiment.polarity
            
            # Enhanced sentiment based on gold-specific keywords
            bullish_score = sum(1 for keyword in self.bullish_keywords if keyword in full_text)
            bearish_score = sum(1 for keyword in self.bearish_keywords if keyword in full_text)
            
            # Calculate weighted sentiment
            keyword_sentiment = (bullish_score - bearish_score) * 0.1
            final_sentiment = (base_sentiment * 0.7) + (keyword_sentiment * 0.3)
            
            # Clamp to -1, 1 range
            final_sentiment = max(-1, min(1, final_sentiment))
            
            # Determine label
            if final_sentiment >= 0.2:
                sentiment_label = "bullish"
            elif final_sentiment <= -0.2:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"
            
            # Calculate confidence based on keyword presence and text length
            confidence = min(1.0, (bullish_score + bearish_score) * 0.1 + abs(base_sentiment) * 0.5 + min(len(full_text) / 1000, 0.3))
            
            return final_sentiment, sentiment_label, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, "neutral", 0.1
    
    def get_current_gold_price(self) -> float:
        """Get current gold price for correlation analysis"""
        try:
            # Try Gold-API first (reliable and unlimited)
            response = requests.get('https://api.gold-api.com/price/XAU', timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('price', 0))
                if price > 1000:  # Reasonable price check
                    return price
        except:
            pass
        
        try:
            # Fallback to yfinance
            gold = yf.Ticker("GC=F")
            hist = gold.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        return 3340.0  # Fallback price
    
    def calculate_price_changes(self, publish_time: datetime, initial_price: float) -> Tuple[float, float, float]:
        """Calculate gold price changes after news publication"""
        try:
            # Get historical data
            gold = yf.Ticker("GC=F")
            
            # Calculate end times for different periods
            end_1h = publish_time + timedelta(hours=1)
            end_4h = publish_time + timedelta(hours=4)
            end_24h = publish_time + timedelta(hours=24)
            
            # Get price data for each period
            hist_1h = gold.history(start=publish_time, end=end_1h, interval="1m")
            hist_4h = gold.history(start=publish_time, end=end_4h, interval="5m")
            hist_24h = gold.history(start=publish_time, end=end_24h, interval="1h")
            
            # Calculate changes
            change_1h = 0.0
            change_4h = 0.0
            change_24h = 0.0
            
            if not hist_1h.empty and len(hist_1h) >= 10:  # At least 10 minutes of data
                final_price_1h = hist_1h['Close'].iloc[-1]
                change_1h = ((final_price_1h - initial_price) / initial_price) * 100
            
            if not hist_4h.empty and len(hist_4h) >= 10:
                final_price_4h = hist_4h['Close'].iloc[-1]
                change_4h = ((final_price_4h - initial_price) / initial_price) * 100
            
            if not hist_24h.empty and len(hist_24h) >= 10:
                final_price_24h = hist_24h['Close'].iloc[-1]
                change_24h = ((final_price_24h - initial_price) / initial_price) * 100
            
            return change_1h, change_4h, change_24h
            
        except Exception as e:
            logger.error(f"Error calculating price changes: {e}")
            return 0.0, 0.0, 0.0
    
    def store_news_analysis(self, article: NewsArticle):
        """Store news article with analysis in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO news_analysis 
                (title, content, source, published_at, url, sentiment_score, sentiment_label,
                 gold_price_at_publish, price_change_1h, price_change_4h, price_change_24h, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.title, article.content, article.source, article.published_at,
                article.url, article.sentiment_score, article.sentiment_label,
                article.gold_price_at_publish, article.price_change_1h,
                article.price_change_4h, article.price_change_24h, article.confidence_score
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored news analysis: {article.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing news analysis: {e}")
    
    def process_news_article(self, title: str, content: str, source: str, published_at: datetime, url: str) -> NewsArticle:
        """Process a news article and return analysis"""
        try:
            # Analyze sentiment
            sentiment_score, sentiment_label, confidence = self.analyze_sentiment(title, content)
            
            # Get current gold price
            gold_price = self.get_current_gold_price()
            
            # Calculate price changes (this will be updated later as time passes)
            change_1h, change_4h, change_24h = self.calculate_price_changes(published_at, gold_price)
            
            article = NewsArticle(
                title=title,
                content=content,
                source=source,
                published_at=published_at,
                url=url,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                gold_price_at_publish=gold_price,
                price_change_1h=change_1h,
                price_change_4h=change_4h,
                price_change_24h=change_24h,
                confidence_score=confidence
            )
            
            # Store in database
            self.store_news_analysis(article)
            
            return article
            
        except Exception as e:
            logger.error(f"Error processing news article: {e}")
            return None
    
    def get_recent_news_with_analysis(self, limit: int = 20) -> List[Dict]:
        """Get recent news articles with sentiment analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, source, published_at, sentiment_score, sentiment_label,
                       gold_price_at_publish, price_change_1h, price_change_4h, 
                       price_change_24h, confidence_score, url
                FROM news_analysis 
                ORDER BY published_at DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            news_list = []
            for row in results:
                news_item = {
                    'title': row[0],
                    'source': row[1],
                    'published_at': row[2],
                    'sentiment_score': row[3],
                    'sentiment_label': row[4],
                    'gold_price_at_publish': row[5],
                    'price_change_1h': row[6],
                    'price_change_4h': row[7],
                    'price_change_24h': row[8],
                    'confidence_score': row[9],
                    'url': row[10],
                    'time_ago': self.calculate_time_ago(row[2])
                }
                news_list.append(news_item)
            
            return news_list
            
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []
    
    def get_recent_articles(self, limit: int = 20) -> List[NewsArticle]:
        """Get recent analyzed articles from database"""
        try:
            articles = []
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query recent articles
            cursor.execute("""
                SELECT id, title, content, source, published_at, url,
                       sentiment_score, sentiment_label, gold_price_at_publish,
                       price_change_1h, price_change_4h, price_change_24h,
                       confidence_score, created_at
                FROM news_articles 
                ORDER BY published_at DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    published_at = datetime.fromisoformat(row[4]) if row[4] else datetime.now()
                    created_at = datetime.fromisoformat(row[13]) if row[13] else datetime.now()
                    
                    article = NewsArticle(
                        id=row[0],
                        title=row[1],
                        content=row[2],
                        source=row[3],
                        published_at=published_at,
                        url=row[5],
                        sentiment_score=row[6] or 0.0,
                        sentiment_label=row[7] or 'neutral',
                        gold_price_at_publish=row[8] or 0.0,
                        price_change_1h=row[9] or 0.0,
                        price_change_4h=row[10] or 0.0,
                        price_change_24h=row[11] or 0.0,
                        confidence_score=row[12] or 0.1,
                        created_at=created_at
                    )
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error parsing article row: {e}")
                    continue
            
            conn.close()
            return articles
            
        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []
    
    def get_recent_news_with_analysis(self, limit: int = 20) -> List[Dict]:
        """Get recent news with analysis in dictionary format"""
        try:
            articles = self.get_recent_articles(limit)
            
            result = []
            for article in articles:
                article_dict = {
                    'id': article.id,
                    'title': article.title,
                    'content': article.content,
                    'source': article.source,
                    'published_at': article.published_at.isoformat(),
                    'url': article.url,
                    'sentiment_score': article.sentiment_score,
                    'sentiment_label': article.sentiment_label,
                    'gold_price_at_publish': article.gold_price_at_publish,
                    'price_change_1h': article.price_change_1h,
                    'price_change_4h': article.price_change_4h,
                    'price_change_24h': article.price_change_24h,
                    'confidence_score': article.confidence_score,
                    'time_ago': self.calculate_time_ago(article.published_at)
                }
                result.append(article_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting recent news with analysis: {e}")
            return []

    def calculate_time_ago(self, published_at: datetime) -> str:
        """Calculate time ago string"""
        try:
            if isinstance(published_at, str):
                pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                pub_time = published_at
                
            now = datetime.now()
            diff = now - pub_time
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            else:
                return f"{diff.seconds // 60}m ago"
                
        except:
            return "Unknown"
    
    def predict_price_movement(self, title: str, content: str) -> Dict:
        """Predict price movement based on similar historical news"""
        try:
            # Analyze current article sentiment
            sentiment_score, sentiment_label, confidence = self.analyze_sentiment(title, content)
            
            # Query similar historical articles
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find articles with similar sentiment
            cursor.execute('''
                SELECT price_change_1h, price_change_4h, price_change_24h, confidence_score
                FROM news_analysis
                WHERE sentiment_label = ? AND confidence_score > 0.3
                ORDER BY published_at DESC
                LIMIT 50
            ''', (sentiment_label,))
            
            similar_articles = cursor.fetchall()
            conn.close()
            
            if len(similar_articles) < 3:
                return {
                    'prediction': 'insufficient_data',
                    'confidence': 0.1,
                    'expected_change_1h': 0.0,
                    'expected_change_4h': 0.0,
                    'expected_change_24h': 0.0
                }
            
            # Calculate average price movements
            avg_1h = sum(row[0] for row in similar_articles) / len(similar_articles)
            avg_4h = sum(row[1] for row in similar_articles) / len(similar_articles)
            avg_24h = sum(row[2] for row in similar_articles) / len(similar_articles)
            
            # Prediction confidence based on historical accuracy and sample size
            prediction_confidence = min(0.9, confidence * 0.5 + min(len(similar_articles) / 20, 0.4))
            
            return {
                'prediction': sentiment_label,
                'confidence': prediction_confidence,
                'expected_change_1h': avg_1h,
                'expected_change_4h': avg_4h,
                'expected_change_24h': avg_24h,
                'sample_size': len(similar_articles)
            }
            
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return {'prediction': 'error', 'confidence': 0.0}

# Global instance
enhanced_news_analyzer = EnhancedNewsAnalyzer()
