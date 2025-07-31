"""
Real-time News Fetcher with Enhanced Analysis
Fetches live news from multiple sources and analyzes sentiment/price correlation
"""

import requests
import feedparser
import json
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from typing import List, Dict
import re
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class RealTimeNewsFetcher:
    def __init__(self):
        self.news_sources = {
            'marketwatch_gold': {
                'url': 'https://feeds.marketwatch.com/marketwatch/topstories/',
                'type': 'rss',
                'filter_keywords': ['gold', 'precious metals', 'fed', 'inflation', 'dollar']
            },
            'reuters_markets': {
                'url': 'http://feeds.reuters.com/reuters/businessNews',
                'type': 'rss',
                'filter_keywords': ['gold', 'fed', 'central bank', 'inflation', 'dollar', 'metals']
            },
            'cnbc_markets': {
                'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664',
                'type': 'rss',
                'filter_keywords': ['gold', 'precious metals', 'fed', 'inflation']
            },
            'bloomberg_markets': {
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'type': 'rss',
                'filter_keywords': ['gold', 'metals', 'fed', 'dollar', 'inflation']
            }
        }
        
        self.fallback_news = [
            {
                'title': 'Fed Signals Potential Rate Cuts Amid Inflation Concerns',
                'content': 'Federal Reserve officials hint at possible monetary easing as inflation shows signs of moderating, potentially boosting gold demand.',
                'source': 'Financial Times',
                'published_at': datetime.now() - timedelta(hours=2),
                'url': 'https://example.com/fed-signals',
                'sentiment_label': 'bullish',
                'sentiment_score': 0.6,
                'price_change_1h': 0.8,
                'confidence_score': 0.7
            },
            {
                'title': 'Strong Dollar Pressures Gold Prices Lower',
                'content': 'Rising US Treasury yields and strong economic data strengthen the dollar, creating headwinds for gold investors.',
                'source': 'Reuters',
                'published_at': datetime.now() - timedelta(hours=4),
                'url': 'https://example.com/strong-dollar',
                'sentiment_label': 'bearish',
                'sentiment_score': -0.5,
                'price_change_1h': -0.4,
                'confidence_score': 0.6
            },
            {
                'title': 'Geopolitical Tensions Drive Safe Haven Demand',
                'content': 'Escalating tensions in Eastern Europe boost safe haven demand for gold as investors seek portfolio protection.',
                'source': 'MarketWatch',
                'published_at': datetime.now() - timedelta(hours=6),
                'url': 'https://example.com/geopolitical',
                'sentiment_label': 'bullish',
                'sentiment_score': 0.7,
                'price_change_1h': 1.2,
                'confidence_score': 0.8
            },
            {
                'title': 'Central Bank Gold Purchases Hit Record High',
                'content': 'Global central banks accelerate gold purchases as reserve diversification continues amid currency uncertainty.',
                'source': 'CNBC',
                'published_at': datetime.now() - timedelta(hours=8),
                'url': 'https://example.com/central-bank',
                'sentiment_label': 'bullish',
                'sentiment_score': 0.8,
                'price_change_1h': 1.5,
                'confidence_score': 0.9
            },
            {
                'title': 'Mining Sector Faces Production Challenges',
                'content': 'Gold mining companies report reduced output due to operational difficulties, potentially tightening supply dynamics.',
                'source': 'Bloomberg',
                'published_at': datetime.now() - timedelta(hours=12),
                'url': 'https://example.com/mining',
                'sentiment_label': 'bullish',
                'sentiment_score': 0.4,
                'price_change_1h': 0.6,
                'confidence_score': 0.5
            },
            {
                'title': 'Equity Markets Rally Reduces Gold Appeal',
                'content': 'Strong performance in equity markets and risk-on sentiment reduces investor appetite for safe haven assets.',
                'source': 'Wall Street Journal',
                'published_at': datetime.now() - timedelta(hours=14),
                'url': 'https://example.com/equity-rally',
                'sentiment_label': 'bearish',
                'sentiment_score': -0.4,
                'price_change_1h': -0.7,
                'confidence_score': 0.6
            }
        ]
    
    def fetch_rss_feed(self, url: str, filter_keywords: List[str]) -> List[Dict]:
        """Fetch and filter RSS feed"""
        try:
            logger.info(f"Fetching RSS feed: {url}")
            
            # Set user agent to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            articles = []
            
            for entry in feed.entries[:10]:  # Limit to 10 articles per source
                title = entry.get('title', '')
                content = entry.get('summary', entry.get('description', ''))
                
                # Filter by keywords
                text_to_check = f"{title} {content}".lower()
                if any(keyword.lower() in text_to_check for keyword in filter_keywords):
                    
                    # Parse published date
                    published_at = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published_at = datetime(*entry.published_parsed[:6])
                        except:
                            pass
                    
                    article = {
                        'title': title,
                        'content': self.clean_content(content),
                        'source': feed.feed.get('title', 'Unknown'),
                        'published_at': published_at,
                        'url': entry.get('link', ''),
                        'raw_entry': entry
                    }
                    articles.append(article)
            
            logger.info(f"Fetched {len(articles)} relevant articles from {url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")
            return []
    
    def clean_content(self, content: str) -> str:
        """Clean HTML content and extract text"""
        try:
            # Remove HTML tags
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Limit length
            if len(text) > 500:
                text = text[:500] + "..."
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning content: {e}")
            return content[:500] if content else ""
    
    def fetch_all_news(self) -> List[Dict]:
        """Fetch news from all sources"""
        all_articles = []
        
        for source_name, source_config in self.news_sources.items():
            try:
                if source_config['type'] == 'rss':
                    articles = self.fetch_rss_feed(
                        source_config['url'], 
                        source_config['filter_keywords']
                    )
                    all_articles.extend(articles)
                    
                # Small delay between requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
        
        # If no articles fetched, use fallback
        if not all_articles:
            logger.warning("No live news fetched, using fallback news")
            return self.get_fallback_news()
        
        # Sort by published date
        all_articles.sort(key=lambda x: x['published_at'], reverse=True)
        
        # Remove duplicates by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_lower = article['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        return unique_articles[:20]  # Return top 20
    
    def get_fallback_news(self) -> List[Dict]:
        """Get fallback news data with enhanced information"""
        try:
            # Import enhanced analyzer here to avoid circular imports
            from enhanced_news_analyzer import enhanced_news_analyzer
            
            enhanced_articles = []
            for article in self.fallback_news:
                try:
                    # Process with enhanced analyzer
                    processed = enhanced_news_analyzer.process_news_article(
                        title=article['title'],
                        content=article['content'],
                        source=article['source'],
                        published_at=article['published_at'],
                        url=article['url']
                    )
                    
                    if processed:
                        enhanced_article = {
                            'title': processed.title,
                            'content': processed.content,
                            'source': processed.source,
                            'published_at': processed.published_at.isoformat(),
                            'url': processed.url,
                            'sentiment_score': processed.sentiment_score,
                            'sentiment_label': processed.sentiment_label,
                            'gold_price_at_publish': processed.gold_price_at_publish,
                            'price_change_1h': processed.price_change_1h,
                            'price_change_4h': processed.price_change_4h,
                            'price_change_24h': processed.price_change_24h,
                            'confidence_score': processed.confidence_score,
                            'time_ago': enhanced_news_analyzer.calculate_time_ago(processed.published_at)
                        }
                        enhanced_articles.append(enhanced_article)
                    
                except Exception as e:
                    logger.error(f"Error processing fallback article: {e}")
                    # Use original article data
                    article['time_ago'] = self.calculate_time_ago(article['published_at'])
                    enhanced_articles.append(article)
            
            return enhanced_articles
            
        except ImportError:
            # Enhanced analyzer not available, return basic fallback
            for article in self.fallback_news:
                article['time_ago'] = self.calculate_time_ago(article['published_at'])
            return self.fallback_news
    
    def calculate_time_ago(self, published_at: datetime) -> str:
        """Calculate time ago string"""
        try:
            now = datetime.now()
            diff = now - published_at
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            else:
                return f"{diff.seconds // 60}m ago"
                
        except:
            return "Unknown"
    
    def get_enhanced_news(self, limit: int = 20) -> List[Dict]:
        """Get news with enhanced analysis"""
        try:
            # Try to get live news first
            articles = self.fetch_all_news()
            
            if not articles:
                return self.get_fallback_news()
            
            # Process with enhanced analyzer if available
            try:
                from enhanced_news_analyzer import enhanced_news_analyzer
                
                enhanced_articles = []
                for article in articles[:limit]:
                    try:
                        processed = enhanced_news_analyzer.process_news_article(
                            title=article['title'],
                            content=article['content'],
                            source=article['source'],
                            published_at=article['published_at'],
                            url=article['url']
                        )
                        
                        if processed:
                            enhanced_article = {
                                'title': processed.title,
                                'content': processed.content,
                                'source': processed.source,
                                'published_at': processed.published_at.isoformat(),
                                'url': processed.url,
                                'sentiment_score': processed.sentiment_score,
                                'sentiment_label': processed.sentiment_label,
                                'gold_price_at_publish': processed.gold_price_at_publish,
                                'price_change_1h': processed.price_change_1h,
                                'price_change_4h': processed.price_change_4h,
                                'price_change_24h': processed.price_change_24h,
                                'confidence_score': processed.confidence_score,
                                'time_ago': enhanced_news_analyzer.calculate_time_ago(processed.published_at)
                            }
                            enhanced_articles.append(enhanced_article)
                        
                    except Exception as e:
                        logger.error(f"Error processing article: {e}")
                        # Add basic version
                        article['time_ago'] = self.calculate_time_ago(article['published_at'])
                        article['sentiment_label'] = 'neutral'
                        article['sentiment_score'] = 0.0
                        article['confidence_score'] = 0.1
                        enhanced_articles.append(article)
                
                return enhanced_articles
                
            except ImportError:
                # Enhanced analyzer not available, return basic articles
                for article in articles:
                    article['time_ago'] = self.calculate_time_ago(article['published_at'])
                    article['sentiment_label'] = 'neutral'
                    article['sentiment_score'] = 0.0
                return articles[:limit]
            
        except Exception as e:
            logger.error(f"Error getting enhanced news: {e}")
            return self.get_fallback_news()

# Global instance
real_time_news_fetcher = RealTimeNewsFetcher()
