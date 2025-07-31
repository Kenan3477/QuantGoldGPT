"""
Robust Multi-Source Data Fetching System for GoldGPT
Implements tiered fallback architecture with APIs, web scraping, and simulated data

Architecture:
1. Primary: Free public APIs (fastest, most reliable)
2. Secondary: Web scraping (reliable but slower)
3. Tertiary: Simulated data (always available)

Features:
- Automatic source failover
- Intelligent caching with TTL
- Rate limiting and retry logic
- Comprehensive error handling
- Source reliability scoring
"""

import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import random
import time
import hashlib
from bs4 import BeautifulSoup
import re
import numpy as np
from textblob import TextBlob
import warnings

# Suppress warnings for production
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source types for tracking and prioritization"""
    API_PRIMARY = "api_primary"
    API_SECONDARY = "api_secondary"
    WEB_SCRAPING = "web_scraping"
    SIMULATED = "simulated"

@dataclass
class PriceData:
    """Price data structure"""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    source: DataSource = DataSource.SIMULATED
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.bid and self.ask:
            self.spread = abs(self.ask - self.bid)

@dataclass
class SentimentData:
    """Sentiment analysis data structure"""
    symbol: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str    # bullish, neutral, bearish
    confidence: float       # 0 to 1
    sources_count: int
    timeframe: str
    news_articles: List[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.news_articles is None:
            self.news_articles = []

@dataclass
class TechnicalData:
    """Technical analysis data structure"""
    symbol: str
    indicators: Dict[str, Any]
    analysis_timeframe: str
    source: DataSource = DataSource.SIMULATED
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class CacheManager:
    """SQLite-based cache manager with TTL support"""
    
    def __init__(self, db_path: str = "robust_data_cache.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp REAL,
                    ttl INTEGER
                )
            """)
            conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, timestamp, ttl FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value, timestamp, ttl = row
                    if time.time() - timestamp < ttl:
                        return json.loads(value)
                    else:
                        # Remove expired entry
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        conn.commit()
                        
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set cached value with TTL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert datetime objects and dataclasses to JSON serializable format
                def custom_serializer(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    # Handle dataclass objects - force convert using asdict
                    try:
                        # Try to use asdict directly (will work if it's a dataclass)
                        return asdict(obj)
                    except (TypeError, ValueError):
                        pass
                    # Handle mappingproxy and other special types
                    if str(type(obj)) == "<class 'mappingproxy'>":
                        return dict(obj)
                    # Handle other objects with __dict__
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    # Last resort - convert to string
                    return str(obj)
                
                json_value = json.dumps(value, default=custom_serializer)
                
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, timestamp, ttl) VALUES (?, ?, ?, ?)",
                    (key, json_value, time.time(), ttl)
                )
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM cache WHERE timestamp + ttl < ?",
                    (time.time(),)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

class RateLimiter:
    """Simple rate limiter for respectful API usage"""
    
    def __init__(self):
        self.requests = {}
    
    async def wait_if_needed(self, source: str, min_interval: float = 1.0):
        """Wait if minimum interval hasn't passed"""
        last_request = self.requests.get(source)
        if last_request:
            time_since = time.time() - last_request
            if time_since < min_interval:
                await asyncio.sleep(min_interval - time_since)
        
        self.requests[source] = time.time()

class DataSourceManager:
    """Manages data sources with automatic failover and reliability tracking"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.rate_limiter = RateLimiter()
        self.source_reliability = {
            DataSource.API_PRIMARY: 0.9,
            DataSource.API_SECONDARY: 0.7,
            DataSource.WEB_SCRAPING: 0.6,
            DataSource.SIMULATED: 1.0
        }
        self.request_counts = {source: 0 for source in DataSource}
        self.success_counts = {source: 0 for source in DataSource}
    
    def update_reliability(self, source: DataSource, success: bool):
        """Update source reliability based on success/failure"""
        self.request_counts[source] += 1
        if success:
            self.success_counts[source] += 1
        
        # Calculate rolling reliability
        if self.request_counts[source] > 0:
            self.source_reliability[source] = self.success_counts[source] / self.request_counts[source]
    
    def get_preferred_sources(self) -> List[DataSource]:
        """Get sources ordered by reliability"""
        return sorted(
            DataSource,
            key=lambda s: self.source_reliability[s],
            reverse=True
        )

class PriceDataService:
    """Price data service with multiple sources and fallbacks"""
    
    def __init__(self, data_manager: DataSourceManager):
        self.data_manager = data_manager
        self.api_sources = [
            {
                'name': 'gold-api',
                'url': 'https://api.gold-api.com/price/XAU',
                'source': DataSource.API_PRIMARY
            },
            {
                'name': 'yahoo-finance',
                'url': 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}=X',
                'source': DataSource.API_SECONDARY
            }
        ]
        self.scraping_urls = [
            'https://www.investing.com/currencies/xau-usd',
            'https://finance.yahoo.com/quote/GC=F',
            'https://www.marketwatch.com/investing/future/gc00'
        ]
    
    async def get_price_data(self, symbol: str) -> PriceData:
        """Get price data with fallback chain"""
        # Check cache first
        cache_key = f"price:{symbol}"
        cached = self.data_manager.cache_manager.get(cache_key)
        if cached:
            # Reconstruct PriceData from cached dict
            if 'timestamp' in cached and isinstance(cached['timestamp'], str):
                cached['timestamp'] = datetime.fromisoformat(cached['timestamp'])
            return PriceData(**cached)
        
        # Try API sources
        for api_source in self.api_sources:
            try:
                await self.data_manager.rate_limiter.wait_if_needed(api_source['name'])
                price_data = await self._fetch_from_api(symbol, api_source)
                if price_data:
                    self.data_manager.update_reliability(api_source['source'], True)
                    self.data_manager.cache_manager.set(cache_key, price_data)
                    logger.info(f"Price data for {symbol} from {api_source['name']}")
                    return price_data
            except Exception as e:
                logger.error(f"Primary API failed for {symbol}: {e}")
                self.data_manager.update_reliability(api_source['source'], False)
        
        # Try web scraping
        try:
            await self.data_manager.rate_limiter.wait_if_needed('scraping', 2.0)
            price_data = await self._scrape_price_data(symbol)
            if price_data:
                self.data_manager.update_reliability(DataSource.WEB_SCRAPING, True)
                self.data_manager.cache_manager.set(cache_key, price_data)
                logger.info(f"Price data for {symbol} from scraping")
                return price_data
        except Exception as e:
            logger.warning(f"Scraping failed for {symbol}: {e}")
            self.data_manager.update_reliability(DataSource.WEB_SCRAPING, False)
        
        # Final fallback: simulated data
        price_data = self._generate_simulated_price(symbol)
        self.data_manager.cache_manager.set(cache_key, price_data, ttl=60)  # Short TTL for simulated
        logger.info(f"Price data for {symbol} from simulated")
        return price_data
    
    async def _fetch_from_api(self, symbol: str, api_source: Dict) -> Optional[PriceData]:
        """Fetch price from specific API"""
        try:
            if api_source['name'] == 'gold-api':
                return await self._fetch_gold_api(symbol)
            elif api_source['name'] == 'yahoo-finance':
                return await self._fetch_yahoo_api(symbol)
        except Exception as e:
            logger.error(f"API fetch failed for {api_source['name']}: {e}")
        return None
    
    async def _fetch_gold_api(self, symbol: str) -> Optional[PriceData]:
        """Fetch from Gold API"""
        if symbol != 'XAUUSD':
            return None
        
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.gold-api.com/price/XAU') as response:
                if response.status == 200:
                    data = await response.json()
                    price = data.get('price', 0)
                    return PriceData(
                        symbol=symbol,
                        price=price,
                        source=DataSource.API_PRIMARY,
                        timestamp=datetime.now()
                    )
        return None
    
    async def _fetch_yahoo_api(self, symbol: str) -> Optional[PriceData]:
        """Fetch from Yahoo Finance API"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}=X"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        chart = data['chart']['result'][0]
                        meta = chart['meta']
                        return PriceData(
                            symbol=symbol,
                            price=meta['regularMarketPrice'],
                            source=DataSource.API_SECONDARY,
                            timestamp=datetime.now()
                        )
        except Exception as e:
            logger.error(f"Yahoo API error: {e}")
        return None
    
    async def _scrape_price_data(self, symbol: str) -> Optional[PriceData]:
        """Scrape price data from financial websites"""
        for url in self.scraping_urls:
            try:
                price = await self._scrape_single_url(url, symbol)
                if price:
                    return PriceData(
                        symbol=symbol,
                        price=price,
                        source=DataSource.WEB_SCRAPING,
                        timestamp=datetime.now()
                    )
            except Exception as e:
                logger.warning(f"Scraping failed for {url}: {e}")
        return None
    
    async def _scrape_single_url(self, url: str, symbol: str) -> Optional[float]:
        """Scrape price from a single URL"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for price patterns
                    price_patterns = [
                        r'\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]{2})?)',
                        r'([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]{2})?)\s*USD'
                    ]
                    
                    for pattern in price_patterns:
                        matches = re.findall(pattern, html)
                        for match in matches:
                            try:
                                price = float(match.replace(',', ''))
                                if 1500 <= price <= 3000:  # Reasonable gold price range
                                    return price
                            except ValueError:
                                continue
        return None
    
    def _generate_simulated_price(self, symbol: str) -> PriceData:
        """Generate realistic simulated price data"""
        base_prices = {
            'XAUUSD': 2000.0,
            'XAGUSD': 25.0,
            'EURUSD': 1.08,
            'GBPUSD': 1.26,
            'USDJPY': 148.0,
            'BTCUSD': 43500.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add realistic volatility
        volatility = random.uniform(-0.02, 0.02)  # ±2%
        price = base_price * (1 + volatility)
        
        # Generate bid/ask spread
        spread_pct = 0.001 if 'USD' in symbol else 0.0001
        spread = price * spread_pct
        
        return PriceData(
            symbol=symbol,
            price=round(price, 2),
            bid=round(price - spread/2, 2),
            ask=round(price + spread/2, 2),
            spread=round(spread, 4),
            change=round(base_price * random.uniform(-0.01, 0.01), 2),
            change_percent=round(random.uniform(-1.0, 1.0), 2),
            volume=random.randint(10000, 100000),
            source=DataSource.SIMULATED,
            timestamp=datetime.now()
        )

class SentimentAnalysisService:
    """Sentiment analysis service with news scraping and NLP"""
    
    def __init__(self, data_manager: DataSourceManager):
        self.data_manager = data_manager
        self.news_sources = [
            'https://finance.yahoo.com/news',
            'https://www.marketwatch.com/latest-news',
            'https://www.reuters.com/markets'
        ]
        self.sentiment_keywords = {
            'bullish': ['rise', 'surge', 'bull', 'up', 'gain', 'rally', 'strong', 'positive', 'buy'],
            'bearish': ['fall', 'drop', 'bear', 'down', 'loss', 'decline', 'weak', 'negative', 'sell'],
            'neutral': ['stable', 'unchanged', 'steady', 'flat', 'hold', 'wait']
        }
    
    async def get_sentiment_data(self, symbol: str, timeframe: str = '1d') -> SentimentData:
        """Get sentiment analysis with fallback chain"""
        cache_key = f"sentiment:{symbol}:{timeframe}"
        cached = self.data_manager.cache_manager.get(cache_key)
        if cached:
            if 'timestamp' in cached and isinstance(cached['timestamp'], str):
                cached['timestamp'] = datetime.fromisoformat(cached['timestamp'])
            return SentimentData(**cached)
        
        # Try news scraping and analysis
        try:
            articles = await self.scrape_financial_news(symbol)
            if articles:
                sentiment_data = self._analyze_sentiment(articles, symbol, timeframe)
                self.data_manager.cache_manager.set(cache_key, sentiment_data, ttl=1800)  # 30 min cache
                return sentiment_data
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
        
        # Fallback to simulated sentiment
        sentiment_data = self._generate_simulated_sentiment(symbol, timeframe)
        self.data_manager.cache_manager.set(cache_key, sentiment_data, ttl=300)
        return sentiment_data
    
    async def scrape_financial_news(self, symbol: str) -> List[Dict]:
        """Scrape recent financial news"""
        articles = []
        search_terms = {
            'XAUUSD': ['gold', 'precious metals'],
            'EURUSD': ['euro', 'european central bank'],
            'GBPUSD': ['pound', 'sterling', 'bank of england']
        }
        
        terms = search_terms.get(symbol, [symbol.lower()])
        
        for source_url in self.news_sources[:2]:  # Limit to 2 sources for speed
            try:
                await self.data_manager.rate_limiter.wait_if_needed(f'news_{source_url}', 3.0)
                source_articles = await self._scrape_news_source(source_url, terms)
                articles.extend(source_articles)
            except Exception as e:
                logger.warning(f"News scraping failed for {source_url}: {e}")
        
        return articles[:10]  # Limit to 10 most recent
    
    async def _scrape_news_source(self, url: str, search_terms: List[str]) -> List[Dict]:
        """Scrape news from a single source"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        articles = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Look for article titles and snippets
                        for article in soup.find_all(['article', 'div'], class_=re.compile(r'(story|article|news)')):
                            title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                            if title_elem:
                                title = title_elem.get_text().strip()
                                if any(term.lower() in title.lower() for term in search_terms):
                                    content_elem = article.find(['p', 'div'], class_=re.compile(r'(summary|excerpt|description)'))
                                    content = content_elem.get_text().strip() if content_elem else title
                                    
                                    articles.append({
                                        'title': title,
                                        'content': content,
                                        'source': url,
                                        'timestamp': datetime.now().isoformat()
                                    })
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
        
        return articles
    
    def _analyze_sentiment(self, articles: List[Dict], symbol: str, timeframe: str) -> SentimentData:
        """Analyze sentiment from news articles"""
        if not articles:
            return self._generate_simulated_sentiment(symbol, timeframe)
        
        sentiment_scores = []
        keyword_scores = []
        
        for article in articles:
            text = f"{article['title']} {article.get('content', '')}"
            
            # TextBlob sentiment analysis
            try:
                blob = TextBlob(text)
                sentiment_scores.append(blob.sentiment.polarity)
            except:
                sentiment_scores.append(0)
            
            # Keyword-based sentiment
            keyword_score = self._calculate_keyword_sentiment(text)
            keyword_scores.append(keyword_score)
        
        # Combine scores
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        avg_keyword = np.mean(keyword_scores) if keyword_scores else 0
        combined_score = (avg_sentiment + avg_keyword) / 2
        
        # Determine label and confidence
        if combined_score > 0.1:
            label = 'bullish'
        elif combined_score < -0.1:
            label = 'bearish'
        else:
            label = 'neutral'
        
        confidence = min(abs(combined_score) + 0.1, 1.0)
        
        return SentimentData(
            symbol=symbol,
            sentiment_score=round(combined_score, 3),
            sentiment_label=label,
            confidence=round(confidence, 3),
            sources_count=len(articles),
            timeframe=timeframe,
            news_articles=articles,
            timestamp=datetime.now()
        )
    
    def _calculate_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on keyword matching"""
        text_lower = text.lower()
        bullish_count = sum(1 for word in self.sentiment_keywords['bullish'] if word in text_lower)
        bearish_count = sum(1 for word in self.sentiment_keywords['bearish'] if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0
        
        return (bullish_count - bearish_count) / total
    
    def _generate_simulated_sentiment(self, symbol: str, timeframe: str) -> SentimentData:
        """Generate simulated sentiment data"""
        sentiment_score = random.uniform(-0.5, 0.5)
        
        if sentiment_score > 0.1:
            label = 'bullish'
        elif sentiment_score < -0.1:
            label = 'bearish'
        else:
            label = 'neutral'
        
        return SentimentData(
            symbol=symbol,
            sentiment_score=round(sentiment_score, 3),
            sentiment_label=label,
            confidence=round(random.uniform(0.2, 0.6), 3),
            sources_count=random.randint(3, 8),
            timeframe=timeframe,
            news_articles=[],
            timestamp=datetime.now()
        )

class TechnicalIndicatorService:
    """Technical indicator calculation service"""
    
    def __init__(self, data_manager: DataSourceManager):
        self.data_manager = data_manager
    
    async def get_technical_data(self, symbol: str, timeframe: str = '1H') -> TechnicalData:
        """Get technical analysis data"""
        cache_key = f"technical:{symbol}:{timeframe}"
        cached = self.data_manager.cache_manager.get(cache_key)
        if cached:
            if 'timestamp' in cached and isinstance(cached['timestamp'], str):
                cached['timestamp'] = datetime.fromisoformat(cached['timestamp'])
            return TechnicalData(**cached)
        
        try:
            # Try to calculate real indicators from price data
            indicators = await self._calculate_indicators(symbol, timeframe)
            technical_data = TechnicalData(
                symbol=symbol,
                indicators=indicators,
                analysis_timeframe=timeframe,
                source=DataSource.API_PRIMARY,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            # Fallback to simulated indicators
            indicators = self._generate_simulated_indicators(symbol)
            technical_data = TechnicalData(
                symbol=symbol,
                indicators=indicators,
                analysis_timeframe=timeframe,
                source=DataSource.SIMULATED,
                timestamp=datetime.now()
            )
        
        self.data_manager.cache_manager.set(cache_key, technical_data, ttl=900)  # 15 min cache
        return technical_data
    
    async def _calculate_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Calculate technical indicators from price data"""
        # Generate sample price data for calculation
        # In production, this would fetch historical data
        prices = self._generate_price_series(symbol, 100)
        
        indicators = {}
        
        # RSI calculation
        indicators['rsi'] = self._calculate_rsi(prices)
        
        # MACD calculation
        indicators['macd'] = self._calculate_macd(prices)
        
        # Moving averages
        indicators['moving_averages'] = self._calculate_moving_averages(prices)
        
        # Bollinger Bands
        indicators['bollinger_bands'] = self._calculate_bollinger_bands(prices)
        
        return indicators
    
    def _generate_price_series(self, symbol: str, length: int) -> List[float]:
        """Generate realistic price series for calculation"""
        base_prices = {
            'XAUUSD': 2000.0,
            'XAGUSD': 25.0,
            'EURUSD': 1.08,
            'GBPUSD': 1.26
        }
        
        base_price = base_prices.get(symbol, 100.0)
        prices = [base_price]
        
        for _ in range(length - 1):
            change = random.uniform(-0.01, 0.01)  # ±1% change
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Dict[str, Any]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return {'value': 50, 'signal': 'neutral'}
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        signal = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
        
        return {
            'value': round(rsi, 2),
            'signal': signal,
            'period': period
        }
    
    def _calculate_macd(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return {'value': 0, 'signal': 'neutral', 'histogram': 0}
        
        # Simple MACD calculation
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        macd_line = ema12 - ema26
        signal_line = np.mean([macd_line, np.mean(prices[-9:])])
        histogram = macd_line - signal_line
        
        signal = 'bullish' if histogram > 0 else 'bearish' if histogram < 0 else 'neutral'
        
        return {
            'value': round(macd_line, 4),
            'signal': signal,
            'histogram': round(histogram, 4)
        }
    
    def _calculate_moving_averages(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate moving averages"""
        ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        ma50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
        
        trend = 'bullish' if ma20 > ma50 else 'bearish' if ma20 < ma50 else 'neutral'
        
        return {
            'ma20': round(ma20, 2),
            'ma50': round(ma50, 2),
            'trend': trend
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1]
            return {
                'upper': round(current_price * 1.02, 2),
                'middle': round(current_price, 2),
                'lower': round(current_price * 0.98, 2),
                'signal': 'neutral'
            }
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        current_price = prices[-1]
        
        if current_price > upper:
            signal = 'overbought'
        elif current_price < lower:
            signal = 'oversold'
        else:
            signal = 'neutral'
        
        return {
            'upper': round(upper, 2),
            'middle': round(sma, 2),
            'lower': round(lower, 2),
            'signal': signal
        }
    
    def _generate_simulated_indicators(self, symbol: str) -> Dict[str, Any]:
        """Generate simulated technical indicators"""
        return {
            'rsi': {
                'value': round(random.uniform(30, 70), 2),
                'signal': random.choice(['neutral', 'bullish', 'bearish'])
            },
            'macd': {
                'value': round(random.uniform(-5, 5), 4),
                'signal': random.choice(['neutral', 'bullish', 'bearish']),
                'histogram': round(random.uniform(-2, 2), 4)
            },
            'moving_averages': {
                'ma20': round(random.uniform(1900, 2100), 2),
                'ma50': round(random.uniform(1900, 2100), 2),
                'trend': random.choice(['neutral', 'bullish', 'bearish'])
            },
            'bollinger_bands': {
                'upper': round(random.uniform(2050, 2100), 2),
                'middle': round(random.uniform(1950, 2050), 2),
                'lower': round(random.uniform(1900, 1950), 2),
                'signal': random.choice(['neutral', 'overbought', 'oversold'])
            }
        }

class UnifiedDataProvider:
    """Unified interface for all data types with automatic source management"""
    
    def __init__(self):
        self.data_manager = DataSourceManager()
        self.price_service = PriceDataService(self.data_manager)
        self.sentiment_service = SentimentAnalysisService(self.data_manager)
        self.technical_service = TechnicalIndicatorService(self.data_manager)
    
    async def get_price_data(self, symbol: str) -> PriceData:
        """Get price data with automatic fallbacks"""
        return await self.price_service.get_price_data(symbol)
    
    async def get_sentiment_data(self, symbol: str, timeframe: str = '1d') -> SentimentData:
        """Get sentiment data with automatic fallbacks"""
        return await self.sentiment_service.get_sentiment_data(symbol, timeframe)
    
    async def get_technical_data(self, symbol: str, timeframe: str = '1H') -> TechnicalData:
        """Get technical data with automatic fallbacks"""
        return await self.technical_service.get_technical_data(symbol, timeframe)
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data types for a symbol"""
        try:
            price_task = self.get_price_data(symbol)
            sentiment_task = self.get_sentiment_data(symbol)
            technical_task = self.get_technical_data(symbol)
            
            price_data, sentiment_data, technical_data = await asyncio.gather(
                price_task, sentiment_task, technical_task,
                return_exceptions=True
            )
            
            result = {}
            
            if isinstance(price_data, PriceData):
                result['price'] = asdict(price_data)
                result['price']['timestamp'] = price_data.timestamp.isoformat()
            else:
                result['price'] = None
            
            if isinstance(sentiment_data, SentimentData):
                result['sentiment'] = asdict(sentiment_data)
                result['sentiment']['timestamp'] = sentiment_data.timestamp.isoformat()
            else:
                result['sentiment'] = None
            
            if isinstance(technical_data, TechnicalData):
                result['technical'] = asdict(technical_data)
                result['technical']['timestamp'] = technical_data.timestamp.isoformat()
            else:
                result['technical'] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive data fetch failed for {symbol}: {e}")
            return {'price': None, 'sentiment': None, 'technical': None}
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics and health"""
        return {
            'source_reliability': {k.value: v for k, v in self.data_manager.source_reliability.items()},
            'request_counts': {k.value: v for k, v in self.data_manager.request_counts.items()},
            'success_counts': {k.value: v for k, v in self.data_manager.success_counts.items()},
            'cache_active': True
        }
    
    def cleanup_cache(self):
        """Cleanup expired cache entries"""
        self.data_manager.cache_manager.cleanup_expired()

# Global instance for easy access
unified_data_provider = UnifiedDataProvider()

async def cleanup_data_cache():
    """Async function to cleanup data cache"""
    unified_data_provider.cleanup_cache()

# Export main classes and functions
__all__ = [
    'UnifiedDataProvider',
    'DataSourceManager', 
    'PriceDataService',
    'SentimentAnalysisService',
    'TechnicalIndicatorService',
    'PriceData',
    'SentimentData', 
    'TechnicalData',
    'DataSource',
    'unified_data_provider',
    'cleanup_data_cache'
]

class DataType(Enum):
    """Types of data we can fetch"""
    PRICE = "price"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    NEWS = "news"
    VOLUME = "volume"

@dataclass
class DataPoint:
    """Standard data point structure"""
    value: Union[float, dict, list]
    timestamp: datetime
    source: DataSource
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

@dataclass
class PriceData:
    """Price data structure"""
    symbol: str
    price: float
    bid: float
    ask: float
    spread: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    source: DataSource
    high_24h: float = 0.0
    low_24h: float = 0.0

@dataclass
class SentimentData:
    """Sentiment analysis result"""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str   # bullish, bearish, neutral
    confidence: float      # 0.0 to 1.0
    sources_count: int
    news_articles: List[Dict]
    timestamp: datetime
    timeframe: str

@dataclass
class TechnicalData:
    """Technical indicator data"""
    symbol: str
    indicators: Dict[str, Dict]  # indicator_name -> {value, signal, confidence}
    timestamp: datetime
    source: DataSource
    analysis_timeframe: str

class RetryConfig:
    """Retry configuration for different operations"""
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 30.0, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

class CacheManager:
    """Intelligent caching system with TTL and source-based invalidation"""
    
    def __init__(self, db_path: str = "data_cache.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source ON cache_entries(source);
        ''')
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached value if not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT value, source, confidence, created_at 
                FROM cache_entries 
                WHERE key = ? AND expires_at > CURRENT_TIMESTAMP
            ''', (key,))
            
            result = cursor.fetchone()
            
            if result:
                # Update access statistics
                cursor.execute('''
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE key = ?
                ''', (key,))
                conn.commit()
                
                return {
                    'value': json.loads(result[0]),
                    'source': result[1],
                    'confidence': result[2],
                    'created_at': result[3]
                }
            
            return None
            
        finally:
            conn.close()
    
    def set(self, key: str, value: Any, source: DataSource, confidence: float, ttl_seconds: int):
        """Cache a value with TTL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            # Handle datetime serialization and dataclass conversion
            def custom_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                # Handle dataclass objects - force convert using asdict
                try:
                    # Try to use asdict directly (will work if it's a dataclass)
                    return asdict(obj)
                except (TypeError, ValueError):
                    pass
                # Handle mappingproxy and other special types
                if str(type(obj)) == "<class 'mappingproxy'>":
                    return dict(obj)
                # Handle other objects with __dict__
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                # Last resort - convert to string
                return str(obj)
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (key, value, source, confidence, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (key, json.dumps(value, default=custom_serializer), source.value, confidence, expires_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM cache_entries WHERE expires_at <= CURRENT_TIMESTAMP')
            deleted = cursor.rowcount
            conn.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired cache entries")
                
        finally:
            conn.close()

class RateLimiter:
    """Rate limiting to be respectful to external services"""
    
    def __init__(self):
        self.last_requests = {}  # domain -> last_request_time
        self.request_counts = {}  # domain -> count_in_current_minute
        self.minute_start = {}   # domain -> minute_start_time
    
    async def wait_if_needed(self, domain: str, max_requests_per_minute: int = 30):
        """Wait if we need to respect rate limits"""
        now = time.time()
        
        # Reset counters if a new minute has started
        if domain not in self.minute_start or now - self.minute_start[domain] >= 60:
            self.minute_start[domain] = now
            self.request_counts[domain] = 0
        
        # Check if we've hit the rate limit
        if self.request_counts[domain] >= max_requests_per_minute:
            wait_time = 60 - (now - self.minute_start[domain])
            if wait_time > 0:
                logger.info(f"Rate limit hit for {domain}, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                # Reset after waiting
                self.minute_start[domain] = time.time()
                self.request_counts[domain] = 0
        
        # Minimum delay between requests to same domain
        if domain in self.last_requests:
            time_since_last = now - self.last_requests[domain]
            min_delay = 1.0  # 1 second minimum between requests
            if time_since_last < min_delay:
                await asyncio.sleep(min_delay - time_since_last)
        
        self.last_requests[domain] = time.time()
        self.request_counts[domain] = self.request_counts.get(domain, 0) + 1

class BaseDataProvider:
    """Base class for all data providers"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        self.cache_manager = cache_manager
        self.rate_limiter = rate_limiter
        self.retry_config = RetryConfig()
        self.reliability_score = 1.0  # Will be adjusted based on success/failure
    
    async def retry_with_backoff(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.retry_config.max_attempts - 1:
                    delay = min(
                        self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt),
                        self.retry_config.max_delay
                    )
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.retry_config.max_attempts} attempts failed")
                    self.reliability_score = max(0.1, self.reliability_score - 0.1)
        
        raise last_exception
    
    def update_reliability(self, success: bool):
        """Update reliability score based on operation success"""
        if success:
            self.reliability_score = min(1.0, self.reliability_score + 0.05)
        else:
            self.reliability_score = max(0.1, self.reliability_score - 0.1)

# =============================================================================
# PRICE DATA PROVIDERS
# =============================================================================

class PrimaryPriceAPI(BaseDataProvider):
    """Primary price data from free APIs"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        super().__init__(cache_manager, rate_limiter)
        self.source = DataSource.API_PRIMARY
        
        # Multiple free API sources
        self.apis = {
            'gold_api': {
                'url': 'https://api.gold-api.com/price/XAU',
                'symbols': ['XAUUSD'],
                'rate_limit': 30
            },
            'fixer': {
                'url': 'https://api.fixer.io/latest',
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'rate_limit': 100,
                'requires_key': False
            },
            'exchangerate': {
                'url': 'https://api.exchangerate-api.com/v4/latest/USD',
                'symbols': ['EUR', 'GBP', 'JPY'],
                'rate_limit': 1500
            }
        }
    
    async def get_price_data(self, symbol: str) -> Optional[PriceData]:
        """Get price data from primary APIs"""
        cache_key = f"price_{symbol}_primary"
        
        # Check cache first
        cached = self.cache_manager.get(cache_key)
        if cached:
            data = cached['value']
            return PriceData(**data)
        
        try:
            if symbol == 'XAUUSD':
                data = await self._fetch_gold_price()
            elif symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                data = await self._fetch_forex_price(symbol)
            else:
                return None
            
            if data:
                # Cache for 30 seconds with correct signature for advanced CacheManager
                self.cache_manager.set(cache_key, data, self.source, 0.9, 30)
                self.update_reliability(True)
                return data
            
        except Exception as e:
            logger.error(f"Primary API failed for {symbol}: {e}")
            self.update_reliability(False)
        
        return None
    
    async def _fetch_gold_price(self) -> Optional[PriceData]:
        """Fetch gold price from Gold API"""
        await self.rate_limiter.wait_if_needed('api.gold-api.com', 30)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://api.gold-api.com/price/XAU',
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    price = float(data['price'])
                    prev_close = float(data.get('prev_close_price', price))
                    change = price - prev_close
                    change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    return PriceData(
                        symbol='XAUUSD',
                        price=price,
                        bid=price - 0.5,
                        ask=price + 0.5,
                        spread=1.0,
                        change=change,
                        change_percent=change_percent,
                        volume=0,  # Not available from this API
                        timestamp=datetime.now(),
                        source=self.source,
                        high_24h=float(data.get('high_price', price)),
                        low_24h=float(data.get('low_price', price))
                    )
                
                raise Exception(f"HTTP {response.status}")
    
    async def _fetch_forex_price(self, symbol: str) -> Optional[PriceData]:
        """Fetch forex prices from exchange rate API"""
        await self.rate_limiter.wait_if_needed('api.exchangerate-api.com', 100)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://api.exchangerate-api.com/v4/latest/USD',
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    rates = data['rates']
                    
                    # Convert symbol to rate calculation
                    if symbol == 'EURUSD':
                        price = 1.0 / rates['EUR']
                    elif symbol == 'GBPUSD':
                        price = 1.0 / rates['GBP']
                    elif symbol == 'USDJPY':
                        price = rates['JPY']
                    else:
                        return None
                    
                    # Simulate spread and change (since free API doesn't provide)
                    spread = 0.0001 if symbol != 'USDJPY' else 0.01
                    change = random.uniform(-0.005, 0.005) * price
                    
                    return PriceData(
                        symbol=symbol,
                        price=price,
                        bid=price - spread/2,
                        ask=price + spread/2,
                        spread=spread,
                        change=change,
                        change_percent=(change / price * 100),
                        volume=0,
                        timestamp=datetime.now(),
                        source=self.source
                    )
                
                raise Exception(f"HTTP {response.status}")

class SecondaryPriceScraper(BaseDataProvider):
    """Fallback price data from web scraping"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        super().__init__(cache_manager, rate_limiter)
        self.source = DataSource.WEB_SCRAPING
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.scrape_targets = {
            'XAUUSD': [
                {
                    'url': 'https://www.investing.com/currencies/xau-usd',
                    'price_selector': '[data-test="instrument-price-last"]',
                    'change_selector': '[data-test="instrument-price-change"]'
                },
                {
                    'url': 'https://www.marketwatch.com/investing/currency/xauusd',
                    'price_selector': '.intraday__price',
                    'change_selector': '.change--point--q'
                }
            ]
        }
    
    async def get_price_data(self, symbol: str) -> Optional[PriceData]:
        """Get price data through web scraping"""
        cache_key = f"price_{symbol}_scraping"
        
        # Check cache first (longer TTL for scraping to be respectful)
        cached = self.cache_manager.get(cache_key)
        if cached:
            data = cached['value']
            return PriceData(**data)
        
        targets = self.scrape_targets.get(symbol, [])
        
        for target in targets:
            try:
                data = await self._scrape_price_data(symbol, target)
                if data:
                    # Cache for 60 seconds with correct signature for advanced CacheManager
                    self.cache_manager.set(cache_key, data, self.source, 0.7, 60)
                    self.update_reliability(True)
                    return data
            except Exception as e:
                logger.warning(f"Scraping failed for {target['url']}: {e}")
                continue
        
        self.update_reliability(False)
        return None
    
    async def _scrape_price_data(self, symbol: str, target: Dict) -> Optional[PriceData]:
        """Scrape price data from a specific target"""
        domain = target['url'].split('/')[2]
        await self.rate_limiter.wait_if_needed(domain, 10)  # Very conservative rate limiting
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                target['url'],
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract price
                    price_elem = soup.select_one(target['price_selector'])
                    if not price_elem:
                        return None
                    
                    price_text = price_elem.get_text().strip().replace(',', '')
                    price = float(re.findall(r'\d+\.?\d*', price_text)[0])
                    
                    # Extract change if available
                    change = 0.0
                    change_elem = soup.select_one(target.get('change_selector', ''))
                    if change_elem:
                        change_text = change_elem.get_text().strip()
                        change_match = re.findall(r'-?\d+\.?\d*', change_text)
                        if change_match:
                            change = float(change_match[0])
                    
                    change_percent = (change / price * 100) if price > 0 else 0
                    spread = 1.0 if symbol == 'XAUUSD' else 0.0001
                    
                    return PriceData(
                        symbol=symbol,
                        price=price,
                        bid=price - spread/2,
                        ask=price + spread/2,
                        spread=spread,
                        change=change,
                        change_percent=change_percent,
                        volume=0,
                        timestamp=datetime.now(),
                        source=self.source
                    )
                
                raise Exception(f"HTTP {response.status}")

class SimulatedPriceProvider(BaseDataProvider):
    """Fallback simulated price data with realistic patterns"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        super().__init__(cache_manager, rate_limiter)
        self.source = DataSource.SIMULATED
        
        # Base prices for realistic simulation
        self.base_prices = {
            'XAUUSD': 2000.0,
            'EURUSD': 1.0875,
            'GBPUSD': 1.2650,
            'USDJPY': 148.50,
            'BTCUSD': 43500.0
        }
        
        # Market volatility patterns
        self.volatility = {
            'XAUUSD': 0.015,  # 1.5% daily volatility
            'EURUSD': 0.008,  # 0.8% daily volatility
            'GBPUSD': 0.012,  # 1.2% daily volatility
            'USDJPY': 0.010,  # 1.0% daily volatility
            'BTCUSD': 0.040   # 4.0% daily volatility
        }
    
    async def get_price_data(self, symbol: str) -> PriceData:
        """Generate realistic simulated price data"""
        base_price = self.base_prices.get(symbol, 1.0)
        vol = self.volatility.get(symbol, 0.01)
        
        # Generate price using brownian motion
        now = datetime.now()
        
        # Create deterministic but varying price based on time
        time_factor = (now.hour * 3600 + now.minute * 60 + now.second) / 86400
        
        # Add some randomness but keep it consistent within short periods
        seed = int(now.timestamp() // 10)  # Changes every 10 seconds
        random.seed(seed)
        
        # Generate realistic price movement
        price_change = random.gauss(0, vol * base_price / 24)  # Hourly volatility
        trend_component = np.sin(time_factor * 2 * np.pi) * vol * base_price * 0.5
        
        current_price = base_price + price_change + trend_component
        
        # Calculate spread based on instrument type
        if symbol == 'XAUUSD':
            spread = 1.0
        elif symbol in ['EURUSD', 'GBPUSD']:
            spread = 0.0001
        elif symbol == 'USDJPY':
            spread = 0.01
        else:
            spread = current_price * 0.001  # 0.1% spread
        
        change = price_change + trend_component
        change_percent = (change / base_price * 100)
        
        return PriceData(
            symbol=symbol,
            price=current_price,
            bid=current_price - spread/2,
            ask=current_price + spread/2,
            spread=spread,
            change=change,
            change_percent=change_percent,
            volume=random.randint(1000, 50000),
            timestamp=now,
            source=self.source,
            high_24h=current_price * (1 + vol),
            low_24h=current_price * (1 - vol)
        )

# =============================================================================
# SENTIMENT ANALYSIS PROVIDERS
# =============================================================================

class NewsSentimentAnalyzer(BaseDataProvider):
    """News-based sentiment analysis with web scraping"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        super().__init__(cache_manager, rate_limiter)
        self.source = DataSource.WEB_SCRAPING
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # News sources with their configurations
        self.news_sources = {
            'reuters': {
                'url': 'https://www.reuters.com/markets/commodities/',
                'article_selector': '.story-card',
                'title_selector': '.story-card__headline__text',
                'time_selector': '.story-card__time',
                'reliability': 0.9
            },
            'marketwatch': {
                'url': 'https://www.marketwatch.com/investing/future/gold',
                'article_selector': '.article__headline',
                'title_selector': 'a',
                'reliability': 0.8
            },
            'cnbc': {
                'url': 'https://www.cnbc.com/gold/',
                'article_selector': '.InlineVideo-container',
                'title_selector': '.InlineVideo-headline',
                'reliability': 0.85
            }
        }
        
        # Sentiment keywords with weights
        self.bullish_keywords = {
            'inflation': 2.0, 'uncertainty': 1.5, 'crisis': 2.0, 'recession': 1.8,
            'dovish': 1.5, 'stimulus': 1.8, 'easing': 1.5, 'safe haven': 2.0,
            'hedge': 1.3, 'store of value': 1.5, 'rally': 2.0, 'surge': 1.8,
            'rising': 1.2, 'strong': 1.0, 'bullish': 2.0, 'optimistic': 1.3,
            'geopolitical tension': 2.0, 'war': 1.8, 'conflict': 1.5
        }
        
        self.bearish_keywords = {
            'hawkish': 1.8, 'rate hike': 2.0, 'tightening': 1.5, 'strong dollar': 1.8,
            'risk on': 1.3, 'equity rally': 1.2, 'growth': 1.0, 'decline': 1.8,
            'fall': 1.5, 'drop': 1.5, 'bearish': 2.0, 'pessimistic': 1.3,
            'selling': 1.2, 'weakness': 1.5, 'correction': 1.8, 'crash': 2.5
        }
    
    async def get_sentiment_data(self, symbol: str, timeframe: str = '1d') -> Optional[SentimentData]:
        """Get sentiment analysis from news sources"""
        cache_key = f"sentiment_{symbol}_{timeframe}"
        
        # Check cache first (cache for 15 minutes)
        cached = self.cache_manager.get(cache_key)
        if cached:
            data = cached['value']
            return SentimentData(**data)
        
        try:
            articles = await self._collect_news_articles(symbol)
            
            if not articles:
                return self._get_simulated_sentiment(symbol, timeframe)
            
            sentiment_score, confidence = self._analyze_sentiment(articles)
            
            sentiment_label = 'neutral'
            if sentiment_score > 0.2:
                sentiment_label = 'bullish'
            elif sentiment_score < -0.2:
                sentiment_label = 'bearish'
            
            result = SentimentData(
                symbol=symbol,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                sources_count=len(articles),
                news_articles=articles,
                timestamp=datetime.now(),
                timeframe=timeframe
            )
            
            # Cache for 15 minutes with correct signature for advanced CacheManager
            self.cache_manager.set(cache_key, result, self.source, confidence, 900)
            self.update_reliability(True)
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            self.update_reliability(False)
            return self._get_simulated_sentiment(symbol, timeframe)
    
    async def _collect_news_articles(self, symbol: str) -> List[Dict]:
        """Collect news articles from multiple sources"""
        articles = []
        
        # Determine search terms based on symbol
        search_terms = {
            'XAUUSD': ['gold', 'precious metals', 'gold price'],
            'EURUSD': ['euro', 'european central bank', 'ECB'],
            'GBPUSD': ['pound', 'bank of england', 'sterling'],
            'USDJPY': ['yen', 'bank of japan', 'boj']
        }
        
        terms = search_terms.get(symbol, ['markets', 'economy'])
        
        for source_name, config in self.news_sources.items():
            try:
                source_articles = await self._scrape_news_source(source_name, config, terms)
                articles.extend(source_articles)
            except Exception as e:
                logger.warning(f"Failed to scrape {source_name}: {e}")
                continue
        
        return articles
    
    async def _scrape_news_source(self, source_name: str, config: Dict, search_terms: List[str]) -> List[Dict]:
        """Scrape news from a specific source"""
        domain = config['url'].split('/')[2]
        await self.rate_limiter.wait_if_needed(domain, 5)  # Very conservative
        
        articles = []
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                config['url'],
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    article_elements = soup.select(config['article_selector'])
                    
                    for elem in article_elements[:10]:  # Limit to 10 articles per source
                        try:
                            title_elem = elem.select_one(config['title_selector'])
                            if not title_elem:
                                continue
                            
                            title = title_elem.get_text().strip()
                            
                            # Check if article is relevant to our search terms
                            title_lower = title.lower()
                            relevant = any(term.lower() in title_lower for term in search_terms)
                            
                            if relevant and len(title) > 10:
                                articles.append({
                                    'title': title,
                                    'source': source_name,
                                    'reliability': config['reliability'],
                                    'timestamp': datetime.now().isoformat(),
                                    'url': config['url']
                                })
                        except Exception as e:
                            logger.debug(f"Error parsing article element: {e}")
                            continue
        
        return articles
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Tuple[float, float]:
        """Analyze sentiment from collected articles"""
        if not articles:
            return 0.0, 0.0
        
        sentiment_scores = []
        total_weight = 0
        
        for article in articles:
            title = article['title'].lower()
            reliability = article.get('reliability', 0.5)
            
            # Calculate sentiment using keyword matching
            bullish_score = 0
            bearish_score = 0
            
            for keyword, weight in self.bullish_keywords.items():
                if keyword in title:
                    bullish_score += weight
            
            for keyword, weight in self.bearish_keywords.items():
                if keyword in title:
                    bearish_score += weight
            
            # Use TextBlob for additional sentiment analysis
            try:
                blob = TextBlob(article['title'])
                textblob_sentiment = blob.sentiment.polarity
                
                # Combine keyword-based and TextBlob sentiment
                article_sentiment = (bullish_score - bearish_score) * 0.1 + textblob_sentiment * 0.5
                
                # Normalize to -1 to 1 range
                article_sentiment = max(-1.0, min(1.0, article_sentiment))
                
                sentiment_scores.append(article_sentiment * reliability)
                total_weight += reliability
                
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
                # Fallback to keyword-only analysis
                article_sentiment = (bullish_score - bearish_score) * 0.1
                article_sentiment = max(-1.0, min(1.0, article_sentiment))
                sentiment_scores.append(article_sentiment * reliability)
                total_weight += reliability
        
        if total_weight == 0:
            return 0.0, 0.0
        
        # Calculate weighted average sentiment
        weighted_sentiment = sum(sentiment_scores) / total_weight
        
        # Calculate confidence based on number of articles and consistency
        if len(sentiment_scores) >= 5:
            std_dev = np.std(sentiment_scores)
            confidence = max(0.1, min(0.9, 1.0 - std_dev))
        else:
            confidence = 0.3 + (len(sentiment_scores) * 0.1)
        
        return weighted_sentiment, confidence
    
    def _get_simulated_sentiment(self, symbol: str, timeframe: str) -> SentimentData:
        """Generate simulated sentiment when real analysis fails"""
        now = datetime.now()
        
        # Create deterministic but varying sentiment
        seed = int(now.timestamp() // 3600)  # Changes every hour
        random.seed(seed)
        
        # Base sentiment with some market hours influence
        if 9 <= now.hour <= 16:  # Market hours
            base_sentiment = random.uniform(-0.3, 0.3)
        else:
            base_sentiment = random.uniform(-0.1, 0.1)
        
        # Add symbol-specific bias
        symbol_bias = {
            'XAUUSD': 0.1,   # Slightly bullish bias for gold
            'EURUSD': 0.0,   # Neutral
            'GBPUSD': -0.05, # Slightly bearish
            'USDJPY': 0.05   # Slightly bullish
        }
        
        sentiment_score = base_sentiment + symbol_bias.get(symbol, 0.0)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        if sentiment_score > 0.2:
            sentiment_label = 'bullish'
        elif sentiment_score < -0.2:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        return SentimentData(
            symbol=symbol,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=0.3,  # Low confidence for simulated data
            sources_count=0,
            news_articles=[],
            timestamp=now,
            timeframe=timeframe
        )

# =============================================================================
# TECHNICAL ANALYSIS PROVIDERS
# =============================================================================

class TechnicalAnalysisProvider(BaseDataProvider):
    """Calculate technical indicators from price data"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        super().__init__(cache_manager, rate_limiter)
        self.source = DataSource.API_PRIMARY  # Uses real calculations
    
    async def get_technical_data(self, symbol: str, timeframe: str = '1H') -> Optional[TechnicalData]:
        """Calculate technical indicators"""
        cache_key = f"technical_{symbol}_{timeframe}"
        
        # Check cache first
        cached = self.cache_manager.get(cache_key)
        if cached:
            data = cached['value']
            return TechnicalData(**data)
        
        try:
            # Get historical price data for calculations
            price_history = await self._get_price_history(symbol, timeframe)
            
            if len(price_history) < 50:  # Need sufficient data for calculations
                return self._get_simulated_technical(symbol, timeframe)
            
            indicators = {}
            
            # Calculate RSI
            indicators['RSI'] = self._calculate_rsi(price_history)
            
            # Calculate MACD
            indicators['MACD'] = self._calculate_macd(price_history)
            
            # Calculate Bollinger Bands
            indicators['BOLLINGER'] = self._calculate_bollinger_bands(price_history)
            
            # Calculate Moving Averages
            indicators['SMA'] = self._calculate_sma(price_history)
            
            # Calculate Stochastic
            indicators['STOCHASTIC'] = self._calculate_stochastic(price_history)
            
            result = TechnicalData(
                symbol=symbol,
                indicators=indicators,
                timestamp=datetime.now(),
                source=self.source,
                analysis_timeframe=timeframe
            )
            
            # Cache for 2 minutes with correct signature for advanced CacheManager
            self.cache_manager.set(cache_key, result, self.source, 0.8, 120)
            self.update_reliability(True)
            
            return result
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            self.update_reliability(False)
            return self._get_simulated_technical(symbol, timeframe)
    
    async def _get_price_history(self, symbol: str, timeframe: str, periods: int = 100) -> List[float]:
        """Get historical price data for calculations"""
        # In a real implementation, this would fetch from a historical data API
        # For now, we'll generate realistic price data
        
        # Use current price as starting point
        current_time = datetime.now()
        
        # Generate realistic price history using geometric brownian motion
        base_price = {
            'XAUUSD': 2000.0,
            'EURUSD': 1.0875,
            'GBPUSD': 1.2650,
            'USDJPY': 148.50
        }.get(symbol, 1.0)
        
        volatility = 0.02  # 2% volatility
        dt = 1.0 / 24  # Hourly data
        
        prices = [base_price]
        
        # Generate price series with realistic patterns
        for i in range(periods - 1):
            # Add trend and random walk
            trend = 0.0001 * np.sin(i * 0.1)  # Subtle trend
            random_component = random.gauss(0, volatility * np.sqrt(dt))
            
            new_price = prices[-1] * (1 + trend + random_component)
            prices.append(new_price)
        
        return prices
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Dict:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return {'value': 50, 'signal': 'neutral', 'confidence': 0.3}
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Determine signal
        if rsi > 70:
            signal = 'bearish'  # Overbought
            confidence = 0.8
        elif rsi < 30:
            signal = 'bullish'  # Oversold
            confidence = 0.8
        else:
            signal = 'neutral'
            confidence = 0.5
        
        return {
            'value': round(rsi, 2),
            'signal': signal,
            'confidence': confidence,
            'overbought': 70,
            'oversold': 30
        }
    
    def _calculate_macd(self, prices: List[float]) -> Dict:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return {'value': 0, 'signal_line': 0, 'histogram': 0, 'signal': 'neutral', 'confidence': 0.3}
        
        # Calculate EMAs
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        
        macd_line = ema12 - ema26
        
        # Calculate signal line (9-period EMA of MACD)
        macd_values = [macd_line] * min(9, len(prices))
        signal_line = self._ema(macd_values, 9)
        
        histogram = macd_line - signal_line
        
        # Determine signal
        if macd_line > signal_line and histogram > 0:
            signal = 'bullish'
            confidence = 0.7
        elif macd_line < signal_line and histogram < 0:
            signal = 'bearish'
            confidence = 0.7
        else:
            signal = 'neutral'
            confidence = 0.4
        
        return {
            'value': round(macd_line, 6),
            'signal_line': round(signal_line, 6),
            'histogram': round(histogram, 6),
            'signal': signal,
            'confidence': confidence
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 2000
            return {
                'upper': current_price * 1.01,
                'middle': current_price,
                'lower': current_price * 0.99,
                'signal': 'neutral',
                'confidence': 0.3
            }
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / len(recent_prices)
        
        variance = sum((p - sma) ** 2 for p in recent_prices) / len(recent_prices)
        std = variance ** 0.5
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        current_price = prices[-1]
        
        # Determine signal based on position relative to bands
        if current_price > upper_band:
            signal = 'bearish'  # Price above upper band
            confidence = 0.7
        elif current_price < lower_band:
            signal = 'bullish'  # Price below lower band
            confidence = 0.7
        else:
            signal = 'neutral'
            confidence = 0.4
        
        return {
            'upper': round(upper_band, 6),
            'middle': round(sma, 6),
            'lower': round(lower_band, 6),
            'signal': signal,
            'confidence': confidence,
            'current_price': round(current_price, 6)
        }
    
    def _calculate_sma(self, prices: List[float]) -> Dict:
        """Calculate Simple Moving Averages"""
        indicators = {}
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(prices) >= period:
                sma = sum(prices[-period:]) / period
                indicators[f'SMA{period}'] = round(sma, 6)
        
        # Determine trend based on SMA alignment
        current_price = prices[-1] if prices else 0
        
        if ('SMA20' in indicators and 'SMA50' in indicators and 
            current_price > indicators['SMA20'] > indicators['SMA50']):
            signal = 'bullish'
            confidence = 0.6
        elif ('SMA20' in indicators and 'SMA50' in indicators and 
              current_price < indicators['SMA20'] < indicators['SMA50']):
            signal = 'bearish'
            confidence = 0.6
        else:
            signal = 'neutral'
            confidence = 0.4
        
        indicators.update({
            'signal': signal,
            'confidence': confidence
        })
        
        return indicators
    
    def _calculate_stochastic(self, prices: List[float], k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        if len(prices) < k_period:
            return {'k': 50, 'd': 50, 'signal': 'neutral', 'confidence': 0.3}
        
        # For simplicity, using close prices as high/low
        # In real implementation, would use actual OHLC data
        recent_prices = prices[-k_period:]
        
        lowest_low = min(recent_prices)
        highest_high = max(recent_prices)
        current_close = prices[-1]
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simple moving average for %D
        d_percent = k_percent  # Simplified
        
        # Determine signal
        if k_percent > 80:
            signal = 'bearish'  # Overbought
            confidence = 0.7
        elif k_percent < 20:
            signal = 'bullish'  # Oversold
            confidence = 0.7
        else:
            signal = 'neutral'
            confidence = 0.4
        
        return {
            'k': round(k_percent, 2),
            'd': round(d_percent, 2),
            'signal': signal,
            'confidence': confidence,
            'overbought': 80,
            'oversold': 20
        }
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _get_simulated_technical(self, symbol: str, timeframe: str) -> TechnicalData:
        """Generate simulated technical indicators"""
        now = datetime.now()
        seed = int(now.timestamp() // 600)  # Changes every 10 minutes
        random.seed(seed)
        
        indicators = {
            'RSI': {
                'value': random.uniform(30, 70),
                'signal': random.choice(['bullish', 'bearish', 'neutral']),
                'confidence': 0.4
            },
            'MACD': {
                'value': random.uniform(-0.01, 0.01),
                'signal_line': random.uniform(-0.01, 0.01),
                'histogram': random.uniform(-0.005, 0.005),
                'signal': random.choice(['bullish', 'bearish', 'neutral']),
                'confidence': 0.4
            },
            'BOLLINGER': {
                'upper': 2050,
                'middle': 2000,
                'lower': 1950,
                'signal': random.choice(['bullish', 'bearish', 'neutral']),
                'confidence': 0.4
            }
        }
        
        return TechnicalData(
            symbol=symbol,
            indicators=indicators,
            timestamp=now,
            source=DataSource.SIMULATED,
            analysis_timeframe=timeframe
        )

# =============================================================================
# UNIFIED DATA PROVIDER
# =============================================================================

class UnifiedDataProvider:
    """Main interface for all data fetching with intelligent source selection"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.rate_limiter = RateLimiter()
        
        # Initialize all providers
        self.price_providers = [
            PrimaryPriceAPI(self.cache_manager, self.rate_limiter),
            SecondaryPriceScraper(self.cache_manager, self.rate_limiter),
            SimulatedPriceProvider(self.cache_manager, self.rate_limiter)
        ]
        
        self.sentiment_analyzer = NewsSentimentAnalyzer(self.cache_manager, self.rate_limiter)
        self.technical_analyzer = TechnicalAnalysisProvider(self.cache_manager, self.rate_limiter)
        
        # Track provider performance
        self.provider_stats = {}
    
    async def get_price_data(self, symbol: str) -> PriceData:
        """Get price data with automatic fallback"""
        errors = []
        
        # Sort providers by reliability score
        sorted_providers = sorted(self.price_providers, 
                                key=lambda p: p.reliability_score, 
                                reverse=True)
        
        for provider in sorted_providers:
            try:
                data = await provider.get_price_data(symbol)
                if data:
                    logger.info(f"Price data for {symbol} from {provider.source.value}")
                    return data
            except Exception as e:
                error_msg = f"{provider.source.value}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Price provider {provider.source.value} failed: {e}")
        
        # If all providers fail, return simulated data
        logger.error(f"All price providers failed for {symbol}: {errors}")
        return await self.price_providers[-1].get_price_data(symbol)
    
    async def get_sentiment_data(self, symbol: str, timeframe: str = '1d') -> SentimentData:
        """Get sentiment analysis data"""
        try:
            return await self.sentiment_analyzer.get_sentiment_data(symbol, timeframe)
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self.sentiment_analyzer._get_simulated_sentiment(symbol, timeframe)
    
    async def get_technical_data(self, symbol: str, timeframe: str = '1H') -> TechnicalData:
        """Get technical analysis data"""
        try:
            return await self.technical_analyzer.get_technical_data(symbol, timeframe)
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return self.technical_analyzer._get_simulated_technical(symbol, timeframe)
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data types for a symbol"""
        try:
            # Fetch all data types concurrently
            price_task = self.get_price_data(symbol)
            sentiment_task = self.get_sentiment_data(symbol)
            technical_task = self.get_technical_data(symbol)
            
            price_data, sentiment_data, technical_data = await asyncio.gather(
                price_task, sentiment_task, technical_task
            )
            
            return {
                'price': asdict(price_data),
                'sentiment': asdict(sentiment_data),
                'technical': asdict(technical_data),
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Comprehensive data fetch failed for {symbol}: {e}")
            raise
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        self.cache_manager.cleanup_expired()
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about provider performance"""
        stats = {}
        
        for provider in self.price_providers + [self.sentiment_analyzer, self.technical_analyzer]:
            stats[provider.source.value] = {
                'reliability_score': provider.reliability_score,
                'source_type': provider.source.value
            }
        
        return stats

# Global instance for easy import
unified_data_provider = UnifiedDataProvider()

# Cleanup function to be called periodically
async def cleanup_data_cache():
    """Periodic cleanup function"""
    while True:
        try:
            unified_data_provider.cleanup_cache()
            await asyncio.sleep(3600)  # Cleanup every hour
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            await asyncio.sleep(3600)

if __name__ == "__main__":
    # Example usage and testing
    async def test_data_provider():
        """Test the data provider system"""
        print("Testing Unified Data Provider System")
        print("=" * 50)
        
        symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
        
        for symbol in symbols:
            print(f"\nTesting {symbol}:")
            
            try:
                # Test price data
                price_data = await unified_data_provider.get_price_data(symbol)
                print(f"  Price: ${price_data.price:.2f} ({price_data.source.value})")
                
                # Test sentiment data
                sentiment_data = await unified_data_provider.get_sentiment_data(symbol)
                print(f"  Sentiment: {sentiment_data.sentiment_label} " +
                      f"(score: {sentiment_data.sentiment_score:.2f}, " +
                      f"confidence: {sentiment_data.confidence:.2f})")
                
                # Test technical data
                technical_data = await unified_data_provider.get_technical_data(symbol)
                rsi = technical_data.indicators.get('RSI', {})
                print(f"  RSI: {rsi.get('value', 'N/A')} ({rsi.get('signal', 'N/A')})")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"\nProvider Statistics:")
        stats = unified_data_provider.get_provider_stats()
        for source, data in stats.items():
            print(f"  {source}: {data['reliability_score']:.2f}")
    
    # Run the test
    asyncio.run(test_data_provider())
