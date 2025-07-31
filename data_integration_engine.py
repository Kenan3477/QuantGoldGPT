#!/usr/bin/env python3
"""
GoldGPT Data Integration Engine
Comprehensive data pipeline for ML prediction system integrating all required sources
"""

import asyncio
import aiohttp
import sqlite3
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CandlestickData:
    """Candlestick data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

@dataclass
class NewsData:
    """News data structure"""
    timestamp: datetime
    title: str
    content: str
    source: str
    sentiment_score: float
    relevance_score: float
    url: str

@dataclass
class EconomicIndicator:
    """Economic indicator data structure"""
    timestamp: datetime
    indicator_name: str
    value: float
    country: str
    impact_level: str
    source: str

@dataclass
class TechnicalIndicator:
    """Technical analysis indicator"""
    timestamp: datetime
    indicator_name: str
    value: float
    signal: str
    timeframe: str

@dataclass
class MarketSentiment:
    """Market sentiment data"""
    timestamp: datetime
    sentiment_type: str
    score: float
    confidence: float
    source: str

class DataCache:
    """Advanced caching system with TTL support"""
    
    def __init__(self, db_path: str = "goldgpt_data_cache.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                timestamp REAL NOT NULL,
                ttl INTEGER NOT NULL,
                data_type TEXT NOT NULL
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON cache_entries(data_type)')
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data, timestamp, ttl FROM cache_entries 
            WHERE key = ? AND (timestamp + ttl) > ?
        ''', (key, time.time()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                return None
        return None
    
    def set(self, key: str, data: Any, ttl: int, data_type: str = "general"):
        """Cache data with TTL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO cache_entries 
            (key, data, timestamp, ttl, data_type) VALUES (?, ?, ?, ?, ?)
        ''', (key, json.dumps(data, default=str), time.time(), ttl, data_type))
        
        conn.commit()
        conn.close()
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM cache_entries WHERE (timestamp + ttl) <= ?', (time.time(),))
        
        conn.commit()
        conn.close()

class CandlestickDataFetcher:
    """Fetches candlestick data from multiple sources"""
    
    def __init__(self, cache: DataCache):
        self.cache = cache
        self.sources = [
            self._fetch_from_gold_api,
            self._fetch_from_yahoo_finance,
            self._fetch_from_alpha_vantage
        ]
    
    async def fetch_candlestick_data(self, timeframes: List[str] = None) -> List[CandlestickData]:
        """Fetch candlestick data for multiple timeframes"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        all_data = []
        
        for timeframe in timeframes:
            cache_key = f"candlestick_{timeframe}_{datetime.now().strftime('%Y%m%d_%H')}"
            cached_data = self.cache.get(cache_key)
            
            if cached_data:
                all_data.extend([CandlestickData(**item) for item in cached_data])
                continue
            
            # Try each source until one works
            data = None
            for source_func in self.sources:
                try:
                    data = await source_func(timeframe)
                    if data:
                        break
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source_func.__name__}: {e}")
                    continue
            
            if data:
                # Cache for appropriate duration based on timeframe
                ttl = self._get_ttl_for_timeframe(timeframe)
                self.cache.set(cache_key, [asdict(item) for item in data], ttl, "candlestick")
                all_data.extend(data)
        
        return all_data
    
    def _get_ttl_for_timeframe(self, timeframe: str) -> int:
        """Get appropriate TTL for timeframe"""
        ttl_map = {
            '1m': 60,      # 1 minute
            '5m': 300,     # 5 minutes
            '15m': 900,    # 15 minutes
            '1h': 3600,    # 1 hour
            '4h': 14400,   # 4 hours
            '1d': 86400    # 1 day
        }
        return ttl_map.get(timeframe, 3600)
    
    async def _fetch_from_gold_api(self, timeframe: str) -> List[CandlestickData]:
        """Fetch from Gold API"""
        async with aiohttp.ClientSession() as session:
            url = "https://api.gold-api.com/price/XAU"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    price = data.get('price', 0)
                    
                    # Simulate candlestick data (in real implementation, use proper OHLCV data)
                    now = datetime.now(timezone.utc)
                    return [CandlestickData(
                        timestamp=now,
                        open=price * 0.999,
                        high=price * 1.001,
                        low=price * 0.998,
                        close=price,
                        volume=1000,
                        timeframe=timeframe
                    )]
        return []
    
    async def _fetch_from_yahoo_finance(self, timeframe: str) -> List[CandlestickData]:
        """Fetch from Yahoo Finance (fallback)"""
        # Implementation would use yfinance or similar
        # For now, return empty list
        return []
    
    async def _fetch_from_alpha_vantage(self, timeframe: str) -> List[CandlestickData]:
        """Fetch from Alpha Vantage (fallback)"""
        # Implementation would use Alpha Vantage API
        # For now, return empty list
        return []

class NewsDataFetcher:
    """Fetches and analyzes news data"""
    
    def __init__(self, cache: DataCache):
        self.cache = cache
        self.news_sources = [
            "https://www.marketwatch.com/markets/us",
            "https://www.investing.com/news/commodities-news",
            "https://www.reuters.com/business/finance"
        ]
    
    async def fetch_news_data(self, hours_back: int = 24) -> List[NewsData]:
        """Fetch and analyze news data"""
        cache_key = f"news_data_{datetime.now().strftime('%Y%m%d_%H')}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return [NewsData(**item) for item in cached_data]
        
        all_news = []
        
        async with aiohttp.ClientSession() as session:
            for source_url in self.news_sources:
                try:
                    news_items = await self._scrape_news_source(session, source_url)
                    all_news.extend(news_items)
                except Exception as e:
                    logger.warning(f"Failed to fetch news from {source_url}: {e}")
        
        # Analyze sentiment for all news items
        for news_item in all_news:
            news_item.sentiment_score = self._analyze_sentiment(news_item.content)
            news_item.relevance_score = self._calculate_relevance(news_item.title, news_item.content)
        
        # Cache for 1 hour
        self.cache.set(cache_key, [asdict(item) for item in all_news], 3600, "news")
        
        return all_news
    
    async def _scrape_news_source(self, session: aiohttp.ClientSession, url: str) -> List[NewsData]:
        """Scrape news from a single source"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_news_html(html, url)
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
        return []
    
    def _parse_news_html(self, html: str, source_url: str) -> List[NewsData]:
        """Parse HTML to extract news items"""
        soup = BeautifulSoup(html, 'html.parser')
        news_items = []
        
        # Generic news extraction (would be customized per source)
        for article in soup.find_all(['article', 'div'], class_=lambda x: x and ('news' in x.lower() or 'article' in x.lower()))[:10]:
            title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
            if title_elem:
                title = title_elem.get_text(strip=True)
                content = article.get_text(strip=True)[:500]  # First 500 chars
                
                if len(title) > 10 and self._is_relevant_to_gold(title + content):
                    news_items.append(NewsData(
                        timestamp=datetime.now(timezone.utc),
                        title=title,
                        content=content,
                        source=source_url,
                        sentiment_score=0.0,  # Will be calculated later
                        relevance_score=0.0,  # Will be calculated later
                        url=source_url
                    ))
        
        return news_items
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns -1 to 1
        except Exception:
            return 0.0
    
    def _calculate_relevance(self, title: str, content: str) -> float:
        """Calculate relevance to gold trading"""
        gold_keywords = [
            'gold', 'xau', 'precious metals', 'inflation', 'fed', 'interest rates',
            'dollar', 'usd', 'monetary policy', 'central bank', 'recession',
            'commodity', 'mining', 'bullion'
        ]
        
        text = (title + ' ' + content).lower()
        relevance = sum(1 for keyword in gold_keywords if keyword in text)
        return min(relevance / len(gold_keywords), 1.0)
    
    def _is_relevant_to_gold(self, text: str) -> bool:
        """Check if news is relevant to gold trading"""
        return self._calculate_relevance("", text) > 0.1

class EconomicDataFetcher:
    """Fetches economic indicators from multiple sources"""
    
    def __init__(self, cache: DataCache):
        self.cache = cache
        self.fred_api_key = None  # Would be set from config
        self.world_bank_base_url = "https://api.worldbank.org/v2"
    
    async def fetch_economic_indicators(self) -> List[EconomicIndicator]:
        """Fetch key economic indicators"""
        cache_key = f"economic_data_{datetime.now().strftime('%Y%m%d')}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return [EconomicIndicator(**item) for item in cached_data]
        
        indicators = []
        
        # Fetch various indicators
        try:
            # USD Index
            usd_data = await self._fetch_usd_index()
            if usd_data:
                indicators.append(usd_data)
            
            # Interest rates
            rates_data = await self._fetch_interest_rates()
            indicators.extend(rates_data)
            
            # Inflation data
            inflation_data = await self._fetch_inflation_data()
            indicators.extend(inflation_data)
            
            # VIX (volatility index)
            vix_data = await self._fetch_vix_data()
            if vix_data:
                indicators.append(vix_data)
                
        except Exception as e:
            logger.error(f"Failed to fetch economic indicators: {e}")
        
        # Cache for 4 hours
        self.cache.set(cache_key, [asdict(item) for item in indicators], 14400, "economic")
        
        return indicators
    
    async def _fetch_usd_index(self) -> Optional[EconomicIndicator]:
        """Fetch USD index data"""
        try:
            # Placeholder implementation - would use real data source
            return EconomicIndicator(
                timestamp=datetime.now(timezone.utc),
                indicator_name="USD_INDEX",
                value=103.5,  # Placeholder
                country="US",
                impact_level="high",
                source="placeholder"
            )
        except Exception as e:
            logger.error(f"Failed to fetch USD index: {e}")
            return None
    
    async def _fetch_interest_rates(self) -> List[EconomicIndicator]:
        """Fetch interest rates data"""
        rates = []
        
        try:
            # Placeholder for Fed funds rate
            rates.append(EconomicIndicator(
                timestamp=datetime.now(timezone.utc),
                indicator_name="FED_FUNDS_RATE",
                value=5.25,  # Placeholder
                country="US",
                impact_level="high",
                source="placeholder"
            ))
        except Exception as e:
            logger.error(f"Failed to fetch interest rates: {e}")
        
        return rates
    
    async def _fetch_inflation_data(self) -> List[EconomicIndicator]:
        """Fetch inflation data"""
        inflation_data = []
        
        try:
            # Placeholder for CPI
            inflation_data.append(EconomicIndicator(
                timestamp=datetime.now(timezone.utc),
                indicator_name="CPI_YOY",
                value=3.2,  # Placeholder
                country="US",
                impact_level="high",
                source="placeholder"
            ))
        except Exception as e:
            logger.error(f"Failed to fetch inflation data: {e}")
        
        return inflation_data
    
    async def _fetch_vix_data(self) -> Optional[EconomicIndicator]:
        """Fetch VIX volatility index"""
        try:
            # Placeholder implementation
            return EconomicIndicator(
                timestamp=datetime.now(timezone.utc),
                indicator_name="VIX",
                value=18.5,  # Placeholder
                country="US",
                impact_level="medium",
                source="placeholder"
            )
        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return None

class TechnicalAnalyzer:
    """Performs technical analysis on price data"""
    
    def __init__(self, cache: DataCache):
        self.cache = cache
    
    def calculate_technical_indicators(self, candlestick_data: List[CandlestickData]) -> List[TechnicalIndicator]:
        """Calculate various technical indicators"""
        if not candlestick_data:
            return []
        
        # Convert to pandas DataFrame for easier calculations
        df = pd.DataFrame([asdict(candle) for candle in candlestick_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        indicators = []
        now = datetime.now(timezone.utc)
        
        try:
            # Moving averages
            if len(df) >= 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['ema_20'] = df['close'].ewm(span=20).mean()
                
                latest_sma = df['sma_20'].iloc[-1]
                latest_ema = df['ema_20'].iloc[-1]
                latest_close = df['close'].iloc[-1]
                
                indicators.extend([
                    TechnicalIndicator(
                        timestamp=now,
                        indicator_name="SMA_20",
                        value=latest_sma,
                        signal="bullish" if latest_close > latest_sma else "bearish",
                        timeframe="1d"
                    ),
                    TechnicalIndicator(
                        timestamp=now,
                        indicator_name="EMA_20",
                        value=latest_ema,
                        signal="bullish" if latest_close > latest_ema else "bearish",
                        timeframe="1d"
                    )
                ])
            
            # RSI
            if len(df) >= 14:
                rsi = self._calculate_rsi(df['close'].values, 14)
                indicators.append(TechnicalIndicator(
                    timestamp=now,
                    indicator_name="RSI_14",
                    value=rsi,
                    signal="overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
                    timeframe="1d"
                ))
            
            # MACD
            if len(df) >= 26:
                macd_line, signal_line, histogram = self._calculate_macd(df['close'].values)
                indicators.append(TechnicalIndicator(
                    timestamp=now,
                    indicator_name="MACD",
                    value=macd_line,
                    signal="bullish" if macd_line > signal_line else "bearish",
                    timeframe="1d"
                ))
            
            # Bollinger Bands
            if len(df) >= 20:
                upper, middle, lower = self._calculate_bollinger_bands(df['close'].values, 20, 2)
                latest_close = df['close'].iloc[-1]
                
                if latest_close > upper:
                    bb_signal = "overbought"
                elif latest_close < lower:
                    bb_signal = "oversold"
                else:
                    bb_signal = "neutral"
                
                indicators.append(TechnicalIndicator(
                    timestamp=now,
                    indicator_name="BOLLINGER_BANDS",
                    value=(upper + lower) / 2,
                    signal=bb_signal,
                    timeframe="1d"
                ))
                
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator"""
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]

class FeatureEngineer:
    """Extracts predictive features from all data sources"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, 
                        candlestick_data: List[CandlestickData],
                        news_data: List[NewsData],
                        economic_data: List[EconomicIndicator],
                        technical_indicators: List[TechnicalIndicator]) -> Dict[str, float]:
        """Extract comprehensive feature set"""
        
        features = {}
        
        # Price-based features
        if candlestick_data:
            price_features = self._extract_price_features(candlestick_data)
            features.update(price_features)
        
        # News sentiment features
        if news_data:
            news_features = self._extract_news_features(news_data)
            features.update(news_features)
        
        # Economic features
        if economic_data:
            economic_features = self._extract_economic_features(economic_data)
            features.update(economic_features)
        
        # Technical features
        if technical_indicators:
            technical_features = self._extract_technical_features(technical_indicators)
            features.update(technical_features)
        
        # Time-based features
        time_features = self._extract_time_features()
        features.update(time_features)
        
        self.feature_names = list(features.keys())
        return features
    
    def _extract_price_features(self, candlestick_data: List[CandlestickData]) -> Dict[str, float]:
        """Extract price-based features"""
        features = {}
        
        if not candlestick_data:
            return features
        
        # Sort by timestamp
        sorted_data = sorted(candlestick_data, key=lambda x: x.timestamp)
        
        # Current price metrics
        latest = sorted_data[-1]
        features['current_price'] = latest.close
        features['current_volume'] = latest.volume
        features['daily_range'] = (latest.high - latest.low) / latest.close
        
        # Price changes
        if len(sorted_data) > 1:
            prev = sorted_data[-2]
            features['price_change'] = (latest.close - prev.close) / prev.close
            features['volume_change'] = (latest.volume - prev.volume) / prev.volume if prev.volume > 0 else 0
        
        # Historical volatility
        if len(sorted_data) >= 10:
            closes = [c.close for c in sorted_data[-10:]]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            features['volatility_10d'] = np.std(returns) if returns else 0
        
        # Price momentum
        if len(sorted_data) >= 5:
            closes = [c.close for c in sorted_data[-5:]]
            features['momentum_5d'] = (closes[-1] - closes[0]) / closes[0]
        
        return features
    
    def _extract_news_features(self, news_data: List[NewsData]) -> Dict[str, float]:
        """Extract news sentiment features"""
        features = {}
        
        if not news_data:
            return {'news_sentiment_avg': 0, 'news_relevance_avg': 0, 'news_count': 0}
        
        # Recent news (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_news = [n for n in news_data if n.timestamp > recent_cutoff]
        
        if recent_news:
            sentiments = [n.sentiment_score for n in recent_news]
            relevances = [n.relevance_score for n in recent_news]
            
            features['news_sentiment_avg'] = np.mean(sentiments)
            features['news_sentiment_std'] = np.std(sentiments)
            features['news_relevance_avg'] = np.mean(relevances)
            features['news_count'] = len(recent_news)
            features['news_positive_ratio'] = len([s for s in sentiments if s > 0.1]) / len(sentiments)
            features['news_negative_ratio'] = len([s for s in sentiments if s < -0.1]) / len(sentiments)
        
        return features
    
    def _extract_economic_features(self, economic_data: List[EconomicIndicator]) -> Dict[str, float]:
        """Extract economic indicator features"""
        features = {}
        
        # Create features for each indicator
        for indicator in economic_data:
            feature_name = f"econ_{indicator.indicator_name.lower()}"
            features[feature_name] = indicator.value
            
            # Impact weighting
            impact_weight = {'high': 1.0, 'medium': 0.5, 'low': 0.25}.get(indicator.impact_level, 0.5)
            features[f"{feature_name}_weighted"] = indicator.value * impact_weight
        
        # USD strength impact on gold (inverse relationship)
        usd_indicators = [i for i in economic_data if 'USD' in i.indicator_name]
        if usd_indicators:
            features['usd_strength_impact'] = -sum(i.value for i in usd_indicators) / len(usd_indicators)
        
        return features
    
    def _extract_technical_features(self, technical_indicators: List[TechnicalIndicator]) -> Dict[str, float]:
        """Extract technical analysis features"""
        features = {}
        
        for indicator in technical_indicators:
            feature_name = f"tech_{indicator.indicator_name.lower()}"
            features[feature_name] = indicator.value
            
            # Signal encoding
            signal_map = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0, 'overbought': 0.8, 'oversold': -0.8}
            features[f"{feature_name}_signal"] = signal_map.get(indicator.signal, 0.0)
        
        # Consensus score (how many indicators agree)
        signals = [indicator.signal for indicator in technical_indicators]
        bullish_count = len([s for s in signals if s in ['bullish']])
        bearish_count = len([s for s in signals if s in ['bearish']])
        total_signals = len(signals)
        
        if total_signals > 0:
            features['tech_consensus_bullish'] = bullish_count / total_signals
            features['tech_consensus_bearish'] = bearish_count / total_signals
            features['tech_consensus_strength'] = abs(bullish_count - bearish_count) / total_signals
        
        return features
    
    def _extract_time_features(self) -> Dict[str, float]:
        """Extract time-based features"""
        now = datetime.now(timezone.utc)
        
        features = {
            'hour_of_day': now.hour / 23.0,  # Normalize to 0-1
            'day_of_week': now.weekday() / 6.0,  # Normalize to 0-1
            'day_of_month': now.day / 31.0,  # Normalize to 0-1
            'month_of_year': now.month / 12.0,  # Normalize to 0-1
            'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
            'is_market_open': self._is_market_open(now),
            'time_to_market_open': self._time_to_market_open(now),
            'time_to_market_close': self._time_to_market_close(now)
        }
        
        return features
    
    def _is_market_open(self, dt: datetime) -> float:
        """Check if major markets are open"""
        # Simplified market hours check (would be more complex in reality)
        hour = dt.hour
        weekday = dt.weekday()
        
        # Assume some market is always open due to global nature
        if weekday < 5:  # Weekday
            return 1.0 if 0 <= hour <= 23 else 0.5
        else:  # Weekend
            return 0.3  # Some markets might be open
    
    def _time_to_market_open(self, dt: datetime) -> float:
        """Hours until next major market opens (normalized)"""
        # Simplified implementation
        hour = dt.hour
        if hour < 9:
            return (9 - hour) / 24.0
        elif hour >= 17:
            return (24 - hour + 9) / 24.0
        else:
            return 0.0
    
    def _time_to_market_close(self, dt: datetime) -> float:
        """Hours until market closes (normalized)"""
        # Simplified implementation
        hour = dt.hour
        if 9 <= hour < 17:
            return (17 - hour) / 24.0
        else:
            return 0.0

class DataIntegrationEngine:
    """Main data integration engine that orchestrates all data fetching"""
    
    def __init__(self, cache_db_path: str = "goldgpt_data_cache.db"):
        self.cache = DataCache(cache_db_path)
        self.candlestick_fetcher = CandlestickDataFetcher(self.cache)
        self.news_fetcher = NewsDataFetcher(self.cache)
        self.economic_fetcher = EconomicDataFetcher(self.cache)
        self.technical_analyzer = TechnicalAnalyzer(self.cache)
        self.feature_engineer = FeatureEngineer()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def fetch_all_data(self) -> Tuple[List[CandlestickData], List[NewsData], 
                                          List[EconomicIndicator], List[TechnicalIndicator]]:
        """Fetch all data from all sources asynchronously"""
        
        logger.info("Starting comprehensive data fetch...")
        
        # Create tasks for concurrent execution
        tasks = [
            self.candlestick_fetcher.fetch_candlestick_data(),
            self.news_fetcher.fetch_news_data(),
            self.economic_fetcher.fetch_economic_indicators()
        ]
        
        try:
            # Execute all fetching tasks concurrently
            candlestick_data, news_data, economic_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(candlestick_data, Exception):
                logger.error(f"Candlestick fetch failed: {candlestick_data}")
                candlestick_data = []
            
            if isinstance(news_data, Exception):
                logger.error(f"News fetch failed: {news_data}")
                news_data = []
            
            if isinstance(economic_data, Exception):
                logger.error(f"Economic data fetch failed: {economic_data}")
                economic_data = []
            
            # Calculate technical indicators
            technical_indicators = self.technical_analyzer.calculate_technical_indicators(candlestick_data)
            
            logger.info(f"Data fetch complete: {len(candlestick_data)} candles, "
                       f"{len(news_data)} news items, {len(economic_data)} economic indicators, "
                       f"{len(technical_indicators)} technical indicators")
            
            return candlestick_data, news_data, economic_data, technical_indicators
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return [], [], [], []
    
    async def get_unified_dataset(self) -> Dict[str, Any]:
        """Get unified dataset with all features extracted"""
        
        # Fetch all data
        candlestick_data, news_data, economic_data, technical_indicators = await self.fetch_all_data()
        
        # Extract features
        features = self.feature_engineer.extract_features(
            candlestick_data, news_data, economic_data, technical_indicators
        )
        
        # Create unified dataset
        dataset = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'features': features,
            'feature_names': self.feature_engineer.feature_names,
            'raw_data': {
                'candlestick_count': len(candlestick_data),
                'news_count': len(news_data),
                'economic_indicators_count': len(economic_data),
                'technical_indicators_count': len(technical_indicators)
            },
            'data_quality': self._assess_data_quality(features)
        }
        
        return dataset
    
    def _assess_data_quality(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Assess the quality of the collected data"""
        
        quality_metrics = {
            'feature_count': len(features),
            'completeness': len([v for v in features.values() if v is not None and not np.isnan(v)]) / len(features) if features else 0,
            'has_price_data': any('price' in k for k in features.keys()),
            'has_news_data': any('news' in k for k in features.keys()),
            'has_economic_data': any('econ' in k for k in features.keys()),
            'has_technical_data': any('tech' in k for k in features.keys()),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Overall quality score
        quality_score = (
            (0.3 * quality_metrics['completeness']) +
            (0.2 * int(quality_metrics['has_price_data'])) +
            (0.2 * int(quality_metrics['has_news_data'])) +
            (0.15 * int(quality_metrics['has_economic_data'])) +
            (0.15 * int(quality_metrics['has_technical_data']))
        )
        
        quality_metrics['overall_score'] = quality_score
        quality_metrics['quality_level'] = 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.5 else 'low'
        
        return quality_metrics
    
    async def cleanup_cache(self):
        """Clean up expired cache entries"""
        self.cache.cleanup_expired()
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)

class DataManager:
    """Unified data manager that provides clean, normalized data to ML engines"""
    
    def __init__(self, integration_engine: DataIntegrationEngine):
        self.integration_engine = integration_engine
        self._last_dataset = None
        self._last_fetch_time = None
        self.min_refresh_interval = 300  # 5 minutes minimum between fetches
    
    async def get_ml_ready_dataset(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get dataset ready for ML consumption"""
        
        # Check if we need to refresh data
        now = datetime.now(timezone.utc)
        if (not force_refresh and self._last_dataset and self._last_fetch_time and 
            (now - self._last_fetch_time).total_seconds() < self.min_refresh_interval):
            logger.info("Returning cached dataset (within refresh interval)")
            return self._last_dataset
        
        # Fetch fresh data
        logger.info("Fetching fresh dataset for ML")
        dataset = await self.integration_engine.get_unified_dataset()
        
        # Validate and normalize
        validated_dataset = self._validate_and_normalize_dataset(dataset)
        
        # Cache the result
        self._last_dataset = validated_dataset
        self._last_fetch_time = now
        
        return validated_dataset
    
    def _validate_and_normalize_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize dataset for ML consumption"""
        
        features = dataset.get('features', {})
        
        # Handle missing values
        normalized_features = {}
        for key, value in features.items():
            if value is None or np.isnan(value) if isinstance(value, (int, float)) else False:
                # Use reasonable defaults for missing values
                if 'sentiment' in key:
                    normalized_features[key] = 0.0
                elif 'price' in key or 'value' in key:
                    normalized_features[key] = 0.0
                elif 'count' in key:
                    normalized_features[key] = 0
                else:
                    normalized_features[key] = 0.0
            else:
                normalized_features[key] = float(value) if isinstance(value, (int, float)) else value
        
        # Ensure we have minimum required features
        required_features = [
            'current_price', 'news_sentiment_avg', 'hour_of_day', 
            'day_of_week', 'is_market_open'
        ]
        
        for feature in required_features:
            if feature not in normalized_features:
                normalized_features[feature] = 0.0
        
        # Update dataset
        validated_dataset = dataset.copy()
        validated_dataset['features'] = normalized_features
        validated_dataset['feature_count'] = len(normalized_features)
        validated_dataset['validation_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return validated_dataset
    
    def get_feature_vector(self, dataset: Dict[str, Any] = None) -> np.ndarray:
        """Get feature vector as numpy array for ML models"""
        
        if dataset is None:
            dataset = self._last_dataset
        
        if not dataset:
            raise ValueError("No dataset available. Call get_ml_ready_dataset() first.")
        
        features = dataset.get('features', {})
        feature_names = sorted(features.keys())  # Ensure consistent ordering
        
        # Convert to numpy array
        feature_vector = np.array([features[name] for name in feature_names])
        
        return feature_vector
    
    def get_feature_names(self, dataset: Dict[str, Any] = None) -> List[str]:
        """Get ordered list of feature names"""
        
        if dataset is None:
            dataset = self._last_dataset
        
        if not dataset:
            raise ValueError("No dataset available. Call get_ml_ready_dataset() first.")
        
        features = dataset.get('features', {})
        return sorted(features.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on data pipeline"""
        
        try:
            # Test data fetching
            start_time = time.time()
            dataset = await self.get_ml_ready_dataset(force_refresh=True)
            fetch_time = time.time() - start_time
            
            data_quality = dataset.get('data_quality', {})
            
            health_status = {
                'status': 'healthy' if data_quality.get('overall_score', 0) > 0.5 else 'degraded',
                'fetch_time_seconds': fetch_time,
                'feature_count': len(dataset.get('features', {})),
                'data_quality_score': data_quality.get('overall_score', 0),
                'data_sources': {
                    'price_data': data_quality.get('has_price_data', False),
                    'news_data': data_quality.get('has_news_data', False),
                    'economic_data': data_quality.get('has_economic_data', False),
                    'technical_data': data_quality.get('has_technical_data', False)
                },
                'last_update': dataset.get('timestamp'),
                'cache_status': 'active'
            }
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

# Example usage and testing
async def main():
    """Example usage of the data integration system"""
    
    # Initialize the system
    integration_engine = DataIntegrationEngine()
    data_manager = DataManager(integration_engine)
    
    try:
        # Get ML-ready dataset
        dataset = await data_manager.get_ml_ready_dataset()
        
        # Print summary
        print("=== GoldGPT Data Integration System ===")
        print(f"Timestamp: {dataset['timestamp']}")
        print(f"Feature Count: {dataset['feature_count']}")
        print(f"Data Quality: {dataset['data_quality']['quality_level']} ({dataset['data_quality']['overall_score']:.2f})")
        print(f"Data Sources: {dataset['raw_data']}")
        
        # Get feature vector for ML
        feature_vector = data_manager.get_feature_vector(dataset)
        feature_names = data_manager.get_feature_names(dataset)
        
        print(f"\nFeature Vector Shape: {feature_vector.shape}")
        print(f"Sample Features:")
        for i, (name, value) in enumerate(zip(feature_names[:10], feature_vector[:10])):
            print(f"  {name}: {value:.4f}")
        
        # Health check
        health = await data_manager.health_check()
        print(f"\nSystem Health: {health['status']}")
        print(f"Fetch Time: {health['fetch_time_seconds']:.2f}s")
        
    finally:
        integration_engine.close()

if __name__ == "__main__":
    asyncio.run(main())
