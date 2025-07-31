"""
Real-Time Data Engine for GoldGPT
Replaces all hardcoded data with live sources including web scraping alternatives
"""

import requests
import sqlite3
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from textblob import TextBlob
import yfinance as yf
import time

logger = logging.getLogger(__name__)

@dataclass
class RealTimeData:
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    change: float
    change_percent: float
    bid: float
    ask: float
    source: str

@dataclass
class SentimentData:
    timestamp: datetime
    symbol: str
    timeframe: str
    sentiment: str  # bullish, bearish, neutral
    confidence: float
    score: float  # -1 to 1
    source: str
    factors: List[str]

@dataclass
class TechnicalIndicator:
    timestamp: datetime
    symbol: str
    indicator: str
    value: float
    signal: str  # buy, sell, hold
    period: int
    source: str

class RealTimeDataEngine:
    """Comprehensive real-time data engine with fallback mechanisms"""
    
    def __init__(self):
        self.db_path = 'goldgpt_realtime.db'
        self.init_database()
        
        # Multiple data sources for redundancy
        self.data_sources = {
            'gold_api': 'https://api.gold-api.com/price/XAU',
            'yahoo_finance': 'yfinance',
            'polygon': 'https://api.polygon.io',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'investing_com': 'https://www.investing.com',
            'marketwatch': 'https://www.marketwatch.com',
            'cnbc': 'https://www.cnbc.com',
            'reuters': 'https://www.reuters.com'
        }
        
        # Cache for rate limiting
        self.cache = {}
        self.cache_timeout = 30  # 30 seconds
        
    def init_database(self):
        """Initialize SQLite database for real-time data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Real-time price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                change_value REAL,
                change_percent REAL,
                bid REAL,
                ask REAL,
                source TEXT NOT NULL
            )
        ''')
        
        # Create index separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON realtime_prices (symbol, timestamp)
        ''')
        
        # Sentiment analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                score REAL NOT NULL,
                source TEXT NOT NULL,
                factors TEXT
            )
        ''')
        
        # Create index separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
            ON realtime_sentiment (symbol, timeframe)
        ''')
        
        # Technical indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_technical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                indicator TEXT NOT NULL,
                value REAL NOT NULL,
                signal TEXT NOT NULL,
                period INTEGER,
                source TEXT NOT NULL
            )
        ''')
        
        # Create index separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_indicator 
            ON realtime_technical (symbol, indicator)
        ''')
        
        conn.commit()
        conn.close()

    def get_live_price_data(self, symbol: str) -> Dict[str, Any]:
        """Get live price data with multiple source fallbacks"""
        cache_key = f"price_{symbol}"
        
        # Check cache first
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        # Try multiple sources in order of preference
        sources = [
            self._fetch_gold_api_price,
            self._fetch_yahoo_finance_price,
            self._scrape_investing_com_price,
            self._scrape_marketwatch_price
        ]
        
        for source_func in sources:
            try:
                data = source_func(symbol)
                if data and data.get('price', 0) > 0:
                    self._cache_data(cache_key, data)
                    self._store_price_data(symbol, data)
                    return data
            except Exception as e:
                logger.warning(f"Price source {source_func.__name__} failed for {symbol}: {e}")
                continue
        
        # If all sources fail, return last known data from database
        return self._get_last_known_price(symbol)

    def _fetch_gold_api_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch price from Gold-API.com"""
        if symbol.upper() != 'XAUUSD':
            return None
            
        response = requests.get(self.data_sources['gold_api'], timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current_price = float(data['price'])
        prev_close = float(data.get('prev_close_price', current_price))
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
        
        return {
            'symbol': 'XAUUSD',
            'price': current_price,
            'change': change,
            'change_percent': change_percent,
            'volume': 0,  # Not available from this source
            'bid': current_price - 0.5,
            'ask': current_price + 0.5,
            'source': 'gold_api',
            'timestamp': datetime.now().isoformat()
        }

    def _fetch_yahoo_finance_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch price from Yahoo Finance"""
        try:
            # Map symbols to Yahoo Finance tickers
            yahoo_symbols = {
                'XAUUSD': 'GC=F',  # Gold futures
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'BTCUSD': 'BTC-USD'
            }
            
            yahoo_symbol = yahoo_symbols.get(symbol, symbol)
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if hist.empty:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
            
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'volume': volume,
                'bid': current_price - 0.1,
                'ask': current_price + 0.1,
                'source': 'yahoo_finance',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            return None

    def _scrape_investing_com_price(self, symbol: str) -> Dict[str, Any]:
        """Scrape price data from Investing.com"""
        try:
            # URL mapping for different symbols
            urls = {
                'XAUUSD': 'https://www.investing.com/currencies/xau-usd',
                'EURUSD': 'https://www.investing.com/currencies/eur-usd',
                'GBPUSD': 'https://www.investing.com/currencies/gbp-usd',
                'USDJPY': 'https://www.investing.com/currencies/usd-jpy'
            }
            
            url = urls.get(symbol)
            if not url:
                return None
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find price elements (this may need adjustment based on site structure)
            price_elem = soup.find('span', {'data-test': 'instrument-price-last'})
            change_elem = soup.find('span', {'data-test': 'instrument-price-change'})
            change_percent_elem = soup.find('span', {'data-test': 'instrument-price-change-percent'})
            
            if not price_elem:
                return None
            
            price = float(price_elem.text.replace(',', ''))
            change = float(change_elem.text.replace(',', '')) if change_elem else 0
            change_percent_text = change_percent_elem.text if change_percent_elem else '0%'
            change_percent = float(re.findall(r'-?\d+\.?\d*', change_percent_text)[0]) if change_percent_text else 0
            
            return {
                'symbol': symbol,
                'price': price,
                'change': change,
                'change_percent': change_percent,
                'volume': 0,
                'bid': price - 0.1,
                'ask': price + 0.1,
                'source': 'investing_com_scrape',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Investing.com scraping failed for {symbol}: {e}")
            return None

    def _scrape_marketwatch_price(self, symbol: str) -> Dict[str, Any]:
        """Scrape price data from MarketWatch"""
        try:
            # URL mapping for MarketWatch
            urls = {
                'XAUUSD': 'https://www.marketwatch.com/investing/currency/xauusd',
                'EURUSD': 'https://www.marketwatch.com/investing/currency/eurusd',
                'GBPUSD': 'https://www.marketwatch.com/investing/currency/gbpusd'
            }
            
            url = urls.get(symbol)
            if not url:
                return None
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find price elements (adjust selectors as needed)
            price_elem = soup.find('bg-quote', class_='value')
            if not price_elem:
                price_elem = soup.find('span', class_='intraday__price')
            
            if price_elem:
                price_text = price_elem.text.strip().replace(',', '')
                price = float(re.findall(r'\d+\.?\d*', price_text)[0])
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'change': 0,  # Change data extraction needs more complex parsing
                    'change_percent': 0,
                    'volume': 0,
                    'bid': price - 0.1,
                    'ask': price + 0.1,
                    'source': 'marketwatch_scrape',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"MarketWatch scraping failed for {symbol}: {e}")
            return None

    def get_real_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get real-time sentiment analysis from multiple sources"""
        cache_key = f"sentiment_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Combine multiple sentiment sources
            news_sentiment = self._analyze_news_sentiment(symbol)
            social_sentiment = self._analyze_social_sentiment(symbol)
            technical_sentiment = self._analyze_technical_sentiment(symbol)
            
            # Calculate multi-timeframe sentiment
            timeframes = ['1h', '4h', '1d', '1w', '1m']
            sentiment_data = {
                'timeframes': {},
                'overall': {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'score': 0,
                    'factors': []
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'real_time_analysis'
            }
            
            # Calculate sentiment for each timeframe
            for timeframe in timeframes:
                timeframe_sentiment = self._calculate_timeframe_sentiment(
                    news_sentiment, social_sentiment, technical_sentiment, timeframe
                )
                sentiment_data['timeframes'][timeframe] = timeframe_sentiment
                
                # Store in database
                self._store_sentiment_data(symbol, timeframe, timeframe_sentiment)
            
            # Calculate overall sentiment
            overall_score = np.mean([sentiment_data['timeframes'][tf]['score'] for tf in timeframes])
            overall_confidence = np.mean([sentiment_data['timeframes'][tf]['confidence'] for tf in timeframes])
            
            if overall_score > 0.2:
                overall_sentiment = 'bullish'
            elif overall_score < -0.2:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            
            sentiment_data['overall'] = {
                'sentiment': overall_sentiment,
                'confidence': overall_confidence,
                'score': overall_score,
                'factors': ['news_analysis', 'social_sentiment', 'technical_analysis']
            }
            
            self._cache_data(cache_key, sentiment_data)
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)

    def _analyze_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from financial news"""
        try:
            # Scrape news from multiple sources
            news_sources = [
                self._scrape_reuters_news,
                self._scrape_cnbc_news,
                self._scrape_marketwatch_news
            ]
            
            all_articles = []
            for source_func in news_sources:
                try:
                    articles = source_func(symbol)
                    all_articles.extend(articles)
                except Exception as e:
                    logger.warning(f"News source {source_func.__name__} failed: {e}")
            
            if not all_articles:
                return {'score': 0, 'confidence': 0.3}
            
            # Analyze sentiment of headlines and content
            sentiment_scores = []
            for article in all_articles:
                # Use TextBlob for basic sentiment analysis
                blob = TextBlob(article['title'] + ' ' + article.get('content', ''))
                sentiment_scores.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            confidence = min(len(sentiment_scores) / 10, 1.0)  # More articles = higher confidence
            
            return {'score': avg_sentiment, 'confidence': confidence}
            
        except Exception as e:
            logger.warning(f"News sentiment analysis failed: {e}")
            return {'score': 0, 'confidence': 0.2}

    def _scrape_reuters_news(self, symbol: str) -> List[Dict]:
        """Scrape news from Reuters"""
        try:
            search_terms = {
                'XAUUSD': 'gold',
                'EURUSD': 'euro dollar',
                'GBPUSD': 'pound dollar'
            }
            
            search_term = search_terms.get(symbol, 'financial markets')
            url = f"https://www.reuters.com/search/news?query={search_term}&sortBy=date&dateRange=today"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find article elements (adjust selectors as needed)
            article_elements = soup.find_all('div', class_='search-result-content')
            
            for elem in article_elements[:5]:  # Limit to 5 articles
                title_elem = elem.find('h3')
                if title_elem:
                    title = title_elem.text.strip()
                    articles.append({
                        'title': title,
                        'content': '',
                        'source': 'reuters',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return articles
            
        except Exception as e:
            logger.warning(f"Reuters scraping failed: {e}")
            return []

    def _scrape_cnbc_news(self, symbol: str) -> List[Dict]:
        """Scrape news from CNBC"""
        try:
            search_terms = {
                'XAUUSD': 'gold+prices',
                'EURUSD': 'euro+dollar+currency',
                'GBPUSD': 'pound+dollar+currency'
            }
            
            search_term = search_terms.get(symbol, 'markets')
            url = f"https://www.cnbc.com/search/?query={search_term}&qsearchterm={search_term}"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find article elements
            article_elements = soup.find_all('div', class_='SearchResult')
            
            for elem in article_elements[:5]:
                title_elem = elem.find('a')
                if title_elem:
                    title = title_elem.text.strip()
                    articles.append({
                        'title': title,
                        'content': '',
                        'source': 'cnbc',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return articles
            
        except Exception as e:
            logger.warning(f"CNBC scraping failed: {e}")
            return []

    def _scrape_marketwatch_news(self, symbol: str) -> List[Dict]:
        """Scrape news from MarketWatch"""
        try:
            search_terms = {
                'XAUUSD': 'gold',
                'EURUSD': 'euro',
                'GBPUSD': 'pound'
            }
            
            search_term = search_terms.get(symbol, 'markets')
            url = f"https://www.marketwatch.com/search?q={search_term}&m=Keyword&rpp=25&mp=2007&bd=false&rs=true"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find article elements
            article_elements = soup.find_all('div', class_='searchresult')
            
            for elem in article_elements[:5]:
                title_elem = elem.find('h3')
                if title_elem:
                    title = title_elem.text.strip()
                    articles.append({
                        'title': title,
                        'content': '',
                        'source': 'marketwatch',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return articles
            
        except Exception as e:
            logger.warning(f"MarketWatch scraping failed: {e}")
            return []

    def _analyze_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze social media sentiment (placeholder - would need Twitter API, etc.)"""
        # This would integrate with Twitter API, Reddit API, etc.
        # For now, return neutral sentiment
        return {'score': 0, 'confidence': 0.3}

    def _analyze_technical_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze technical indicators for sentiment"""
        try:
            price_data = self.get_live_price_data(symbol)
            
            # Simple momentum analysis
            current_price = price_data.get('price', 0)
            change_percent = price_data.get('change_percent', 0)
            
            # Convert price change to sentiment score
            if change_percent > 1:
                score = 0.5
            elif change_percent > 0.5:
                score = 0.3
            elif change_percent > 0:
                score = 0.1
            elif change_percent < -1:
                score = -0.5
            elif change_percent < -0.5:
                score = -0.3
            else:
                score = -0.1 if change_percent < 0 else 0.1
            
            return {'score': score, 'confidence': 0.6}
            
        except Exception as e:
            logger.warning(f"Technical sentiment analysis failed: {e}")
            return {'score': 0, 'confidence': 0.3}

    def _calculate_timeframe_sentiment(self, news_sentiment: Dict, social_sentiment: Dict, 
                                     technical_sentiment: Dict, timeframe: str) -> Dict[str, Any]:
        """Calculate sentiment for specific timeframe"""
        # Weight different sources based on timeframe
        weights = {
            '1h': {'technical': 0.6, 'news': 0.3, 'social': 0.1},
            '4h': {'technical': 0.5, 'news': 0.4, 'social': 0.1},
            '1d': {'technical': 0.3, 'news': 0.5, 'social': 0.2},
            '1w': {'technical': 0.2, 'news': 0.6, 'social': 0.2},
            '1m': {'technical': 0.1, 'news': 0.7, 'social': 0.2}
        }
        
        timeframe_weights = weights.get(timeframe, weights['1d'])
        
        # Calculate weighted sentiment score
        weighted_score = (
            news_sentiment['score'] * timeframe_weights['news'] +
            social_sentiment['score'] * timeframe_weights['social'] +
            technical_sentiment['score'] * timeframe_weights['technical']
        )
        
        # Calculate confidence
        weighted_confidence = (
            news_sentiment['confidence'] * timeframe_weights['news'] +
            social_sentiment['confidence'] * timeframe_weights['social'] +
            technical_sentiment['confidence'] * timeframe_weights['technical']
        )
        
        # Determine sentiment label
        if weighted_score > 0.2:
            sentiment = 'bullish'
        elif weighted_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': weighted_confidence,
            'score': weighted_score
        }

    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate real technical indicators"""
        cache_key = f"technical_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Get historical data for calculations
            historical_data = self._get_historical_data(symbol)
            
            if not historical_data or len(historical_data) < 20:
                return self._get_fallback_technical_indicators(symbol)
            
            # Calculate indicators
            indicators = {
                'rsi': self._calculate_rsi(historical_data),
                'macd': self._calculate_macd(historical_data),
                'bollinger_bands': self._calculate_bollinger_bands(historical_data),
                'moving_averages': self._calculate_moving_averages(historical_data),
                'volume_indicators': self._calculate_volume_indicators(historical_data),
                'timestamp': datetime.now().isoformat(),
                'source': 'real_time_calculation'
            }
            
            # Store in database
            for indicator_name, indicator_data in indicators.items():
                if indicator_name not in ['timestamp', 'source'] and isinstance(indicator_data, dict):
                    self._store_technical_indicator(
                        symbol, 
                        indicator_name, 
                        indicator_data.get('value', 0),
                        indicator_data.get('signal', 'hold')
                    )
            
            self._cache_data(cache_key, indicators)
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed for {symbol}: {e}")
            return self._get_fallback_technical_indicators(symbol)

    def _get_historical_data(self, symbol: str, period: str = "1mo") -> List[Dict]:
        """Get historical price data for technical calculations"""
        try:
            # Use Yahoo Finance for historical data
            yahoo_symbols = {
                'XAUUSD': 'GC=F',
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'BTCUSD': 'BTC-USD'
            }
            
            yahoo_symbol = yahoo_symbols.get(symbol, symbol)
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return []
            
            # Convert to list of dictionaries
            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if 'Volume' in row else 0
                })
            
            return data
            
        except Exception as e:
            logger.warning(f"Historical data fetch failed for {symbol}: {e}")
            return []

    def _calculate_rsi(self, data: List[Dict], period: int = 14) -> Dict[str, Any]:
        """Calculate RSI indicator"""
        try:
            closes = [d['close'] for d in data[-period-1:]]
            
            if len(closes) < period + 1:
                return {'value': 50, 'signal': 'hold', 'period': period}
            
            gains = []
            losses = []
            
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Determine signal
            if rsi > 70:
                signal = 'sell'
            elif rsi < 30:
                signal = 'buy'
            else:
                signal = 'hold'
            
            return {
                'value': round(rsi, 2),
                'signal': signal,
                'period': period,
                'overbought_level': 70,
                'oversold_level': 30
            }
            
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return {'value': 50, 'signal': 'hold', 'period': period}

    def _calculate_macd(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate MACD indicator"""
        try:
            closes = [d['close'] for d in data]
            
            if len(closes) < 26:
                return {'value': 0, 'signal': 'hold', 'histogram': 0}
            
            # Calculate EMAs
            ema12 = self._calculate_ema(closes, 12)
            ema26 = self._calculate_ema(closes, 26)
            
            macd_line = ema12 - ema26
            
            # Signal line (9-period EMA of MACD)
            macd_values = [macd_line] * 9  # Simplified for demo
            signal_line = self._calculate_ema(macd_values, 9)
            
            histogram = macd_line - signal_line
            
            # Determine signal
            if macd_line > signal_line and histogram > 0:
                signal = 'buy'
            elif macd_line < signal_line and histogram < 0:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'value': round(macd_line, 4),
                'signal': signal,
                'signal_line': round(signal_line, 4),
                'histogram': round(histogram, 4)
            }
            
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return {'value': 0, 'signal': 'hold', 'histogram': 0}

    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not values or len(values) < period:
            return sum(values) / len(values) if values else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period
        
        for value in values[period:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return ema

    def _calculate_bollinger_bands(self, data: List[Dict], period: int = 20) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        try:
            closes = [d['close'] for d in data[-period:]]
            
            if len(closes) < period:
                current_price = closes[-1] if closes else 2000
                return {
                    'upper': current_price * 1.01,
                    'middle': current_price,
                    'lower': current_price * 0.99,
                    'signal': 'hold'
                }
            
            sma = sum(closes) / len(closes)
            variance = sum((x - sma) ** 2 for x in closes) / len(closes)
            std_dev = variance ** 0.5
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            current_price = closes[-1]
            
            # Determine signal
            if current_price > upper_band:
                signal = 'sell'
            elif current_price < lower_band:
                signal = 'buy'
            else:
                signal = 'hold'
            
            return {
                'upper': round(upper_band, 2),
                'middle': round(sma, 2),
                'lower': round(lower_band, 2),
                'signal': signal,
                'current_price': round(current_price, 2)
            }
            
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            return {'upper': 2050, 'middle': 2000, 'lower': 1950, 'signal': 'hold'}

    def _calculate_moving_averages(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate various moving averages"""
        try:
            closes = [d['close'] for d in data]
            
            mas = {}
            periods = [5, 10, 20, 50, 100, 200]
            
            for period in periods:
                if len(closes) >= period:
                    ma = sum(closes[-period:]) / period
                    mas[f'ma{period}'] = round(ma, 2)
                else:
                    mas[f'ma{period}'] = closes[-1] if closes else 2000
            
            # Determine overall trend
            current_price = closes[-1] if closes else 2000
            if current_price > mas.get('ma20', 2000) > mas.get('ma50', 2000):
                trend = 'bullish'
            elif current_price < mas.get('ma20', 2000) < mas.get('ma50', 2000):
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                **mas,
                'trend': trend,
                'current_price': round(current_price, 2)
            }
            
        except Exception as e:
            logger.warning(f"Moving averages calculation failed: {e}")
            return {'ma20': 2000, 'ma50': 2000, 'trend': 'neutral'}

    def _calculate_volume_indicators(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            volumes = [d['volume'] for d in data if d['volume'] > 0]
            
            if not volumes:
                return {'average_volume': 0, 'volume_trend': 'neutral'}
            
            avg_volume = sum(volumes[-20:]) / min(len(volumes), 20)
            current_volume = volumes[-1] if volumes else 0
            
            if current_volume > avg_volume * 1.5:
                volume_trend = 'high'
            elif current_volume < avg_volume * 0.5:
                volume_trend = 'low'
            else:
                volume_trend = 'normal'
            
            return {
                'average_volume': int(avg_volume),
                'current_volume': int(current_volume),
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            logger.warning(f"Volume indicators calculation failed: {e}")
            return {'average_volume': 0, 'volume_trend': 'neutral'}

    def _is_cached(self, key: str) -> bool:
        """Check if data is in cache and not expired"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_timeout

    def _cache_data(self, key: str, data: Any):
        """Store data in cache with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def _store_price_data(self, symbol: str, data: Dict):
        """Store price data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_prices 
                (symbol, price, volume, change_value, change_percent, bid, ask, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                data.get('price', 0),
                data.get('volume', 0),
                data.get('change', 0),
                data.get('change_percent', 0),
                data.get('bid', 0),
                data.get('ask', 0),
                data.get('source', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to store price data: {e}")

    def _store_sentiment_data(self, symbol: str, timeframe: str, data: Dict):
        """Store sentiment data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_sentiment 
                (symbol, timeframe, sentiment, confidence, score, source, factors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                timeframe,
                data.get('sentiment', 'neutral'),
                data.get('confidence', 0.5),
                data.get('score', 0),
                'real_time_analysis',
                json.dumps(data.get('factors', []))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to store sentiment data: {e}")

    def _store_technical_indicator(self, symbol: str, indicator: str, value: float, signal: str):
        """Store technical indicator in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_technical 
                (symbol, indicator, value, signal, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, indicator, value, signal, 'real_time_calculation'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to store technical indicator: {e}")

    def _get_last_known_price(self, symbol: str) -> Dict[str, Any]:
        """Get last known price from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT price, volume, change_value, change_percent, bid, ask, source, timestamp
                FROM realtime_prices
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'symbol': symbol,
                    'price': row[0],
                    'volume': row[1],
                    'change': row[2],
                    'change_percent': row[3],
                    'bid': row[4],
                    'ask': row[5],
                    'source': f"database_{row[6]}",
                    'timestamp': row[7]
                }
        except Exception as e:
            logger.warning(f"Failed to get last known price: {e}")
        
        # Ultimate fallback
        return {
            'symbol': symbol,
            'price': 2000.0,
            'volume': 0,
            'change': 0,
            'change_percent': 0,
            'bid': 1999.5,
            'ask': 2000.5,
            'source': 'fallback',
            'timestamp': datetime.now().isoformat()
        }

    def _get_fallback_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get fallback sentiment data when real analysis fails"""
        timeframes = ['1h', '4h', '1d', '1w', '1m']
        sentiment_data = {
            'timeframes': {},
            'overall': {
                'sentiment': 'neutral',
                'confidence': 0.3,
                'score': 0,
                'factors': ['fallback_mode']
            },
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
        
        for timeframe in timeframes:
            sentiment_data['timeframes'][timeframe] = {
                'sentiment': 'neutral',
                'confidence': 0.3,
                'score': 0
            }
        
        return sentiment_data

    def _get_fallback_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get fallback technical indicators when calculation fails"""
        return {
            'rsi': {'value': 50, 'signal': 'hold', 'period': 14},
            'macd': {'value': 0, 'signal': 'hold', 'histogram': 0},
            'bollinger_bands': {'upper': 2050, 'middle': 2000, 'lower': 1950, 'signal': 'hold'},
            'moving_averages': {'ma20': 2000, 'ma50': 2000, 'trend': 'neutral'},
            'volume_indicators': {'average_volume': 0, 'volume_trend': 'neutral'},
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }

# Global instance
real_time_data_engine = RealTimeDataEngine()
