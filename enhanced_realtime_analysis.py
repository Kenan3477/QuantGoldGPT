"""
ENHANCED REAL-TIME ANALYSIS ENGINE
==================================
This system provides real-time factor integration including:
1. Live news sentiment analysis with immediate price impact correlation
2. Real-time candlestick convergence/divergence detection
3. Dynamic volatility analysis based on breaking events
4. Multi-timeframe technical analysis synchronization
5. Event-driven prediction updates
"""

import asyncio
import aiohttp
import requests
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import feedparser
from textblob import TextBlob
import re
import threading
import time
try:
    import websocket
except ImportError:
    websocket = None  # Graceful fallback if websocket not available
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

@dataclass
class RealTimeEvent:
    """Real-time market event that can affect predictions"""
    event_type: str  # 'news', 'technical', 'volume_spike', 'breakout'
    timestamp: datetime
    impact_score: float  # -1 to 1
    confidence: float
    description: str
    price_at_event: float
    expected_duration: int  # minutes
    data: Dict[str, Any]

@dataclass
class ConvergenceDivergenceSignal:
    """Advanced technical convergence/divergence analysis"""
    signal_type: str  # 'bullish_convergence', 'bearish_divergence', etc.
    timeframe: str
    strength: float  # 0 to 1
    indicators_involved: List[str]
    confidence: float
    target_price: float
    invalidation_price: float
    expected_timeframe: str

class EnhancedRealTimeAnalyzer:
    """Advanced real-time analysis with live news and technical convergence"""
    
    def __init__(self, db_path='enhanced_realtime.db'):
        self.db_path = db_path
        self.init_database()
        self.active_events = []
        self.convergence_signals = []
        self.news_cache = {}
        self.last_analysis_time = datetime.now()
        self.is_monitoring = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Real-time news sources
        self.news_sources = {
            'reuters': 'http://feeds.reuters.com/reuters/businessNews',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
        }
        
        # Keywords for immediate impact detection
        self.high_impact_keywords = {
            'fed': ['federal reserve', 'fed chair', 'powell', 'fomc', 'interest rate'],
            'geopolitical': ['war', 'conflict', 'sanctions', 'crisis', 'tension'],
            'economic': ['inflation', 'cpi', 'ppi', 'gdp', 'employment', 'unemployment'],
            'gold_specific': ['gold reserves', 'central bank buying', 'mining', 'jewelry demand']
        }
        
        self.start_real_time_monitoring()
    
    def init_database(self):
        """Initialize enhanced database for real-time analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Real-time events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    impact_score REAL,
                    confidence REAL,
                    description TEXT,
                    price_at_event REAL,
                    expected_duration INTEGER,
                    data TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Convergence/Divergence signals
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS convergence_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT NOT NULL,
                    timeframe TEXT,
                    strength REAL,
                    indicators_involved TEXT,
                    confidence REAL,
                    target_price REAL,
                    invalidation_price REAL,
                    expected_timeframe TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # News sentiment tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT,
                    published_at TIMESTAMP,
                    sentiment_score REAL,
                    impact_prediction REAL,
                    price_before REAL,
                    price_after_1h REAL,
                    price_after_4h REAL,
                    accuracy_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Enhanced real-time analysis database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing real-time database: {e}")
    
    def start_real_time_monitoring(self):
        """Start background monitoring of real-time factors"""
        if not self.is_monitoring:
            self.is_monitoring = True
            
            # Start monitoring threads
            threading.Thread(target=self._monitor_news_feeds, daemon=True).start()
            threading.Thread(target=self._monitor_technical_convergence, daemon=True).start()
            threading.Thread(target=self._monitor_volume_spikes, daemon=True).start()
            
            logger.info("🔄 Started real-time monitoring threads")
    
    def _monitor_news_feeds(self):
        """Continuously monitor news feeds for breaking news"""
        while self.is_monitoring:
            try:
                for source_name, feed_url in self.news_sources.items():
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries[:5]:  # Check latest 5 entries
                            if self._is_new_article(entry):
                                impact_score = self._analyze_news_impact(entry)
                                if abs(impact_score) > 0.3:  # Only significant news
                                    self._create_news_event(entry, impact_score, source_name)
                    except Exception as e:
                        logger.warning(f"⚠️ Error fetching {source_name}: {e}")
                        continue
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in news monitoring: {e}")
                time.sleep(60)
    
    def _monitor_technical_convergence(self):
        """Monitor for technical convergence/divergence patterns"""
        while self.is_monitoring:
            try:
                # Get fresh market data
                gold_data = self._get_live_gold_data()
                if len(gold_data) > 50:
                    # Analyze convergence/divergence
                    convergence_signals = self._detect_convergence_divergence(gold_data)
                    for signal in convergence_signals:
                        self._create_technical_event(signal)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error in convergence monitoring: {e}")
                time.sleep(120)
    
    def _monitor_volume_spikes(self):
        """Monitor for unusual volume spikes that might indicate events"""
        while self.is_monitoring:
            try:
                gold_data = self._get_live_gold_data()
                if len(gold_data) > 20:
                    volume_spike = self._detect_volume_spike(gold_data)
                    if volume_spike:
                        self._create_volume_event(volume_spike)
                
                time.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in volume monitoring: {e}")
                time.sleep(180)
    
    def _is_new_article(self, entry) -> bool:
        """Check if article is new and not already processed"""
        article_id = entry.get('id', entry.get('link', ''))
        if article_id in self.news_cache:
            return False
        
        # Check if published in last 2 hours
        published = entry.get('published_parsed', None)
        if published:
            pub_time = datetime(*published[:6])
            if datetime.now() - pub_time < timedelta(hours=2):
                self.news_cache[article_id] = pub_time
                return True
        
        return False
    
    def _analyze_news_impact(self, entry) -> float:
        """Analyze potential market impact of news article"""
        title = entry.get('title', '').lower()
        summary = entry.get('summary', '').lower()
        full_text = f"{title} {summary}"
        
        impact_score = 0.0
        
        # Check for high-impact keywords
        for category, keywords in self.high_impact_keywords.items():
            for keyword in keywords:
                if keyword in full_text:
                    if category == 'fed':
                        impact_score += 0.4
                    elif category == 'geopolitical':
                        impact_score += 0.3
                    elif category == 'economic':
                        impact_score += 0.25
                    elif category == 'gold_specific':
                        impact_score += 0.35
        
        # Sentiment analysis
        blob = TextBlob(full_text)
        sentiment = blob.sentiment.polarity
        
        # Combine impact and sentiment
        if 'fed' in full_text and 'rate' in full_text:
            if 'cut' in full_text or 'lower' in full_text:
                impact_score += 0.6  # Bullish for gold
            elif 'hike' in full_text or 'raise' in full_text:
                impact_score -= 0.6  # Bearish for gold
        
        if 'inflation' in full_text:
            if 'high' in full_text or 'rising' in full_text:
                impact_score += 0.4  # Bullish for gold
            elif 'low' in full_text or 'falling' in full_text:
                impact_score -= 0.3  # Bearish for gold
        
        return max(-1.0, min(1.0, impact_score * sentiment))
    
    def _detect_convergence_divergence(self, data: pd.DataFrame) -> List[ConvergenceDivergenceSignal]:
        """Detect technical convergence/divergence patterns"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        try:
            # Calculate technical indicators
            prices = data['close']
            
            # RSI
            rsi = self._calculate_rsi(prices)
            
            # MACD
            macd, macd_signal, macd_hist = self._calculate_macd(prices)
            
            # Price momentum
            price_momentum = prices.pct_change(periods=14).rolling(5).mean()
            
            # RSI Divergence Detection
            recent_prices = prices[-20:]
            recent_rsi = rsi[-20:]
            
            # Bullish divergence: Price makes lower lows, RSI makes higher lows
            price_lows = recent_prices.rolling(5).min()
            rsi_lows = recent_rsi.rolling(5).min()
            
            if len(price_lows) > 10 and len(rsi_lows) > 10:
                price_trend = price_lows[-5:].iloc[-1] < price_lows[-10:].iloc[-1]
                rsi_trend = rsi_lows[-5:].iloc[-1] > rsi_lows[-10:].iloc[-1]
                
                if price_trend and rsi_trend and recent_rsi.iloc[-1] < 40:
                    signals.append(ConvergenceDivergenceSignal(
                        signal_type='bullish_rsi_divergence',
                        timeframe='1h',
                        strength=0.7,
                        indicators_involved=['RSI', 'Price'],
                        confidence=0.75,
                        target_price=prices.iloc[-1] * 1.02,
                        invalidation_price=prices.iloc[-1] * 0.985,
                        expected_timeframe='4-8 hours'
                    ))
            
            # MACD Convergence
            if len(macd) > 20 and len(macd_signal) > 20:
                macd_diff = macd - macd_signal
                if (macd_diff.iloc[-2] < 0 and macd_diff.iloc[-1] > 0 and 
                    macd.iloc[-1] > macd.iloc[-5]):
                    signals.append(ConvergenceDivergenceSignal(
                        signal_type='bullish_macd_convergence',
                        timeframe='1h',
                        strength=0.6,
                        indicators_involved=['MACD'],
                        confidence=0.65,
                        target_price=prices.iloc[-1] * 1.015,
                        invalidation_price=prices.iloc[-1] * 0.99,
                        expected_timeframe='2-4 hours'
                    ))
            
            # Multi-timeframe alignment
            if (rsi.iloc[-1] > 30 and rsi.iloc[-1] < 70 and 
                macd_diff.iloc[-1] > 0 and 
                price_momentum.iloc[-1] > 0):
                signals.append(ConvergenceDivergenceSignal(
                    signal_type='multi_timeframe_bullish_alignment',
                    timeframe='multiple',
                    strength=0.8,
                    indicators_involved=['RSI', 'MACD', 'Momentum'],
                    confidence=0.8,
                    target_price=prices.iloc[-1] * 1.025,
                    invalidation_price=prices.iloc[-1] * 0.98,
                    expected_timeframe='6-12 hours'
                ))
            
        except Exception as e:
            logger.error(f"❌ Error detecting convergence/divergence: {e}")
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _get_live_gold_data(self) -> pd.DataFrame:
        """Get live gold data for analysis"""
        try:
            # Use yfinance for real-time data
            gold = yf.Ticker("GC=F")  # Gold futures
            data = gold.history(period="5d", interval="1h")
            
            if data.empty:
                # Fallback to gold ETF
                gold_etf = yf.Ticker("GLD")
                data = gold_etf.history(period="5d", interval="1h")
            
            if not data.empty:
                data = data.reset_index()
                # Defensively handle column assignment based on actual columns
                expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if len(data.columns) == len(expected_columns):
                    data.columns = expected_columns
                elif len(data.columns) == 6:  # Standard case
                    data.columns = expected_columns
                else:
                    # Map columns by position, fill missing with NaN
                    column_mapping = {}
                    for i, col in enumerate(expected_columns[:len(data.columns)]):
                        column_mapping[data.columns[i]] = col
                    data = data.rename(columns=column_mapping)
                    # Add missing columns
                    for col in expected_columns:
                        if col not in data.columns:
                            data[col] = np.nan
                return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching live gold data: {e}")
        
        return pd.DataFrame()
    
    def _detect_volume_spike(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect unusual volume spikes"""
        if len(data) < 20:
            return None
        
        try:
            volumes = data['volume']
            recent_volume = volumes.iloc[-1]
            avg_volume = volumes.iloc[-20:-1].mean()
            
            if recent_volume > avg_volume * 2.5:  # 250% above average
                return {
                    'spike_ratio': recent_volume / avg_volume,
                    'volume': recent_volume,
                    'avg_volume': avg_volume,
                    'timestamp': data['timestamp'].iloc[-1],
                    'price': data['close'].iloc[-1]
                }
        
        except Exception as e:
            logger.error(f"❌ Error detecting volume spike: {e}")
        
        return None
    
    def _create_news_event(self, entry, impact_score: float, source: str):
        """Create a news-based real-time event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_price = self._get_current_gold_price()
            
            event = RealTimeEvent(
                event_type='news',
                timestamp=datetime.now(),
                impact_score=impact_score,
                confidence=min(0.9, abs(impact_score) + 0.1),
                description=f"Breaking: {entry.get('title', '')[:100]}",
                price_at_event=current_price,
                expected_duration=240,  # 4 hours
                data={
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'source': source,
                    'link': entry.get('link', '')
                }
            )
            
            cursor.execute('''
                INSERT INTO realtime_events 
                (event_type, timestamp, impact_score, confidence, description, 
                 price_at_event, expected_duration, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_type, event.timestamp, event.impact_score,
                event.confidence, event.description, event.price_at_event,
                event.expected_duration, json.dumps(event.data)
            ))
            
            conn.commit()
            conn.close()
            
            self.active_events.append(event)
            logger.info(f"📰 Created news event: {event.description} (Impact: {impact_score:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error creating news event: {e}")
    
    def _create_technical_event(self, signal: ConvergenceDivergenceSignal):
        """Create a technical convergence/divergence event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO convergence_signals 
                (signal_type, timeframe, strength, indicators_involved, confidence,
                 target_price, invalidation_price, expected_timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.signal_type, signal.timeframe, signal.strength,
                json.dumps(signal.indicators_involved), signal.confidence,
                signal.target_price, signal.invalidation_price, signal.expected_timeframe
            ))
            
            conn.commit()
            conn.close()
            
            self.convergence_signals.append(signal)
            logger.info(f"📊 Detected convergence signal: {signal.signal_type} (Strength: {signal.strength:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error creating technical event: {e}")
    
    def _create_volume_event(self, volume_data: Dict):
        """Create a volume spike event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            event = RealTimeEvent(
                event_type='volume_spike',
                timestamp=datetime.now(),
                impact_score=min(0.8, volume_data['spike_ratio'] / 5),
                confidence=0.6,
                description=f"Volume spike: {volume_data['spike_ratio']:.1f}x average",
                price_at_event=volume_data['price'],
                expected_duration=60,  # 1 hour
                data=volume_data
            )
            
            cursor.execute('''
                INSERT INTO realtime_events 
                (event_type, timestamp, impact_score, confidence, description, 
                 price_at_event, expected_duration, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_type, event.timestamp, event.impact_score,
                event.confidence, event.description, event.price_at_event,
                event.expected_duration, json.dumps(event.data)
            ))
            
            conn.commit()
            conn.close()
            
            self.active_events.append(event)
            logger.info(f"📈 Volume spike detected: {volume_data['spike_ratio']:.1f}x average volume")
            
        except Exception as e:
            logger.error(f"❌ Error creating volume event: {e}")
    
    def _get_current_gold_price(self) -> float:
        """Get current gold price"""
        try:
            # Try multiple sources for current price
            gold = yf.Ticker("GC=F")
            data = gold.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            # Fallback to ETF
            gold_etf = yf.Ticker("GLD")
            data = gold_etf.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1]) * 10  # Approximate conversion
            
        except Exception as e:
            logger.error(f"❌ Error getting current price: {e}")
        
        return 2050.0  # Default fallback
    
    def get_active_real_time_factors(self) -> Dict[str, Any]:
        """Get all active real-time factors affecting predictions"""
        try:
            # Clean up expired events
            current_time = datetime.now()
            self.active_events = [
                event for event in self.active_events
                if (current_time - event.timestamp).total_seconds() < event.expected_duration * 60
            ]
            
            # Get recent convergence signals
            active_convergence = [
                signal for signal in self.convergence_signals
                if (current_time - datetime.now()).total_seconds() < 3600  # Last hour
            ]
            
            # Calculate combined impact
            total_news_impact = sum(event.impact_score for event in self.active_events if event.event_type == 'news')
            total_technical_impact = sum(signal.strength * (1 if 'bullish' in signal.signal_type else -1) 
                                       for signal in active_convergence)
            
            return {
                'active_events': len(self.active_events),
                'news_impact': total_news_impact,
                'technical_impact': total_technical_impact,
                'convergence_signals': len(active_convergence),
                'combined_impact': total_news_impact + (total_technical_impact * 0.5),
                'last_update': current_time.isoformat(),
                'events': [
                    {
                        'type': event.event_type,
                        'impact': event.impact_score,
                        'description': event.description,
                        'age_minutes': (current_time - event.timestamp).total_seconds() / 60
                    }
                    for event in self.active_events[-5:]  # Last 5 events
                ],
                'technical_signals': [
                    {
                        'type': signal.signal_type,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'target': signal.target_price
                    }
                    for signal in active_convergence[-3:]  # Last 3 signals
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting real-time factors: {e}")
            return {
                'active_events': 0,
                'news_impact': 0,
                'technical_impact': 0,
                'combined_impact': 0,
                'last_update': datetime.now().isoformat(),
                'events': [],
                'technical_signals': []
            }
    
    def cleanup(self):
        """Cleanup and stop monitoring"""
        self.is_monitoring = False
        logger.info("🛑 Stopped real-time monitoring")

# Global instance
enhanced_realtime_analyzer = EnhancedRealTimeAnalyzer()

def get_real_time_factors() -> Dict[str, Any]:
    """Public function to get real-time factors"""
    return enhanced_realtime_analyzer.get_active_real_time_factors()
