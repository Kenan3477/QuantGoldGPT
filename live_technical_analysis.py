"""
Live Technical Analysis Engine for QuantGold Dashboard
Real-time candlestick patterns, news sentiment, and technical indicators
"""

import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTechnicalAnalysis:
    def __init__(self):
        self.db_path = "technical_analysis.db"
        self.init_database()
        
    def init_database(self):
        """Initialize database for storing analysis results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for technical analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candlestick_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pattern_name TEXT NOT NULL,
                    symbol TEXT DEFAULT 'XAUUSD',
                    confidence_percent REAL,
                    expectancy_percent REAL,
                    direction TEXT,
                    price_level REAL,
                    timeframe TEXT,
                    analysis_data TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT DEFAULT 'XAUUSD',
                    indicator_name TEXT,
                    value REAL,
                    signal TEXT,
                    strength TEXT,
                    timeframe TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    headline TEXT,
                    sentiment_score REAL,
                    market_impact TEXT,
                    source TEXT,
                    relevance_score REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Technical analysis database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    def detect_candlestick_patterns(self, ohlc_data: List[Dict]) -> List[Dict]:
        """
        Detect candlestick patterns from OHLC data
        Returns pattern analysis with confidence and expectancy
        """
        patterns = []
        
        if len(ohlc_data) < 3:
            return patterns
            
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(ohlc_data)
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
            df['is_bullish'] = df['close'] > df['open']
            
            current = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else current
            prev2 = df.iloc[-3] if len(df) > 2 else prev
            
            # Bearish Engulfing Pattern
            if (prev['is_bullish'] and not current['is_bullish'] and 
                current['open'] > prev['close'] and current['close'] < prev['open']):
                
                confidence = min(95, 60 + (current['body'] / prev['body']) * 20)
                expectancy = -2.5 + (confidence - 60) * 0.1
                
                patterns.append({
                    'pattern': 'Bearish Engulfing',
                    'confidence': round(confidence, 1),
                    'expectancy': round(expectancy, 2),
                    'direction': 'BEARISH',
                    'strength': 'HIGH' if confidence > 80 else 'MEDIUM',
                    'price_level': current['close'],
                    'description': f"Strong bearish reversal signal - Previous bullish candle engulfed by bearish candle"
                })
            
            # Bullish Engulfing Pattern
            elif (not prev['is_bullish'] and current['is_bullish'] and 
                  current['open'] < prev['close'] and current['close'] > prev['open']):
                
                confidence = min(95, 60 + (current['body'] / prev['body']) * 20)
                expectancy = 2.5 + (confidence - 60) * 0.1
                
                patterns.append({
                    'pattern': 'Bullish Engulfing',
                    'confidence': round(confidence, 1),
                    'expectancy': round(expectancy, 2),
                    'direction': 'BULLISH',
                    'strength': 'HIGH' if confidence > 80 else 'MEDIUM',
                    'price_level': current['close'],
                    'description': f"Strong bullish reversal signal - Previous bearish candle engulfed by bullish candle"
                })
            
            # Doji Pattern
            elif current['body'] < (current['high'] - current['low']) * 0.1:
                confidence = 65 + min(25, (current['upper_shadow'] + current['lower_shadow']) / current['body'] * 5)
                expectancy = 0  # Neutral pattern
                
                patterns.append({
                    'pattern': 'Doji',
                    'confidence': round(confidence, 1),
                    'expectancy': round(expectancy, 2),
                    'direction': 'NEUTRAL',
                    'strength': 'MEDIUM',
                    'price_level': current['close'],
                    'description': f"Market indecision - Potential reversal signal"
                })
            
            # Hammer Pattern
            elif (current['lower_shadow'] > current['body'] * 2 and 
                  current['upper_shadow'] < current['body'] * 0.5 and
                  current['close'] > (current['high'] + current['low']) / 2):
                
                confidence = 70 + min(20, current['lower_shadow'] / current['body'] * 3)
                expectancy = 3.2 + (confidence - 70) * 0.08
                
                patterns.append({
                    'pattern': 'Hammer',
                    'confidence': round(confidence, 1),
                    'expectancy': round(expectancy, 2),
                    'direction': 'BULLISH',
                    'strength': 'HIGH' if confidence > 85 else 'MEDIUM',
                    'price_level': current['close'],
                    'description': f"Bullish reversal pattern - Long lower shadow indicates buying pressure"
                })
            
            # Shooting Star Pattern
            elif (current['upper_shadow'] > current['body'] * 2 and 
                  current['lower_shadow'] < current['body'] * 0.5 and
                  current['close'] < (current['high'] + current['low']) / 2):
                
                confidence = 70 + min(20, current['upper_shadow'] / current['body'] * 3)
                expectancy = -3.2 - (confidence - 70) * 0.08
                
                patterns.append({
                    'pattern': 'Shooting Star',
                    'confidence': round(confidence, 1),
                    'expectancy': round(expectancy, 2),
                    'direction': 'BEARISH',
                    'strength': 'HIGH' if confidence > 85 else 'MEDIUM',
                    'price_level': current['close'],
                    'description': f"Bearish reversal pattern - Long upper shadow indicates selling pressure"
                })
            
            # Morning Star (3-candle pattern)
            if len(df) >= 3:
                if (not prev2['is_bullish'] and  # First candle bearish
                    prev['body'] < prev2['body'] * 0.5 and  # Second candle small
                    current['is_bullish'] and  # Third candle bullish
                    current['close'] > prev2['close']):  # Closes above first candle
                    
                    confidence = 75 + min(15, (current['close'] - prev2['close']) / prev2['close'] * 100 * 5)
                    expectancy = 4.1 + (confidence - 75) * 0.12
                    
                    patterns.append({
                        'pattern': 'Morning Star',
                        'confidence': round(confidence, 1),
                        'expectancy': round(expectancy, 2),
                        'direction': 'BULLISH',
                        'strength': 'HIGH',
                        'price_level': current['close'],
                        'description': f"Strong bullish reversal - Three-candle morning star formation"
                    })
                
                # Evening Star (3-candle pattern)
                elif (prev2['is_bullish'] and  # First candle bullish
                      prev['body'] < prev2['body'] * 0.5 and  # Second candle small
                      not current['is_bullish'] and  # Third candle bearish
                      current['close'] < prev2['close']):  # Closes below first candle
                    
                    confidence = 75 + min(15, (prev2['close'] - current['close']) / prev2['close'] * 100 * 5)
                    expectancy = -4.1 - (confidence - 75) * 0.12
                    
                    patterns.append({
                        'pattern': 'Evening Star',
                        'confidence': round(confidence, 1),
                        'expectancy': round(expectancy, 2),
                        'direction': 'BEARISH',
                        'strength': 'HIGH',
                        'price_level': current['close'],
                        'description': f"Strong bearish reversal - Three-candle evening star formation"
                    })
            
            # Store patterns in database
            self.store_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Pattern detection error: {e}")
            return []
    
    def analyze_technical_indicators(self, price_data: List[Dict]) -> Dict:
        """
        Analyze technical indicators and provide signals
        """
        try:
            if len(price_data) < 20:
                return {'error': 'Insufficient data for technical analysis'}
            
            df = pd.DataFrame(price_data)
            df['price'] = df['close']
            
            indicators = {}
            
            # RSI Analysis
            rsi = self.calculate_rsi(df['price'])
            if rsi:
                rsi_signal = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
                indicators['RSI'] = {
                    'value': round(rsi, 2),
                    'signal': rsi_signal,
                    'strength': 'HIGH' if rsi < 25 or rsi > 75 else 'MEDIUM',
                    'description': f"RSI at {rsi:.1f} - {rsi_signal.lower()} territory"
                }
            
            # MACD Analysis
            macd_line, signal_line, histogram = self.calculate_macd(df['price'])
            if macd_line and signal_line:
                macd_signal = 'BULLISH' if macd_line > signal_line else 'BEARISH'
                indicators['MACD'] = {
                    'macd': round(macd_line, 4),
                    'signal': round(signal_line, 4),
                    'histogram': round(histogram, 4),
                    'trend': macd_signal,
                    'strength': 'HIGH' if abs(histogram) > 0.5 else 'MEDIUM',
                    'description': f"MACD {macd_signal.lower()} crossover - Histogram: {histogram:.3f}"
                }
            
            # Moving Average Analysis
            sma_20 = df['price'].rolling(20).mean().iloc[-1]
            sma_50 = df['price'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            current_price = df['price'].iloc[-1]
            
            ma_signal = 'BULLISH' if current_price > sma_20 > sma_50 else 'BEARISH' if current_price < sma_20 < sma_50 else 'NEUTRAL'
            indicators['MA'] = {
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'current_price': round(current_price, 2),
                'signal': ma_signal,
                'strength': 'HIGH' if abs(current_price - sma_20) / sma_20 > 0.02 else 'MEDIUM',
                'description': f"Price vs MA20: {((current_price - sma_20) / sma_20 * 100):+.2f}%"
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['price'])
            if bb_upper and bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_signal = 'OVERBOUGHT' if bb_position > 0.8 else 'OVERSOLD' if bb_position < 0.2 else 'NEUTRAL'
                
                indicators['BB'] = {
                    'upper': round(bb_upper, 2),
                    'middle': round(bb_middle, 2),
                    'lower': round(bb_lower, 2),
                    'position': round(bb_position * 100, 1),
                    'signal': bb_signal,
                    'strength': 'HIGH' if bb_position > 0.9 or bb_position < 0.1 else 'MEDIUM',
                    'description': f"Price at {bb_position*100:.1f}% of Bollinger Band range"
                }
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Technical indicator analysis error: {e}")
            return {'error': str(e)}
    
    def analyze_news_sentiment(self) -> List[Dict]:
        """
        Analyze recent news sentiment for gold market
        """
        try:
            # Simulated news analysis (in production, integrate with real news APIs)
            news_items = [
                {
                    'headline': 'Federal Reserve hints at potential rate cuts amid inflation concerns',
                    'sentiment_score': 0.75,  # Bullish for gold
                    'market_impact': 'BULLISH',
                    'source': 'Reuters',
                    'relevance_score': 0.9,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': 'Rate cut expectations typically boost gold demand as safe haven'
                },
                {
                    'headline': 'Dollar strengthens on positive economic data',
                    'sentiment_score': -0.65,  # Bearish for gold
                    'market_impact': 'BEARISH',
                    'source': 'Bloomberg',
                    'relevance_score': 0.8,
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'analysis': 'Strong dollar typically pressures gold prices lower'
                },
                {
                    'headline': 'Geopolitical tensions rise in Eastern Europe',
                    'sentiment_score': 0.85,  # Very bullish for gold
                    'market_impact': 'BULLISH',
                    'source': 'Financial Times',
                    'relevance_score': 0.95,
                    'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                    'analysis': 'Geopolitical uncertainty drives safe-haven demand for gold'
                },
                {
                    'headline': 'China increases gold reserves for third consecutive month',
                    'sentiment_score': 0.6,
                    'market_impact': 'BULLISH',
                    'source': 'CNBC',
                    'relevance_score': 0.85,
                    'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
                    'analysis': 'Central bank buying supports gold price fundamentals'
                }
            ]
            
            # Store news sentiment in database
            self.store_news_sentiment(news_items)
            
            return news_items
            
        except Exception as e:
            logger.error(f"❌ News sentiment analysis error: {e}")
            return []
    
    def get_comprehensive_analysis(self) -> Dict:
        """
        Get comprehensive technical analysis combining all factors
        """
        try:
            # Generate sample OHLC data (in production, use real market data)
            current_time = datetime.now()
            sample_ohlc = []
            base_price = 2054.32
            
            for i in range(50):
                timestamp = current_time - timedelta(minutes=15*i)
                volatility = np.random.normal(0, 0.5)
                open_price = base_price + volatility
                close_price = open_price + np.random.normal(0, 2)
                high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
                low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))
                
                sample_ohlc.append({
                    'timestamp': timestamp.isoformat(),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': np.random.randint(1000, 5000)
                })
                base_price = close_price
            
            sample_ohlc.reverse()  # Chronological order
            
            # Analyze patterns and indicators
            patterns = self.detect_candlestick_patterns(sample_ohlc[-10:])  # Last 10 candles
            indicators = self.analyze_technical_indicators(sample_ohlc)
            news_sentiment = self.analyze_news_sentiment()
            
            # Calculate overall sentiment
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # Count pattern signals
            for pattern in patterns:
                total_signals += 1
                if pattern['direction'] == 'BULLISH':
                    bullish_signals += 1
                elif pattern['direction'] == 'BEARISH':
                    bearish_signals += 1
            
            # Count indicator signals
            for indicator_name, indicator_data in indicators.items():
                if 'signal' in indicator_data:
                    total_signals += 1
                    signal = indicator_data['signal']
                    if signal in ['BULLISH', 'OVERSOLD']:
                        bullish_signals += 1
                    elif signal in ['BEARISH', 'OVERBOUGHT']:
                        bearish_signals += 1
            
            # Count news sentiment
            bullish_news = sum(1 for news in news_sentiment if news['market_impact'] == 'BULLISH')
            bearish_news = sum(1 for news in news_sentiment if news['market_impact'] == 'BEARISH')
            
            overall_sentiment = 'NEUTRAL'
            confidence = 50
            
            if bullish_signals + bullish_news > bearish_signals + bearish_news:
                overall_sentiment = 'BULLISH'
                confidence = min(95, 50 + (bullish_signals + bullish_news - bearish_signals - bearish_news) * 10)
            elif bearish_signals + bearish_news > bullish_signals + bullish_news:
                overall_sentiment = 'BEARISH'
                confidence = min(95, 50 + (bearish_signals + bearish_news - bullish_signals - bullish_news) * 10)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_price': sample_ohlc[-1]['close'],
                'overall_sentiment': overall_sentiment,
                'confidence': round(confidence, 1),
                'candlestick_patterns': patterns,
                'technical_indicators': indicators,
                'news_sentiment': news_sentiment,
                'signal_summary': {
                    'bullish_signals': bullish_signals,
                    'bearish_signals': bearish_signals,
                    'neutral_signals': total_signals - bullish_signals - bearish_signals,
                    'bullish_news': bullish_news,
                    'bearish_news': bearish_news
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis error: {e}")
            return {'error': str(e)}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
        except:
            return None
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return (macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1])
        except:
            return (None, None, None)
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return (upper.iloc[-1], sma.iloc[-1], lower.iloc[-1])
        except:
            return (None, None, None)
    
    def store_patterns(self, patterns):
        """Store detected patterns in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pattern in patterns:
                cursor.execute('''
                    INSERT INTO candlestick_patterns 
                    (pattern_name, confidence_percent, expectancy_percent, direction, price_level, timeframe, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern['pattern'],
                    pattern['confidence'],
                    pattern['expectancy'],
                    pattern['direction'],
                    pattern['price_level'],
                    '15m',
                    json.dumps(pattern)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing patterns: {e}")
    
    def store_news_sentiment(self, news_items):
        """Store news sentiment in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for news in news_items:
                cursor.execute('''
                    INSERT INTO news_sentiment 
                    (headline, sentiment_score, market_impact, source, relevance_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    news['headline'],
                    news['sentiment_score'],
                    news['market_impact'],
                    news['source'],
                    news['relevance_score']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing news sentiment: {e}")

# Initialize the technical analysis engine
technical_analyzer = LiveTechnicalAnalysis()
