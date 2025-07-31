#!/usr/bin/env python3
"""
AI Trade Signal Generator - Generates high-ROI trade signals using multi-factor analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from price_storage_manager import get_current_gold_price, get_comprehensive_price_data
from ai_analysis_api import SimplifiedSentimentAnalyzer
from real_time_data_engine import RealTimeDataEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#!/usr/bin/env python3
"""
AI Trade Signal Generator - Generates high-ROI trade signals using multi-factor analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from price_storage_manager import get_current_gold_price, get_comprehensive_price_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AITradeSignalGenerator:
    """Advanced AI system for generating high-ROI trade signals"""
    
    def __init__(self):
        self.db_path = 'goldgpt_signals.db'
        self._initialize_db()
        self.last_signal = None
        self.min_signal_interval = 4  # hours
        
    def _initialize_db(self):
        """Initialize the signals database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create signals table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_signals (
            id INTEGER PRIMARY KEY,
            signal_type TEXT,
            entry_price REAL,
            target_price REAL,
            stop_loss REAL, 
            risk_reward_ratio REAL,
            confidence REAL,
            timestamp TEXT,
            timeframe TEXT,
            analysis_summary TEXT,
            factors_json TEXT,
            status TEXT DEFAULT 'open',
            exit_price REAL,
            profit_loss REAL,
            exit_timestamp TEXT
        )
        ''')
        conn.commit()
        conn.close()

    def generate_signal(self):
        """Generate a high-ROI trade signal based on multi-factor analysis"""
        try:
            # Get current market data
            current_price = get_current_gold_price()
            
            # Only generate new signal if enough time has passed
            if self._should_generate_new_signal():
                # Get all analysis factors
                technical_factors = self._analyze_technical_factors()
                fundamental_factors = self._analyze_fundamental_factors()
                pattern_factors = self._analyze_chart_patterns()
                sentiment_factors = self._analyze_sentiment()
                momentum_factors = self._analyze_momentum()
                
                # Calculate combined signal strength
                signal_strength = (
                    technical_factors['weight'] * technical_factors['signal'] +
                    fundamental_factors['weight'] * fundamental_factors['signal'] +
                    pattern_factors['weight'] * pattern_factors['signal'] + 
                    sentiment_factors['weight'] * sentiment_factors['signal'] +
                    momentum_factors['weight'] * momentum_factors['signal']
                )
                
                # Normalize to -1.0 to 1.0 range
                signal_strength = max(min(signal_strength, 1.0), -1.0)
                
                # Determine signal type
                signal_type = 'buy' if signal_strength > 0.2 else 'sell' if signal_strength < -0.2 else 'neutral'
                
                # Calculate confidence (0-100%)
                confidence = (abs(signal_strength) * 80) + 20 if abs(signal_strength) > 0.2 else 0
                
                # Skip neutral signals
                if signal_type == 'neutral':
                    return None
                    
                # Calculate take profit and stop loss
                volatility = self._calculate_volatility()
                risk_ratio = 0.01  # Risk 1%
                
                if signal_type == 'buy':
                    stop_loss = current_price * (1 - (volatility * risk_ratio))
                    target_price = current_price + (current_price - stop_loss) * 3  # 1:3 risk-reward ratio
                else:  # sell
                    stop_loss = current_price * (1 + (volatility * risk_ratio))
                    target_price = current_price - (stop_loss - current_price) * 3  # 1:3 risk-reward ratio
                
                # Create combined analysis factors
                all_factors = {
                    "technical": technical_factors,
                    "fundamental": fundamental_factors,
                    "patterns": pattern_factors,
                    "sentiment": sentiment_factors,
                    "momentum": momentum_factors,
                    "volatility": volatility,
                    "price": current_price
                }
                
                # Create analysis summary
                summary = self._create_analysis_summary(signal_type, all_factors, confidence)
                
                # Save signal to database
                signal_id = self._save_signal(
                    signal_type=signal_type,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=3.0,  # Fixed 1:3 risk-reward
                    confidence=confidence,
                    timeframe="1D",
                    analysis_summary=summary,
                    factors_json=json.dumps(all_factors)
                )
                
                # Return the signal data
                self.last_signal = {
                    "id": signal_id,
                    "type": signal_type,
                    "entry_price": round(current_price, 2),
                    "target_price": round(target_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "risk_reward": 3.0,
                    "confidence": round(confidence, 1),
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary,
                    "timeframe": "1D"
                }
                
                logger.info(f"Generated {signal_type} signal with {confidence:.1f}% confidence")
                return self.last_signal
            
            return self.last_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _should_generate_new_signal(self):
        """Check if we should generate a new signal based on time elapsed"""
        try:
            # Get the most recent signal
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT timestamp FROM trade_signals ORDER BY id DESC LIMIT 1')
            last_signal = cursor.fetchone()
            conn.close()
            
            if not last_signal:
                return True
                
            last_time = datetime.fromisoformat(last_signal[0])
            time_elapsed = datetime.now() - last_time
            
            # Only generate new signal if enough time has passed
            return time_elapsed.total_seconds() > (self.min_signal_interval * 3600)
        except Exception as e:
            logger.error(f"Error checking signal timing: {e}")
            return True
    
    def _analyze_technical_factors(self):
        """Analyze technical indicators for signal generation"""
        try:
            # Get price data for technical analysis
            price_data = get_comprehensive_price_data()
            historical_data = price_data.get('historical', [])
            
            if len(historical_data) < 20:
                return {'weight': 0.4, 'signal': 0, 'details': {}}
            
            # Convert to price list
            prices = [float(item.get('price', 0)) for item in historical_data[-50:]]
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices)
            
            # Calculate MACD
            macd_signal = self._calculate_macd(prices)
            
            # Calculate moving average crossover
            ma_cross = self._calculate_ma_crossover(prices)
            
            # Calculate Bollinger Bands position
            bb_position = self._calculate_bollinger_position(prices)
            
            # Calculate signal from technical indicators
            rsi_signal = 1.0 if rsi < 30 else -1.0 if rsi > 70 else 0
            macd_signal_val = 1.0 if macd_signal > 0 else -1.0 if macd_signal < 0 else 0
            ma_signal = ma_cross  # Already -1 to 1
            bb_signal = 1.0 if bb_position < 0.2 else -1.0 if bb_position > 0.8 else 0
            
            # Weighted average of signals
            signal_value = (rsi_signal * 0.3 + macd_signal_val * 0.3 + ma_signal * 0.2 + bb_signal * 0.2)
            
            return {
                'weight': 0.4,  # Technical analysis has 40% weight in overall signal
                'signal': signal_value,
                'details': {
                    'rsi': rsi,
                    'macd': macd_signal,
                    'ma_cross': ma_cross,
                    'bb_position': bb_position
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing technical factors: {e}")
            return {'weight': 0.4, 'signal': 0, 'details': {}}
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50
            
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
            
            if len(gains) < period:
                return 50
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return 50
    
    def _calculate_macd(self, prices):
        """Calculate MACD signal"""
        try:
            if len(prices) < 26:
                return 0
            
            # Calculate EMAs
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            
            macd_line = ema12 - ema26
            
            # Normalize to -1 to 1 range
            return max(min(macd_line / prices[-1] * 100, 1), -1)
        except:
            return 0
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return sum(prices) / len(prices)
            
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
        except:
            return prices[-1] if prices else 0
    
    def _calculate_ma_crossover(self, prices):
        """Calculate moving average crossover signal"""
        try:
            if len(prices) < 20:
                return 0
            
            ma10 = sum(prices[-10:]) / 10
            ma20 = sum(prices[-20:]) / 20
            
            # Normalize crossover signal
            crossover = (ma10 / ma20 - 1) * 100
            return max(min(crossover, 1), -1)
        except:
            return 0
    
    def _calculate_bollinger_position(self, prices, period=20):
        """Calculate position within Bollinger Bands"""
        try:
            if len(prices) < period:
                return 0.5
            
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / period
            
            # Calculate standard deviation
            variance = sum([(p - sma) ** 2 for p in recent_prices]) / period
            std_dev = variance ** 0.5
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            
            current_price = prices[-1]
            
            # Position within bands (0 = lower band, 1 = upper band)
            if upper_band == lower_band:
                return 0.5
            
            position = (current_price - lower_band) / (upper_band - lower_band)
            return max(min(position, 1), 0)
        except:
            return 0.5
    
    def _analyze_fundamental_factors(self):
        """Analyze fundamental/economic factors"""
        try:
            # Simulate economic indicators analysis
            # In production, this would connect to real economic data sources
            
            # Simulate USD strength (DXY trend)
            dxy_trend = np.random.uniform(-1, 1)
            
            # Simulate interest rate environment
            interest_rate_trend = np.random.uniform(-1, 1)
            
            # Simulate inflation impact
            inflation_impact = np.random.uniform(-0.5, 1)
            
            # Gold is inversely correlated with USD strength
            dxy_signal = -dxy_trend  # Invert DXY trend for gold
            
            # Higher interest rates are negative for gold
            rates_signal = -interest_rate_trend * 0.5
            
            # Higher inflation is positive for gold (hedge)
            inflation_signal = inflation_impact
            
            # Combined fundamental signal
            signal_value = (dxy_signal * 0.5 + rates_signal * 0.3 + inflation_signal * 0.2)
            
            return {
                'weight': 0.2,  # Fundamentals have 20% weight
                'signal': signal_value,
                'details': {
                    'dxy': dxy_trend,
                    'interest_rates': interest_rate_trend,
                    'inflation': inflation_impact
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing fundamental factors: {e}")
            return {'weight': 0.2, 'signal': 0, 'details': {}}
    
    def _analyze_chart_patterns(self):
        """Analyze candlestick and chart patterns"""
        try:
            # Get price data for pattern analysis
            price_data = get_comprehensive_price_data()
            historical_data = price_data.get('historical', [])
            
            if len(historical_data) < 10:
                return {'weight': 0.15, 'signal': 0, 'details': {}}
            
            # Convert to price list
            prices = [float(item.get('price', 0)) for item in historical_data[-30:]]
            
            # Check for key reversal patterns
            doji_pattern = self._detect_doji(prices)
            engulfing_pattern = self._detect_engulfing(prices)
            hammer_pattern = self._detect_hammer(prices)
            
            # Check for key continuation patterns
            flag_pattern = self._detect_flag_pattern(prices)
            triangle_pattern = self._detect_triangle_pattern(prices)
            
            # Calculate weighted pattern signal
            pattern_signal = (
                doji_pattern['signal'] * 0.2 +
                engulfing_pattern['signal'] * 0.3 +
                hammer_pattern['signal'] * 0.2 +
                flag_pattern['signal'] * 0.15 +
                triangle_pattern['signal'] * 0.15
            )
            
            return {
                'weight': 0.15,  # Patterns have 15% weight
                'signal': pattern_signal,
                'details': {
                    'doji': doji_pattern['found'],
                    'engulfing': engulfing_pattern['found'],
                    'hammer': hammer_pattern['found'],
                    'flag': flag_pattern['found'],
                    'triangle': triangle_pattern['found']
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing chart patterns: {e}")
            return {'weight': 0.15, 'signal': 0, 'details': {}}
        
    def _analyze_sentiment(self):
        """Analyze news and market sentiment"""
        try:
            # Use SimplifiedSentimentAnalyzer for real sentiment data
            sentiment_analyzer = SimplifiedSentimentAnalyzer()
            sentiment_data = sentiment_analyzer.analyze_sentiment("XAUUSD")
            
            # Extract sentiment values
            news_sentiment = sentiment_data.news_sentiment
            social_sentiment = sentiment_data.social_sentiment
            overall_sentiment = sentiment_data.overall_sentiment
            
            # Calculate composite sentiment signal
            sentiment_signal = (
                news_sentiment * 0.4 +
                social_sentiment * 0.3 +
                overall_sentiment * 0.3
            )
            
            return {
                'weight': 0.15,  # Sentiment has 15% weight
                'signal': sentiment_signal,
                'details': {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'overall': overall_sentiment,
                    'sentiment_score': sentiment_data.sentiment_score
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            # Fallback to default values if sentiment analysis fails
            return {
                'weight': 0.15, 
                'signal': 0, 
                'details': {
                    'news': 0,
                    'social': 0,
                    'overall': 0,
                    'sentiment_score': 0
                }
            }
        
    def _analyze_momentum(self):
        """Analyze price momentum factors"""
        try:
            # Get historical prices for momentum calculation
            price_data = get_comprehensive_price_data()
            historical_data = price_data.get('historical', [])
            
            if len(historical_data) < 20:
                return {'weight': 0.1, 'signal': 0, 'details': {}}
            
            # Convert to price list
            prices = [float(item.get('price', 0)) for item in historical_data[-20:]]
            
            # Calculate rate of change
            price_change = (prices[-1] - prices[0]) / prices[0]
            
            # Calculate momentum indicators
            price_roc = price_change * 10  # Rate of change
            
            # Create simple moving averages
            sma5 = sum(prices[-5:]) / 5
            sma20 = sum(prices) / 20
            
            # SMA alignment signal (-1 to 1)
            sma_alignment = min(max((sma5 / sma20 - 1) * 10, -1), 1)
            
            # Calculate momentum trend strength (-1 to 1)
            momentum_strength = min(max(price_roc, -1), 1)
            
            # Combined momentum signal
            momentum_signal = (momentum_strength * 0.7 + sma_alignment * 0.3)
            
            return {
                'weight': 0.1,  # Momentum has 10% weight
                'signal': momentum_signal,
                'details': {
                    'price_change': price_change,
                    'roc': price_roc,
                    'sma5': sma5,
                    'sma20': sma20
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {'weight': 0.1, 'signal': 0, 'details': {}}
        
    def _calculate_volatility(self):
        """Calculate current market volatility for position sizing"""
        try:
            # Get historical prices
            price_data = get_comprehensive_price_data()
            historical_data = price_data.get('historical', [])
            
            if len(historical_data) < 20:
                return 0.02  # Default volatility
            
            # Convert to price list
            prices = [float(item.get('price', 0)) for item in historical_data[-20:]]
            
            # Calculate daily returns
            returns = [((prices[i] / prices[i-1]) - 1) for i in range(1, len(prices))]
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns) if returns else 0.02
            
            return max(volatility, 0.005)  # Minimum volatility floor
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02  # Default volatility
    
    def _create_analysis_summary(self, signal_type, factors, confidence):
        """Create a human-readable analysis summary"""
        top_factors = []
        
        try:
            # Add technical factors
            tech = factors["technical"]
            tech_details = tech.get("details", {})
            
            rsi = tech_details.get("rsi", 50)
            if abs(rsi - 50) > 15:
                rsi_state = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
                top_factors.append(f"RSI is {rsi_state} at {rsi:.1f}")
            
            macd = tech_details.get("macd", 0)
            if abs(macd) > 0.1:
                macd_state = "bullish" if macd > 0 else "bearish"
                top_factors.append(f"MACD showing {macd_state} momentum")
            
            # Add sentiment if strong
            sentiment = factors["sentiment"]["signal"]
            if abs(sentiment) > 0.3:
                sentiment_state = "positive" if sentiment > 0 else "negative"
                top_factors.append(f"Market sentiment is strongly {sentiment_state}")
            
            # Add strongest pattern if any
            patterns = factors["patterns"]["details"]
            for pattern, found in patterns.items():
                if found:
                    top_factors.append(f"Detected {pattern} pattern")
                    break
            
            # Create summary with signal type and confidence
            signal_text = signal_type.upper()
            summary = f"{signal_text} signal with {confidence:.1f}% confidence. "
            
            if top_factors:
                summary += "Key factors: " + "; ".join(top_factors) + "."
            else:
                summary += "Based on multi-factor analysis of technical and fundamental indicators."
            
            return summary
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return f"{signal_type.upper()} signal with {confidence:.1f}% confidence based on AI analysis."
        
    def _save_signal(self, **kwargs):
        """Save signal to database and return ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO trade_signals 
                (signal_type, entry_price, target_price, stop_loss, 
                 risk_reward_ratio, confidence, timestamp, timeframe, 
                 analysis_summary, factors_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['signal_type'],
                kwargs['entry_price'],
                kwargs['target_price'],
                kwargs['stop_loss'],
                kwargs['risk_reward_ratio'],
                kwargs['confidence'],
                timestamp,
                kwargs['timeframe'],
                kwargs['analysis_summary'],
                kwargs['factors_json']
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return signal_id
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return None
        
    # Pattern detection helpers
    def _detect_doji(self, prices):
        """Detect doji candlestick pattern"""
        try:
            if len(prices) < 2:
                return {'found': False, 'signal': 0}
                
            # Simple implementation - check if consecutive prices are very close
            price_diff = abs(prices[-1] - prices[-2]) / prices[-2]
            is_doji = price_diff < 0.0005  # Very small body
            
            # Doji can be reversal or indecision
            signal = 0
            if is_doji:
                # Check trend before doji
                trend = self._calculate_trend(prices[:-1])
                signal = -trend * 0.5  # Potential reversal of previous trend, but weak signal
                
            return {'found': is_doji, 'signal': signal}
        except Exception as e:
            logger.error(f"Error detecting doji: {e}")
            return {'found': False, 'signal': 0}
        
    def _detect_engulfing(self, prices):
        """Detect bullish or bearish engulfing pattern"""
        try:
            if len(prices) < 4:
                return {'found': False, 'signal': 0}
                
            # Simplified implementation without OHLC data
            p1, p2, p3 = prices[-4], prices[-3], prices[-2]
            current = prices[-1]
            
            # Bullish engulfing: down trend, then current candle engulfs previous
            if p1 > p2 and p2 > p3 and current > p3 and (current - p3) > (p2 - p3):
                return {'found': True, 'signal': 0.8}  # Strong bullish signal
                
            # Bearish engulfing: up trend, then current candle engulfs previous
            if p1 < p2 and p2 < p3 and current < p3 and (p3 - current) > (p3 - p2):
                return {'found': True, 'signal': -0.8}  # Strong bearish signal
                
            return {'found': False, 'signal': 0}
        except Exception as e:
            logger.error(f"Error detecting engulfing: {e}")
            return {'found': False, 'signal': 0}
        
    def _detect_hammer(self, prices):
        """Detect hammer pattern - a potential reversal pattern"""
        try:
            if len(prices) < 5:
                return {'found': False, 'signal': 0}
                
            p1, p2, p3, p4 = prices[-5], prices[-4], prices[-3], prices[-2]
            current = prices[-1]
            
            # Downtrend followed by potential reversal
            if p1 > p2 and p2 > p3 and p3 > p4 and current > p4:
                # Hammer strength based on reversal magnitude
                strength = min((current - p4) / p4 * 5, 1.0)
                return {'found': True, 'signal': strength}  # Bullish signal
                
            return {'found': False, 'signal': 0}
        except Exception as e:
            logger.error(f"Error detecting hammer: {e}")
            return {'found': False, 'signal': 0}
        
    def _detect_flag_pattern(self, prices):
        """Detect bull/bear flag pattern (continuation)"""
        try:
            if len(prices) < 10:
                return {'found': False, 'signal': 0}
                
            # Calculate trend
            trend = self._calculate_trend(prices[:-5])
            
            # Check for consolidation (flag)
            recent_prices = prices[-5:]
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = sum(recent_prices) / len(recent_prices)
            
            # Flag is a tight consolidation (low volatility)
            is_consolidation = price_range / avg_price < 0.01
            
            if is_consolidation and abs(trend) > 0.5:
                # Flag strength is based on prior trend strength
                return {'found': True, 'signal': trend * 0.7}  # Continuation signal in trend direction
                
            return {'found': False, 'signal': 0}
        except Exception as e:
            logger.error(f"Error detecting flag pattern: {e}")
            return {'found': False, 'signal': 0}
        
    def _detect_triangle_pattern(self, prices):
        """Detect triangle pattern"""
        try:
            if len(prices) < 15:
                return {'found': False, 'signal': 0}
                
            # Check for decreasing volatility
            volatility = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            
            early_vol = sum(volatility[:7]) / 7
            recent_vol = sum(volatility[-7:]) / 7
            
            is_converging = recent_vol < early_vol * 0.7
            
            if is_converging:
                # Triangle breakout direction often follows the main trend
                trend = self._calculate_trend(prices)
                return {'found': True, 'signal': trend * 0.5}
                
            return {'found': False, 'signal': 0}
        except Exception as e:
            logger.error(f"Error detecting triangle pattern: {e}")
            return {'found': False, 'signal': 0}
        
    def _calculate_trend(self, prices):
        """Calculate trend strength from -1 (down) to 1 (up)"""
        try:
            if len(prices) < 2:
                return 0
                
            # Linear regression slope would be better, but for simplicity:
            first, last = prices[0], prices[-1]
            change = (last - first) / first
            
            # Normalize to -1 to 1 range
            return max(min(change * 20, 1.0), -1.0)
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0
        
    def get_open_signals(self):
        """Get all open trade signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM trade_signals 
            WHERE status = 'open'
            ORDER BY id DESC
            ''')
            
            signals = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return signals
        except Exception as e:
            logger.error(f"Error getting open signals: {e}")
            return []
        
    def update_signal_status(self, signal_id, current_price):
        """Update signal status based on current price"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get signal details
            cursor.execute('SELECT * FROM trade_signals WHERE id = ?', (signal_id,))
            signal = cursor.fetchone()
            
            if not signal:
                conn.close()
                return False
                
            # Extract relevant data
            signal_type = signal[1]  # signal_type
            entry_price = signal[2]  # entry_price
            target_price = signal[3]  # target_price
            stop_loss = signal[4]    # stop_loss
            
            # Check if target or stop loss hit
            status = 'open'
            exit_price = None
            profit_loss = None
            
            if signal_type == 'buy':
                if current_price >= target_price:
                    status = 'target_hit'
                    exit_price = target_price
                    profit_loss = (exit_price - entry_price) / entry_price * 100
                elif current_price <= stop_loss:
                    status = 'stop_loss'
                    exit_price = stop_loss
                    profit_loss = (exit_price - entry_price) / entry_price * 100
                    
            elif signal_type == 'sell':
                if current_price <= target_price:
                    status = 'target_hit'
                    exit_price = target_price
                    profit_loss = (entry_price - exit_price) / entry_price * 100
                elif current_price >= stop_loss:
                    status = 'stop_loss'
                    exit_price = stop_loss
                    profit_loss = (entry_price - exit_price) / entry_price * 100
            
            # Update signal if status changed
            if status != 'open':
                cursor.execute('''
                UPDATE trade_signals 
                SET status = ?, exit_price = ?, profit_loss = ?, exit_timestamp = ?
                WHERE id = ?
                ''', (status, exit_price, profit_loss, datetime.now().isoformat(), signal_id))
                conn.commit()
                
            conn.close()
            return status != 'open'
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return False
        
    def get_signal_statistics(self):
        """Get performance statistics for signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total counts
            cursor.execute('SELECT COUNT(*) FROM trade_signals WHERE status != "open"')
            total_closed = cursor.fetchone()[0]
            
            if total_closed == 0:
                conn.close()
                return {
                    'total_signals': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_return': 0
                }
            
            # Get wins
            cursor.execute('SELECT COUNT(*) FROM trade_signals WHERE status = "target_hit"')
            total_wins = cursor.fetchone()[0]
            
            # Get average profit
            cursor.execute('''
            SELECT AVG(profit_loss) FROM trade_signals 
            WHERE status = "target_hit"
            ''')
            avg_profit = cursor.fetchone()[0] or 0
            
            # Get average loss
            cursor.execute('''
            SELECT AVG(profit_loss) FROM trade_signals 
            WHERE status = "stop_loss"
            ''')
            avg_loss = cursor.fetchone()[0] or 0
            
            # Calculate profit factor
            cursor.execute('''
            SELECT SUM(profit_loss) FROM trade_signals 
            WHERE status = "target_hit"
            ''')
            total_profit = cursor.fetchone()[0] or 0
            
            cursor.execute('''
            SELECT SUM(profit_loss) FROM trade_signals 
            WHERE status = "stop_loss"
            ''')
            total_loss = abs(cursor.fetchone()[0] or 0.001)  # Avoid division by zero
            
            profit_factor = total_profit / total_loss if total_loss > 0 else total_profit
            
            # Get total return
            cursor.execute('SELECT SUM(profit_loss) FROM trade_signals WHERE status != "open"')
            total_return = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_signals': total_closed,
                'win_rate': (total_wins / total_closed) * 100 if total_closed > 0 else 0,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_return': total_return
            }
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {
                'total_signals': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0
            }

# Global instance
try:
    signal_generator = AITradeSignalGenerator()
    logger.info("✅ AI Signal Generator initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize AI Signal Generator: {e}")
    signal_generator = None

def get_trade_signal():
    """Get the latest trade signal"""
    if signal_generator is None:
        logger.error("Signal generator not initialized")
        return None
    return signal_generator.generate_signal()

def get_open_trade_signals():
    """Get all open trade signals"""
    if signal_generator is None:
        logger.warning("Signal generator not initialized")
        return []
    return signal_generator.get_open_signals()

def get_signal_stats():
    """Get signal performance statistics"""
    if signal_generator is None:
        logger.warning("Signal generator not initialized")
        return {
            'total_signals': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_return': 0
        }
    return signal_generator.get_signal_statistics()

def update_signals_status():
    """Update status of all open signals"""
    try:
        current_price = get_current_gold_price()
        open_signals = signal_generator.get_open_signals()
        
        for signal in open_signals:
            signal_generator.update_signal_status(signal['id'], current_price)
            
        return True
    except Exception as e:
        logger.error(f"Error updating signals status: {e}")
        return False
