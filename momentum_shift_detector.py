#!/usr/bin/env python3
"""
Enhanced Momentum Shift Detection System - Fixed Version
Detects and alerts on bullish/bearish momentum shifts using multiple indicators
"""
import os
import sys
import logging
import time
import sqlite3
import asyncio
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

# Try to import talib, fallback to pandas_ta if not available
try:
    import talib as ta_lib
    USE_TALIB = True
except ImportError:
    ta_lib = None
    USE_TALIB = False

try:
    import pandas_ta as ta
    USE_PANDAS_TA = True
except ImportError:
    print("Warning: Neither talib nor pandas_ta available. Using basic calculations.")
    ta = None
    USE_PANDAS_TA = False

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from bot_modules.config import DB_PATH, TELEGRAM_BOT_TOKEN, CHAT_ID

class MomentumShiftDetector:
    """Detects momentum shifts using multiple indicators and sends Telegram alerts"""
    
    def __init__(self):
        self.last_momentum_state = None
        self.momentum_history = []
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        self.last_alert_time = {}
        
        # Momentum detection thresholds
        self.bullish_threshold = 0.6  # 60% bullish signals
        self.bearish_threshold = -0.6  # 60% bearish signals
        self.momentum_confirmation_periods = 3  # Need 3 consecutive periods
        
        # Initialize database table
        self._init_momentum_table()
    
    def _init_momentum_table(self):
        """Initialize momentum tracking table"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS momentum_shifts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        shift_type TEXT NOT NULL,
                        momentum_score REAL,
                        confidence REAL,
                        indicators TEXT,
                        alert_sent BOOLEAN DEFAULT 0
                    )
                """)
                conn.commit()
        except Exception as e:
            logging.error(f"Error initializing momentum table: {e}")
    
    def fetch_market_data(self) -> Dict:
        """Fetch current market data for analysis"""
        try:
            # Fetch gold price data
            df = yf.download("GC=F", period="5d", interval="15m", progress=False)
            if df.empty or len(df) < 50:
                logging.warning("Insufficient data for momentum analysis")
                return {}
            
            # Ensure we have valid OHLC data
            required_columns = ['Open', 'High', 'Low', 'Close']
            for col in required_columns:
                if col not in df.columns:
                    logging.error(f"Missing required column: {col}")
                    return {}
            
            # Convert to numpy arrays and ensure they're contiguous
            open_prices = np.ascontiguousarray(df['Open'].values, dtype=np.float64)
            high = np.ascontiguousarray(df['High'].values, dtype=np.float64)
            low = np.ascontiguousarray(df['Low'].values, dtype=np.float64)
            close = np.ascontiguousarray(df['Close'].values, dtype=np.float64)
            volume = np.ascontiguousarray(df['Volume'].values, dtype=np.float64) if 'Volume' in df.columns else None
            
            return {
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'timestamp': df.index[-1]
            }
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return {}
    
    def calculate_momentum_indicators(self, data: Dict) -> Dict:
        """Calculate momentum indicators using preferred library"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            indicators = {}
            
            # Ensure we have enough data points
            if len(close) < 50:
                logging.warning("Insufficient data for reliable momentum indicators")
                return {
                    'rsi': 50,
                    'rsi_trend': 'neutral',
                    'macd': 0,
                    'macd_signal': 0,
                    'macd_histogram': 0,
                    'macd_crossover': 'none',
                    'ema20': close[-1],
                    'ema50': close[-1],
                    'ma_trend': 'neutral',
                    'stoch_k': 50,
                    'stoch_d': 50,
                    'adx': 25,
                    'bb_position': 0.5,
                    'bb_squeeze': False,
                    'price_momentum_5m': 0,
                    'price_momentum_15m': 0
                }
            
            # Try pandas_ta first
            if USE_PANDAS_TA and ta is not None:
                indicators = self._calculate_with_pandas_ta(data)
            
            # Fallback to talib if pandas_ta fails or isn't available
            if not indicators and USE_TALIB and ta_lib is not None:
                indicators = self._calculate_with_talib(data)
            
            # Final fallback to basic calculations
            if not indicators:
                indicators = self._calculate_basic_indicators(data)
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_with_pandas_ta(self, data: Dict) -> Dict:
        """Calculate indicators using pandas_ta"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Ensure we have 1-dimensional arrays
            close_1d = np.ravel(close) if hasattr(close, 'shape') and len(close.shape) > 1 else close
            high_1d = np.ravel(high) if hasattr(high, 'shape') and len(high.shape) > 1 else high
            low_1d = np.ravel(low) if hasattr(low, 'shape') and len(low.shape) > 1 else low
            open_1d = np.ravel(data.get('open', close_1d)) if 'open' in data else close_1d
            
            df = pd.DataFrame({
                'close': close_1d, 
                'high': high_1d, 
                'low': low_1d,
                'open': open_1d
            })
            
            indicators = {}
            
            # RSI
            try:
                rsi_series = ta.rsi(df['close'], length=14)
                if rsi_series is not None and len(rsi_series) > 0:
                    indicators['rsi'] = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
                    if len(rsi_series) >= 5:
                        indicators['rsi_trend'] = 'bullish' if rsi_series.iloc[-1] > rsi_series.iloc[-5] else 'bearish' if rsi_series.iloc[-1] < rsi_series.iloc[-5] else 'neutral'
                    else:
                        indicators['rsi_trend'] = 'neutral'
                else:
                    indicators['rsi'] = self._calculate_basic_rsi(close_1d)
                    indicators['rsi_trend'] = 'neutral'
            except Exception as e:
                logging.warning(f"RSI calculation failed: {e}")
                indicators['rsi'] = self._calculate_basic_rsi(close_1d)
                indicators['rsi_trend'] = 'neutral'
            
            # MACD
            try:
                macd_data = ta.macd(df['close'])
                if macd_data is not None and len(macd_data) > 0:
                    macd_col = 'MACD_12_26_9'
                    signal_col = 'MACDs_12_26_9'
                    hist_col = 'MACDh_12_26_9'
                    
                    indicators['macd'] = macd_data[macd_col].iloc[-1] if macd_col in macd_data.columns and not pd.isna(macd_data[macd_col].iloc[-1]) else 0
                    indicators['macd_signal'] = macd_data[signal_col].iloc[-1] if signal_col in macd_data.columns and not pd.isna(macd_data[signal_col].iloc[-1]) else 0
                    indicators['macd_histogram'] = macd_data[hist_col].iloc[-1] if hist_col in macd_data.columns and not pd.isna(macd_data[hist_col].iloc[-1]) else 0
                    
                    # Calculate crossover
                    if len(macd_data) >= 2 and macd_col in macd_data.columns and signal_col in macd_data.columns:
                        current_macd = macd_data[macd_col].iloc[-1]
                        current_signal = macd_data[signal_col].iloc[-1]
                        prev_macd = macd_data[macd_col].iloc[-2]
                        prev_signal = macd_data[signal_col].iloc[-2]
                        
                        if current_macd > current_signal and prev_macd <= prev_signal:
                            indicators['macd_crossover'] = 'bullish'
                        elif current_macd < current_signal and prev_macd >= prev_signal:
                            indicators['macd_crossover'] = 'bearish'
                        else:
                            indicators['macd_crossover'] = 'none'
                    else:
                        indicators['macd_crossover'] = 'none'
                else:
                    indicators['macd'] = 0
                    indicators['macd_signal'] = 0
                    indicators['macd_histogram'] = 0
                    indicators['macd_crossover'] = 'none'
            except Exception as e:
                logging.warning(f"MACD calculation failed: {e}")
                indicators['macd'] = 0
                indicators['macd_signal'] = 0
                indicators['macd_histogram'] = 0
                indicators['macd_crossover'] = 'none'
            
            # EMAs
            try:
                ema20 = ta.ema(df['close'], length=20)
                ema50 = ta.ema(df['close'], length=50)
                
                indicators['ema20'] = ema20.iloc[-1] if ema20 is not None and len(ema20) > 0 and not pd.isna(ema20.iloc[-1]) else close_1d[-1]
                indicators['ema50'] = ema50.iloc[-1] if ema50 is not None and len(ema50) > 0 and not pd.isna(ema50.iloc[-1]) else close_1d[-1]
                indicators['ma_trend'] = 'bullish' if indicators['ema20'] > indicators['ema50'] else 'bearish'
            except Exception as e:
                logging.warning(f"EMA calculation failed: {e}")
                indicators['ema20'] = np.mean(close_1d[-20:]) if len(close_1d) >= 20 else close_1d[-1]
                indicators['ema50'] = np.mean(close_1d[-50:]) if len(close_1d) >= 50 else close_1d[-1]
                indicators['ma_trend'] = 'bullish' if indicators['ema20'] > indicators['ema50'] else 'bearish'
            
            # Stochastic
            try:
                stoch = ta.stoch(df['high'], df['low'], df['close'])
                if stoch is not None and len(stoch) > 0:
                    k_col = 'STOCHk_14_3_3'
                    d_col = 'STOCHd_14_3_3'
                    indicators['stoch_k'] = stoch[k_col].iloc[-1] if k_col in stoch.columns and not pd.isna(stoch[k_col].iloc[-1]) else 50
                    indicators['stoch_d'] = stoch[d_col].iloc[-1] if d_col in stoch.columns and not pd.isna(stoch[d_col].iloc[-1]) else 50
                else:
                    indicators['stoch_k'] = 50
                    indicators['stoch_d'] = 50
            except Exception as e:
                logging.warning(f"Stochastic calculation failed: {e}")
                indicators['stoch_k'] = 50
                indicators['stoch_d'] = 50
            
            # ADX - Enhanced error handling for insufficient data
            try:
                # ADX requires at least 14 periods, plus additional for proper calculation
                if len(df) >= 20:  # Require more data for stable ADX
                    adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
                    if adx_data is not None and len(adx_data) > 0:
                        adx_col = 'ADX_14'
                        if adx_col in adx_data.columns:
                            adx_value = adx_data[adx_col].iloc[-1]
                            indicators['adx'] = adx_value if not pd.isna(adx_value) else 20
                        else:
                            # Try alternative column name
                            adx_cols = [col for col in adx_data.columns if 'ADX' in col]
                            if adx_cols:
                                adx_value = adx_data[adx_cols[0]].iloc[-1]
                                indicators['adx'] = adx_value if not pd.isna(adx_value) else 20
                            else:
                                indicators['adx'] = 20
                    else:
                        indicators['adx'] = 20
                else:
                    # Insufficient data for ADX calculation
                    indicators['adx'] = 20
                    logging.debug(f"Insufficient data for ADX: {len(df)} rows (need at least 20)")
            except Exception as e:
                logging.warning(f"Error calculating ADX: {e}")
                indicators['adx'] = 20
            
            # Bollinger Bands
            try:
                bb = ta.bbands(df['close'])
                if bb is not None and len(bb) > 0:
                    # Find column names dynamically
                    upper_cols = [col for col in bb.columns if 'upper' in col.lower() or 'bbu' in col.lower()]
                    lower_cols = [col for col in bb.columns if 'lower' in col.lower() or 'bbl' in col.lower()]
                    
                    if upper_cols and lower_cols:
                        bb_upper = bb[upper_cols[0]].iloc[-1]
                        bb_lower = bb[lower_cols[0]].iloc[-1]
                        current_price = close_1d[-1]
                        
                        if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                            indicators['bb_position'] = max(0, min(1, bb_position))
                            
                            # Check for squeeze
                            if len(bb) >= 10:
                                prev_range = bb[upper_cols[0]].iloc[-10] - bb[lower_cols[0]].iloc[-10]
                                current_range = bb_upper - bb_lower
                                indicators['bb_squeeze'] = current_range < prev_range if not pd.isna(prev_range) else False
                            else:
                                indicators['bb_squeeze'] = False
                        else:
                            indicators['bb_position'] = 0.5
                            indicators['bb_squeeze'] = False
                    else:
                        indicators['bb_position'] = 0.5
                        indicators['bb_squeeze'] = False
                else:
                    indicators['bb_position'] = 0.5
                    indicators['bb_squeeze'] = False
            except Exception as e:
                logging.warning(f"Bollinger Bands calculation failed: {e}")
                indicators['bb_position'] = 0.5
                indicators['bb_squeeze'] = False
            
            # Price momentum
            price_change_5m = (close_1d[-1] - close_1d[-5]) / close_1d[-5] * 100 if len(close_1d) >= 5 else 0
            price_change_15m = (close_1d[-1] - close_1d[-15]) / close_1d[-15] * 100 if len(close_1d) >= 15 else 0
            indicators['price_momentum_5m'] = price_change_5m
            indicators['price_momentum_15m'] = price_change_15m
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error in pandas_ta calculations: {e}")
            return {}
    
    def _calculate_with_talib(self, data: Dict) -> Dict:
        """Calculate indicators using TA-Lib"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            indicators = {}
            
            # RSI
            try:
                rsi = ta_lib.RSI(close, timeperiod=14)
                indicators['rsi'] = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50
                if len(rsi) >= 5:
                    indicators['rsi_trend'] = 'bullish' if rsi[-1] > rsi[-5] else 'bearish' if rsi[-1] < rsi[-5] else 'neutral'
                else:
                    indicators['rsi_trend'] = 'neutral'
            except Exception as e:
                logging.warning(f"RSI calculation failed: {e}")
                indicators['rsi'] = self._calculate_basic_rsi(close)
                indicators['rsi_trend'] = 'neutral'
            
            # MACD
            try:
                macd, macd_signal, macd_hist = ta_lib.MACD(close)
                indicators['macd'] = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
                indicators['macd_signal'] = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
                indicators['macd_histogram'] = macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0
                
                # Calculate crossover
                if len(macd) >= 2 and len(macd_signal) >= 2:
                    if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                        indicators['macd_crossover'] = 'bullish'
                    elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                        indicators['macd_crossover'] = 'bearish'
                    else:
                        indicators['macd_crossover'] = 'none'
                else:
                    indicators['macd_crossover'] = 'none'
            except Exception as e:
                logging.warning(f"MACD calculation failed: {e}")
                indicators['macd'] = 0
                indicators['macd_signal'] = 0
                indicators['macd_histogram'] = 0
                indicators['macd_crossover'] = 'none'
            
            # EMAs
            try:
                ema20 = ta_lib.EMA(close, timeperiod=20)
                ema50 = ta_lib.EMA(close, timeperiod=50)
                indicators['ema20'] = ema20[-1] if len(ema20) > 0 and not np.isnan(ema20[-1]) else close[-1]
                indicators['ema50'] = ema50[-1] if len(ema50) > 0 and not np.isnan(ema50[-1]) else close[-1]
                indicators['ma_trend'] = 'bullish' if indicators['ema20'] > indicators['ema50'] else 'bearish'
            except Exception as e:
                logging.warning(f"EMA calculation failed: {e}")
                indicators['ema20'] = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
                indicators['ema50'] = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
                indicators['ma_trend'] = 'bullish' if indicators['ema20'] > indicators['ema50'] else 'bearish'
            
            # Stochastic
            try:
                stoch_k, stoch_d = ta_lib.STOCH(high, low, close)
                indicators['stoch_k'] = stoch_k[-1] if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]) else 50
                indicators['stoch_d'] = stoch_d[-1] if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else 50
            except Exception as e:
                logging.warning(f"Stochastic calculation failed: {e}")
                indicators['stoch_k'] = 50
                indicators['stoch_d'] = 50
            
            # ADX
            try:
                adx = ta_lib.ADX(high, low, close, timeperiod=14)
                indicators['adx'] = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 20
            except Exception as e:
                logging.warning(f"ADX calculation failed: {e}")
                indicators['adx'] = 20
            
            # Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = ta_lib.BBANDS(close)
                current_price = close[-1]
                if len(bb_upper) > 0 and len(bb_lower) > 0 and not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
                    bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                    indicators['bb_position'] = max(0, min(1, bb_position))
                    
                    # Check for squeeze
                    if len(bb_upper) >= 10:
                        indicators['bb_squeeze'] = (bb_upper[-1] - bb_lower[-1]) < (bb_upper[-10] - bb_lower[-10])
                    else:
                        indicators['bb_squeeze'] = False
                else:
                    indicators['bb_position'] = 0.5
                    indicators['bb_squeeze'] = False
            except Exception as e:
                logging.warning(f"Bollinger Bands calculation failed: {e}")
                indicators['bb_position'] = 0.5
                indicators['bb_squeeze'] = False
            
            # Price momentum
            close_1d = np.ravel(close) if hasattr(close, 'shape') and len(close.shape) > 1 else close
            price_change_5m = (close_1d[-1] - close_1d[-5]) / close_1d[-5] * 100 if len(close_1d) >= 5 else 0
            price_change_15m = (close_1d[-1] - close_1d[-15]) / close_1d[-15] * 100 if len(close_1d) >= 15 else 0
            indicators['price_momentum_5m'] = price_change_5m
            indicators['price_momentum_15m'] = price_change_15m
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error in TA-Lib calculations: {e}")
            return {}
    
    def _calculate_basic_indicators(self, data: Dict) -> Dict:
        """Basic fallback calculations without external libraries"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            indicators = {}
            
            # Basic RSI
            indicators['rsi'] = self._calculate_basic_rsi(close)
            indicators['rsi_trend'] = 'neutral'
            
            # Basic moving averages
            indicators['ema20'] = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
            indicators['ema50'] = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
            indicators['ma_trend'] = 'bullish' if indicators['ema20'] > indicators['ema50'] else 'bearish'
            
            # Basic values for other indicators
            indicators['macd'] = 0
            indicators['macd_signal'] = 0
            indicators['macd_histogram'] = 0
            indicators['macd_crossover'] = 'none'
            indicators['stoch_k'] = 50
            indicators['stoch_d'] = 50
            indicators['adx'] = 25
            indicators['bb_position'] = 0.5
            indicators['bb_squeeze'] = False
            
            # Price momentum
            close_1d = np.ravel(close) if hasattr(close, 'shape') and len(close.shape) > 1 else close
            price_change_5m = (close_1d[-1] - close_1d[-5]) / close_1d[-5] * 100 if len(close_1d) >= 5 else 0
            price_change_15m = (close_1d[-1] - close_1d[-15]) / close_1d[-15] * 100 if len(close_1d) >= 15 else 0
            indicators['price_momentum_5m'] = price_change_5m
            indicators['price_momentum_15m'] = price_change_15m
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error in basic calculations: {e}")
            return {}
    
    def _calculate_basic_rsi(self, close_prices, period=14):
        """Calculate basic RSI without external libraries"""
        try:
            close_1d = np.ravel(close_prices) if hasattr(close_prices, 'shape') and len(close_prices.shape) > 1 else close_prices
            
            if len(close_1d) < period + 1:
                return 50
            
            deltas = np.diff(close_1d)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return 50
    
    def calculate_momentum_score(self, indicators: Dict) -> Tuple[float, str]:
        """Calculate overall momentum score (-1 to 1, where -1 is bearish, 1 is bullish)"""
        try:
            score = 0.0
            reasons = []
            
            # RSI contribution (20% weight)
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                score -= 0.15
                reasons.append("RSI overbought")
            elif rsi > 60:
                score += 0.1
                reasons.append("RSI bullish")
            elif rsi < 30:
                score += 0.15
                reasons.append("RSI oversold (bullish)")
            elif rsi < 40:
                score -= 0.1
                reasons.append("RSI bearish")
            
            # MACD contribution (25% weight)
            macd_crossover = indicators.get('macd_crossover', 'none')
            macd_hist = indicators.get('macd_histogram', 0)
            if macd_crossover == 'bullish':
                score += 0.2
                reasons.append("MACD bullish crossover")
            elif macd_crossover == 'bearish':
                score -= 0.2
                reasons.append("MACD bearish crossover")
            elif macd_hist > 0:
                score += 0.1
                reasons.append("MACD histogram positive")
            elif macd_hist < 0:
                score -= 0.1
                reasons.append("MACD histogram negative")
            
            # Moving average trend (20% weight)
            ma_trend = indicators.get('ma_trend', 'neutral')
            if ma_trend == 'bullish':
                score += 0.15
                reasons.append("EMA trend bullish")
            elif ma_trend == 'bearish':
                score -= 0.15
                reasons.append("EMA trend bearish")
            
            # Price momentum (15% weight)
            price_momentum_5m = indicators.get('price_momentum_5m', 0)
            price_momentum_15m = indicators.get('price_momentum_15m', 0)
            if price_momentum_5m > 0.5:
                score += 0.1
                reasons.append("Strong 5m price momentum")
            elif price_momentum_5m < -0.5:
                score -= 0.1
                reasons.append("Weak 5m price momentum")
            
            if price_momentum_15m > 1.0:
                score += 0.05
                reasons.append("Strong 15m momentum")
            elif price_momentum_15m < -1.0:
                score -= 0.05
                reasons.append("Weak 15m momentum")
            
            # ADX trend strength (10% weight)
            adx = indicators.get('adx', 20)
            if adx > 30:
                # Strong trend - amplify other signals
                score *= 1.2
                reasons.append("Strong trend confirmation")
            
            # Stochastic (10% weight)
            stoch_k = indicators.get('stoch_k', 50)
            if stoch_k > 80:
                score -= 0.05
                reasons.append("Stochastic overbought")
            elif stoch_k < 20:
                score += 0.05
                reasons.append("Stochastic oversold")
            
            # Ensure score stays within bounds
            score = max(-1.0, min(1.0, score))
            
            return score, "; ".join(reasons)
            
        except Exception as e:
            logging.error(f"Error calculating momentum score: {e}")
            return 0.0, "Error in calculation"
    
    def detect_momentum_shift(self, current_score: float, reasons: str) -> Optional[Dict]:
        """Detect if there's been a significant momentum shift"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Add to momentum history
            self.momentum_history.append({
                'timestamp': current_time,
                'score': current_score,
                'reasons': reasons
            })
            
            # Keep only recent history (last 2 hours)
            cutoff_time = current_time - timedelta(hours=2)
            self.momentum_history = [
                h for h in self.momentum_history 
                if h['timestamp'] > cutoff_time
            ]
            
            # Need at least 3 periods for momentum shift detection
            if len(self.momentum_history) < self.momentum_confirmation_periods:
                return None
            
            # Get recent scores
            recent_scores = [h['score'] for h in self.momentum_history[-self.momentum_confirmation_periods:]]
            avg_recent_score = np.mean(recent_scores)
            
            # Determine current momentum state
            if avg_recent_score >= self.bullish_threshold:
                current_state = 'bullish'
            elif avg_recent_score <= self.bearish_threshold:
                current_state = 'bearish'
            else:
                current_state = 'neutral'
            
            # Check for momentum shift
            if self.last_momentum_state is None:
                self.last_momentum_state = current_state
                return None
            
            # Detect shift
            shift_detected = False
            shift_type = None
            
            if self.last_momentum_state != current_state:
                if current_state == 'bullish' and self.last_momentum_state in ['bearish', 'neutral']:
                    shift_type = 'bullish_shift'
                    shift_detected = True
                elif current_state == 'bearish' and self.last_momentum_state in ['bullish', 'neutral']:
                    shift_type = 'bearish_shift'
                    shift_detected = True
            
            if shift_detected:
                self.last_momentum_state = current_state
                
                return {
                    'shift_type': shift_type,
                    'momentum_score': avg_recent_score,
                    'confidence': abs(avg_recent_score),
                    'reasons': reasons,
                    'timestamp': current_time
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error detecting momentum shift: {e}")
            return None
    
    def format_momentum_alert(self, shift_data: Dict) -> str:
        """Format momentum shift alert message"""
        try:
            shift_type = shift_data['shift_type']
            score = shift_data['momentum_score']
            confidence = shift_data['confidence']
            reasons = shift_data['reasons']
            timestamp = shift_data['timestamp']
            
            # Determine emoji and direction
            if shift_type == 'bullish_shift':
                emoji = "🟢📈"
                direction = "BULLISH"
            else:
                emoji = "🔴📉"
                direction = "BEARISH"
            
            # Format confidence level
            conf_stars = "⭐" * min(5, int(confidence * 5))
            
            message = f"""
{emoji} **MOMENTUM SHIFT DETECTED** {emoji}

🎯 **Direction**: {direction} Momentum Shift
📊 **Score**: {score:.3f}
🎪 **Confidence**: {conf_stars} ({confidence:.1%})

📋 **Analysis**:
{reasons}

⏰ **Time**: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}

💡 **Recommendation**: Monitor for trading opportunities in the {direction.lower()} direction.

🔔 Automated momentum shift alert from Gold Trading AI Bot.
"""
            
            return message.strip()
            
        except Exception as e:
            logging.error(f"Error formatting alert: {e}")
            return f"Momentum shift detected: {shift_data.get('shift_type', 'unknown')}"
    
    def send_telegram_alert(self, message: str, shift_type: str):
        """Send alert to Telegram"""
        try:
            # Check cooldown
            current_time = time.time()
            last_alert = self.last_alert_time.get(shift_type, 0)
            
            if current_time - last_alert < self.alert_cooldown:
                logging.info(f"Skipping {shift_type} alert due to cooldown")
                return
            
            # Send alert
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                self.last_alert_time[shift_type] = current_time
                logging.info(f"⚡ Momentum alert sent: {shift_type}")
            else:
                logging.error(f"Failed to send alert: {response.status_code}")
            
        except Exception as e:
            logging.error(f"Error sending Telegram alert: {e}")
    
    def store_momentum_shift(self, shift_data: Dict):
        """Store momentum shift in database"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO momentum_shifts 
                    (timestamp, shift_type, momentum_score, confidence, indicators, alert_sent)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    shift_data['timestamp'].isoformat(),
                    shift_data['shift_type'],
                    shift_data['momentum_score'],
                    shift_data['confidence'],
                    shift_data['reasons'],
                    1
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error storing momentum shift: {e}")
    
    def analyze_momentum(self):
        """Main analysis function - analyze current momentum and detect shifts"""
        try:
            # Fetch market data
            market_data = self.fetch_market_data()
            if not market_data:
                return
            
            # Calculate indicators
            indicators = self.calculate_momentum_indicators(market_data)
            if not indicators:
                return
            
            # Calculate momentum score
            momentum_score, reasons = self.calculate_momentum_score(indicators)
            
            # Detect momentum shift
            shift = self.detect_momentum_shift(momentum_score, reasons)
            
            if shift:
                logging.info(f"⚡ Momentum shift detected: {shift['shift_type']} (score: {shift['momentum_score']:.3f})")
                
                # Format and send alert
                alert_message = self.format_momentum_alert(shift)
                self.send_telegram_alert(alert_message, shift['shift_type'])
                
                # Store in database
                self.store_momentum_shift(shift)
            
        except Exception as e:
            logging.error(f"Error in momentum analysis: {e}")

# Global instance
momentum_detector = MomentumShiftDetector()

if __name__ == "__main__":
    print("🔍 Enhanced Momentum Shift Detector")
    print("Running analysis...")
    momentum_detector.analyze_momentum()
    print("Analysis complete.")
