#!/usr/bin/env python3
"""
GoldGPT Technical Indicator Service
25+ technical indicators across multiple timeframes with real-time calculations
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from price_data_service import price_service

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    indicator: str
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    value: float
    description: str
    timeframe: str

@dataclass
class IndicatorResult:
    """Result of technical indicator calculation"""
    name: str
    values: List[float]
    signal: str
    current_value: float
    interpretation: str
    timeframe: str
    confidence: float

class TechnicalIndicatorService:
    """Advanced technical analysis service with 25+ indicators"""
    
    def __init__(self, db_path: str = "goldgpt_technical.db"):
        self.db_path = db_path
        self.indicator_cache = {}
        self.supported_timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize technical indicators database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Technical indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                indicator_name TEXT NOT NULL,
                indicator_value REAL,
                signal TEXT,
                confidence REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX(symbol, timeframe, indicator_name, timestamp)
            )
        ''')
        
        # Signal aggregations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_aggregations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                bullish_signals INTEGER DEFAULT 0,
                bearish_signals INTEGER DEFAULT 0,
                neutral_signals INTEGER DEFAULT 0,
                overall_signal TEXT,
                signal_strength REAL,
                total_indicators INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Technical indicators database initialized")
    
    async def calculate_all_indicators(self, symbol: str = 'XAU', 
                                     timeframe: str = '1h', 
                                     limit: int = 200) -> Dict[str, IndicatorResult]:
        """Calculate all technical indicators for given parameters"""
        
        # Get historical price data
        historical_data = await price_service.get_historical_data(symbol, timeframe, limit)
        
        if not historical_data or len(historical_data) < 20:
            logger.warning(f"Insufficient data for technical analysis: {len(historical_data)} points")
            return {}
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        results = {}
        
        # Moving Averages
        results['SMA_20'] = self.calculate_sma(df, 20)
        results['SMA_50'] = self.calculate_sma(df, 50)
        results['SMA_200'] = self.calculate_sma(df, 200)
        results['EMA_12'] = self.calculate_ema(df, 12)
        results['EMA_26'] = self.calculate_ema(df, 26)
        
        # Momentum Indicators
        results['RSI'] = self.calculate_rsi(df)
        results['MACD'] = self.calculate_macd(df)
        results['Stochastic'] = self.calculate_stochastic(df)
        results['Williams_R'] = self.calculate_williams_r(df)
        results['ROC'] = self.calculate_roc(df)
        results['MOM'] = self.calculate_momentum(df)
        
        # Volume Indicators
        results['OBV'] = self.calculate_obv(df)
        results['Volume_SMA'] = self.calculate_volume_sma(df)
        
        # Volatility Indicators
        results['Bollinger_Bands'] = self.calculate_bollinger_bands(df)
        results['ATR'] = self.calculate_atr(df)
        results['Standard_Deviation'] = self.calculate_standard_deviation(df)
        
        # Trend Indicators
        results['ADX'] = self.calculate_adx(df)
        results['CCI'] = self.calculate_cci(df)
        results['Aroon'] = self.calculate_aroon(df)
        results['PSAR'] = self.calculate_parabolic_sar(df)
        
        # Support/Resistance
        results['Pivot_Points'] = self.calculate_pivot_points(df)
        results['Fibonacci'] = self.calculate_fibonacci_retracements(df)
        
        # Custom Gold-Specific Indicators
        results['Gold_Strength'] = self.calculate_gold_strength_index(df)
        results['Dollar_Correlation'] = self.calculate_dollar_correlation_signal(df)
        
        # Store results in database
        self.store_indicator_results(results, symbol, timeframe)
        
        logger.info(f"📊 Calculated {len(results)} technical indicators for {symbol} {timeframe}")
        return results
    
    def calculate_sma(self, df: pd.DataFrame, period: int) -> IndicatorResult:
        """Simple Moving Average"""
        if len(df) < period:
            return self.create_insufficient_data_result(f'SMA_{period}')
        
        sma_values = df['close'].rolling(window=period).mean()
        current_sma = sma_values.iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Generate signal
        if current_price > current_sma:
            signal = "BUY"
            interpretation = f"Price above SMA({period}), bullish trend"
        else:
            signal = "SELL"
            interpretation = f"Price below SMA({period}), bearish trend"
        
        # Calculate confidence based on how far price is from SMA
        distance_ratio = abs(current_price - current_sma) / current_sma
        confidence = min(1.0, distance_ratio * 10)  # Higher distance = higher confidence
        
        return IndicatorResult(
            name=f'SMA_{period}',
            values=sma_values.dropna().tolist(),
            signal=signal,
            current_value=current_sma,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> IndicatorResult:
        """Exponential Moving Average"""
        if len(df) < period:
            return self.create_insufficient_data_result(f'EMA_{period}')
        
        ema_values = df['close'].ewm(span=period).mean()
        current_ema = ema_values.iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > current_ema:
            signal = "BUY"
            interpretation = f"Price above EMA({period}), bullish momentum"
        else:
            signal = "SELL"
            interpretation = f"Price below EMA({period}), bearish momentum"
        
        distance_ratio = abs(current_price - current_ema) / current_ema
        confidence = min(1.0, distance_ratio * 10)
        
        return IndicatorResult(
            name=f'EMA_{period}',
            values=ema_values.tolist(),
            signal=signal,
            current_value=current_ema,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """Relative Strength Index"""
        if len(df) < period + 1:
            return self.create_insufficient_data_result('RSI')
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi_values = 100 - (100 / (1 + rs))
        current_rsi = rsi_values.iloc[-1]
        
        # RSI signals
        if current_rsi > 70:
            signal = "SELL"
            interpretation = f"RSI {current_rsi:.1f} - Overbought condition"
        elif current_rsi < 30:
            signal = "BUY"
            interpretation = f"RSI {current_rsi:.1f} - Oversold condition"
        else:
            signal = "HOLD"
            interpretation = f"RSI {current_rsi:.1f} - Neutral zone"
        
        # Confidence based on how extreme the RSI is
        if current_rsi > 70:
            confidence = (current_rsi - 70) / 30
        elif current_rsi < 30:
            confidence = (30 - current_rsi) / 30
        else:
            confidence = 0.3
        
        return IndicatorResult(
            name='RSI',
            values=rsi_values.dropna().tolist(),
            signal=signal,
            current_value=current_rsi,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=min(1.0, confidence)
        )
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> IndicatorResult:
        """MACD (Moving Average Convergence Divergence)"""
        if len(df) < slow + signal_period:
            return self.create_insufficient_data_result('MACD')
        
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # MACD signals
        if current_macd > current_signal and current_histogram > 0:
            signal = "BUY"
            interpretation = "MACD above signal line with positive histogram - Bullish momentum"
        elif current_macd < current_signal and current_histogram < 0:
            signal = "SELL"
            interpretation = "MACD below signal line with negative histogram - Bearish momentum"
        else:
            signal = "HOLD"
            interpretation = "MACD signals mixed - No clear trend"
        
        confidence = min(1.0, abs(current_histogram) / (df['close'].std() * 0.01))
        
        return IndicatorResult(
            name='MACD',
            values=macd_line.tolist(),
            signal=signal,
            current_value=current_macd,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult:
        """Stochastic Oscillator"""
        if len(df) < k_period + d_period:
            return self.create_insufficient_data_result('Stochastic')
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if current_k > 80 and current_d > 80:
            signal = "SELL"
            interpretation = f"Stochastic %K={current_k:.1f} %D={current_d:.1f} - Overbought"
        elif current_k < 20 and current_d < 20:
            signal = "BUY"
            interpretation = f"Stochastic %K={current_k:.1f} %D={current_d:.1f} - Oversold"
        else:
            signal = "HOLD"
            interpretation = f"Stochastic %K={current_k:.1f} %D={current_d:.1f} - Neutral"
        
        confidence = 0.7 if signal != "HOLD" else 0.3
        
        return IndicatorResult(
            name='Stochastic',
            values=k_percent.dropna().tolist(),
            signal=signal,
            current_value=current_k,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """Williams %R"""
        if len(df) < period:
            return self.create_insufficient_data_result('Williams_R')
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        current_wr = williams_r.iloc[-1]
        
        if current_wr > -20:
            signal = "SELL"
            interpretation = f"Williams %R {current_wr:.1f} - Overbought"
        elif current_wr < -80:
            signal = "BUY"
            interpretation = f"Williams %R {current_wr:.1f} - Oversold"
        else:
            signal = "HOLD"
            interpretation = f"Williams %R {current_wr:.1f} - Neutral"
        
        confidence = 0.7 if signal != "HOLD" else 0.3
        
        return IndicatorResult(
            name='Williams_R',
            values=williams_r.dropna().tolist(),
            signal=signal,
            current_value=current_wr,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> IndicatorResult:
        """Bollinger Bands"""
        if len(df) < period:
            return self.create_insufficient_data_result('Bollinger_Bands')
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = df['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]
        
        # Bollinger Band signals
        if current_price >= current_upper:
            signal = "SELL"
            interpretation = "Price at upper Bollinger Band - Potential reversal"
        elif current_price <= current_lower:
            signal = "BUY"
            interpretation = "Price at lower Bollinger Band - Potential reversal"
        elif current_price > current_middle:
            signal = "HOLD"
            interpretation = "Price above middle line - Bullish bias"
        else:
            signal = "HOLD"
            interpretation = "Price below middle line - Bearish bias"
        
        # Calculate position within bands for confidence
        band_width = current_upper - current_lower
        position = (current_price - current_lower) / band_width
        
        if position >= 0.95 or position <= 0.05:
            confidence = 0.8
        else:
            confidence = 0.4
        
        return IndicatorResult(
            name='Bollinger_Bands',
            values=sma.tolist(),
            signal=signal,
            current_value=current_middle,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """Average True Range"""
        if len(df) < period + 1:
            return self.create_insufficient_data_result('ATR')
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_values = true_range.rolling(window=period).mean()
        
        current_atr = atr_values.iloc[-1]
        atr_mean = atr_values.mean()
        
        if current_atr > atr_mean * 1.5:
            signal = "HOLD"
            interpretation = f"ATR {current_atr:.2f} - High volatility, caution advised"
        elif current_atr < atr_mean * 0.5:
            signal = "HOLD"
            interpretation = f"ATR {current_atr:.2f} - Low volatility, potential breakout"
        else:
            signal = "HOLD"
            interpretation = f"ATR {current_atr:.2f} - Normal volatility"
        
        return IndicatorResult(
            name='ATR',
            values=atr_values.dropna().tolist(),
            signal=signal,
            current_value=current_atr,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.5
        )
    
    def calculate_roc(self, df: pd.DataFrame, period: int = 12) -> IndicatorResult:
        """Rate of Change"""
        if len(df) < period + 1:
            return self.create_insufficient_data_result('ROC')
        
        roc_values = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        current_roc = roc_values.iloc[-1]
        
        if current_roc > 5:
            signal = "BUY"
            interpretation = f"ROC {current_roc:.2f}% - Strong positive momentum"
        elif current_roc < -5:
            signal = "SELL"
            interpretation = f"ROC {current_roc:.2f}% - Strong negative momentum"
        else:
            signal = "HOLD"
            interpretation = f"ROC {current_roc:.2f}% - Neutral momentum"
        
        confidence = min(1.0, abs(current_roc) / 10)
        
        return IndicatorResult(
            name='ROC',
            values=roc_values.dropna().tolist(),
            signal=signal,
            current_value=current_roc,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> IndicatorResult:
        """Momentum"""
        if len(df) < period + 1:
            return self.create_insufficient_data_result('Momentum')
        
        momentum_values = df['close'] - df['close'].shift(period)
        current_momentum = momentum_values.iloc[-1]
        
        if current_momentum > 0:
            signal = "BUY"
            interpretation = f"Momentum {current_momentum:.2f} - Positive momentum"
        elif current_momentum < 0:
            signal = "SELL"
            interpretation = f"Momentum {current_momentum:.2f} - Negative momentum"
        else:
            signal = "HOLD"
            interpretation = "Momentum neutral"
        
        confidence = min(1.0, abs(current_momentum) / df['close'].std())
        
        return IndicatorResult(
            name='Momentum',
            values=momentum_values.dropna().tolist(),
            signal=signal,
            current_value=current_momentum,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_obv(self, df: pd.DataFrame) -> IndicatorResult:
        """On-Balance Volume"""
        if len(df) < 2:
            return self.create_insufficient_data_result('OBV')
        
        obv_values = []
        obv = 0
        
        for i in range(len(df)):
            if i == 0:
                obv_values.append(0)
            else:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv += df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv -= df['volume'].iloc[i]
                obv_values.append(obv)
        
        # Calculate OBV trend
        recent_obv = obv_values[-10:]  # Last 10 periods
        obv_trend = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
        
        if obv_trend > 0:
            signal = "BUY"
            interpretation = "OBV trending up - Volume supporting price gains"
        elif obv_trend < 0:
            signal = "SELL"
            interpretation = "OBV trending down - Volume supporting price declines"
        else:
            signal = "HOLD"
            interpretation = "OBV neutral - Mixed volume signals"
        
        return IndicatorResult(
            name='OBV',
            values=obv_values,
            signal=signal,
            current_value=obv_values[-1],
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.6
        )
    
    def calculate_volume_sma(self, df: pd.DataFrame, period: int = 20) -> IndicatorResult:
        """Volume Simple Moving Average"""
        if len(df) < period:
            return self.create_insufficient_data_result('Volume_SMA')
        
        volume_sma = df['volume'].rolling(window=period).mean()
        current_volume = df['volume'].iloc[-1]
        current_sma = volume_sma.iloc[-1]
        
        if current_volume > current_sma * 1.5:
            signal = "BUY"
            interpretation = "High volume - Strong interest"
        elif current_volume < current_sma * 0.5:
            signal = "HOLD"
            interpretation = "Low volume - Weak interest"
        else:
            signal = "HOLD"
            interpretation = "Normal volume"
        
        return IndicatorResult(
            name='Volume_SMA',
            values=volume_sma.dropna().tolist(),
            signal=signal,
            current_value=current_sma,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.5
        )
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """Average Directional Index"""
        if len(df) < period * 2:
            return self.create_insufficient_data_result('ADX')
        
        # Simplified ADX calculation
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().abs()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = np.maximum(df['high'] - df['low'], 
                       np.maximum(abs(df['high'] - df['close'].shift()),
                                 abs(df['low'] - df['close'].shift())))
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_values = dx.rolling(window=period).mean()
        
        current_adx = adx_values.iloc[-1]
        
        if current_adx > 25:
            signal = "BUY"
            interpretation = f"ADX {current_adx:.1f} - Strong trend"
        elif current_adx > 20:
            signal = "HOLD"
            interpretation = f"ADX {current_adx:.1f} - Moderate trend"
        else:
            signal = "HOLD"
            interpretation = f"ADX {current_adx:.1f} - Weak trend"
        
        return IndicatorResult(
            name='ADX',
            values=adx_values.dropna().tolist(),
            signal=signal,
            current_value=current_adx,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.7
        )
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> IndicatorResult:
        """Commodity Channel Index"""
        if len(df) < period:
            return self.create_insufficient_data_result('CCI')
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci_values = (typical_price - sma_tp) / (0.015 * mad)
        current_cci = cci_values.iloc[-1]
        
        if current_cci > 100:
            signal = "SELL"
            interpretation = f"CCI {current_cci:.1f} - Overbought"
        elif current_cci < -100:
            signal = "BUY"
            interpretation = f"CCI {current_cci:.1f} - Oversold"
        else:
            signal = "HOLD"
            interpretation = f"CCI {current_cci:.1f} - Neutral"
        
        confidence = min(1.0, abs(current_cci) / 200)
        
        return IndicatorResult(
            name='CCI',
            values=cci_values.dropna().tolist(),
            signal=signal,
            current_value=current_cci,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=confidence
        )
    
    def calculate_aroon(self, df: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """Aroon Indicator"""
        if len(df) < period:
            return self.create_insufficient_data_result('Aroon')
        
        aroon_up = []
        aroon_down = []
        
        for i in range(period - 1, len(df)):
            high_period = df['high'].iloc[i - period + 1:i + 1]
            low_period = df['low'].iloc[i - period + 1:i + 1]
            
            periods_since_high = period - 1 - high_period.argmax()
            periods_since_low = period - 1 - low_period.argmin()
            
            aroon_up.append(((period - periods_since_high) / period) * 100)
            aroon_down.append(((period - periods_since_low) / period) * 100)
        
        current_up = aroon_up[-1]
        current_down = aroon_down[-1]
        
        if current_up > 70 and current_up > current_down:
            signal = "BUY"
            interpretation = f"Aroon Up {current_up:.1f} - Strong uptrend"
        elif current_down > 70 and current_down > current_up:
            signal = "SELL"
            interpretation = f"Aroon Down {current_down:.1f} - Strong downtrend"
        else:
            signal = "HOLD"
            interpretation = f"Aroon mixed - No clear trend"
        
        return IndicatorResult(
            name='Aroon',
            values=aroon_up,
            signal=signal,
            current_value=current_up,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.7
        )
    
    def calculate_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> IndicatorResult:
        """Parabolic SAR"""
        if len(df) < 10:
            return self.create_insufficient_data_result('PSAR')
        
        # Simplified PSAR calculation
        psar_values = []
        af = acceleration
        ep = df['high'].iloc[0]
        psar = df['low'].iloc[0]
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(len(df)):
            if i == 0:
                psar_values.append(psar)
                continue
            
            if trend == 1:  # Uptrend
                if df['low'].iloc[i] <= psar:
                    trend = -1
                    psar = ep
                    af = acceleration
                    ep = df['low'].iloc[i]
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + acceleration, maximum)
                    psar = psar + af * (ep - psar)
            else:  # Downtrend
                if df['high'].iloc[i] >= psar:
                    trend = 1
                    psar = ep
                    af = acceleration
                    ep = df['high'].iloc[i]
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + acceleration, maximum)
                    psar = psar + af * (ep - psar)
            
            psar_values.append(psar)
        
        current_psar = psar_values[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > current_psar:
            signal = "BUY"
            interpretation = f"Price above PSAR - Bullish trend"
        else:
            signal = "SELL"
            interpretation = f"Price below PSAR - Bearish trend"
        
        return IndicatorResult(
            name='PSAR',
            values=psar_values,
            signal=signal,
            current_value=current_psar,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.8
        )
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> IndicatorResult:
        """Pivot Points"""
        if len(df) < 2:
            return self.create_insufficient_data_result('Pivot_Points')
        
        # Use previous day's data for pivot calculation
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        
        current_price = df['close'].iloc[-1]
        
        if current_price > r1:
            signal = "BUY"
            interpretation = f"Price above R1 ({r1:.2f}) - Bullish breakout"
        elif current_price < s1:
            signal = "SELL"
            interpretation = f"Price below S1 ({s1:.2f}) - Bearish breakdown"
        else:
            signal = "HOLD"
            interpretation = f"Price between S1 and R1 - Neutral zone"
        
        return IndicatorResult(
            name='Pivot_Points',
            values=[pivot],
            signal=signal,
            current_value=pivot,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.6
        )
    
    def calculate_fibonacci_retracements(self, df: pd.DataFrame) -> IndicatorResult:
        """Fibonacci Retracements"""
        if len(df) < 20:
            return self.create_insufficient_data_result('Fibonacci')
        
        # Find recent swing high and low
        recent_high = df['high'].rolling(window=10).max().iloc[-1]
        recent_low = df['low'].rolling(window=10).min().iloc[-1]
        
        fib_levels = {
            '23.6%': recent_low + 0.236 * (recent_high - recent_low),
            '38.2%': recent_low + 0.382 * (recent_high - recent_low),
            '50%': recent_low + 0.5 * (recent_high - recent_low),
            '61.8%': recent_low + 0.618 * (recent_high - recent_low),
            '78.6%': recent_low + 0.786 * (recent_high - recent_low)
        }
        
        current_price = df['close'].iloc[-1]
        
        # Find nearest Fibonacci level
        distances = {level: abs(current_price - price) for level, price in fib_levels.items()}
        nearest_level = min(distances, key=distances.get)
        
        if current_price > fib_levels['61.8%']:
            signal = "BUY"
            interpretation = f"Price above 61.8% Fibonacci - Strong bullish signal"
        elif current_price < fib_levels['38.2%']:
            signal = "SELL"
            interpretation = f"Price below 38.2% Fibonacci - Strong bearish signal"
        else:
            signal = "HOLD"
            interpretation = f"Price near {nearest_level} Fibonacci level"
        
        return IndicatorResult(
            name='Fibonacci',
            values=list(fib_levels.values()),
            signal=signal,
            current_value=fib_levels['50%'],
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.5
        )
    
    def calculate_standard_deviation(self, df: pd.DataFrame, period: int = 20) -> IndicatorResult:
        """Standard Deviation"""
        if len(df) < period:
            return self.create_insufficient_data_result('Standard_Deviation')
        
        std_values = df['close'].rolling(window=period).std()
        current_std = std_values.iloc[-1]
        avg_std = std_values.mean()
        
        if current_std > avg_std * 1.5:
            signal = "HOLD"
            interpretation = f"High volatility (σ={current_std:.2f}) - Exercise caution"
        elif current_std < avg_std * 0.5:
            signal = "HOLD"
            interpretation = f"Low volatility (σ={current_std:.2f}) - Potential breakout"
        else:
            signal = "HOLD"
            interpretation = f"Normal volatility (σ={current_std:.2f})"
        
        return IndicatorResult(
            name='Standard_Deviation',
            values=std_values.dropna().tolist(),
            signal=signal,
            current_value=current_std,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.4
        )
    
    def calculate_gold_strength_index(self, df: pd.DataFrame) -> IndicatorResult:
        """Custom Gold Strength Index"""
        if len(df) < 20:
            return self.create_insufficient_data_result('Gold_Strength')
        
        # Combine multiple factors specific to gold
        price_momentum = df['close'].pct_change(10).iloc[-1] * 100
        volatility = df['close'].rolling(10).std().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Gold-specific strength calculation
        strength_score = (
            (price_momentum * 0.4) +  # Price momentum weight
            (min(2, volume_ratio) * 30 - 30) +  # Volume factor
            (-volatility * 0.5)  # Lower volatility = higher strength for gold
        )
        
        if strength_score > 10:
            signal = "BUY"
            interpretation = f"Gold Strength Index {strength_score:.1f} - Strong bullish signal"
        elif strength_score < -10:
            signal = "SELL"
            interpretation = f"Gold Strength Index {strength_score:.1f} - Strong bearish signal"
        else:
            signal = "HOLD"
            interpretation = f"Gold Strength Index {strength_score:.1f} - Neutral"
        
        return IndicatorResult(
            name='Gold_Strength',
            values=[strength_score],
            signal=signal,
            current_value=strength_score,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=min(1.0, abs(strength_score) / 20)
        )
    
    def calculate_dollar_correlation_signal(self, df: pd.DataFrame) -> IndicatorResult:
        """Dollar Correlation Signal (Simulated)"""
        # In a real implementation, this would correlate with DXY
        # For now, we'll simulate based on gold price action
        
        price_changes = df['close'].pct_change(5).dropna()
        correlation_strength = abs(price_changes.corr(price_changes.shift(1)))
        
        # Simulate inverse dollar correlation
        recent_change = df['close'].pct_change().iloc[-1]
        dollar_signal = -recent_change * 100  # Inverse relationship
        
        if dollar_signal > 2:
            signal = "SELL"
            interpretation = f"Dollar strength signal {dollar_signal:.1f} - Bearish for gold"
        elif dollar_signal < -2:
            signal = "BUY"
            interpretation = f"Dollar weakness signal {dollar_signal:.1f} - Bullish for gold"
        else:
            signal = "HOLD"
            interpretation = f"Dollar neutral signal {dollar_signal:.1f}"
        
        return IndicatorResult(
            name='Dollar_Correlation',
            values=[dollar_signal],
            signal=signal,
            current_value=dollar_signal,
            interpretation=interpretation,
            timeframe=df.get('timeframe', '1h'),
            confidence=0.6
        )
    
    def create_insufficient_data_result(self, indicator_name: str) -> IndicatorResult:
        """Create result for insufficient data"""
        return IndicatorResult(
            name=indicator_name,
            values=[],
            signal="HOLD",
            current_value=0.0,
            interpretation=f"Insufficient data for {indicator_name} calculation",
            timeframe="unknown",
            confidence=0.0
        )
    
    def store_indicator_results(self, results: Dict[str, IndicatorResult], symbol: str, timeframe: str):
        """Store indicator results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for indicator_name, result in results.items():
            if result.values:  # Only store if we have data
                cursor.execute('''
                    INSERT OR REPLACE INTO technical_indicators 
                    (symbol, timeframe, timestamp, indicator_name, indicator_value, 
                     signal, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timeframe,
                    timestamp,
                    indicator_name,
                    result.current_value,
                    result.signal,
                    result.confidence,
                    json.dumps({
                        'interpretation': result.interpretation,
                        'values_count': len(result.values)
                    })
                ))
        
        conn.commit()
        conn.close()
    
    def aggregate_signals(self, results: Dict[str, IndicatorResult]) -> Dict:
        """Aggregate all technical signals into overall assessment"""
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        total_confidence = 0
        signal_details = []
        
        for name, result in results.items():
            if result.signal == "BUY":
                buy_signals += result.confidence
            elif result.signal == "SELL":
                sell_signals += result.confidence
            else:
                hold_signals += result.confidence
            
            total_confidence += result.confidence
            
            signal_details.append({
                'indicator': name,
                'signal': result.signal,
                'confidence': result.confidence,
                'value': result.current_value,
                'interpretation': result.interpretation
            })
        
        # Determine overall signal
        if buy_signals > sell_signals and buy_signals > hold_signals:
            overall_signal = "BUY"
            signal_strength = buy_signals / (buy_signals + sell_signals + hold_signals)
        elif sell_signals > buy_signals and sell_signals > hold_signals:
            overall_signal = "SELL"
            signal_strength = sell_signals / (buy_signals + sell_signals + hold_signals)
        else:
            overall_signal = "HOLD"
            signal_strength = hold_signals / (buy_signals + sell_signals + hold_signals)
        
        return {
            'overall_signal': overall_signal,
            'signal_strength': round(signal_strength, 3),
            'buy_signals': round(buy_signals, 2),
            'sell_signals': round(sell_signals, 2),
            'hold_signals': round(hold_signals, 2),
            'total_indicators': len(results),
            'average_confidence': round(total_confidence / len(results), 3) if results else 0,
            'signal_details': signal_details,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_technical_analysis(self, symbol: str = 'XAU', timeframe: str = '1h') -> Dict:
        """Get comprehensive technical analysis"""
        logger.info(f"🔍 Running technical analysis for {symbol} {timeframe}")
        
        # Calculate all indicators
        indicators = await self.calculate_all_indicators(symbol, timeframe)
        
        if not indicators:
            return {
                'error': 'Unable to calculate technical indicators',
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }
        
        # Aggregate signals
        aggregation = self.aggregate_signals(indicators)
        
        # Get top indicators by confidence
        top_indicators = sorted(
            aggregation['signal_details'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )[:5]
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'overall_assessment': {
                'signal': aggregation['overall_signal'],
                'strength': aggregation['signal_strength'],
                'confidence': aggregation['average_confidence']
            },
            'signal_breakdown': {
                'buy_strength': aggregation['buy_signals'],
                'sell_strength': aggregation['sell_signals'],
                'hold_strength': aggregation['hold_signals'],
                'total_indicators': aggregation['total_indicators']
            },
            'top_indicators': top_indicators,
            'all_indicators': {name: {
                'signal': result.signal,
                'value': result.current_value,
                'confidence': result.confidence,
                'interpretation': result.interpretation
            } for name, result in indicators.items()},
            'timestamp': datetime.now().isoformat()
        }

# Global instance
technical_service = TechnicalIndicatorService()

if __name__ == "__main__":
    # Test the technical service
    async def test_technical_service():
        print("🧪 Testing Technical Indicator Service...")
        
        analysis = await technical_service.get_technical_analysis('XAU', '1h')
        
        if 'error' not in analysis:
            print(f"📊 Overall Signal: {analysis['overall_assessment']['signal']}")
            print(f"💪 Signal Strength: {analysis['overall_assessment']['strength']:.2f}")
            print(f"🎯 Confidence: {analysis['overall_assessment']['confidence']:.2f}")
            print(f"📈 Total Indicators: {analysis['signal_breakdown']['total_indicators']}")
            
            print("\n🔝 Top 3 Indicators:")
            for i, indicator in enumerate(analysis['top_indicators'][:3], 1):
                print(f"  {i}. {indicator['indicator']}: {indicator['signal']} "
                      f"(Confidence: {indicator['confidence']:.2f})")
        else:
            print(f"❌ Error: {analysis['error']}")
    
    asyncio.run(test_technical_service())
