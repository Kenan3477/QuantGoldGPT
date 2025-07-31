#!/usr/bin/env python3
"""
GoldGPT Technical Indicator Service
Advanced technical analysis with 25+ indicators across multiple timeframes
"""

import asyncio
import sqlite3
import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import deque
from data_pipeline_core import DataPipelineCore, DataType
from advanced_price_data_service import AdvancedPriceDataService, OHLCVData

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicatorValue:
    """Technical indicator value with metadata"""
    indicator_name: str
    value: float
    signal: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0.0 to 1.0
    timeframe: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class TechnicalAnalysisResult:
    """Complete technical analysis result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    indicators: Dict[str, TechnicalIndicatorValue]
    overall_signal: str
    signal_strength: float
    trend_direction: str
    support_levels: List[float]
    resistance_levels: List[float]

class AdvancedTechnicalIndicatorService:
    """Advanced technical indicator service with 25+ indicators"""
    
    def __init__(self, price_service: AdvancedPriceDataService, db_path: str = "goldgpt_technical_indicators.db"):
        self.price_service = price_service
        self.db_path = db_path
        
        # Indicator configurations
        self.indicator_configs = {
            'sma': {'periods': [10, 20, 50, 200]},
            'ema': {'periods': [12, 26, 50, 200]},
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'williams_r': {'period': 14},
            'cci': {'period': 20},
            'atr': {'period': 14},
            'adx': {'period': 14},
            'aroon': {'period': 25},
            'momentum': {'period': 10},
            'roc': {'period': 12},
            'trix': {'period': 14},
            'ultimate_oscillator': {'short': 7, 'medium': 14, 'long': 28},
            'commodity_channel_index': {'period': 20},
            'chaikin_money_flow': {'period': 20},
            'volume_oscillator': {'short': 5, 'long': 10},
            'price_oscillator': {'short': 12, 'long': 26},
            'detrended_price_oscillator': {'period': 20},
            'mass_index': {'period': 25},
            'vortex_indicator': {'period': 14},
            'parabolic_sar': {'acceleration': 0.02, 'maximum': 0.2},
            'pivot_points': {},
            'fibonacci_retracements': {}
        }
        
        # Timeframes to analyze
        self.timeframes = ['1h', '4h', '1d']
        
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
                indicator_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                value REAL NOT NULL,
                signal TEXT NOT NULL,
                strength REAL NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, indicator_name, timestamp)
            )
        ''')
        
        # Technical analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                overall_signal TEXT NOT NULL,
                signal_strength REAL NOT NULL,
                trend_direction TEXT NOT NULL,
                support_levels TEXT,
                resistance_levels TEXT,
                indicator_count INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # Signal tracking for accuracy
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                signal_timestamp DATETIME NOT NULL,
                signal_type TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                actual_outcome TEXT,
                accuracy_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Technical indicators database initialized")
    
    async def calculate_all_indicators(self, symbol: str, timeframe: str, 
                                       lookback_periods: int = 200) -> TechnicalAnalysisResult:
        """Calculate all technical indicators for given symbol and timeframe"""
        try:
            # Get historical price data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_periods)
            
            ohlcv_data = await self.price_service.get_historical_ohlcv(
                symbol, timeframe, start_time, end_time, limit=lookback_periods
            )
            
            if len(ohlcv_data) < 50:  # Need minimum data for indicators
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(ohlcv_data)} periods")
                return self.generate_empty_analysis(symbol, timeframe)
            
            # Convert to pandas DataFrame for easier calculation
            df = self.ohlcv_to_dataframe(ohlcv_data)
            
            # Calculate all indicators
            indicators = {}
            
            # Trend Indicators
            indicators.update(await self.calculate_moving_averages(df))
            indicators.update(await self.calculate_macd(df))
            indicators.update(await self.calculate_adx(df))
            indicators.update(await self.calculate_aroon(df))
            indicators.update(await self.calculate_parabolic_sar(df))
            
            # Momentum Indicators
            indicators.update(await self.calculate_rsi(df))
            indicators.update(await self.calculate_stochastic(df))
            indicators.update(await self.calculate_williams_r(df))
            indicators.update(await self.calculate_cci(df))
            indicators.update(await self.calculate_momentum(df))
            indicators.update(await self.calculate_roc(df))
            indicators.update(await self.calculate_trix(df))
            indicators.update(await self.calculate_ultimate_oscillator(df))
            
            # Volatility Indicators
            indicators.update(await self.calculate_bollinger_bands(df))
            indicators.update(await self.calculate_atr(df))
            indicators.update(await self.calculate_mass_index(df))
            
            # Volume Indicators
            indicators.update(await self.calculate_chaikin_money_flow(df))
            indicators.update(await self.calculate_volume_oscillator(df))
            
            # Other Indicators
            indicators.update(await self.calculate_detrended_price_oscillator(df))
            indicators.update(await self.calculate_vortex_indicator(df))
            
            # Support/Resistance Levels
            support_levels, resistance_levels = await self.calculate_support_resistance(df)
            
            # Determine overall signal
            overall_signal, signal_strength, trend_direction = self.calculate_overall_signal(indicators)
            
            # Create analysis result
            result = TechnicalAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                indicators=indicators,
                overall_signal=overall_signal,
                signal_strength=signal_strength,
                trend_direction=trend_direction,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
            # Store results
            await self.store_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {e}")
            return self.generate_empty_analysis(symbol, timeframe)
    
    def ohlcv_to_dataframe(self, ohlcv_data: List[OHLCVData]) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame"""
        data = []
        for ohlcv in sorted(ohlcv_data, key=lambda x: x.timestamp):
            data.append({
                'timestamp': ohlcv.timestamp,
                'open': ohlcv.open_price,
                'high': ohlcv.high_price,
                'low': ohlcv.low_price,
                'close': ohlcv.close_price,
                'volume': ohlcv.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    async def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Simple and Exponential Moving Averages"""
        indicators = {}
        
        try:
            # Simple Moving Averages
            for period in self.indicator_configs['sma']['periods']:
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean()
                    current_sma = sma.iloc[-1]
                    current_price = df['close'].iloc[-1]
                    
                    # Determine signal
                    if current_price > current_sma:
                        signal = 'buy'
                        strength = min(1.0, (current_price - current_sma) / current_sma * 10)
                    else:
                        signal = 'sell'
                        strength = min(1.0, (current_sma - current_price) / current_sma * 10)
                    
                    indicators[f'sma_{period}'] = TechnicalIndicatorValue(
                        indicator_name=f'sma_{period}',
                        value=current_sma,
                        signal=signal,
                        strength=abs(strength),
                        timeframe='',
                        timestamp=datetime.now(),
                        metadata={'period': period, 'current_price': current_price}
                    )
            
            # Exponential Moving Averages
            for period in self.indicator_configs['ema']['periods']:
                if len(df) >= period:
                    ema = df['close'].ewm(span=period).mean()
                    current_ema = ema.iloc[-1]
                    current_price = df['close'].iloc[-1]
                    
                    # Determine signal
                    if current_price > current_ema:
                        signal = 'buy'
                        strength = min(1.0, (current_price - current_ema) / current_ema * 10)
                    else:
                        signal = 'sell'
                        strength = min(1.0, (current_ema - current_price) / current_ema * 10)
                    
                    indicators[f'ema_{period}'] = TechnicalIndicatorValue(
                        indicator_name=f'ema_{period}',
                        value=current_ema,
                        signal=signal,
                        strength=abs(strength),
                        timeframe='',
                        timestamp=datetime.now(),
                        metadata={'period': period, 'current_price': current_price}
                    )
                    
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
        
        return indicators
    
    async def calculate_rsi(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Relative Strength Index"""
        indicators = {}
        
        try:
            period = self.indicator_configs['rsi']['period']
            if len(df) >= period + 1:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # Determine signal
                if current_rsi > 70:
                    signal = 'sell'  # Overbought
                    strength = min(1.0, (current_rsi - 70) / 30)
                elif current_rsi < 30:
                    signal = 'buy'  # Oversold
                    strength = min(1.0, (30 - current_rsi) / 30)
                else:
                    signal = 'neutral'
                    strength = 0.5
                
                indicators['rsi'] = TechnicalIndicatorValue(
                    indicator_name='rsi',
                    value=current_rsi,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period, 'interpretation': self.rsi_interpretation(current_rsi)}
                )
                
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        return indicators
    
    def rsi_interpretation(self, rsi_value: float) -> str:
        """Interpret RSI value"""
        if rsi_value > 80:
            return 'extremely_overbought'
        elif rsi_value > 70:
            return 'overbought'
        elif rsi_value > 60:
            return 'bullish'
        elif rsi_value > 40:
            return 'neutral'
        elif rsi_value > 30:
            return 'bearish'
        elif rsi_value > 20:
            return 'oversold'
        else:
            return 'extremely_oversold'
    
    async def calculate_macd(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        indicators = {}
        
        try:
            config = self.indicator_configs['macd']
            if len(df) >= config['slow'] + config['signal']:
                ema_fast = df['close'].ewm(span=config['fast']).mean()
                ema_slow = df['close'].ewm(span=config['slow']).mean()
                
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=config['signal']).mean()
                histogram = macd_line - signal_line
                
                current_macd = macd_line.iloc[-1]
                current_signal = signal_line.iloc[-1]
                current_histogram = histogram.iloc[-1]
                
                # Determine signal
                if current_macd > current_signal and current_histogram > 0:
                    signal = 'buy'
                    strength = min(1.0, abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0.5)
                elif current_macd < current_signal and current_histogram < 0:
                    signal = 'sell'
                    strength = min(1.0, abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0.5)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['macd'] = TechnicalIndicatorValue(
                    indicator_name='macd',
                    value=current_macd,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={
                        'macd_line': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_histogram
                    }
                )
                
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        return indicators
    
    async def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Bollinger Bands"""
        indicators = {}
        
        try:
            config = self.indicator_configs['bollinger_bands']
            period = config['period']
            std_dev = config['std_dev']
            
            if len(df) >= period:
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                
                upper_band = sma + (std * std_dev)
                lower_band = sma - (std * std_dev)
                
                current_price = df['close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                current_middle = sma.iloc[-1]
                
                # Calculate Bollinger Band position
                bb_position = (current_price - current_lower) / (current_upper - current_lower)
                
                # Determine signal
                if bb_position > 0.8:
                    signal = 'sell'  # Near upper band
                    strength = bb_position
                elif bb_position < 0.2:
                    signal = 'buy'  # Near lower band
                    strength = 1.0 - bb_position
                else:
                    signal = 'neutral'
                    strength = 0.5
                
                indicators['bollinger_bands'] = TechnicalIndicatorValue(
                    indicator_name='bollinger_bands',
                    value=bb_position,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={
                        'upper_band': current_upper,
                        'middle_band': current_middle,
                        'lower_band': current_lower,
                        'current_price': current_price,
                        'position': bb_position
                    }
                )
                
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
        
        return indicators
    
    async def calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Stochastic Oscillator"""
        indicators = {}
        
        try:
            config = self.indicator_configs['stochastic']
            k_period = config['k_period']
            d_period = config['d_period']
            
            if len(df) >= k_period + d_period:
                low_min = df['low'].rolling(window=k_period).min()
                high_max = df['high'].rolling(window=k_period).max()
                
                k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
                d_percent = k_percent.rolling(window=d_period).mean()
                
                current_k = k_percent.iloc[-1]
                current_d = d_percent.iloc[-1]
                
                # Determine signal
                if current_k > 80 and current_d > 80:
                    signal = 'sell'  # Overbought
                    strength = min(1.0, (current_k - 80) / 20)
                elif current_k < 20 and current_d < 20:
                    signal = 'buy'  # Oversold
                    strength = min(1.0, (20 - current_k) / 20)
                else:
                    signal = 'neutral'
                    strength = 0.4
                
                indicators['stochastic'] = TechnicalIndicatorValue(
                    indicator_name='stochastic',
                    value=current_k,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'k_percent': current_k, 'd_percent': current_d}
                )
                
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
        
        return indicators
    
    async def calculate_atr(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Average True Range"""
        indicators = {}
        
        try:
            period = self.indicator_configs['atr']['period']
            if len(df) >= period + 1:
                high_low = df['high'] - df['low']
                high_close_prev = abs(df['high'] - df['close'].shift(1))
                low_close_prev = abs(df['low'] - df['close'].shift(1))
                
                true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = true_range.rolling(window=period).mean()
                
                current_atr = atr.iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # ATR as percentage of price for normalization
                atr_percentage = (current_atr / current_price) * 100
                
                # Determine volatility level
                if atr_percentage > 2.0:
                    volatility = 'high'
                    strength = min(1.0, atr_percentage / 3.0)
                elif atr_percentage > 1.0:
                    volatility = 'medium'
                    strength = 0.6
                else:
                    volatility = 'low'
                    strength = 0.3
                
                indicators['atr'] = TechnicalIndicatorValue(
                    indicator_name='atr',
                    value=current_atr,
                    signal='neutral',  # ATR doesn't give buy/sell signals
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={
                        'atr_percentage': atr_percentage,
                        'volatility_level': volatility
                    }
                )
                
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
        
        return indicators
    
    async def calculate_adx(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Average Directional Index"""
        indicators = {}
        
        try:
            period = self.indicator_configs['adx']['period']
            if len(df) >= period * 2:
                # Calculate Directional Movement
                plus_dm = df['high'].diff()
                minus_dm = df['low'].diff().abs()
                
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
                
                # Calculate True Range
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift(1))
                tr3 = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Smooth the values
                plus_di = 100 * (plus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
                minus_di = 100 * (minus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
                
                # Calculate ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=period).mean()
                
                current_adx = adx.iloc[-1]
                current_plus_di = plus_di.iloc[-1]
                current_minus_di = minus_di.iloc[-1]
                
                # Determine signal
                if current_adx > 25:
                    if current_plus_di > current_minus_di:
                        signal = 'buy'
                        strength = min(1.0, current_adx / 50)
                    else:
                        signal = 'sell'
                        strength = min(1.0, current_adx / 50)
                else:
                    signal = 'neutral'  # Weak trend
                    strength = 0.3
                
                indicators['adx'] = TechnicalIndicatorValue(
                    indicator_name='adx',
                    value=current_adx,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={
                        'plus_di': current_plus_di,
                        'minus_di': current_minus_di,
                        'trend_strength': 'strong' if current_adx > 25 else 'weak'
                    }
                )
                
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
        
        return indicators
    
    async def calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        try:
            # Use recent highs and lows to identify levels
            recent_data = df.tail(50)  # Last 50 periods
            
            # Find local minima (support) and maxima (resistance)
            highs = recent_data['high']
            lows = recent_data['low']
            
            # Simple method: use rolling max/min with different windows
            resistance_levels = []
            support_levels = []
            
            for window in [5, 10, 20]:
                rolling_max = highs.rolling(window=window, center=True).max()
                rolling_min = lows.rolling(window=window, center=True).min()
                
                # Find peaks and troughs
                for i in range(window, len(highs) - window):
                    if highs.iloc[i] == rolling_max.iloc[i]:
                        resistance_levels.append(highs.iloc[i])
                    if lows.iloc[i] == rolling_min.iloc[i]:
                        support_levels.append(lows.iloc[i])
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)))[:5]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return [], []
    
    # Additional indicator methods would go here...
    # (Williams %R, CCI, Aroon, etc.)
    
    async def calculate_williams_r(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Williams %R"""
        indicators = {}
        try:
            period = self.indicator_configs['williams_r']['period']
            if len(df) >= period:
                high_max = df['high'].rolling(window=period).max()
                low_min = df['low'].rolling(window=period).min()
                williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
                
                current_wr = williams_r.iloc[-1]
                
                if current_wr > -20:
                    signal = 'sell'  # Overbought
                    strength = min(1.0, (-current_wr) / 20)
                elif current_wr < -80:
                    signal = 'buy'  # Oversold
                    strength = min(1.0, (current_wr + 100) / 20)
                else:
                    signal = 'neutral'
                    strength = 0.4
                
                indicators['williams_r'] = TechnicalIndicatorValue(
                    indicator_name='williams_r',
                    value=current_wr,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period}
                )
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
        return indicators
    
    async def calculate_cci(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Commodity Channel Index"""
        indicators = {}
        try:
            period = self.indicator_configs['cci']['period']
            if len(df) >= period:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(window=period).mean()
                mean_deviation = typical_price.rolling(window=period).apply(
                    lambda x: np.mean(np.abs(x - x.mean()))
                )
                cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
                
                current_cci = cci.iloc[-1]
                
                if current_cci > 100:
                    signal = 'sell'  # Overbought
                    strength = min(1.0, (current_cci - 100) / 100)
                elif current_cci < -100:
                    signal = 'buy'  # Oversold
                    strength = min(1.0, (-current_cci - 100) / 100)
                else:
                    signal = 'neutral'
                    strength = 0.4
                
                indicators['cci'] = TechnicalIndicatorValue(
                    indicator_name='cci',
                    value=current_cci,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period}
                )
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
        return indicators
    
    # Additional indicator calculations...
    async def calculate_aroon(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Aroon Indicator"""
        indicators = {}
        try:
            period = self.indicator_configs['aroon']['period']
            if len(df) >= period:
                aroon_up = df['high'].rolling(window=period).apply(
                    lambda x: ((period - 1 - np.argmax(x)) / (period - 1)) * 100
                )
                aroon_down = df['low'].rolling(window=period).apply(
                    lambda x: ((period - 1 - np.argmin(x)) / (period - 1)) * 100
                )
                
                current_up = aroon_up.iloc[-1]
                current_down = aroon_down.iloc[-1]
                
                if current_up > 70 and current_up > current_down:
                    signal = 'buy'
                    strength = current_up / 100
                elif current_down > 70 and current_down > current_up:
                    signal = 'sell'
                    strength = current_down / 100
                else:
                    signal = 'neutral'
                    strength = 0.4
                
                indicators['aroon'] = TechnicalIndicatorValue(
                    indicator_name='aroon',
                    value=(current_up - current_down),
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'aroon_up': current_up, 'aroon_down': current_down}
                )
        except Exception as e:
            logger.error(f"Error calculating Aroon: {e}")
        return indicators
    
    async def calculate_momentum(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Momentum"""
        indicators = {}
        try:
            period = self.indicator_configs['momentum']['period']
            if len(df) >= period:
                momentum = df['close'] / df['close'].shift(period) * 100
                current_momentum = momentum.iloc[-1]
                
                if current_momentum > 105:
                    signal = 'buy'
                    strength = min(1.0, (current_momentum - 100) / 10)
                elif current_momentum < 95:
                    signal = 'sell'
                    strength = min(1.0, (100 - current_momentum) / 10)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['momentum'] = TechnicalIndicatorValue(
                    indicator_name='momentum',
                    value=current_momentum,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period}
                )
        except Exception as e:
            logger.error(f"Error calculating Momentum: {e}")
        return indicators
    
    async def calculate_roc(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Rate of Change"""
        indicators = {}
        try:
            period = self.indicator_configs['roc']['period']
            if len(df) >= period:
                roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
                current_roc = roc.iloc[-1]
                
                if current_roc > 5:
                    signal = 'buy'
                    strength = min(1.0, current_roc / 10)
                elif current_roc < -5:
                    signal = 'sell'
                    strength = min(1.0, abs(current_roc) / 10)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['roc'] = TechnicalIndicatorValue(
                    indicator_name='roc',
                    value=current_roc,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period}
                )
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
        return indicators
    
    async def calculate_trix(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate TRIX"""
        indicators = {}
        try:
            period = self.indicator_configs['trix']['period']
            if len(df) >= period * 3:
                ema1 = df['close'].ewm(span=period).mean()
                ema2 = ema1.ewm(span=period).mean()
                ema3 = ema2.ewm(span=period).mean()
                trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000
                
                current_trix = trix.iloc[-1]
                
                if current_trix > 0:
                    signal = 'buy'
                    strength = min(1.0, abs(current_trix) / 50)
                elif current_trix < 0:
                    signal = 'sell'
                    strength = min(1.0, abs(current_trix) / 50)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['trix'] = TechnicalIndicatorValue(
                    indicator_name='trix',
                    value=current_trix,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period}
                )
        except Exception as e:
            logger.error(f"Error calculating TRIX: {e}")
        return indicators
    
    async def calculate_ultimate_oscillator(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Ultimate Oscillator"""
        indicators = {}
        try:
            config = self.indicator_configs['ultimate_oscillator']
            short = config['short']
            medium = config['medium']
            long = config['long']
            
            if len(df) >= long + 1:
                # Calculate buying pressure
                bp = df['close'] - df[['low', 'close']].shift(1).min(axis=1)
                tr = pd.concat([
                    df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                ], axis=1).max(axis=1)
                
                # Calculate averages
                avg_short = bp.rolling(window=short).sum() / tr.rolling(window=short).sum()
                avg_medium = bp.rolling(window=medium).sum() / tr.rolling(window=medium).sum()
                avg_long = bp.rolling(window=long).sum() / tr.rolling(window=long).sum()
                
                uo = 100 * ((4 * avg_short) + (2 * avg_medium) + avg_long) / 7
                current_uo = uo.iloc[-1]
                
                if current_uo > 70:
                    signal = 'sell'
                    strength = min(1.0, (current_uo - 70) / 30)
                elif current_uo < 30:
                    signal = 'buy'
                    strength = min(1.0, (30 - current_uo) / 30)
                else:
                    signal = 'neutral'
                    strength = 0.4
                
                indicators['ultimate_oscillator'] = TechnicalIndicatorValue(
                    indicator_name='ultimate_oscillator',
                    value=current_uo,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'short': short, 'medium': medium, 'long': long}
                )
        except Exception as e:
            logger.error(f"Error calculating Ultimate Oscillator: {e}")
        return indicators
    
    async def calculate_chaikin_money_flow(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Chaikin Money Flow"""
        indicators = {}
        try:
            period = self.indicator_configs['chaikin_money_flow']['period']
            if len(df) >= period:
                money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
                money_flow_volume = money_flow_multiplier * df['volume']
                cmf = money_flow_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
                
                current_cmf = cmf.iloc[-1]
                
                if current_cmf > 0.1:
                    signal = 'buy'
                    strength = min(1.0, current_cmf * 5)
                elif current_cmf < -0.1:
                    signal = 'sell'
                    strength = min(1.0, abs(current_cmf) * 5)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['chaikin_money_flow'] = TechnicalIndicatorValue(
                    indicator_name='chaikin_money_flow',
                    value=current_cmf,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period}
                )
        except Exception as e:
            logger.error(f"Error calculating Chaikin Money Flow: {e}")
        return indicators
    
    async def calculate_volume_oscillator(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Volume Oscillator"""
        indicators = {}
        try:
            config = self.indicator_configs['volume_oscillator']
            short = config['short']
            long = config['long']
            
            if len(df) >= long:
                short_avg = df['volume'].rolling(window=short).mean()
                long_avg = df['volume'].rolling(window=long).mean()
                vo = ((short_avg - long_avg) / long_avg) * 100
                
                current_vo = vo.iloc[-1]
                
                if current_vo > 5:
                    signal = 'buy'
                    strength = min(1.0, current_vo / 20)
                elif current_vo < -5:
                    signal = 'sell'
                    strength = min(1.0, abs(current_vo) / 20)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['volume_oscillator'] = TechnicalIndicatorValue(
                    indicator_name='volume_oscillator',
                    value=current_vo,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'short': short, 'long': long}
                )
        except Exception as e:
            logger.error(f"Error calculating Volume Oscillator: {e}")
        return indicators
    
    async def calculate_detrended_price_oscillator(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Detrended Price Oscillator"""
        indicators = {}
        try:
            period = self.indicator_configs['detrended_price_oscillator']['period']
            if len(df) >= period:
                sma = df['close'].rolling(window=period).mean()
                dpo = df['close'] - sma.shift(period // 2 + 1)
                
                current_dpo = dpo.iloc[-1]
                avg_price = df['close'].iloc[-1]
                dpo_percentage = (current_dpo / avg_price) * 100 if avg_price != 0 else 0
                
                if dpo_percentage > 1:
                    signal = 'buy'
                    strength = min(1.0, dpo_percentage / 3)
                elif dpo_percentage < -1:
                    signal = 'sell'
                    strength = min(1.0, abs(dpo_percentage) / 3)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                indicators['detrended_price_oscillator'] = TechnicalIndicatorValue(
                    indicator_name='detrended_price_oscillator',
                    value=current_dpo,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period, 'percentage': dpo_percentage}
                )
        except Exception as e:
            logger.error(f"Error calculating Detrended Price Oscillator: {e}")
        return indicators
    
    async def calculate_mass_index(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Mass Index"""
        indicators = {}
        try:
            period = self.indicator_configs['mass_index']['period']
            if len(df) >= period + 9:
                high_low = df['high'] - df['low']
                ema9 = high_low.ewm(span=9).mean()
                ema9_of_ema9 = ema9.ewm(span=9).mean()
                mass_index = (ema9 / ema9_of_ema9).rolling(window=period).sum()
                
                current_mi = mass_index.iloc[-1]
                
                # Mass Index interpretation
                if current_mi > 27:
                    signal = 'sell'  # Potential reversal
                    strength = min(1.0, (current_mi - 27) / 10)
                elif current_mi < 26.5:
                    signal = 'neutral'
                    strength = 0.3
                else:
                    signal = 'neutral'
                    strength = 0.5
                
                indicators['mass_index'] = TechnicalIndicatorValue(
                    indicator_name='mass_index',
                    value=current_mi,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'period': period, 'interpretation': 'reversal_signal' if current_mi > 27 else 'normal'}
                )
        except Exception as e:
            logger.error(f"Error calculating Mass Index: {e}")
        return indicators
    
    async def calculate_vortex_indicator(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Vortex Indicator"""
        indicators = {}
        try:
            period = self.indicator_configs['vortex_indicator']['period']
            if len(df) >= period + 1:
                vm_plus = abs(df['high'] - df['low'].shift(1)).rolling(window=period).sum()
                vm_minus = abs(df['low'] - df['high'].shift(1)).rolling(window=period).sum()
                
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift(1))
                tr3 = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                tr_sum = true_range.rolling(window=period).sum()
                
                vi_plus = vm_plus / tr_sum
                vi_minus = vm_minus / tr_sum
                
                current_vi_plus = vi_plus.iloc[-1]
                current_vi_minus = vi_minus.iloc[-1]
                
                if current_vi_plus > current_vi_minus and current_vi_plus > 1.0:
                    signal = 'buy'
                    strength = min(1.0, (current_vi_plus - 1.0) / 0.3)
                elif current_vi_minus > current_vi_plus and current_vi_minus > 1.0:
                    signal = 'sell'
                    strength = min(1.0, (current_vi_minus - 1.0) / 0.3)
                else:
                    signal = 'neutral'
                    strength = 0.4
                
                indicators['vortex_indicator'] = TechnicalIndicatorValue(
                    indicator_name='vortex_indicator',
                    value=current_vi_plus - current_vi_minus,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'vi_plus': current_vi_plus, 'vi_minus': current_vi_minus}
                )
        except Exception as e:
            logger.error(f"Error calculating Vortex Indicator: {e}")
        return indicators
    
    async def calculate_parabolic_sar(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicatorValue]:
        """Calculate Parabolic SAR"""
        indicators = {}
        try:
            config = self.indicator_configs['parabolic_sar']
            acceleration = config['acceleration']
            maximum = config['maximum']
            
            if len(df) >= 10:
                # Simplified Parabolic SAR calculation
                sar = []
                af = acceleration
                ep = df['high'].iloc[0]
                trend = 1  # 1 for uptrend, -1 for downtrend
                
                for i in range(len(df)):
                    if i == 0:
                        sar.append(df['low'].iloc[i])
                    else:
                        prev_sar = sar[i-1]
                        high = df['high'].iloc[i]
                        low = df['low'].iloc[i]
                        
                        if trend == 1:  # Uptrend
                            new_sar = prev_sar + af * (ep - prev_sar)
                            if low <= new_sar:
                                trend = -1
                                new_sar = ep
                                af = acceleration
                                ep = low
                            else:
                                if high > ep:
                                    ep = high
                                    af = min(af + acceleration, maximum)
                        else:  # Downtrend
                            new_sar = prev_sar - af * (prev_sar - ep)
                            if high >= new_sar:
                                trend = 1
                                new_sar = ep
                                af = acceleration
                                ep = high
                            else:
                                if low < ep:
                                    ep = low
                                    af = min(af + acceleration, maximum)
                        
                        sar.append(new_sar)
                
                current_sar = sar[-1]
                current_price = df['close'].iloc[-1]
                
                if current_price > current_sar:
                    signal = 'buy'
                    strength = min(1.0, (current_price - current_sar) / current_price * 10)
                else:
                    signal = 'sell'
                    strength = min(1.0, (current_sar - current_price) / current_price * 10)
                
                indicators['parabolic_sar'] = TechnicalIndicatorValue(
                    indicator_name='parabolic_sar',
                    value=current_sar,
                    signal=signal,
                    strength=strength,
                    timeframe='',
                    timestamp=datetime.now(),
                    metadata={'current_price': current_price, 'trend': 'up' if current_price > current_sar else 'down'}
                )
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {e}")
        return indicators
    
    def calculate_overall_signal(self, indicators: Dict[str, TechnicalIndicatorValue]) -> Tuple[str, float, str]:
        """Calculate overall signal from all indicators"""
        try:
            buy_signals = 0
            sell_signals = 0
            neutral_signals = 0
            total_strength = 0.0
            
            # Weight different indicator types
            weights = {
                'trend': 1.2,      # Moving averages, MACD, etc.
                'momentum': 1.0,   # RSI, Stochastic, etc.
                'volatility': 0.8, # Bollinger Bands, ATR, etc.
                'volume': 0.9      # Volume-based indicators
            }
            
            trend_indicators = ['sma_', 'ema_', 'macd', 'adx', 'aroon', 'parabolic_sar']
            momentum_indicators = ['rsi', 'stochastic', 'williams_r', 'cci', 'momentum', 'roc', 'trix', 'ultimate_oscillator']
            volatility_indicators = ['bollinger_bands', 'atr', 'mass_index']
            volume_indicators = ['chaikin_money_flow', 'volume_oscillator']
            
            for name, indicator in indicators.items():
                # Determine indicator type and weight
                weight = 1.0
                for prefix in trend_indicators:
                    if name.startswith(prefix):
                        weight = weights['trend']
                        break
                else:
                    if name in momentum_indicators:
                        weight = weights['momentum']
                    elif name in volatility_indicators:
                        weight = weights['volatility']
                    elif name in volume_indicators:
                        weight = weights['volume']
                
                # Count weighted signals
                weighted_strength = indicator.strength * weight
                
                if indicator.signal == 'buy':
                    buy_signals += weighted_strength
                elif indicator.signal == 'sell':
                    sell_signals += weighted_strength
                else:
                    neutral_signals += weighted_strength
                
                total_strength += weighted_strength
            
            # Determine overall signal
            if total_strength == 0:
                return 'neutral', 0.0, 'sideways'
            
            buy_percentage = buy_signals / total_strength
            sell_percentage = sell_signals / total_strength
            
            if buy_percentage > 0.6:
                overall_signal = 'buy'
                signal_strength = buy_percentage
                trend_direction = 'bullish'
            elif sell_percentage > 0.6:
                overall_signal = 'sell'
                signal_strength = sell_percentage
                trend_direction = 'bearish'
            else:
                overall_signal = 'neutral'
                signal_strength = max(buy_percentage, sell_percentage)
                trend_direction = 'sideways'
            
            return overall_signal, signal_strength, trend_direction
            
        except Exception as e:
            logger.error(f"Error calculating overall signal: {e}")
            return 'neutral', 0.0, 'sideways'
    
    def generate_empty_analysis(self, symbol: str, timeframe: str) -> TechnicalAnalysisResult:
        """Generate empty analysis result when insufficient data"""
        return TechnicalAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            indicators={},
            overall_signal='neutral',
            signal_strength=0.0,
            trend_direction='unknown',
            support_levels=[],
            resistance_levels=[]
        )
    
    async def store_analysis_result(self, result: TechnicalAnalysisResult):
        """Store technical analysis result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store individual indicators
        for name, indicator in result.indicators.items():
            cursor.execute('''
                INSERT OR REPLACE INTO technical_indicators
                (symbol, timeframe, indicator_name, timestamp, value,
                 signal, strength, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.symbol,
                result.timeframe,
                name,
                result.timestamp.isoformat(),
                indicator.value,
                indicator.signal,
                indicator.strength,
                json.dumps(indicator.metadata)
            ))
        
        # Store overall analysis
        cursor.execute('''
            INSERT OR REPLACE INTO technical_analysis_results
            (symbol, timeframe, timestamp, overall_signal, signal_strength,
             trend_direction, support_levels, resistance_levels, indicator_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.symbol,
            result.timeframe,
            result.timestamp.isoformat(),
            result.overall_signal,
            result.signal_strength,
            result.trend_direction,
            json.dumps(result.support_levels),
            json.dumps(result.resistance_levels),
            len(result.indicators)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Stored technical analysis for {result.symbol} {result.timeframe}")
    
    async def get_multi_timeframe_analysis(self, symbol: str) -> Dict[str, TechnicalAnalysisResult]:
        """Get technical analysis across multiple timeframes"""
        results = {}
        
        for timeframe in self.timeframes:
            try:
                analysis = await self.calculate_all_indicators(symbol, timeframe)
                results[timeframe] = analysis
                logger.info(f"📊 {timeframe}: {analysis.overall_signal} "
                           f"({analysis.signal_strength:.2f}) - {len(analysis.indicators)} indicators")
            except Exception as e:
                logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
                results[timeframe] = self.generate_empty_analysis(symbol, timeframe)
        
        return results

# Global instance
technical_service = AdvancedTechnicalIndicatorService(
    AdvancedPriceDataService(DataPipelineCore())
)

if __name__ == "__main__":
    async def test_technical_service():
        print("🧪 Testing Advanced Technical Indicator Service...")
        
        # Test multi-timeframe analysis
        analysis = await technical_service.get_multi_timeframe_analysis('XAU')
        
        for timeframe, result in analysis.items():
            print(f"\n📊 {timeframe} Analysis:")
            print(f"   Overall Signal: {result.overall_signal}")
            print(f"   Signal Strength: {result.signal_strength:.3f}")
            print(f"   Trend Direction: {result.trend_direction}")
            print(f"   Indicators Count: {len(result.indicators)}")
            
            # Show top 5 indicators
            for i, (name, indicator) in enumerate(list(result.indicators.items())[:5]):
                print(f"   {name}: {indicator.signal} ({indicator.strength:.2f})")
    
    asyncio.run(test_technical_service())
