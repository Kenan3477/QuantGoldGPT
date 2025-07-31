"""
Advanced Technical Analysis Module for GoldGPT
Provides comprehensive technical indicators and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf

class TechnicalAnalyzer:
    """Advanced Technical Analysis with multiple indicators"""
    
    def __init__(self):
        self.indicators = {}
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators for the given data"""
        try:
            indicators = {}
            
            # Ensure we have enough data
            if len(df) < 50:
                return self._get_default_indicators()
            
            # Price-based indicators
            indicators['sma_20'] = self.sma(df['Close'], 20)
            indicators['sma_50'] = self.sma(df['Close'], 50)
            indicators['ema_12'] = self.ema(df['Close'], 12)
            indicators['ema_26'] = self.ema(df['Close'], 26)
            
            # Momentum indicators
            indicators['rsi'] = self.rsi(df['Close'], 14)
            indicators['macd'], indicators['macd_signal'], indicators['macd_histogram'] = self.macd(df['Close'])
            indicators['stochastic_k'], indicators['stochastic_d'] = self.stochastic(df)
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.bollinger_bands(df['Close'])
            indicators['atr'] = self.atr(df)
            
            # Volume indicators
            indicators['volume_sma'] = self.sma(df['Volume'], 20) if 'Volume' in df.columns else [0] * len(df)
            indicators['volume_ratio'] = self.volume_ratio(df) if 'Volume' in df.columns else [1] * len(df)
            
            # Support/Resistance levels
            indicators['support_levels'], indicators['resistance_levels'] = self.support_resistance(df)
            
            # Trend analysis
            indicators['trend_direction'] = self.trend_analysis(df)
            indicators['trend_strength'] = self.trend_strength(df)
            
            # Current values (latest)
            current_values = {
                'rsi_current': float(indicators['rsi'][-1]) if len(indicators['rsi']) > 0 else 50,
                'macd_current': float(indicators['macd'][-1]) if len(indicators['macd']) > 0 else 0,
                'bb_position': self.bb_position(df['Close'][-1], indicators['bb_upper'][-1], indicators['bb_lower'][-1]),
                'trend_current': indicators['trend_direction'][-1] if indicators['trend_direction'] else 'NEUTRAL',
                'volume_trend': 'HIGH' if indicators['volume_ratio'][-1] > 1.2 else 'LOW' if indicators['volume_ratio'][-1] < 0.8 else 'NORMAL'
            }
            
            indicators.update(current_values)
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return self._get_default_indicators()
    
    def sma(self, data: pd.Series, period: int) -> List[float]:
        """Simple Moving Average"""
        return data.rolling(window=period).mean().fillna(0).tolist()
    
    def ema(self, data: pd.Series, period: int) -> List[float]:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean().fillna(0).tolist()
    
    def rsi(self, data: pd.Series, period: int = 14) -> List[float]:
        """Relative Strength Index"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).tolist()
        except:
            return [50] * len(data)
    
    def macd(self, data: pd.Series) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        try:
            ema_12 = data.ewm(span=12).mean()
            ema_26 = data.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            return (macd_line.fillna(0).tolist(), 
                   signal_line.fillna(0).tolist(), 
                   histogram.fillna(0).tolist())
        except:
            return ([0] * len(data), [0] * len(data), [0] * len(data))
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands"""
        try:
            sma = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return (upper_band.fillna(0).tolist(), 
                   sma.fillna(0).tolist(), 
                   lower_band.fillna(0).tolist())
        except:
            return ([0] * len(data), [0] * len(data), [0] * len(data))
    
    def stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """Stochastic Oscillator"""
        try:
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return (k_percent.fillna(50).tolist(), d_percent.fillna(50).tolist())
        except:
            return ([50] * len(df), [50] * len(df))
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Average True Range"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            return atr.fillna(0).tolist()
        except:
            return [0] * len(df)
    
    def volume_ratio(self, df: pd.DataFrame) -> List[float]:
        """Volume ratio compared to average"""
        try:
            volume_avg = df['Volume'].rolling(window=20).mean()
            ratio = df['Volume'] / volume_avg
            return ratio.fillna(1).tolist()
        except:
            return [1] * len(df)
    
    def support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        try:
            # Simple pivot point calculation
            pivot_high = df['High'].rolling(window=5, center=True).max() == df['High']
            pivot_low = df['Low'].rolling(window=5, center=True).min() == df['Low']
            
            resistance_levels = df[pivot_high]['High'].values[-3:] if pivot_high.any() else []
            support_levels = df[pivot_low]['Low'].values[-3:] if pivot_low.any() else []
            
            return (support_levels.tolist(), resistance_levels.tolist())
        except:
            current_price = df['Close'].iloc[-1] if len(df) > 0 else 2000
            return ([current_price * 0.99], [current_price * 1.01])
    
    def trend_analysis(self, df: pd.DataFrame) -> List[str]:
        """Analyze trend direction"""
        try:
            sma_short = df['Close'].rolling(window=10).mean()
            sma_long = df['Close'].rolling(window=20).mean()
            
            trends = []
            for i in range(len(df)):
                if i < 20:
                    trends.append('NEUTRAL')
                elif sma_short.iloc[i] > sma_long.iloc[i]:
                    trends.append('BULLISH')
                elif sma_short.iloc[i] < sma_long.iloc[i]:
                    trends.append('BEARISH')
                else:
                    trends.append('NEUTRAL')
            
            return trends
        except:
            return ['NEUTRAL'] * len(df)
    
    def trend_strength(self, df: pd.DataFrame) -> List[float]:
        """Calculate trend strength (0-1)"""
        try:
            # Use ADX-like calculation
            periods = min(14, len(df))
            strength = []
            
            for i in range(len(df)):
                if i < periods:
                    strength.append(0.5)
                else:
                    price_range = df['High'].iloc[i-periods:i].max() - df['Low'].iloc[i-periods:i].min()
                    current_move = abs(df['Close'].iloc[i] - df['Close'].iloc[i-periods])
                    trend_str = min(current_move / price_range if price_range > 0 else 0, 1.0)
                    strength.append(trend_str)
            
            return strength
        except:
            return [0.5] * len(df)
    
    def bb_position(self, price: float, upper: float, lower: float) -> str:
        """Determine position relative to Bollinger Bands"""
        try:
            if price > upper:
                return 'ABOVE_UPPER'
            elif price < lower:
                return 'BELOW_LOWER'
            elif price > (upper + lower) / 2:
                return 'UPPER_HALF'
            else:
                return 'LOWER_HALF'
        except:
            return 'MIDDLE'
    
    def _get_default_indicators(self) -> Dict:
        """Return default indicator values when calculation fails"""
        return {
            'rsi_current': 50,
            'macd_current': 0,
            'bb_position': 'MIDDLE',
            'trend_current': 'NEUTRAL',
            'volume_trend': 'NORMAL',
            'sma_20': [2000],
            'sma_50': [2000],
            'ema_12': [2000],
            'ema_26': [2000],
            'rsi': [50],
            'macd': [0],
            'macd_signal': [0],
            'macd_histogram': [0],
            'bb_upper': [2010],
            'bb_middle': [2000],
            'bb_lower': [1990],
            'support_levels': [1990],
            'resistance_levels': [2010],
            'trend_direction': ['NEUTRAL'],
            'trend_strength': [0.5]
        }
    
    def get_trading_signals(self, indicators: Dict) -> Dict:
        """Generate trading signals based on indicators"""
        try:
            signals = {
                'overall_signal': 'NEUTRAL',
                'strength': 0.5,
                'signals': []
            }
            
            signal_count = 0
            bullish_signals = 0
            
            # RSI signals
            rsi = indicators.get('rsi_current', 50)
            if rsi < 30:
                signals['signals'].append('RSI Oversold (Bullish)')
                bullish_signals += 1
            elif rsi > 70:
                signals['signals'].append('RSI Overbought (Bearish)')
            signal_count += 1
            
            # MACD signals
            macd = indicators.get('macd_current', 0)
            if macd > 0:
                signals['signals'].append('MACD Above Zero (Bullish)')
                bullish_signals += 1
            elif macd < 0:
                signals['signals'].append('MACD Below Zero (Bearish)')
            signal_count += 1
            
            # Bollinger Band signals
            bb_pos = indicators.get('bb_position', 'MIDDLE')
            if bb_pos == 'BELOW_LOWER':
                signals['signals'].append('Below Lower BB (Bullish)')
                bullish_signals += 1
            elif bb_pos == 'ABOVE_UPPER':
                signals['signals'].append('Above Upper BB (Bearish)')
            signal_count += 1
            
            # Trend signals
            trend = indicators.get('trend_current', 'NEUTRAL')
            if trend == 'BULLISH':
                signals['signals'].append('Bullish Trend')
                bullish_signals += 1
            elif trend == 'BEARISH':
                signals['signals'].append('Bearish Trend')
            signal_count += 1
            
            # Calculate overall signal
            if signal_count > 0:
                bullish_ratio = bullish_signals / signal_count
                if bullish_ratio >= 0.6:
                    signals['overall_signal'] = 'BUY'
                    signals['strength'] = bullish_ratio
                elif bullish_ratio <= 0.4:
                    signals['overall_signal'] = 'SELL'
                    signals['strength'] = 1 - bullish_ratio
                else:
                    signals['overall_signal'] = 'NEUTRAL'
                    signals['strength'] = 0.5
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return {'overall_signal': 'NEUTRAL', 'strength': 0.5, 'signals': ['Analysis unavailable']}

# Global instance
technical_analyzer = TechnicalAnalyzer()
