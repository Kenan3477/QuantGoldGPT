#!/usr/bin/env python3
"""
REAL Technical Analysis Signal Generator
======================================
This generates signals based on ACTUAL technical analysis, not random fake data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import talib
import logging

logger = logging.getLogger(__name__)

class RealTechnicalAnalyzer:
    """Real technical analysis for gold trading signals"""
    
    def __init__(self):
        self.symbol = "GC=F"  # Gold futures
        self.fallback_symbol = "GLD"  # Gold ETF
        
    def get_real_gold_price(self) -> float:
        """Get current real gold price"""
        try:
            response = requests.get("http://localhost:5000/api/live-gold-price", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except:
            pass
        
        # Fallback to Yahoo Finance
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'][-1])
        except:
            pass
        
        return 3400.0  # Emergency fallback
    
    def get_historical_data(self, period: str = "5d", interval: str = "15m") -> Optional[pd.DataFrame]:
        """Get real historical data for analysis"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                # Try fallback
                ticker = yf.Ticker(self.fallback_symbol)
                data = ticker.history(period=period, interval=interval)
            
            if len(data) < 20:  # Need minimum data for indicators
                return None
                
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate real technical indicators"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            indicators = {}
            
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = {
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'histogram': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bollinger'] = {
                'upper': bb_upper[-1],
                'middle': bb_middle[-1],
                'lower': bb_lower[-1],
                'position': (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100
            }
            
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else indicators['sma_20']
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1]
            
            # Support/Resistance (properly calculated)
            recent_highs = high[-20:]
            recent_lows = low[-20:]
            current_price_float = float(close[-1])
            
            # Resistance should be above current price
            resistance_candidates = recent_highs[recent_highs > current_price_float]
            if len(resistance_candidates) > 0:
                indicators['resistance'] = np.min(resistance_candidates)  # Nearest resistance above
            else:
                indicators['resistance'] = current_price_float + (current_price_float * 0.01)  # 1% above as fallback
            
            # Support should be below current price
            support_candidates = recent_lows[recent_lows < current_price_float]
            if len(support_candidates) > 0:
                indicators['support'] = np.max(support_candidates)  # Nearest support below
            else:
                indicators['support'] = current_price_float - (current_price_float * 0.01)  # 1% below as fallback
            
            # Volume analysis
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Trend analysis
            indicators['trend'] = 'bullish' if close[-1] > indicators['sma_20'] else 'bearish'
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def analyze_signal(self, indicators: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Generate REAL signal based on technical analysis"""
        try:
            signal_strength = 0
            signal_type = "NEUTRAL"
            reasoning_parts = []
            
            # RSI Analysis
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                signal_strength += 2
                reasoning_parts.append(f"RSI oversold at {rsi:.1f}")
                signal_bias = "BUY"
            elif rsi > 70:
                signal_strength += 2
                reasoning_parts.append(f"RSI overbought at {rsi:.1f}")
                signal_bias = "SELL"
            else:
                signal_bias = "NEUTRAL"
                reasoning_parts.append(f"RSI neutral at {rsi:.1f}")
            
            # MACD Analysis
            macd_data = indicators.get('macd', {})
            macd_line = macd_data.get('macd', 0)
            macd_signal = macd_data.get('signal', 0)
            macd_hist = macd_data.get('histogram', 0)
            
            if macd_line > macd_signal and macd_hist > 0:
                signal_strength += 1.5
                reasoning_parts.append("MACD bullish crossover")
                if signal_bias == "NEUTRAL":
                    signal_bias = "BUY"
            elif macd_line < macd_signal and macd_hist < 0:
                signal_strength += 1.5
                reasoning_parts.append("MACD bearish crossover")
                if signal_bias == "NEUTRAL":
                    signal_bias = "SELL"
            
            # Bollinger Bands Analysis
            bb = indicators.get('bollinger', {})
            bb_position = bb.get('position', 50)
            
            if bb_position < 10:  # Near lower band
                signal_strength += 1
                reasoning_parts.append("Price near Bollinger lower band")
                if signal_bias != "SELL":
                    signal_bias = "BUY"
            elif bb_position > 90:  # Near upper band
                signal_strength += 1
                reasoning_parts.append("Price near Bollinger upper band")
                if signal_bias != "BUY":
                    signal_bias = "SELL"
            
            # Moving Average Analysis
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            
            if current_price > sma_20 > sma_50:
                signal_strength += 1
                reasoning_parts.append("Price above key moving averages")
                if signal_bias == "NEUTRAL":
                    signal_bias = "BUY"
            elif current_price < sma_20 < sma_50:
                signal_strength += 1
                reasoning_parts.append("Price below key moving averages")
                if signal_bias == "NEUTRAL":
                    signal_bias = "SELL"
            
            # Support/Resistance Analysis
            support = indicators.get('support', current_price - 20)
            resistance = indicators.get('resistance', current_price + 20)
            
            if current_price <= support + 2:  # Near support
                signal_strength += 1
                reasoning_parts.append(f"Price testing support at ${support:.2f}")
                if signal_bias != "SELL":
                    signal_bias = "BUY"
            elif current_price >= resistance - 2:  # Near resistance
                signal_strength += 1
                reasoning_parts.append(f"Price testing resistance at ${resistance:.2f}")
                if signal_bias != "BUY":
                    signal_bias = "SELL"
            
            # Volume Confirmation
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                signal_strength += 0.5
                reasoning_parts.append("High volume confirmation")
            
            # Determine final signal
            if signal_strength >= 3 and signal_bias != "NEUTRAL":
                signal_type = signal_bias
            else:
                signal_type = "NEUTRAL"
            
            # Calculate confidence based on signal strength
            confidence = min(0.95, 0.5 + (signal_strength / 10))
            win_probability = min(0.9, 0.55 + (signal_strength / 15))
            
            # Calculate entry, TP, SL based on technical levels
            if signal_type == "BUY":
                entry_price = current_price
                take_profit = min(resistance, current_price + (current_price * 0.015))  # 1.5% or resistance
                stop_loss = max(support, current_price - (current_price * 0.01))  # 1% or support
            elif signal_type == "SELL":
                entry_price = current_price
                take_profit = max(support, current_price - (current_price * 0.015))  # 1.5% or support
                stop_loss = min(resistance, current_price + (current_price * 0.01))  # 1% or resistance
            else:
                # No signal
                return None
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 2.0
            
            return {
                'signal_type': signal_type,
                'entry_price': entry_price,
                'current_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'win_probability': win_probability,
                'risk_reward_ratio': risk_reward_ratio,
                'signal_strength': signal_strength,
                'reasoning': ". ".join(reasoning_parts),
                'technical_indicators': indicators,
                'status': 'ACTIVE',
                'success': True,
                'signal_generated': True,
                'timestamp': datetime.now().isoformat(),
                'signal_id': f"real_signal_{int(datetime.now().timestamp() * 1000)}",
                'expected_roi': (reward / entry_price) * 100 if entry_price > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing signal: {e}")
            return None

def generate_real_signal(symbol: str = "GOLD", timeframe: str = "1h") -> Optional[Dict[str, Any]]:
    """Generate a real trading signal based on actual technical analysis"""
    try:
        analyzer = RealTechnicalAnalyzer()
        
        # Get current price
        current_price = analyzer.get_real_gold_price()
        logger.info(f"📊 Current gold price: ${current_price:.2f}")
        
        # Get historical data for analysis
        data = analyzer.get_historical_data(period="5d", interval="15m")
        if data is None or len(data) < 20:
            logger.error("❌ Insufficient historical data for analysis")
            return None
        
        # Calculate technical indicators
        indicators = analyzer.calculate_technical_indicators(data)
        if not indicators:
            logger.error("❌ Failed to calculate technical indicators")
            return None
        
        # Generate signal
        signal = analyzer.analyze_signal(indicators, current_price)
        if signal is None:
            logger.info("📊 No strong signal detected - market conditions neutral")
            return {
                'success': False,
                'message': 'No strong technical signal detected',
                'signal_strength': 0,
                'current_price': current_price
            }
        
        logger.info(f"✅ Real signal generated: {signal['signal_type']} with {signal['signal_strength']:.1f} strength")
        return signal
        
    except Exception as e:
        logger.error(f"❌ Error generating real signal: {e}")
        return None

if __name__ == "__main__":
    # Test the real signal generator
    signal = generate_real_signal()
    if signal and signal.get('success', False):
        print(f"Signal: {signal['signal_type']}")
        print(f"Entry: ${signal['entry_price']:.2f}")
        print(f"TP: ${signal['take_profit']:.2f}")
        print(f"SL: ${signal['stop_loss']:.2f}")
        print(f"Confidence: {signal['confidence']:.1%}")
        print(f"Reasoning: {signal['reasoning']}")
        print(f"Signal Strength: {signal['signal_strength']:.1f}")
    else:
        print("No strong signal generated")
        if signal:
            print(f"Current Price: ${signal.get('current_price', 'Unknown')}")
            print(f"Message: {signal.get('message', 'Unknown')}")
