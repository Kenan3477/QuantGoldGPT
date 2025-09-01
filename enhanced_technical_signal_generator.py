#!/usr/bin/env python3
"""
Enhanced Technical Analysis Signal Generator
==========================================
More sensitive signal detection for testing while maintaining real analysis
"""

from real_technical_signal_generator import RealTechnicalAnalyzer
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedTechnicalAnalyzer(RealTechnicalAnalyzer):
    """Enhanced version with more sensitive signal detection"""
    
    def analyze_signal(self, indicators: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Generate signal with lower threshold for demo purposes"""
        try:
            signal_strength = 0
            signal_type = "NEUTRAL"
            reasoning_parts = []
            
            # RSI Analysis (more sensitive)
            rsi = indicators.get('rsi', 50)
            if rsi < 40:  # Lowered from 30
                signal_strength += 1.5
                reasoning_parts.append(f"RSI indicates potential oversold at {rsi:.1f}")
                signal_bias = "BUY"
            elif rsi > 60:  # Lowered from 70
                signal_strength += 1.5
                reasoning_parts.append(f"RSI indicates potential overbought at {rsi:.1f}")
                signal_bias = "SELL"
            else:
                signal_bias = "NEUTRAL"
                reasoning_parts.append(f"RSI neutral at {rsi:.1f}")
            
            # MACD Analysis
            macd_data = indicators.get('macd', {})
            macd_line = macd_data.get('macd', 0)
            macd_signal = macd_data.get('signal', 0)
            macd_hist = macd_data.get('histogram', 0)
            
            if macd_line > macd_signal:
                signal_strength += 1
                reasoning_parts.append("MACD showing bullish momentum")
                if signal_bias == "NEUTRAL":
                    signal_bias = "BUY"
            elif macd_line < macd_signal:
                signal_strength += 1
                reasoning_parts.append("MACD showing bearish momentum")
                if signal_bias == "NEUTRAL":
                    signal_bias = "SELL"
            
            # Bollinger Bands Analysis
            bb = indicators.get('bollinger', {})
            bb_position = bb.get('position', 50)
            
            if bb_position < 25:  # Lowered threshold
                signal_strength += 1
                reasoning_parts.append("Price in lower Bollinger Band region")
                if signal_bias != "SELL":
                    signal_bias = "BUY"
            elif bb_position > 75:  # Lowered threshold
                signal_strength += 1
                reasoning_parts.append("Price in upper Bollinger Band region")
                if signal_bias != "BUY":
                    signal_bias = "SELL"
            
            # Moving Average Analysis
            sma_20 = indicators.get('sma_20', current_price)
            ema_12 = indicators.get('ema_12', current_price)
            
            if current_price > sma_20 and current_price > ema_12:
                signal_strength += 0.5
                reasoning_parts.append("Price above moving averages")
                if signal_bias == "NEUTRAL":
                    signal_bias = "BUY"
            elif current_price < sma_20 and current_price < ema_12:
                signal_strength += 0.5
                reasoning_parts.append("Price below moving averages")
                if signal_bias == "NEUTRAL":
                    signal_bias = "SELL"
            
            # Support/Resistance Analysis
            support = indicators.get('support', current_price - 20)
            resistance = indicators.get('resistance', current_price + 20)
            
            distance_to_support = current_price - support
            distance_to_resistance = resistance - current_price
            
            if distance_to_support < 5:  # Within $5 of support
                signal_strength += 1
                reasoning_parts.append(f"Price near support level at ${support:.2f}")
                if signal_bias != "SELL":
                    signal_bias = "BUY"
            elif distance_to_resistance < 5:  # Within $5 of resistance
                signal_strength += 1
                reasoning_parts.append(f"Price near resistance level at ${resistance:.2f}")
                if signal_bias != "BUY":
                    signal_bias = "SELL"
            
            # Market structure analysis
            trend = indicators.get('trend', 'neutral')
            if trend == 'bullish':
                signal_strength += 0.5
                reasoning_parts.append("Overall trend is bullish")
            elif trend == 'bearish':
                signal_strength += 0.5
                reasoning_parts.append("Overall trend is bearish")
            
            # Volume Confirmation
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:  # Lowered threshold
                signal_strength += 0.5
                reasoning_parts.append("Volume above average")
            
            # Determine final signal (lowered threshold from 3 to 2)
            if signal_strength >= 2 and signal_bias != "NEUTRAL":
                signal_type = signal_bias
            else:
                signal_type = "NEUTRAL"
            
            # If still no signal, create a weak signal for demo
            if signal_type == "NEUTRAL" and signal_strength >= 1:
                signal_type = signal_bias if signal_bias != "NEUTRAL" else "BUY"
                reasoning_parts.append("WEAK SIGNAL - Demo purposes only")
            
            if signal_type == "NEUTRAL":
                return None
            
            # Calculate confidence based on signal strength
            confidence = min(0.95, 0.4 + (signal_strength / 8))  # Adjusted for lower thresholds
            win_probability = min(0.85, 0.50 + (signal_strength / 10))
            
            # Calculate entry, TP, SL based on technical levels
            if signal_type == "BUY":
                entry_price = current_price
                take_profit = min(resistance, current_price + (current_price * 0.012))  # 1.2%
                stop_loss = max(support, current_price - (current_price * 0.008))  # 0.8%
            else:  # SELL
                entry_price = current_price
                take_profit = max(support, current_price - (current_price * 0.012))  # 1.2% DOWN
                stop_loss = min(resistance, current_price + (current_price * 0.008))  # 0.8% UP
            
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
                'signal_id': f"enhanced_signal_{int(datetime.now().timestamp() * 1000)}",
                'expected_roi': (reward / entry_price) * 100 if entry_price > 0 else 1.0,
                'analysis_type': 'REAL_TECHNICAL_ENHANCED'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced signal: {e}")
            return None

def generate_enhanced_signal(symbol: str = "GOLD", timeframe: str = "1h") -> Optional[Dict[str, Any]]:
    """Generate enhanced technical signal with more sensitive detection"""
    try:
        analyzer = EnhancedTechnicalAnalyzer()
        
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
            logger.info("📊 No signal detected - market conditions neutral")
            return {
                'success': False,
                'message': 'No technical signal detected',
                'signal_strength': 0,
                'current_price': current_price
            }
        
        logger.info(f"✅ Enhanced signal generated: {signal['signal_type']} with {signal['signal_strength']:.1f} strength")
        return signal
        
    except Exception as e:
        logger.error(f"❌ Error generating enhanced signal: {e}")
        return None

if __name__ == "__main__":
    # Test the enhanced signal generator
    signal = generate_enhanced_signal()
    if signal and signal.get('success', False):
        print(f"Signal: {signal['signal_type']}")
        print(f"Entry: ${signal['entry_price']:.2f}")
        print(f"TP: ${signal['take_profit']:.2f}")
        print(f"SL: ${signal['stop_loss']:.2f}")
        print(f"Confidence: {signal['confidence']:.1%}")
        print(f"Reasoning: {signal['reasoning']}")
        print(f"Signal Strength: {signal['signal_strength']:.1f}")
        print(f"Analysis Type: {signal['analysis_type']}")
    else:
        print("No signal generated")
        if signal:
            print(f"Current Price: ${signal.get('current_price', 'Unknown')}")
            print(f"Message: {signal.get('message', 'Unknown')}")
