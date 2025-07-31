"""
Advanced Candlestick Pattern Detection Module
Detects and analyzes candlestick patterns with confidence scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class CandlestickPatternDetector:
    """Advanced candlestick pattern detection and analysis"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_library = self._initialize_pattern_library()
        
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect all candlestick patterns in the data"""
        try:
            if len(df) < 3:
                return {'patterns': [], 'current_patterns': [], 'pattern_signals': []}
                
            patterns_found = []
            current_patterns = []
            
            # Single candle patterns
            single_patterns = self._detect_single_candle_patterns(df)
            patterns_found.extend(single_patterns)
            
            # Two candle patterns
            two_patterns = self._detect_two_candle_patterns(df)
            patterns_found.extend(two_patterns)
            
            # Three candle patterns
            three_patterns = self._detect_three_candle_patterns(df)
            patterns_found.extend(three_patterns)
            
            # Multi-candle formations
            formation_patterns = self._detect_formations(df)
            patterns_found.extend(formation_patterns)
            
            # Get current patterns (last 5 candles)
            current_patterns = [p for p in patterns_found if p['index'] >= len(df) - 5]
            
            # Generate pattern-based signals
            pattern_signals = self._generate_pattern_signals(current_patterns)
            
            return {
                'patterns': patterns_found,
                'current_patterns': current_patterns,
                'pattern_signals': pattern_signals,
                'pattern_count': len(patterns_found),
                'bullish_patterns': len([p for p in current_patterns if p['bias'] == 'BULLISH']),
                'bearish_patterns': len([p for p in current_patterns if p['bias'] == 'BEARISH'])
            }
            
        except Exception as e:
            print(f"Error detecting patterns: {e}")
            return {'patterns': [], 'current_patterns': [], 'pattern_signals': []}
    
    def _detect_single_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect single candlestick patterns"""
        patterns = []
        
        for i in range(len(df)):
            candle = df.iloc[i]
            
            # Doji patterns
            if self._is_doji(candle):
                patterns.append({
                    'name': 'Doji',
                    'type': 'single',
                    'index': i,
                    'timestamp': candle.name,
                    'bias': 'NEUTRAL',
                    'confidence': 0.7,
                    'description': 'Market indecision, potential reversal'
                })
            
            # Hammer
            elif self._is_hammer(candle, df, i):
                patterns.append({
                    'name': 'Hammer',
                    'type': 'single',
                    'index': i,
                    'timestamp': candle.name,
                    'bias': 'BULLISH',
                    'confidence': 0.75,
                    'description': 'Bullish reversal pattern'
                })
            
            # Shooting Star
            elif self._is_shooting_star(candle, df, i):
                patterns.append({
                    'name': 'Shooting Star',
                    'type': 'single',
                    'index': i,
                    'timestamp': candle.name,
                    'bias': 'BEARISH',
                    'confidence': 0.75,
                    'description': 'Bearish reversal pattern'
                })
            
            # Spinning Top
            elif self._is_spinning_top(candle):
                patterns.append({
                    'name': 'Spinning Top',
                    'type': 'single',
                    'index': i,
                    'timestamp': candle.name,
                    'bias': 'NEUTRAL',
                    'confidence': 0.6,
                    'description': 'Market indecision'
                })
            
            # Marubozu
            elif self._is_marubozu(candle):
                bias = 'BULLISH' if candle['Close'] > candle['Open'] else 'BEARISH'
                patterns.append({
                    'name': 'Marubozu',
                    'type': 'single',
                    'index': i,
                    'timestamp': candle.name,
                    'bias': bias,
                    'confidence': 0.8,
                    'description': f'{bias.lower()} continuation pattern'
                })
        
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect two-candle patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            candle1 = df.iloc[i-1]
            candle2 = df.iloc[i]
            
            # Engulfing patterns
            if self._is_bullish_engulfing(candle1, candle2):
                patterns.append({
                    'name': 'Bullish Engulfing',
                    'type': 'two_candle',
                    'index': i,
                    'timestamp': candle2.name,
                    'bias': 'BULLISH',
                    'confidence': 0.85,
                    'description': 'Strong bullish reversal signal'
                })
            
            elif self._is_bearish_engulfing(candle1, candle2):
                patterns.append({
                    'name': 'Bearish Engulfing',
                    'type': 'two_candle',
                    'index': i,
                    'timestamp': candle2.name,
                    'bias': 'BEARISH',
                    'confidence': 0.85,
                    'description': 'Strong bearish reversal signal'
                })
            
            # Harami patterns
            elif self._is_bullish_harami(candle1, candle2):
                patterns.append({
                    'name': 'Bullish Harami',
                    'type': 'two_candle',
                    'index': i,
                    'timestamp': candle2.name,
                    'bias': 'BULLISH',
                    'confidence': 0.7,
                    'description': 'Potential bullish reversal'
                })
            
            elif self._is_bearish_harami(candle1, candle2):
                patterns.append({
                    'name': 'Bearish Harami',
                    'type': 'two_candle',
                    'index': i,
                    'timestamp': candle2.name,
                    'bias': 'BEARISH',
                    'confidence': 0.7,
                    'description': 'Potential bearish reversal'
                })
            
            # Tweezer patterns
            elif self._is_tweezer_top(candle1, candle2):
                patterns.append({
                    'name': 'Tweezer Top',
                    'type': 'two_candle',
                    'index': i,
                    'timestamp': candle2.name,
                    'bias': 'BEARISH',
                    'confidence': 0.65,
                    'description': 'Bearish reversal pattern'
                })
            
            elif self._is_tweezer_bottom(candle1, candle2):
                patterns.append({
                    'name': 'Tweezer Bottom',
                    'type': 'two_candle',
                    'index': i,
                    'timestamp': candle2.name,
                    'bias': 'BULLISH',
                    'confidence': 0.65,
                    'description': 'Bullish reversal pattern'
                })
        
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect three-candle patterns"""
        patterns = []
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Morning Star
            if self._is_morning_star(candle1, candle2, candle3):
                patterns.append({
                    'name': 'Morning Star',
                    'type': 'three_candle',
                    'index': i,
                    'timestamp': candle3.name,
                    'bias': 'BULLISH',
                    'confidence': 0.9,
                    'description': 'Very strong bullish reversal'
                })
            
            # Evening Star
            elif self._is_evening_star(candle1, candle2, candle3):
                patterns.append({
                    'name': 'Evening Star',
                    'type': 'three_candle',
                    'index': i,
                    'timestamp': candle3.name,
                    'bias': 'BEARISH',
                    'confidence': 0.9,
                    'description': 'Very strong bearish reversal'
                })
            
            # Three White Soldiers
            elif self._is_three_white_soldiers(candle1, candle2, candle3):
                patterns.append({
                    'name': 'Three White Soldiers',
                    'type': 'three_candle',
                    'index': i,
                    'timestamp': candle3.name,
                    'bias': 'BULLISH',
                    'confidence': 0.85,
                    'description': 'Strong bullish continuation'
                })
            
            # Three Black Crows
            elif self._is_three_black_crows(candle1, candle2, candle3):
                patterns.append({
                    'name': 'Three Black Crows',
                    'type': 'three_candle',
                    'index': i,
                    'timestamp': candle3.name,
                    'bias': 'BEARISH',
                    'confidence': 0.85,
                    'description': 'Strong bearish continuation'
                })
            
            # Inside Bar (Three Inside Up/Down)
            elif self._is_three_inside_up(candle1, candle2, candle3):
                patterns.append({
                    'name': 'Three Inside Up',
                    'type': 'three_candle',
                    'index': i,
                    'timestamp': candle3.name,
                    'bias': 'BULLISH',
                    'confidence': 0.8,
                    'description': 'Bullish reversal confirmation'
                })
            
            elif self._is_three_inside_down(candle1, candle2, candle3):
                patterns.append({
                    'name': 'Three Inside Down',
                    'type': 'three_candle',
                    'index': i,
                    'timestamp': candle3.name,
                    'bias': 'BEARISH',
                    'confidence': 0.8,
                    'description': 'Bearish reversal confirmation'
                })
        
        return patterns
    
    def _detect_formations(self, df: pd.DataFrame) -> List[Dict]:
        """Detect multi-candle formations"""
        patterns = []
        
        # Double top/bottom detection (simplified)
        if len(df) >= 10:
            highs = df['High'].rolling(window=5, center=True).max() == df['High']
            lows = df['Low'].rolling(window=5, center=True).min() == df['Low']
            
            # Find double tops
            high_indices = df[highs].index[-4:] if highs.any() else []
            if len(high_indices) >= 2:
                for i in range(len(high_indices)-1):
                    idx1, idx2 = high_indices[i], high_indices[i+1]
                    price1, price2 = df.loc[idx1, 'High'], df.loc[idx2, 'High']
                    
                    if abs(price1 - price2) / price1 < 0.02:  # Within 2%
                        patterns.append({
                            'name': 'Double Top',
                            'type': 'formation',
                            'index': df.index.get_loc(idx2),
                            'timestamp': idx2,
                            'bias': 'BEARISH',
                            'confidence': 0.75,
                            'description': 'Bearish reversal formation'
                        })
            
            # Find double bottoms
            low_indices = df[lows].index[-4:] if lows.any() else []
            if len(low_indices) >= 2:
                for i in range(len(low_indices)-1):
                    idx1, idx2 = low_indices[i], low_indices[i+1]
                    price1, price2 = df.loc[idx1, 'Low'], df.loc[idx2, 'Low']
                    
                    if abs(price1 - price2) / price1 < 0.02:  # Within 2%
                        patterns.append({
                            'name': 'Double Bottom',
                            'type': 'formation',
                            'index': df.index.get_loc(idx2),
                            'timestamp': idx2,
                            'bias': 'BULLISH',
                            'confidence': 0.75,
                            'description': 'Bullish reversal formation'
                        })
        
        return patterns
    
    # Pattern detection helper methods
    def _is_doji(self, candle) -> bool:
        """Check if candle is a Doji"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        return total_range > 0 and body_size / total_range < 0.1
    
    def _is_hammer(self, candle, df: pd.DataFrame, index: int) -> bool:
        """Check if candle is a Hammer"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        
        if total_range == 0:
            return False
            
        # In downtrend, small body, long lower shadow, small upper shadow
        is_downtrend = index > 0 and df.iloc[index-1]['Close'] > candle['Close']
        return (is_downtrend and 
                body_size / total_range < 0.3 and 
                lower_shadow > 2 * body_size and 
                upper_shadow < body_size)
    
    def _is_shooting_star(self, candle, df: pd.DataFrame, index: int) -> bool:
        """Check if candle is a Shooting Star"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        
        if total_range == 0:
            return False
            
        # In uptrend, small body, long upper shadow, small lower shadow
        is_uptrend = index > 0 and df.iloc[index-1]['Close'] < candle['Close']
        return (is_uptrend and 
                body_size / total_range < 0.3 and 
                upper_shadow > 2 * body_size and 
                lower_shadow < body_size)
    
    def _is_spinning_top(self, candle) -> bool:
        """Check if candle is a Spinning Top"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return False
            
        return (body_size / total_range < 0.3 and 
                body_size / total_range > 0.05)
    
    def _is_marubozu(self, candle) -> bool:
        """Check if candle is a Marubozu"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return False
            
        return body_size / total_range > 0.95
    
    def _is_bullish_engulfing(self, candle1, candle2) -> bool:
        """Check for Bullish Engulfing pattern"""
        return (candle1['Close'] < candle1['Open'] and  # First candle bearish
                candle2['Close'] > candle2['Open'] and  # Second candle bullish
                candle2['Open'] < candle1['Close'] and  # Gap down
                candle2['Close'] > candle1['Open'])     # Engulfs first candle
    
    def _is_bearish_engulfing(self, candle1, candle2) -> bool:
        """Check for Bearish Engulfing pattern"""
        return (candle1['Close'] > candle1['Open'] and  # First candle bullish
                candle2['Close'] < candle2['Open'] and  # Second candle bearish
                candle2['Open'] > candle1['Close'] and  # Gap up
                candle2['Close'] < candle1['Open'])     # Engulfs first candle
    
    def _is_bullish_harami(self, candle1, candle2) -> bool:
        """Check for Bullish Harami pattern"""
        return (candle1['Close'] < candle1['Open'] and  # First candle bearish
                candle2['Close'] > candle2['Open'] and  # Second candle bullish
                candle2['Open'] > candle1['Close'] and  # Inside first candle
                candle2['Close'] < candle1['Open'])
    
    def _is_bearish_harami(self, candle1, candle2) -> bool:
        """Check for Bearish Harami pattern"""
        return (candle1['Close'] > candle1['Open'] and  # First candle bullish
                candle2['Close'] < candle2['Open'] and  # Second candle bearish
                candle2['Open'] < candle1['Close'] and  # Inside first candle
                candle2['Close'] > candle1['Open'])
    
    def _is_tweezer_top(self, candle1, candle2) -> bool:
        """Check for Tweezer Top pattern"""
        return (abs(candle1['High'] - candle2['High']) / candle1['High'] < 0.005 and
                candle1['Close'] > candle1['Open'] and
                candle2['Close'] < candle2['Open'])
    
    def _is_tweezer_bottom(self, candle1, candle2) -> bool:
        """Check for Tweezer Bottom pattern"""
        return (abs(candle1['Low'] - candle2['Low']) / candle1['Low'] < 0.005 and
                candle1['Close'] < candle1['Open'] and
                candle2['Close'] > candle2['Open'])
    
    def _is_morning_star(self, candle1, candle2, candle3) -> bool:
        """Check for Morning Star pattern"""
        return (candle1['Close'] < candle1['Open'] and  # First bearish
                abs(candle2['Close'] - candle2['Open']) < abs(candle1['Close'] - candle1['Open']) * 0.3 and  # Small middle
                candle3['Close'] > candle3['Open'] and  # Third bullish
                candle3['Close'] > (candle1['Open'] + candle1['Close']) / 2)  # Closes above midpoint
    
    def _is_evening_star(self, candle1, candle2, candle3) -> bool:
        """Check for Evening Star pattern"""
        return (candle1['Close'] > candle1['Open'] and  # First bullish
                abs(candle2['Close'] - candle2['Open']) < abs(candle1['Close'] - candle1['Open']) * 0.3 and  # Small middle
                candle3['Close'] < candle3['Open'] and  # Third bearish
                candle3['Close'] < (candle1['Open'] + candle1['Close']) / 2)  # Closes below midpoint
    
    def _is_three_white_soldiers(self, candle1, candle2, candle3) -> bool:
        """Check for Three White Soldiers pattern"""
        return (candle1['Close'] > candle1['Open'] and
                candle2['Close'] > candle2['Open'] and
                candle3['Close'] > candle3['Open'] and
                candle2['Close'] > candle1['Close'] and
                candle3['Close'] > candle2['Close'])
    
    def _is_three_black_crows(self, candle1, candle2, candle3) -> bool:
        """Check for Three Black Crows pattern"""
        return (candle1['Close'] < candle1['Open'] and
                candle2['Close'] < candle2['Open'] and
                candle3['Close'] < candle3['Open'] and
                candle2['Close'] < candle1['Close'] and
                candle3['Close'] < candle2['Close'])
    
    def _is_three_inside_up(self, candle1, candle2, candle3) -> bool:
        """Check for Three Inside Up pattern"""
        return (self._is_bullish_harami(candle1, candle2) and
                candle3['Close'] > candle3['Open'] and
                candle3['Close'] > candle2['Close'])
    
    def _is_three_inside_down(self, candle1, candle2, candle3) -> bool:
        """Check for Three Inside Down pattern"""
        return (self._is_bearish_harami(candle1, candle2) and
                candle3['Close'] < candle3['Open'] and
                candle3['Close'] < candle2['Close'])
    
    def _generate_pattern_signals(self, patterns: List[Dict]) -> List[Dict]:
        """Generate trading signals based on detected patterns"""
        signals = []
        
        if not patterns:
            return signals
        
        # Aggregate pattern signals
        bullish_confidence = 0
        bearish_confidence = 0
        pattern_names = []
        
        for pattern in patterns:
            pattern_names.append(pattern['name'])
            if pattern['bias'] == 'BULLISH':
                bullish_confidence += pattern['confidence']
            elif pattern['bias'] == 'BEARISH':
                bearish_confidence += pattern['confidence']
        
        # Generate overall signal
        if bullish_confidence > bearish_confidence and bullish_confidence > 0.5:
            signals.append({
                'signal': 'BUY',
                'confidence': min(bullish_confidence, 1.0),
                'reason': f"Bullish patterns detected: {', '.join(set(pattern_names))}",
                'patterns': [p for p in patterns if p['bias'] == 'BULLISH']
            })
        elif bearish_confidence > bullish_confidence and bearish_confidence > 0.5:
            signals.append({
                'signal': 'SELL',
                'confidence': min(bearish_confidence, 1.0),
                'reason': f"Bearish patterns detected: {', '.join(set(pattern_names))}",
                'patterns': [p for p in patterns if p['bias'] == 'BEARISH']
            })
        else:
            signals.append({
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'reason': f"Mixed or weak patterns: {', '.join(set(pattern_names))}",
                'patterns': patterns
            })
        
        return signals
    
    def _initialize_pattern_library(self) -> Dict:
        """Initialize the pattern library with definitions"""
        return {
            'single_candle': [
                'Doji', 'Hammer', 'Shooting Star', 'Spinning Top', 'Marubozu'
            ],
            'two_candle': [
                'Bullish Engulfing', 'Bearish Engulfing', 'Bullish Harami', 'Bearish Harami',
                'Tweezer Top', 'Tweezer Bottom'
            ],
            'three_candle': [
                'Morning Star', 'Evening Star', 'Three White Soldiers', 'Three Black Crows',
                'Three Inside Up', 'Three Inside Down'
            ],
            'formations': [
                'Double Top', 'Double Bottom', 'Head and Shoulders', 'Inverse Head and Shoulders'
            ]
        }

# Global instance
pattern_detector = CandlestickPatternDetector()
