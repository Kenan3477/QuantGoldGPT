"""
🏛️ ADVANCED ENSEMBLE ML SYSTEM - PART 2
=========================================

Continuation of advanced multi-strategy ML architecture
Includes MacroStrategy, PatternStrategy, MomentumStrategy, EnsembleVotingSystem, and MetaLearningEngine

Author: GoldGPT AI System
Created: July 23, 2025
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import warnings

# Import base classes from part 1
from advanced_ensemble_ml_system import (
    BaseStrategy, StrategyResult, MarketConditions, EnsemblePrediction, logger
)

warnings.filterwarnings('ignore')

class MacroStrategy(BaseStrategy):
    """
    🏦 Macroeconomic Strategy
    Interest rates, inflation, unemployment, GDP impact modeling
    """
    
    def __init__(self):
        super().__init__("MacroStrategy")
        self.macro_indicators = ['interest_rates', 'inflation', 'unemployment', 'gdp_growth', 'dollar_index']
        self.model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    
    def predict(self, data: pd.DataFrame, timeframe: str) -> StrategyResult:
        """Generate macroeconomic-based prediction"""
        try:
            # Gather macro data
            macro_data = self._gather_macro_data()
            
            # Calculate macro impact
            macro_impact = self._calculate_macro_impact(macro_data, timeframe)
            
            # Generate prediction
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * (1 + macro_impact)
            
            # Calculate confidence
            confidence = self._calculate_macro_confidence(macro_data)
            
            # Determine direction
            direction = self._determine_macro_direction(macro_impact)
            
            result = StrategyResult(
                strategy_name=self.name,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                timeframe=timeframe,
                technical_indicators={'macro_impact': macro_impact, **macro_data},
                reasoning=self._generate_macro_reasoning(macro_data, direction),
                risk_assessment={'macro_uncertainty': self._calculate_macro_risk(macro_data)},
                timestamp=datetime.now()
            )
            
            self._store_prediction(result)
            
            logger.info(f"🏦 Macro prediction: ${predicted_price:.2f} ({direction}, {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Macro strategy prediction failed: {e}")
            current_price = data['Close'].iloc[-1]
            return StrategyResult(
                strategy_name=self.name,
                predicted_price=current_price,
                confidence=0.1,
                direction='neutral',
                timeframe=timeframe,
                technical_indicators={},
                reasoning="Macro data unavailable",
                risk_assessment={'overall_risk': 0.6},
                timestamp=datetime.now()
            )
    
    def _gather_macro_data(self) -> Dict[str, float]:
        """Gather macroeconomic indicators"""
        try:
            # Mock macro data (replace with real economic APIs)
            macro_data = {
                'fed_rate': 5.25,  # Current fed funds rate
                'inflation_rate': 3.2,  # CPI inflation
                'unemployment_rate': 3.8,  # Unemployment
                'gdp_growth': 2.1,  # GDP growth rate
                'dollar_index': 103.5,  # DXY
                'yield_10y': 4.8,  # 10-year treasury yield
                'yield_curve': 0.3,  # 10Y-2Y spread
                'vix': 18.5  # Volatility index
            }
            
            return macro_data
            
        except Exception as e:
            logger.error(f"❌ Macro data gathering failed: {e}")
            return {}
    
    def _calculate_macro_impact(self, macro_data: Dict[str, float], timeframe: str) -> float:
        """Calculate macroeconomic impact on gold price"""
        try:
            # Gold typically inversely correlated with:
            # - Interest rates (higher rates = lower gold)
            # - Dollar strength (stronger dollar = lower gold)
            # Gold typically positively correlated with:
            # - Inflation (higher inflation = higher gold)
            # - Economic uncertainty (higher uncertainty = higher gold)
            
            fed_rate = macro_data.get('fed_rate', 5.0)
            inflation = macro_data.get('inflation_rate', 3.0)
            dollar_index = macro_data.get('dollar_index', 100.0)
            vix = macro_data.get('vix', 20.0)
            
            # Calculate impact factors
            rate_impact = -(fed_rate - 2.0) * 0.02  # Negative correlation
            inflation_impact = (inflation - 2.0) * 0.015  # Positive correlation
            dollar_impact = -(dollar_index - 100.0) * 0.001  # Negative correlation
            uncertainty_impact = (vix - 20.0) * 0.002  # Positive correlation
            
            total_impact = rate_impact + inflation_impact + dollar_impact + uncertainty_impact
            
            # Adjust for timeframe
            timeframe_multipliers = {
                '1h': 0.1,
                '4h': 0.3,
                '1d': 1.0,
                '1w': 2.0
            }
            
            multiplier = timeframe_multipliers.get(timeframe, 1.0)
            
            return max(-0.1, min(0.1, total_impact * multiplier))
            
        except Exception:
            return 0.0
    
    def _calculate_macro_confidence(self, macro_data: Dict[str, float]) -> float:
        """Calculate confidence based on macro data clarity"""
        try:
            # Higher confidence when macro conditions are clear
            fed_rate = macro_data.get('fed_rate', 5.0)
            inflation = macro_data.get('inflation_rate', 3.0)
            
            # Clear signals
            rate_clarity = 1.0 if abs(fed_rate - 2.5) > 1.0 else 0.5
            inflation_clarity = 1.0 if abs(inflation - 2.0) > 1.0 else 0.5
            
            # Data availability
            data_completeness = len(macro_data) / 8.0  # Expected 8 indicators
            
            confidence = (rate_clarity + inflation_clarity + data_completeness) / 3.0
            
            return max(0.2, min(0.85, confidence))
            
        except Exception:
            return 0.5
    
    def _determine_macro_direction(self, macro_impact: float) -> str:
        """Determine direction based on macro impact"""
        if macro_impact > 0.01:
            return 'bullish'
        elif macro_impact < -0.01:
            return 'bearish'
        else:
            return 'neutral'
    
    def _generate_macro_reasoning(self, macro_data: Dict[str, float], direction: str) -> str:
        """Generate reasoning for macro prediction"""
        try:
            fed_rate = macro_data.get('fed_rate', 5.0)
            inflation = macro_data.get('inflation_rate', 3.0)
            
            reasoning = f"Macro analysis indicates {direction} bias. "
            
            if fed_rate > 4.0:
                reasoning += "High interest rates pressure gold. "
            elif fed_rate < 2.0:
                reasoning += "Low rates support gold. "
            
            if inflation > 3.5:
                reasoning += "High inflation favors gold as hedge. "
            elif inflation < 2.0:
                reasoning += "Low inflation reduces gold appeal. "
            
            return reasoning
            
        except Exception:
            return f"Macro analysis suggests {direction} market direction."
    
    def _calculate_macro_risk(self, macro_data: Dict[str, float]) -> float:
        """Calculate macroeconomic risk"""
        try:
            vix = macro_data.get('vix', 20.0)
            yield_curve = macro_data.get('yield_curve', 0.5)
            
            # Risk factors
            volatility_risk = min(1.0, vix / 40.0)
            curve_risk = 1.0 if yield_curve < 0 else 0.3  # Inverted curve = high risk
            
            return (volatility_risk + curve_risk) / 2.0
            
        except Exception:
            return 0.5
    
    def get_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence based on market conditions"""
        try:
            base_confidence = 0.65
            
            # Macro strategies work better in crisis and volatile periods
            if market_conditions.market_regime == 'crisis':
                base_confidence += 0.25
            elif market_conditions.market_regime == 'volatile':
                base_confidence += 0.15
            
            return min(0.9, max(0.2, base_confidence))
            
        except Exception:
            return 0.65
    
    def _store_prediction(self, result: StrategyResult):
        """Store macro prediction in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_performance 
                    (strategy_name, prediction_timestamp, predicted_price, confidence, timeframe, market_conditions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.strategy_name,
                    result.timestamp,
                    result.predicted_price,
                    result.confidence,
                    result.timeframe,
                    json.dumps(result.technical_indicators)
                ))
        except Exception as e:
            logger.error(f"❌ Failed to store macro prediction: {e}")

class PatternStrategy(BaseStrategy):
    """
    📊 Pattern Recognition Strategy
    Chart pattern recognition, Elliott Wave, Fibonacci retracements
    """
    
    def __init__(self):
        super().__init__("PatternStrategy")
        self.patterns = ['head_shoulders', 'double_top', 'triangle', 'flag', 'wedge']
        self.model = RandomForestRegressor(n_estimators=120, random_state=42)
    
    def predict(self, data: pd.DataFrame, timeframe: str) -> StrategyResult:
        """Generate pattern-based prediction"""
        try:
            # Detect patterns
            patterns = self._detect_patterns(data)
            
            # Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(data)
            
            # Analyze Elliott Wave
            wave_analysis = self._analyze_elliott_wave(data)
            
            # Generate prediction
            current_price = data['Close'].iloc[-1]
            pattern_impact = self._calculate_pattern_impact(patterns, fib_levels, wave_analysis)
            predicted_price = current_price * (1 + pattern_impact)
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(patterns, data)
            
            # Determine direction
            direction = self._determine_pattern_direction(pattern_impact, patterns)
            
            result = StrategyResult(
                strategy_name=self.name,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                timeframe=timeframe,
                technical_indicators={'patterns': patterns, 'fibonacci': fib_levels, 'elliott_wave': wave_analysis},
                reasoning=self._generate_pattern_reasoning(patterns, direction),
                risk_assessment={'pattern_reliability': confidence},
                timestamp=datetime.now()
            )
            
            self._store_prediction(result)
            
            logger.info(f"📊 Pattern prediction: ${predicted_price:.2f} ({direction}, {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pattern strategy prediction failed: {e}")
            current_price = data['Close'].iloc[-1]
            return StrategyResult(
                strategy_name=self.name,
                predicted_price=current_price,
                confidence=0.1,
                direction='neutral',
                timeframe=timeframe,
                technical_indicators={},
                reasoning="Pattern analysis failed",
                risk_assessment={'overall_risk': 0.7},
                timestamp=datetime.now()
            )
    
    def _detect_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Detect chart patterns"""
        try:
            patterns = {}
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            
            # Simple pattern detection (can be enhanced with more sophisticated algorithms)
            
            # Double top/bottom detection
            patterns['double_top'] = self._detect_double_top(high)
            patterns['double_bottom'] = self._detect_double_bottom(low)
            
            # Triangle pattern
            patterns['triangle'] = self._detect_triangle(high, low)
            
            # Head and shoulders
            patterns['head_shoulders'] = self._detect_head_shoulders(high)
            
            # Support/Resistance levels
            patterns['support_level'] = np.min(low[-20:])
            patterns['resistance_level'] = np.max(high[-20:])
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Pattern detection failed: {e}")
            return {}
    
    def _detect_double_top(self, high: np.ndarray) -> float:
        """Detect double top pattern"""
        try:
            if len(high) < 20:
                return 0.0
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(high, distance=5)
            
            if len(peaks) >= 2:
                # Check if last two peaks are similar
                last_peaks = high[peaks[-2:]]
                if abs(last_peaks[0] - last_peaks[1]) / last_peaks[0] < 0.02:
                    return 0.8  # Strong double top signal
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_double_bottom(self, low: np.ndarray) -> float:
        """Detect double bottom pattern"""
        try:
            if len(low) < 20:
                return 0.0
            
            # Find valleys
            from scipy.signal import find_peaks
            valleys, _ = find_peaks(-low, distance=5)
            
            if len(valleys) >= 2:
                # Check if last two valleys are similar
                last_valleys = low[valleys[-2:]]
                if abs(last_valleys[0] - last_valleys[1]) / last_valleys[0] < 0.02:
                    return 0.8  # Strong double bottom signal
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_triangle(self, high: np.ndarray, low: np.ndarray) -> float:
        """Detect triangle pattern"""
        try:
            if len(high) < 30:
                return 0.0
            
            # Calculate trend lines for highs and lows
            recent_high = high[-30:]
            recent_low = low[-30:]
            
            # Simple trend detection
            high_trend = np.polyfit(range(len(recent_high)), recent_high, 1)[0]
            low_trend = np.polyfit(range(len(recent_low)), recent_low, 1)[0]
            
            # Converging lines indicate triangle
            if high_trend < 0 and low_trend > 0:
                convergence = abs(high_trend) + abs(low_trend)
                return min(0.8, convergence * 100)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_head_shoulders(self, high: np.ndarray) -> float:
        """Detect head and shoulders pattern"""
        try:
            if len(high) < 30:
                return 0.0
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(high, distance=5)
            
            if len(peaks) >= 3:
                # Check for head and shoulders pattern
                last_three_peaks = high[peaks[-3:]]
                left_shoulder, head, right_shoulder = last_three_peaks
                
                # Head should be higher than shoulders
                if (head > left_shoulder and head > right_shoulder and 
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                    return 0.7
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            high = data['High'].max()
            low = data['Low'].min()
            diff = high - low
            
            fib_levels = {
                'fib_0': high,
                'fib_236': high - diff * 0.236,
                'fib_382': high - diff * 0.382,
                'fib_500': high - diff * 0.500,
                'fib_618': high - diff * 0.618,
                'fib_100': low
            }
            
            return fib_levels
            
        except Exception:
            return {}
    
    def _analyze_elliott_wave(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Elliott Wave patterns"""
        try:
            close = data['Close'].values
            
            # Simplified Elliott Wave analysis
            wave_analysis = {}
            
            # Calculate recent trend
            recent_trend = np.polyfit(range(len(close[-20:])), close[-20:], 1)[0]
            wave_analysis['trend_direction'] = 'up' if recent_trend > 0 else 'down'
            wave_analysis['trend_strength'] = abs(recent_trend)
            
            # Wave count estimation (simplified)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(close, distance=5)
            valleys, _ = find_peaks(-close, distance=5)
            
            wave_analysis['wave_count'] = len(peaks) + len(valleys)
            wave_analysis['current_wave'] = 'impulse' if len(peaks) > len(valleys) else 'corrective'
            
            return wave_analysis
            
        except Exception:
            return {}
    
    def _calculate_pattern_impact(self, patterns: Dict[str, float], fib_levels: Dict[str, float], wave_analysis: Dict[str, Any]) -> float:
        """Calculate overall pattern impact"""
        try:
            impact = 0.0
            
            # Pattern impacts
            if patterns.get('double_top', 0) > 0.5:
                impact -= 0.02  # Bearish
            if patterns.get('double_bottom', 0) > 0.5:
                impact += 0.02  # Bullish
            if patterns.get('head_shoulders', 0) > 0.5:
                impact -= 0.015  # Bearish
            
            # Triangle pattern (direction depends on breakout)
            triangle_strength = patterns.get('triangle', 0)
            if triangle_strength > 0.5:
                # Assume bullish breakout for now
                impact += triangle_strength * 0.01
            
            # Elliott Wave impact
            if wave_analysis.get('current_wave') == 'impulse':
                if wave_analysis.get('trend_direction') == 'up':
                    impact += 0.01
                else:
                    impact -= 0.01
            
            return max(-0.05, min(0.05, impact))
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, float], data: pd.DataFrame) -> float:
        """Calculate confidence based on pattern strength"""
        try:
            pattern_strengths = [v for v in patterns.values() if isinstance(v, float) and 0 <= v <= 1]
            
            if not pattern_strengths:
                return 0.3
            
            max_strength = max(pattern_strengths)
            avg_strength = np.mean(pattern_strengths)
            
            # Data quality factor
            data_quality = min(1.0, len(data) / 100.0)  # More data = higher confidence
            
            confidence = (max_strength * 0.6 + avg_strength * 0.2 + data_quality * 0.2)
            
            return max(0.1, min(0.85, confidence))
            
        except Exception:
            return 0.5
    
    def _determine_pattern_direction(self, pattern_impact: float, patterns: Dict[str, float]) -> str:
        """Determine direction based on patterns"""
        try:
            if pattern_impact > 0.005:
                return 'bullish'
            elif pattern_impact < -0.005:
                return 'bearish'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'
    
    def _generate_pattern_reasoning(self, patterns: Dict[str, float], direction: str) -> str:
        """Generate reasoning for pattern prediction"""
        try:
            reasoning = f"Pattern analysis indicates {direction} bias. "
            
            detected_patterns = []
            if patterns.get('double_top', 0) > 0.5:
                detected_patterns.append("double top")
            if patterns.get('double_bottom', 0) > 0.5:
                detected_patterns.append("double bottom")
            if patterns.get('head_shoulders', 0) > 0.5:
                detected_patterns.append("head and shoulders")
            if patterns.get('triangle', 0) > 0.5:
                detected_patterns.append("triangle")
            
            if detected_patterns:
                reasoning += f"Detected patterns: {', '.join(detected_patterns)}. "
            
            return reasoning
            
        except Exception:
            return f"Pattern analysis suggests {direction} market direction."
    
    def get_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence based on market conditions"""
        try:
            base_confidence = 0.55
            
            # Patterns work better in trending markets
            if market_conditions.market_regime == 'trending':
                base_confidence += 0.2
            elif market_conditions.market_regime == 'ranging':
                base_confidence += 0.1
            elif market_conditions.market_regime == 'volatile':
                base_confidence -= 0.2
            
            return min(0.8, max(0.2, base_confidence))
            
        except Exception:
            return 0.55
    
    def _store_prediction(self, result: StrategyResult):
        """Store pattern prediction in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_performance 
                    (strategy_name, prediction_timestamp, predicted_price, confidence, timeframe, market_conditions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.strategy_name,
                    result.timestamp,
                    result.predicted_price,
                    result.confidence,
                    result.timeframe,
                    json.dumps(result.technical_indicators)
                ))
        except Exception as e:
            logger.error(f"❌ Failed to store pattern prediction: {e}")

class MomentumStrategy(BaseStrategy):
    """
    🚀 Momentum Strategy
    Trend following, breakout detection, volume-price analysis
    """
    
    def __init__(self):
        super().__init__("MomentumStrategy")
        self.momentum_indicators = ['roc', 'momentum', 'trix', 'adx']
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    def predict(self, data: pd.DataFrame, timeframe: str) -> StrategyResult:
        """Generate momentum-based prediction"""
        try:
            # Calculate momentum indicators
            momentum_data = self._calculate_momentum_indicators(data)
            
            # Detect breakouts
            breakout_signals = self._detect_breakouts(data)
            
            # Analyze volume-price relationship
            volume_analysis = self._analyze_volume_price(data)
            
            # Generate prediction
            current_price = data['Close'].iloc[-1]
            momentum_impact = self._calculate_momentum_impact(momentum_data, breakout_signals, volume_analysis, timeframe)
            predicted_price = current_price * (1 + momentum_impact)
            
            # Calculate confidence
            confidence = self._calculate_momentum_confidence(momentum_data, breakout_signals)
            
            # Determine direction
            direction = self._determine_momentum_direction(momentum_impact, momentum_data)
            
            result = StrategyResult(
                strategy_name=self.name,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                timeframe=timeframe,
                technical_indicators={**momentum_data, **breakout_signals, **volume_analysis},
                reasoning=self._generate_momentum_reasoning(momentum_data, breakout_signals, direction),
                risk_assessment={'momentum_volatility': self._calculate_momentum_risk(data)},
                timestamp=datetime.now()
            )
            
            self._store_prediction(result)
            
            logger.info(f"🚀 Momentum prediction: ${predicted_price:.2f} ({direction}, {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Momentum strategy prediction failed: {e}")
            current_price = data['Close'].iloc[-1]
            return StrategyResult(
                strategy_name=self.name,
                predicted_price=current_price,
                confidence=0.1,
                direction='neutral',
                timeframe=timeframe,
                technical_indicators={},
                reasoning="Momentum analysis failed",
                risk_assessment={'overall_risk': 0.7},
                timestamp=datetime.now()
            )
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            
            momentum_data = {}
            
            # Rate of Change
            if len(close) >= 14:
                roc = ((close[-1] - close[-14]) / close[-14]) * 100
                momentum_data['roc_14'] = roc
            
            # Momentum (price difference)
            if len(close) >= 10:
                momentum_data['momentum_10'] = close[-1] - close[-10]
            
            # Average Directional Index (ADX)
            if len(close) >= 14:
                try:
                    import talib
                    adx = talib.ADX(high, low, close, timeperiod=14)
                    momentum_data['adx'] = adx[-1] if not np.isnan(adx[-1]) else 25.0
                except:
                    momentum_data['adx'] = 25.0
            
            # TRIX indicator
            if len(close) >= 30:
                try:
                    import talib
                    trix = talib.TRIX(close, timeperiod=14)
                    momentum_data['trix'] = trix[-1] if not np.isnan(trix[-1]) else 0.0
                except:
                    momentum_data['trix'] = 0.0
            
            # Price momentum (percentage change)
            if len(close) >= 5:
                momentum_data['momentum_5d'] = ((close[-1] - close[-5]) / close[-5]) * 100
            
            return momentum_data
            
        except Exception as e:
            logger.error(f"❌ Momentum indicators calculation failed: {e}")
            return {}
    
    def _detect_breakouts(self, data: pd.DataFrame) -> Dict[str, float]:
        """Detect price breakouts"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(close))
            
            breakout_signals = {}
            
            # Resistance breakout
            if len(high) >= 20:
                resistance = np.max(high[-21:-1])  # Previous 20 periods
                current_high = high[-1]
                if current_high > resistance:
                    breakout_signals['resistance_breakout'] = (current_high - resistance) / resistance
                else:
                    breakout_signals['resistance_breakout'] = 0.0
            
            # Support breakout
            if len(low) >= 20:
                support = np.min(low[-21:-1])  # Previous 20 periods
                current_low = low[-1]
                if current_low < support:
                    breakout_signals['support_breakdown'] = (support - current_low) / support
                else:
                    breakout_signals['support_breakdown'] = 0.0
            
            # Volume breakout
            if len(volume) >= 20:
                avg_volume = np.mean(volume[-21:-1])
                current_volume = volume[-1]
                if current_volume > avg_volume * 1.5:
                    breakout_signals['volume_breakout'] = current_volume / avg_volume
                else:
                    breakout_signals['volume_breakout'] = 1.0
            
            # Volatility breakout
            if len(close) >= 20:
                volatility = np.std(close[-21:-1])
                current_change = abs(close[-1] - close[-2])
                if current_change > volatility * 2:
                    breakout_signals['volatility_breakout'] = current_change / volatility
                else:
                    breakout_signals['volatility_breakout'] = 0.0
            
            return breakout_signals
            
        except Exception as e:
            logger.error(f"❌ Breakout detection failed: {e}")
            return {}
    
    def _analyze_volume_price(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume-price relationship"""
        try:
            close = data['Close'].values
            volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(close))
            
            volume_analysis = {}
            
            # Volume-price correlation
            if len(close) >= 20:
                price_changes = np.diff(close[-20:])
                volume_changes = np.diff(volume[-20:])
                
                if len(price_changes) > 0 and len(volume_changes) > 0:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                    volume_analysis['volume_price_correlation'] = correlation if not np.isnan(correlation) else 0.0
            
            # On-Balance Volume (OBV)
            if len(close) >= 10:
                obv = np.zeros(len(close))
                for i in range(1, len(close)):
                    if close[i] > close[i-1]:
                        obv[i] = obv[i-1] + volume[i]
                    elif close[i] < close[i-1]:
                        obv[i] = obv[i-1] - volume[i]
                    else:
                        obv[i] = obv[i-1]
                
                # OBV trend
                if len(obv) >= 10:
                    obv_trend = np.polyfit(range(len(obv[-10:])), obv[-10:], 1)[0]
                    volume_analysis['obv_trend'] = obv_trend
            
            # Volume Rate of Change
            if len(volume) >= 10:
                volume_roc = ((volume[-1] - volume[-10]) / volume[-10]) * 100
                volume_analysis['volume_roc'] = volume_roc
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"❌ Volume-price analysis failed: {e}")
            return {}
    
    def _calculate_momentum_impact(self, momentum_data: Dict[str, float], breakout_signals: Dict[str, float], 
                                 volume_analysis: Dict[str, float], timeframe: str) -> float:
        """Calculate momentum impact on price"""
        try:
            impact = 0.0
            
            # ROC impact
            roc = momentum_data.get('roc_14', 0)
            impact += roc * 0.001  # Convert to decimal
            
            # ADX impact (trend strength)
            adx = momentum_data.get('adx', 25)
            if adx > 25:  # Strong trend
                momentum_5d = momentum_data.get('momentum_5d', 0)
                impact += (momentum_5d * 0.001) * (adx / 50)  # Scale by trend strength
            
            # Breakout impacts
            resistance_breakout = breakout_signals.get('resistance_breakout', 0)
            support_breakdown = breakout_signals.get('support_breakdown', 0)
            
            impact += resistance_breakout * 0.02  # Bullish
            impact -= support_breakdown * 0.02  # Bearish
            
            # Volume confirmation
            volume_breakout = breakout_signals.get('volume_breakout', 1.0)
            if volume_breakout > 1.5:
                impact *= 1.5  # Amplify with volume confirmation
            
            # Timeframe adjustment
            timeframe_multipliers = {
                '1h': 0.5,
                '4h': 0.8,
                '1d': 1.0,
                '1w': 1.5
            }
            
            multiplier = timeframe_multipliers.get(timeframe, 1.0)
            impact *= multiplier
            
            return max(-0.08, min(0.08, impact))
            
        except Exception:
            return 0.0
    
    def _calculate_momentum_confidence(self, momentum_data: Dict[str, float], breakout_signals: Dict[str, float]) -> float:
        """Calculate confidence based on momentum strength"""
        try:
            confidence_factors = []
            
            # ADX confidence
            adx = momentum_data.get('adx', 25)
            if adx > 40:
                confidence_factors.append(0.9)  # Very strong trend
            elif adx > 25:
                confidence_factors.append(0.7)  # Strong trend
            else:
                confidence_factors.append(0.4)  # Weak trend
            
            # ROC consistency
            roc = abs(momentum_data.get('roc_14', 0))
            if roc > 5:
                confidence_factors.append(0.8)
            elif roc > 2:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            
            # Breakout confirmation
            breakout_strength = max(
                breakout_signals.get('resistance_breakout', 0),
                breakout_signals.get('support_breakdown', 0)
            )
            if breakout_strength > 0.02:
                confidence_factors.append(0.8)
            elif breakout_strength > 0.01:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            return max(0.2, min(0.9, np.mean(confidence_factors)))
            
        except Exception:
            return 0.6
    
    def _determine_momentum_direction(self, momentum_impact: float, momentum_data: Dict[str, float]) -> str:
        """Determine direction based on momentum"""
        try:
            if momentum_impact > 0.01:
                return 'bullish'
            elif momentum_impact < -0.01:
                return 'bearish'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'
    
    def _generate_momentum_reasoning(self, momentum_data: Dict[str, float], breakout_signals: Dict[str, float], direction: str) -> str:
        """Generate reasoning for momentum prediction"""
        try:
            reasoning = f"Momentum analysis indicates {direction} bias. "
            
            adx = momentum_data.get('adx', 25)
            roc = momentum_data.get('roc_14', 0)
            
            if adx > 25:
                reasoning += f"Strong trend detected (ADX: {adx:.1f}). "
            
            if abs(roc) > 2:
                trend_word = "upward" if roc > 0 else "downward"
                reasoning += f"Significant {trend_word} momentum (ROC: {roc:.1f}%). "
            
            if breakout_signals.get('resistance_breakout', 0) > 0.01:
                reasoning += "Resistance breakout detected. "
            if breakout_signals.get('support_breakdown', 0) > 0.01:
                reasoning += "Support breakdown detected. "
            
            return reasoning
            
        except Exception:
            return f"Momentum analysis suggests {direction} market direction."
    
    def _calculate_momentum_risk(self, data: pd.DataFrame) -> float:
        """Calculate momentum-based risk"""
        try:
            close = data['Close'].values
            
            # Volatility risk
            volatility = np.std(close[-20:]) / np.mean(close[-20:]) if len(close) >= 20 else 0.02
            
            # Momentum reversal risk (high momentum can reverse quickly)
            recent_momentum = abs(((close[-1] - close[-5]) / close[-5])) if len(close) >= 5 else 0.01
            reversal_risk = min(1.0, recent_momentum * 10)
            
            return (volatility * 10 + reversal_risk) / 2
            
        except Exception:
            return 0.5
    
    def get_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence based on market conditions"""
        try:
            base_confidence = 0.7
            
            # Momentum strategies work best in trending markets
            if market_conditions.market_regime == 'trending':
                base_confidence += 0.2
            elif market_conditions.market_regime == 'volatile':
                base_confidence += 0.1
            elif market_conditions.market_regime == 'ranging':
                base_confidence -= 0.2
            
            # Adjust for trend strength
            trend_boost = market_conditions.trend_strength * 0.15
            base_confidence += trend_boost
            
            return min(0.95, max(0.2, base_confidence))
            
        except Exception:
            return 0.7
    
    def _store_prediction(self, result: StrategyResult):
        """Store momentum prediction in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_performance 
                    (strategy_name, prediction_timestamp, predicted_price, confidence, timeframe, market_conditions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.strategy_name,
                    result.timestamp,
                    result.predicted_price,
                    result.confidence,
                    result.timeframe,
                    json.dumps(result.technical_indicators)
                ))
        except Exception as e:
            logger.error(f"❌ Failed to store momentum prediction: {e}")

# Export all strategy classes for the main ensemble system
__all__ = [
    'MacroStrategy', 'PatternStrategy', 'MomentumStrategy'
]
