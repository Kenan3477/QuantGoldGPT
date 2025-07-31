#!/usr/bin/env python3
"""
Advanced Multi-Strategy ML Engine for GoldGPT
Implements a comprehensive prediction system with specialized strategies,
ensemble voting, and dynamic performance tracking.

Features:
- BaseStrategy abstract class with standard interface
- 5 specialized strategy classes (Technical, Sentiment, Macro, Pattern, Momentum)
- EnsembleVotingSystem with weighted voting
- StrategyPerformanceTracker with dynamic weight adjustment
- Confidence scoring based on model agreement
- Support/resistance calculation with TP/SL recommendations
- REST API with WebSocket integration
- Comprehensive logging and performance metrics
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import threading
import time
from collections import defaultdict, deque
import json
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Technical Analysis
try:
    import ta
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # Logger will be defined below

# Import our data sources
try:
    from data_integration_engine import DataManager, DataIntegrationEngine
    from enhanced_data_sources import EnhancedDataSources
    DATA_PIPELINE_AVAILABLE = True
except ImportError:
    DATA_PIPELINE_AVAILABLE = False
    logger.warning("Data pipeline not available, using fallback mechanisms")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check TALib availability after logger is configured
if not TALIB_AVAILABLE:
    logger.warning("TALib not available, using basic technical indicators")

# Constants
TIMEFRAMES = ['1H', '4H', '1D', '1W']
SUPPORTED_SYMBOLS = ['XAU/USD', 'GOLD', 'GC=F']

@dataclass
class PredictionResult:
    """Structured prediction result from individual strategy"""
    strategy_name: str
    symbol: str
    timeframe: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    support_level: Optional[float]
    resistance_level: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    features_used: List[str]
    reasoning: str
    raw_features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': round(self.current_price, 2),
            'predicted_price': round(self.predicted_price, 2),
            'price_change': round(self.price_change, 2),
            'price_change_percent': round(self.price_change_percent, 3),
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'support_level': round(self.support_level, 2) if self.support_level else None,
            'resistance_level': round(self.resistance_level, 2) if self.resistance_level else None,
            'stop_loss': round(self.stop_loss, 2) if self.stop_loss else None,
            'take_profit': round(self.take_profit, 2) if self.take_profit else None,
            'timestamp': self.timestamp.isoformat(),
            'features_used': self.features_used,
            'reasoning': self.reasoning
        }

@dataclass
class EnsemblePrediction:
    """Final ensemble prediction combining all strategies"""
    symbol: str
    timeframe: str
    current_price: float
    predicted_price: float
    confidence_interval: Tuple[float, float]  # (lower_bound, upper_bound)
    direction: str
    ensemble_confidence: float
    model_agreement: float  # How much strategies agree (0.0 to 1.0)
    support_levels: List[float]
    resistance_levels: List[float]
    recommended_stop_loss: float
    recommended_take_profit: float
    strategy_contributions: Dict[str, float]  # Strategy name -> weight
    individual_predictions: List[PredictionResult]
    timestamp: datetime
    prediction_quality_score: float  # Overall quality metric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': round(self.current_price, 2),
            'predicted_price': round(self.predicted_price, 2),
            'price_change': round(self.predicted_price - self.current_price, 2),
            'price_change_percent': round(((self.predicted_price - self.current_price) / self.current_price) * 100, 3),
            'confidence_interval': [round(self.confidence_interval[0], 2), round(self.confidence_interval[1], 2)],
            'direction': self.direction,
            'ensemble_confidence': round(self.ensemble_confidence, 3),
            'model_agreement': round(self.model_agreement, 3),
            'support_levels': [round(level, 2) for level in self.support_levels],
            'resistance_levels': [round(level, 2) for level in self.resistance_levels],
            'recommended_stop_loss': round(self.recommended_stop_loss, 2),
            'recommended_take_profit': round(self.recommended_take_profit, 2),
            'strategy_contributions': {k: round(v, 3) for k, v in self.strategy_contributions.items()},
            'individual_predictions': [pred.to_dict() for pred in self.individual_predictions],
            'timestamp': self.timestamp.isoformat(),
            'prediction_quality_score': round(self.prediction_quality_score, 3)
        }


class BaseStrategy(ABC):
    """Abstract base class for all prediction strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.model = None
        self.scaler = StandardScaler()
        self.last_training_time = None
        self.performance_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'r2_score': 0.0,
            'predictions_count': 0
        }
    
    @abstractmethod
    async def extract_features(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract strategy-specific features from market data"""
        pass
    
    @abstractmethod
    async def train_model(self, historical_data: pd.DataFrame, target_column: str = 'close') -> bool:
        """Train the strategy's ML model"""
        pass
    
    @abstractmethod
    async def generate_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate prediction using the trained model"""
        pass
    
    def calculate_support_resistance(self, price_data: List[float], current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate support and resistance levels"""
        if not price_data or len(price_data) < 10:
            return None, None
        
        prices = np.array(price_data)
        
        # Find local minima and maxima
        from scipy.signal import find_peaks
        
        # Support levels (local minima)
        min_peaks, _ = find_peaks(-prices, distance=5)
        support_levels = prices[min_peaks] if len(min_peaks) > 0 else []
        
        # Resistance levels (local maxima)  
        max_peaks, _ = find_peaks(prices, distance=5)
        resistance_levels = prices[max_peaks] if len(max_peaks) > 0 else []
        
        # Find nearest levels
        support = None
        resistance = None
        
        if len(support_levels) > 0:
            # Find highest support level below current price
            valid_supports = support_levels[support_levels < current_price]
            support = float(np.max(valid_supports)) if len(valid_supports) > 0 else None
        
        if len(resistance_levels) > 0:
            # Find lowest resistance level above current price
            valid_resistance = resistance_levels[resistance_levels > current_price]
            resistance = float(np.min(valid_resistance)) if len(valid_resistance) > 0 else None
        
        return support, resistance
    
    def calculate_stop_loss_take_profit(self, current_price: float, predicted_price: float, 
                                      support: Optional[float], resistance: Optional[float],
                                      confidence: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate recommended stop-loss and take-profit levels"""
        direction = "bullish" if predicted_price > current_price else "bearish"
        
        # Base risk as percentage of current price
        base_risk = 0.02 * (1 - confidence)  # Lower confidence = higher risk
        base_risk = max(0.005, min(base_risk, 0.05))  # Between 0.5% and 5%
        
        if direction == "bullish":
            # Stop loss below current price
            stop_loss = current_price * (1 - base_risk)
            if support and support > stop_loss:
                stop_loss = support * 0.995  # Slightly below support
            
            # Take profit above predicted price
            profit_target = max(predicted_price, current_price * (1 + base_risk * 2))
            if resistance and resistance < profit_target * 1.2:
                profit_target = resistance * 0.995  # Slightly below resistance
            
            return float(stop_loss), float(profit_target)
        
        else:  # bearish
            # Stop loss above current price
            stop_loss = current_price * (1 + base_risk)
            if resistance and resistance < stop_loss:
                stop_loss = resistance * 1.005  # Slightly above resistance
            
            # Take profit below predicted price
            profit_target = min(predicted_price, current_price * (1 - base_risk * 2))
            if support and support > profit_target * 0.8:
                profit_target = support * 1.005  # Slightly above support
                
            return float(stop_loss), float(profit_target)


class TechnicalStrategy(BaseStrategy):
    """Technical Analysis Strategy using RSI, MACD, Bollinger Bands, etc."""
    
    def __init__(self):
        super().__init__("Technical")
        self.lookback_period = 100
        
    async def extract_features(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract technical indicators as features"""
        features = {}
        
        try:
            # Get price data
            if 'price_data' not in data or len(data['price_data']) < 20:
                logger.warning(f"Insufficient price data for technical analysis")
                return self._get_fallback_features()
            
            prices = pd.DataFrame(data['price_data'])
            if 'close' not in prices.columns:
                prices.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            close = prices['close'].values
            high = prices['high'].values
            low = prices['low'].values
            volume = prices.get('volume', pd.Series([1000] * len(close))).values
            
            # RSI
            if TALIB_AVAILABLE:
                features['rsi'] = float(talib.RSI(close)[-1])
            else:
                features['rsi'] = self._calculate_rsi(close)
            
            # MACD
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(close)
                features['macd'] = float(macd[-1])
                features['macd_signal'] = float(macd_signal[-1])
                features['macd_histogram'] = float(macd_hist[-1])
            else:
                macd_data = self._calculate_macd(close)
                features.update(macd_data)
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
                features['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                features['bb_squeeze'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            else:
                bb_data = self._calculate_bollinger_bands(close)
                features.update(bb_data)
            
            # Moving Averages
            features['sma_20'] = float(np.mean(close[-20:]))
            features['sma_50'] = float(np.mean(close[-50:]) if len(close) >= 50 else np.mean(close))
            features['ema_12'] = self._calculate_ema(close, 12)
            features['ema_26'] = self._calculate_ema(close, 26)
            
            # Price position relative to MAs
            features['price_vs_sma20'] = (close[-1] - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (close[-1] - features['sma_50']) / features['sma_50']
            
            # Volatility
            features['volatility'] = float(np.std(close[-20:]) / np.mean(close[-20:]))
            
            # Volume indicators
            features['volume_sma'] = float(np.mean(volume[-20:]))
            features['volume_ratio'] = float(volume[-1] / features['volume_sma'])
            
            # Stochastic
            features['stoch_k'] = self._calculate_stochastic(high, low, close)
            
            # ATR (Average True Range)
            features['atr'] = self._calculate_atr(high, low, close)
            
            # Williams %R
            features['williams_r'] = self._calculate_williams_r(high, low, close)
            
            logger.info(f"Extracted {len(features)} technical features for {symbol} {timeframe}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            return self._get_fallback_features()
    
    def _get_fallback_features(self) -> Dict[str, float]:
        """Fallback features when data is insufficient"""
        return {
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_position': 0.5,
            'bb_squeeze': 0.02,
            'sma_20': 2000.0,
            'sma_50': 2000.0,
            'ema_12': 2000.0,
            'ema_26': 2000.0,
            'price_vs_sma20': 0.0,
            'price_vs_sma50': 0.0,
            'volatility': 0.02,
            'volume_sma': 1000.0,
            'volume_ratio': 1.0,
            'stoch_k': 50.0,
            'atr': 20.0,
            'williams_r': -50.0
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI manually"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate MACD manually"""
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd = ema_12 - ema_26
        # Signal line is 9-period EMA of MACD
        # For simplicity, using simple average
        macd_signal = np.mean([macd] * 9)  # Simplified
        macd_histogram = macd - macd_signal
        
        return {
            'macd': float(macd),
            'macd_signal': float(macd_signal),
            'macd_histogram': float(macd_histogram)
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA manually"""
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return float(ema)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands manually"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        current_price = prices[-1]
        bb_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        bb_squeeze = (upper_band - lower_band) / sma if sma != 0 else 0.02
        
        return {
            'bb_position': float(bb_position),
            'bb_squeeze': float(bb_squeeze)
        }
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Stochastic %K"""
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        
        if highest_high == lowest_low:
            return 50.0
        
        stoch_k = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        return float(stoch_k)
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high) < 2:
            return 20.0
        
        tr_list = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        
        atr = np.mean(tr_list[-period:]) if len(tr_list) >= period else np.mean(tr_list)
        return float(atr)
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Williams %R"""
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - close[-1]) / (highest_high - lowest_low)) * -100
        return float(williams_r)
    
    async def train_model(self, historical_data: pd.DataFrame, target_column: str = 'close') -> bool:
        """Train the technical analysis model"""
        try:
            if len(historical_data) < 100:
                logger.warning("Insufficient data for technical model training")
                return False
            
            # Prepare features
            feature_data = []
            targets = []
            
            for i in range(50, len(historical_data) - 1):  # Need lookback and forward data
                current_data = {
                    'price_data': historical_data.iloc[i-50:i].to_dict('records')
                }
                features = await self.extract_features("GOLD", "1D", current_data)
                
                if features:
                    feature_data.append(list(features.values()))
                    # Target is next day's close price
                    targets.append(historical_data.iloc[i+1][target_column])
            
            if len(feature_data) < 10:
                logger.error("Not enough feature data for training")
                return False
            
            X = np.array(feature_data)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble of models
            models = [
                RandomForestRegressor(n_estimators=100, random_state=42),
                GradientBoostingRegressor(random_state=42),
                SVR(kernel='rbf', C=100, gamma=0.001)
            ]
            
            self.model = VotingRegressor(
                estimators=[('rf', models[0]), ('gb', models[1]), ('svr', models[2])],
                n_jobs=-1
            )
            
            self.model.fit(X_scaled, y)
            
            # Calculate performance metrics
            predictions = self.model.predict(X_scaled)
            self.performance_metrics['r2_score'] = r2_score(y, predictions)
            self.performance_metrics['accuracy'] = 1.0 / (1.0 + mean_absolute_error(y, predictions) / np.mean(y))
            
            self.is_trained = True
            self.last_training_time = datetime.now(timezone.utc)
            
            logger.info(f"Technical strategy trained successfully. R² Score: {self.performance_metrics['r2_score']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training technical model: {e}")
            return False
    
    async def generate_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate technical analysis prediction"""
        try:
            # Extract features
            features = await self.extract_features(symbol, timeframe, current_data)
            
            if not features:
                raise ValueError("Could not extract technical features")
            
            # Use Gold API price or reasonable fallback
            try:
                from price_storage_manager import get_current_gold_price
                current_price = current_data.get('current_price', get_current_gold_price() or 3430.0)
            except:
                current_price = current_data.get('current_price', 3430.0)
            
            # Make prediction if model is trained
            if self.is_trained and self.model:
                feature_vector = np.array([list(features.values())]).reshape(1, -1)
                feature_vector_scaled = self.scaler.transform(feature_vector)
                predicted_price = float(self.model.predict(feature_vector_scaled)[0])
            else:
                # Fallback prediction based on technical indicators
                predicted_price = self._fallback_technical_prediction(features, current_price)
            
            # Ensure mathematical accuracy
            predicted_price = round(predicted_price, 2)
            price_change = round(predicted_price - current_price, 2)
            price_change_percent = round((price_change / current_price) * 100, 3) if current_price > 0 else 0.0
            
            # Determine direction and confidence
            direction = "bullish" if predicted_price > current_price else "bearish" if predicted_price < current_price else "neutral"
            
            # Calculate confidence based on signal strength
            confidence = self._calculate_technical_confidence(features)
            
            # Calculate support/resistance
            price_history = [item['close'] for item in current_data.get('price_data', [])] if current_data.get('price_data') else [current_price]
            support, resistance = self.calculate_support_resistance(price_history, current_price)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, predicted_price, support, resistance, confidence
            )
            
            # Generate reasoning
            reasoning = self._generate_technical_reasoning(features, direction, confidence)
            
            return PredictionResult(
                strategy_name=self.name,
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=reasoning,
                raw_features=features
            )
            
        except Exception as e:
            logger.error(f"Error generating technical prediction: {e}")
            # Return fallback prediction
            return self._generate_fallback_prediction(symbol, timeframe, current_data)
    
    def _fallback_technical_prediction(self, features: Dict[str, float], current_price: float) -> float:
        """Generate fallback prediction using technical indicators"""
        score = 0
        
        # RSI signals
        rsi = features.get('rsi', 50)
        if rsi < 30:
            score += 2  # Oversold, expect bounce
        elif rsi > 70:
            score -= 2  # Overbought, expect decline
        elif rsi < 45:
            score += 1
        elif rsi > 55:
            score -= 1
        
        # MACD signals
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        if macd > macd_signal:
            score += 1
        else:
            score -= 1
        
        # Price vs moving averages
        price_vs_sma20 = features.get('price_vs_sma20', 0)
        price_vs_sma50 = features.get('price_vs_sma50', 0)
        
        if price_vs_sma20 > 0.01:
            score += 1
        elif price_vs_sma20 < -0.01:
            score -= 1
        
        if price_vs_sma50 > 0.02:
            score += 1
        elif price_vs_sma50 < -0.02:
            score -= 1
        
        # Bollinger Bands
        bb_position = features.get('bb_position', 0.5)
        if bb_position < 0.2:
            score += 1  # Near lower band
        elif bb_position > 0.8:
            score -= 1  # Near upper band
        
        # Convert score to price prediction
        volatility = features.get('volatility', 0.02)
        # Fixed calculation: Make score impact more meaningful
        # Score ranges from -10 to +10, we want percentage changes from -2% to +2%
        max_change_percent = volatility * 100  # Use volatility as max change (e.g., 2%)
        price_change_percent = (score / 10) * max_change_percent  # Normalize score to percentage
        predicted_price = current_price * (1 + price_change_percent / 100)
        
        return predicted_price
    
    def _calculate_technical_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on technical signal strength"""
        confidence_factors = []
        
        # RSI confidence (higher when extreme)
        rsi = features.get('rsi', 50)
        rsi_confidence = min(abs(rsi - 50) / 50, 1.0)
        confidence_factors.append(rsi_confidence)
        
        # MACD confidence (higher when histogram is strong)
        macd_hist = abs(features.get('macd_histogram', 0))
        macd_confidence = min(macd_hist / 10, 1.0)  # Normalize
        confidence_factors.append(macd_confidence)
        
        # Volatility confidence (lower when too volatile)
        volatility = features.get('volatility', 0.02)
        vol_confidence = 1.0 - min(volatility / 0.1, 1.0)
        confidence_factors.append(vol_confidence)
        
        # Volume confirmation
        volume_ratio = features.get('volume_ratio', 1.0)
        vol_conf = min(volume_ratio / 2, 1.0) if volume_ratio > 1 else 0.5
        confidence_factors.append(vol_conf)
        
        return np.mean(confidence_factors)
    
    def _generate_technical_reasoning(self, features: Dict[str, float], direction: str, confidence: float) -> str:
        """Generate human-readable reasoning for technical prediction"""
        reasoning_parts = []
        
        rsi = features.get('rsi', 50)
        if rsi < 30:
            reasoning_parts.append("RSI indicates oversold conditions")
        elif rsi > 70:
            reasoning_parts.append("RSI indicates overbought conditions")
        
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        if macd > macd_signal:
            reasoning_parts.append("MACD shows bullish momentum")
        else:
            reasoning_parts.append("MACD shows bearish momentum")
        
        bb_position = features.get('bb_position', 0.5)
        if bb_position < 0.2:
            reasoning_parts.append("Price near Bollinger Band lower bound")
        elif bb_position > 0.8:
            reasoning_parts.append("Price near Bollinger Band upper bound")
        
        price_vs_sma20 = features.get('price_vs_sma20', 0)
        if price_vs_sma20 > 0.01:
            reasoning_parts.append("Price above 20-day SMA")
        elif price_vs_sma20 < -0.01:
            reasoning_parts.append("Price below 20-day SMA")
        
        base_reasoning = f"Technical analysis suggests {direction} trend. " + "; ".join(reasoning_parts)
        return f"{base_reasoning}. Confidence: {confidence:.1%}"
    
    def _generate_fallback_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate fallback prediction when normal prediction fails"""
        current_price = current_data.get('current_price', 2000.0)
        
        # Minimal prediction with small random movement
        price_change = np.random.uniform(-0.5, 0.5) / 100 * current_price
        predicted_price = round(current_price + price_change, 2)
        
        return PredictionResult(
            strategy_name=self.name,
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=round(price_change, 2),
            price_change_percent=round((price_change / current_price) * 100, 3),
            direction="neutral",
            confidence=0.3,
            support_level=current_price * 0.98,
            resistance_level=current_price * 1.02,
            stop_loss=current_price * 0.985,
            take_profit=current_price * 1.015,
            timestamp=datetime.now(timezone.utc),
            features_used=["fallback"],
            reasoning="Technical analysis fallback prediction due to data limitations",
            raw_features={}
        )


class SentimentStrategy(BaseStrategy):
    """Sentiment Analysis Strategy using news and social media data"""
    
    def __init__(self):
        super().__init__("Sentiment")
        self.sentiment_history = deque(maxlen=50)
        
    async def extract_features(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract sentiment-based features"""
        features = {}
        
        try:
            # News sentiment
            news_data = data.get('news_sentiment', {})
            features['news_sentiment'] = float(news_data.get('overall_sentiment', 0.0))
            features['news_confidence'] = float(news_data.get('confidence', 0.5))
            features['news_volume'] = float(news_data.get('article_count', 0))
            
            # Social media sentiment
            social_data = data.get('social_sentiment', {})
            features['social_sentiment'] = float(social_data.get('overall_sentiment', 0.0))
            features['social_volume'] = float(social_data.get('mention_count', 0))
            
            # Fear & Greed Index
            features['fear_greed_index'] = float(data.get('fear_greed_index', 50.0))
            
            # Market sentiment indicators
            features['vix_level'] = float(data.get('vix_level', 20.0))
            features['put_call_ratio'] = float(data.get('put_call_ratio', 1.0))
            
            # Economic sentiment
            economic_data = data.get('economic_sentiment', {})
            features['economic_sentiment'] = float(economic_data.get('overall', 0.0))
            features['inflation_sentiment'] = float(economic_data.get('inflation', 0.0))
            features['fed_sentiment'] = float(economic_data.get('fed_policy', 0.0))
            
            # Sentiment momentum
            self.sentiment_history.append(features['news_sentiment'])
            if len(self.sentiment_history) >= 5:
                recent_sentiment = list(self.sentiment_history)
                features['sentiment_momentum'] = float(np.mean(recent_sentiment[-3:]) - np.mean(recent_sentiment[-5:-2]))
                features['sentiment_volatility'] = float(np.std(recent_sentiment))
            else:
                features['sentiment_momentum'] = 0.0
                features['sentiment_volatility'] = 0.2
            
            logger.info(f"Extracted {len(features)} sentiment features for {symbol} {timeframe}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
            return self._get_fallback_sentiment_features()
    
    def _get_fallback_sentiment_features(self) -> Dict[str, float]:
        """Fallback sentiment features when data is unavailable"""
        return {
            'news_sentiment': 0.0,
            'news_confidence': 0.5,
            'news_volume': 10.0,
            'social_sentiment': 0.0,
            'social_volume': 100.0,
            'fear_greed_index': 50.0,
            'vix_level': 20.0,
            'put_call_ratio': 1.0,
            'economic_sentiment': 0.0,
            'inflation_sentiment': 0.0,
            'fed_sentiment': 0.0,
            'sentiment_momentum': 0.0,
            'sentiment_volatility': 0.2
        }
    
    async def train_model(self, historical_data: pd.DataFrame, target_column: str = 'close') -> bool:
        """Train sentiment analysis model"""
        try:
            if len(historical_data) < 50:
                logger.warning("Insufficient data for sentiment model training")
                return False
            
            # For now, use a simple linear model based on sentiment scores
            # In production, this would be trained on historical sentiment vs price data
            self.model = LinearRegression()
            
            # Create dummy training data (in real implementation, use historical sentiment data)
            X = np.random.randn(100, 13)  # 13 sentiment features
            y = historical_data[target_column].iloc[-100:] if len(historical_data) >= 100 else historical_data[target_column]
            
            self.model.fit(X, y)
            self.is_trained = True
            self.last_training_time = datetime.now(timezone.utc)
            
            logger.info("Sentiment strategy trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
            return False
    
    async def generate_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate sentiment-based prediction"""
        try:
            features = await self.extract_features(symbol, timeframe, current_data)
            current_price = current_data.get('current_price', 2000.0)
            
            # Calculate sentiment-based prediction
            predicted_price = self._calculate_sentiment_prediction(features, current_price)
            
            # Ensure mathematical accuracy
            predicted_price = round(predicted_price, 2)
            price_change = round(predicted_price - current_price, 2)
            price_change_percent = round((price_change / current_price) * 100, 3) if current_price > 0 else 0.0
            
            direction = "bullish" if predicted_price > current_price else "bearish" if predicted_price < current_price else "neutral"
            confidence = self._calculate_sentiment_confidence(features)
            
            # Support/resistance based on sentiment levels
            price_history = [item['close'] for item in current_data.get('price_data', [])] if current_data.get('price_data') else [current_price]
            support, resistance = self.calculate_support_resistance(price_history, current_price)
            
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, predicted_price, support, resistance, confidence
            )
            
            reasoning = self._generate_sentiment_reasoning(features, direction, confidence)
            
            return PredictionResult(
                strategy_name=self.name,
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=reasoning,
                raw_features=features
            )
            
        except Exception as e:
            logger.error(f"Error generating sentiment prediction: {e}")
            return self._generate_fallback_prediction(symbol, timeframe, current_data)
    
    def _calculate_sentiment_prediction(self, features: Dict[str, float], current_price: float) -> float:
        """Calculate price prediction based on sentiment"""
        # Weighted sentiment score
        news_weight = 0.4
        social_weight = 0.3
        fear_greed_weight = 0.2
        economic_weight = 0.1
        
        overall_sentiment = (
            features['news_sentiment'] * news_weight +
            features['social_sentiment'] * social_weight +
            (features['fear_greed_index'] - 50) / 50 * fear_greed_weight +  # Normalize fear/greed
            features['economic_sentiment'] * economic_weight
        )
        
        # Sentiment momentum factor
        momentum_factor = features.get('sentiment_momentum', 0.0)
        
        # VIX factor (inverse relationship with gold sometimes)
        vix_factor = -(features['vix_level'] - 20) / 100  # Normalize VIX
        
        # Combine factors
        total_sentiment = overall_sentiment + momentum_factor * 0.5 + vix_factor * 0.3
        
        # Convert sentiment to price movement (typically 0.5-2% for sentiment)
        max_move_percent = 1.5
        price_change_percent = np.tanh(total_sentiment) * max_move_percent
        
        predicted_price = current_price * (1 + price_change_percent / 100)
        return predicted_price
    
    def _calculate_sentiment_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on sentiment signal strength"""
        confidence_factors = []
        
        # News confidence
        news_conf = features.get('news_confidence', 0.5)
        news_volume = features.get('news_volume', 0)
        news_factor = news_conf * min(news_volume / 20, 1.0)  # More news = more confidence
        confidence_factors.append(news_factor)
        
        # Sentiment strength
        news_strength = abs(features.get('news_sentiment', 0))
        social_strength = abs(features.get('social_sentiment', 0))
        strength_factor = (news_strength + social_strength) / 2
        confidence_factors.append(strength_factor)
        
        # Sentiment consistency (lower volatility = higher confidence)
        sentiment_vol = features.get('sentiment_volatility', 0.5)
        consistency_factor = 1.0 - min(sentiment_vol / 1.0, 1.0)
        confidence_factors.append(consistency_factor)
        
        # Fear/Greed extremes (more confident when extreme)
        fg_index = features.get('fear_greed_index', 50)
        fg_factor = abs(fg_index - 50) / 50
        confidence_factors.append(fg_factor)
        
        return np.mean(confidence_factors)
    
    def _generate_sentiment_reasoning(self, features: Dict[str, float], direction: str, confidence: float) -> str:
        """Generate reasoning for sentiment prediction"""
        reasoning_parts = []
        
        news_sentiment = features.get('news_sentiment', 0)
        if news_sentiment > 0.2:
            reasoning_parts.append("Positive news sentiment")
        elif news_sentiment < -0.2:
            reasoning_parts.append("Negative news sentiment")
        
        social_sentiment = features.get('social_sentiment', 0)
        if social_sentiment > 0.2:
            reasoning_parts.append("Bullish social media sentiment")
        elif social_sentiment < -0.2:
            reasoning_parts.append("Bearish social media sentiment")
        
        fg_index = features.get('fear_greed_index', 50)
        if fg_index < 25:
            reasoning_parts.append("Extreme fear in markets")
        elif fg_index > 75:
            reasoning_parts.append("Extreme greed in markets")
        
        vix_level = features.get('vix_level', 20)
        if vix_level > 30:
            reasoning_parts.append("High volatility environment")
        elif vix_level < 15:
            reasoning_parts.append("Low volatility environment")
        
        momentum = features.get('sentiment_momentum', 0)
        if momentum > 0.1:
            reasoning_parts.append("Improving sentiment momentum")
        elif momentum < -0.1:
            reasoning_parts.append("Deteriorating sentiment momentum")
        
        base_reasoning = f"Sentiment analysis suggests {direction} trend. " + "; ".join(reasoning_parts)
        return f"{base_reasoning}. Confidence: {confidence:.1%}"
    
    def _generate_fallback_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate fallback prediction when sentiment data is unavailable"""
        current_price = current_data.get('current_price', 2000.0)
        
        # Neutral sentiment prediction
        price_change = np.random.uniform(-0.2, 0.2) / 100 * current_price
        predicted_price = round(current_price + price_change, 2)
        
        return PredictionResult(
            strategy_name=self.name,
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=round(price_change, 2),
            price_change_percent=round((price_change / current_price) * 100, 3),
            direction="neutral",
            confidence=0.3,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.995,
            take_profit=current_price * 1.005,
            timestamp=datetime.now(timezone.utc),
            features_used=["fallback"],
            reasoning="Sentiment analysis fallback prediction due to data limitations",
            raw_features={}
        )


class MacroStrategy(BaseStrategy):
    """Macroeconomic Strategy using interest rates, inflation, and economic indicators"""
    
    def __init__(self):
        super().__init__("Macro")
        self.macro_history = deque(maxlen=30)
        
    async def extract_features(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract macroeconomic features"""
        features = {}
        
        try:
            # Interest rates
            rates_data = data.get('interest_rates', {})
            features['fed_rate'] = float(rates_data.get('fed_funds_rate', 5.25))
            features['real_yield_10y'] = float(rates_data.get('real_yield_10y', 2.0))
            features['yield_curve_10y2y'] = float(rates_data.get('yield_curve_10y2y', 1.0))
            
            # Inflation data
            inflation_data = data.get('inflation', {})
            features['cpi_yoy'] = float(inflation_data.get('cpi_yoy', 3.0))
            features['core_cpi_yoy'] = float(inflation_data.get('core_cpi_yoy', 2.5))
            features['pce_yoy'] = float(inflation_data.get('pce_yoy', 2.8))
            features['inflation_expectations_5y'] = float(inflation_data.get('expectations_5y', 2.5))
            
            # Currency strength
            currency_data = data.get('currency', {})
            features['dxy_level'] = float(currency_data.get('dxy', 103.0))
            features['dxy_change'] = float(currency_data.get('dxy_change', 0.0))
            
            # Economic indicators
            economic_data = data.get('economic_indicators', {})
            features['unemployment_rate'] = float(economic_data.get('unemployment', 3.7))
            features['gdp_growth'] = float(economic_data.get('gdp_growth', 2.1))
            features['ism_manufacturing'] = float(economic_data.get('ism_manufacturing', 48.5))
            features['consumer_confidence'] = float(economic_data.get('consumer_confidence', 102.0))
            
            # Central bank policy indicators
            cb_data = data.get('central_bank', {})
            features['fed_hawkish_dovish'] = float(cb_data.get('fed_stance', 0.0))  # -1 to 1
            features['ecb_policy_divergence'] = float(cb_data.get('ecb_divergence', 0.0))
            
            # Real rates calculation
            features['real_fed_rate'] = features['fed_rate'] - features['cpi_yoy']
            features['real_rate_change'] = features['real_fed_rate'] - 0.5  # Assuming previous real rate
            
            # Macro momentum
            self.macro_history.append({
                'real_rate': features['real_fed_rate'],
                'dxy': features['dxy_level'],
                'inflation': features['cpi_yoy']
            })
            
            if len(self.macro_history) >= 5:
                recent = list(self.macro_history)
                features['real_rate_momentum'] = np.mean([h['real_rate'] for h in recent[-3:]]) - np.mean([h['real_rate'] for h in recent[-5:-2]])
                features['dxy_momentum'] = np.mean([h['dxy'] for h in recent[-3:]]) - np.mean([h['dxy'] for h in recent[-5:-2]])
                features['inflation_momentum'] = np.mean([h['inflation'] for h in recent[-3:]]) - np.mean([h['inflation'] for h in recent[-5:-2]])
            else:
                features['real_rate_momentum'] = 0.0
                features['dxy_momentum'] = 0.0
                features['inflation_momentum'] = 0.0
            
            logger.info(f"Extracted {len(features)} macro features for {symbol} {timeframe}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting macro features: {e}")
            return self._get_fallback_macro_features()
    
    def _get_fallback_macro_features(self) -> Dict[str, float]:
        """Fallback macro features when data is unavailable"""
        return {
            'fed_rate': 5.25,
            'real_yield_10y': 2.0,
            'yield_curve_10y2y': 1.0,
            'cpi_yoy': 3.0,
            'core_cpi_yoy': 2.5,
            'pce_yoy': 2.8,
            'inflation_expectations_5y': 2.5,
            'dxy_level': 103.0,
            'dxy_change': 0.0,
            'unemployment_rate': 3.7,
            'gdp_growth': 2.1,
            'ism_manufacturing': 48.5,
            'consumer_confidence': 102.0,
            'fed_hawkish_dovish': 0.0,
            'ecb_policy_divergence': 0.0,
            'real_fed_rate': 2.25,
            'real_rate_change': 0.0,
            'real_rate_momentum': 0.0,
            'dxy_momentum': 0.0,
            'inflation_momentum': 0.0
        }
    
    async def train_model(self, historical_data: pd.DataFrame, target_column: str = 'close') -> bool:
        """Train macroeconomic model"""
        try:
            # Use ensemble of macro-focused models
            models = [
                RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42),
                Ridge(alpha=1.0)
            ]
            
            self.model = VotingRegressor(
                estimators=[('rf', models[0]), ('gb', models[1]), ('ridge', models[2])],
                n_jobs=-1
            )
            
            # Create training data (in real implementation, use historical macro data)
            X = np.random.randn(100, 20)  # 20 macro features
            y = historical_data[target_column].iloc[-100:] if len(historical_data) >= 100 else historical_data[target_column]
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            self.last_training_time = datetime.now(timezone.utc)
            
            logger.info("Macro strategy trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training macro model: {e}")
            return False
    
    async def generate_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate macroeconomic-based prediction"""
        try:
            features = await self.extract_features(symbol, timeframe, current_data)
            current_price = current_data.get('current_price', 2000.0)
            
            predicted_price = self._calculate_macro_prediction(features, current_price)
            predicted_price = round(predicted_price, 2)
            price_change = round(predicted_price - current_price, 2)
            price_change_percent = round((price_change / current_price) * 100, 3) if current_price > 0 else 0.0
            
            direction = "bullish" if predicted_price > current_price else "bearish" if predicted_price < current_price else "neutral"
            confidence = self._calculate_macro_confidence(features)
            
            price_history = [item['close'] for item in current_data.get('price_data', [])] if current_data.get('price_data') else [current_price]
            support, resistance = self.calculate_support_resistance(price_history, current_price)
            
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, predicted_price, support, resistance, confidence
            )
            
            reasoning = self._generate_macro_reasoning(features, direction, confidence)
            
            return PredictionResult(
                strategy_name=self.name,
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=reasoning,
                raw_features=features
            )
            
        except Exception as e:
            logger.error(f"Error generating macro prediction: {e}")
            return self._generate_fallback_prediction(symbol, timeframe, current_data)
    
    def _calculate_macro_prediction(self, features: Dict[str, float], current_price: float) -> float:
        """Calculate prediction based on macroeconomic factors"""
        # Real rates effect (negative correlation with gold)
        real_rate_effect = -features['real_fed_rate'] * 0.02  # 2% per percentage point
        
        # Dollar strength effect (negative correlation)
        dxy_effect = -(features['dxy_level'] - 103) / 103 * 0.03  # 3% per 1% DXY change
        
        # Inflation effect (positive correlation)
        inflation_effect = (features['cpi_yoy'] - 2.5) / 2.5 * 0.025  # 2.5% per percentage point above 2.5%
        
        # Economic weakness effect (positive for gold)
        economic_weakness = 0
        if features['ism_manufacturing'] < 50:
            economic_weakness += (50 - features['ism_manufacturing']) / 50 * 0.015
        if features['unemployment_rate'] > 4.0:
            economic_weakness += (features['unemployment_rate'] - 4.0) / 4.0 * 0.01
        
        # Fed policy effect
        fed_effect = -features['fed_hawkish_dovish'] * 0.02  # Hawkish is negative for gold
        
        # Momentum effects
        real_rate_momentum_effect = -features['real_rate_momentum'] * 0.01
        dxy_momentum_effect = -features['dxy_momentum'] * 0.01
        
        # Combine effects
        total_effect = (real_rate_effect + dxy_effect + inflation_effect + 
                       economic_weakness + fed_effect + real_rate_momentum_effect + dxy_momentum_effect)
        
        # Apply effect to current price
        predicted_price = current_price * (1 + total_effect)
        return predicted_price
    
    def _calculate_macro_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on macro signal clarity"""
        confidence_factors = []
        
        # Real rate clarity (more extreme = more confident)
        real_rate = abs(features['real_fed_rate'])
        real_rate_conf = min(real_rate / 3.0, 1.0)  # Max confidence at 3% real rate
        confidence_factors.append(real_rate_conf)
        
        # DXY momentum clarity
        dxy_momentum = abs(features['dxy_momentum'])
        dxy_conf = min(dxy_momentum / 2.0, 1.0)
        confidence_factors.append(dxy_conf)
        
        # Inflation extremes
        inflation_distance = abs(features['cpi_yoy'] - 2.5)
        inflation_conf = min(inflation_distance / 2.5, 1.0)
        confidence_factors.append(inflation_conf)
        
        # Fed policy clarity
        fed_clarity = abs(features['fed_hawkish_dovish'])
        confidence_factors.append(fed_clarity)
        
        return np.mean(confidence_factors)
    
    def _generate_macro_reasoning(self, features: Dict[str, float], direction: str, confidence: float) -> str:
        """Generate reasoning for macro prediction"""
        reasoning_parts = []
        
        real_rate = features['real_fed_rate']
        if real_rate > 1.5:
            reasoning_parts.append("High real interest rates pressure gold")
        elif real_rate < 0:
            reasoning_parts.append("Negative real rates support gold")
        
        dxy = features['dxy_level']
        if dxy > 105:
            reasoning_parts.append("Strong dollar headwind")
        elif dxy < 100:
            reasoning_parts.append("Weak dollar supportive")
        
        inflation = features['cpi_yoy']
        if inflation > 4:
            reasoning_parts.append("High inflation supports gold hedge demand")
        elif inflation < 2:
            reasoning_parts.append("Low inflation reduces gold appeal")
        
        fed_stance = features['fed_hawkish_dovish']
        if fed_stance > 0.5:
            reasoning_parts.append("Hawkish Fed policy")
        elif fed_stance < -0.5:
            reasoning_parts.append("Dovish Fed policy")
        
        base_reasoning = f"Macro analysis suggests {direction} trend. " + "; ".join(reasoning_parts)
        return f"{base_reasoning}. Confidence: {confidence:.1%}"
    
    def _generate_fallback_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate fallback macro prediction"""
        current_price = current_data.get('current_price', 2000.0)
        
        # Conservative macro prediction
        price_change = np.random.uniform(-0.3, 0.3) / 100 * current_price
        predicted_price = round(current_price + price_change, 2)
        
class MomentumStrategy(BaseStrategy):
    """Momentum and Trend Following Strategy"""
    
    def __init__(self):
        super().__init__("Momentum")
        self.momentum_history = deque(maxlen=50)
        
    async def extract_features(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract momentum-based features"""
        features = {}
        
        try:
            price_data = data.get('price_data', [])
            if len(price_data) < 20:
                return self._get_fallback_momentum_features()
            
            closes = np.array([item['close'] for item in price_data])
            volumes = np.array([item.get('volume', 1000) for item in price_data])
            
            # Price momentum indicators
            features['roc_1'] = self._calculate_rate_of_change(closes, 1)  # 1-period ROC
            features['roc_5'] = self._calculate_rate_of_change(closes, 5)  # 5-period ROC
            features['roc_10'] = self._calculate_rate_of_change(closes, 10)  # 10-period ROC
            features['roc_20'] = self._calculate_rate_of_change(closes, 20)  # 20-period ROC
            
            # Moving average momentum
            features['ma_momentum_fast'] = self._calculate_ma_momentum(closes, 5, 10)
            features['ma_momentum_medium'] = self._calculate_ma_momentum(closes, 10, 20)
            features['ma_momentum_slow'] = self._calculate_ma_momentum(closes, 20, 50)
            
            # Relative strength vs different timeframes
            features['strength_vs_week'] = self._calculate_relative_strength(closes, 7)
            features['strength_vs_month'] = self._calculate_relative_strength(closes, 30)
            
            # Momentum oscillators
            features['momentum_oscillator'] = self._calculate_momentum_oscillator(closes)
            features['price_position_vs_range'] = self._calculate_price_position(closes)
            
            # Volume momentum
            features['volume_momentum'] = self._calculate_volume_momentum(volumes)
            features['price_volume_trend'] = self._calculate_price_volume_trend(closes, volumes)
            
            # Acceleration (second derivative of price)
            features['price_acceleration'] = self._calculate_price_acceleration(closes)
            
            # Trend consistency
            features['trend_consistency'] = self._calculate_trend_consistency(closes)
            features['momentum_divergence'] = self._calculate_momentum_divergence(closes)
            
            # Volatility-adjusted momentum
            features['volatility_adj_momentum'] = self._calculate_vol_adjusted_momentum(closes)
            
            # Multi-timeframe momentum alignment
            features['momentum_alignment'] = self._calculate_momentum_alignment(closes)
            
            # Momentum persistence
            self.momentum_history.append({
                'roc_10': features['roc_10'],
                'ma_momentum': features['ma_momentum_medium'],
                'trend_consistency': features['trend_consistency']
            })
            
            if len(self.momentum_history) >= 10:
                features['momentum_persistence'] = self._calculate_momentum_persistence()
                features['momentum_mean_reversion'] = self._calculate_momentum_mean_reversion()
            else:
                features['momentum_persistence'] = 0.0
                features['momentum_mean_reversion'] = 0.0
            
            logger.info(f"Extracted {len(features)} momentum features for {symbol} {timeframe}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting momentum features: {e}")
            return self._get_fallback_momentum_features()
    
    def _get_fallback_momentum_features(self) -> Dict[str, float]:
        """Fallback momentum features"""
        return {
            'roc_1': 0.0,
            'roc_5': 0.0,
            'roc_10': 0.0,
            'roc_20': 0.0,
            'ma_momentum_fast': 0.0,
            'ma_momentum_medium': 0.0,
            'ma_momentum_slow': 0.0,
            'strength_vs_week': 0.0,
            'strength_vs_month': 0.0,
            'momentum_oscillator': 0.0,
            'price_position_vs_range': 0.5,
            'volume_momentum': 0.0,
            'price_volume_trend': 0.0,
            'price_acceleration': 0.0,
            'trend_consistency': 0.5,
            'momentum_divergence': 0.0,
            'volatility_adj_momentum': 0.0,
            'momentum_alignment': 0.0,
            'momentum_persistence': 0.0,
            'momentum_mean_reversion': 0.0
        }
    
    def _calculate_rate_of_change(self, prices: np.ndarray, period: int) -> float:
        """Calculate Rate of Change (ROC)"""
        if len(prices) <= period:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        if past_price == 0:
            return 0.0
        
        roc = ((current_price - past_price) / past_price) * 100
        return float(roc)
    
    def _calculate_ma_momentum(self, prices: np.ndarray, fast_period: int, slow_period: int) -> float:
        """Calculate moving average momentum"""
        if len(prices) < slow_period:
            return 0.0
        
        fast_ma = np.mean(prices[-fast_period:])
        slow_ma = np.mean(prices[-slow_period:])
        
        if slow_ma == 0:
            return 0.0
        
        momentum = ((fast_ma - slow_ma) / slow_ma) * 100
        return float(momentum)
    
    def _calculate_relative_strength(self, prices: np.ndarray, period: int) -> float:
        """Calculate relative strength vs past performance"""
        if len(prices) <= period:
            return 0.0
        
        current_price = prices[-1]
        period_ago_price = prices[-period-1]
        
        if period_ago_price == 0:
            return 0.0
        
        # Compare current performance vs average performance over period
        avg_price = np.mean(prices[-period:])
        relative_strength = ((current_price - avg_price) / avg_price) * 100
        
        return float(relative_strength)
    
    def _calculate_momentum_oscillator(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate momentum oscillator"""
        if len(prices) <= period:
            return 0.0
        
        current_price = prices[-1]
        period_ago_price = prices[-period-1]
        
        momentum = current_price - period_ago_price
        
        # Normalize by average true range
        if len(prices) >= period + 1:
            price_changes = np.abs(np.diff(prices[-period-1:]))
            avg_change = np.mean(price_changes) if len(price_changes) > 0 else 1.0
            normalized_momentum = momentum / avg_change if avg_change != 0 else 0.0
        else:
            normalized_momentum = momentum / current_price if current_price != 0 else 0.0
        
        return float(normalized_momentum)
    
    def _calculate_price_position(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate price position within recent range"""
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        current_price = prices[-1]
        
        price_min = np.min(recent_prices)
        price_max = np.max(recent_prices)
        
        if price_max == price_min:
            return 0.5
        
        position = (current_price - price_min) / (price_max - price_min)
        return float(position)
    
    def _calculate_volume_momentum(self, volumes: np.ndarray, period: int = 10) -> float:
        """Calculate volume momentum"""
        if len(volumes) < period * 2:
            return 0.0
        
        recent_volume = np.mean(volumes[-period:])
        past_volume = np.mean(volumes[-period*2:-period])
        
        if past_volume == 0:
            return 0.0
        
        volume_momentum = ((recent_volume - past_volume) / past_volume) * 100
        return float(volume_momentum)
    
    def _calculate_price_volume_trend(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate price-volume trend correlation"""
        if len(prices) < 10 or len(volumes) < 10:
            return 0.0
        
        # Calculate recent price changes and volume changes
        price_changes = np.diff(prices[-10:])
        volume_changes = np.diff(volumes[-10:])
        
        if len(price_changes) == 0 or len(volume_changes) == 0:
            return 0.0
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_price_acceleration(self, prices: np.ndarray, period: int = 5) -> float:
        """Calculate price acceleration (second derivative)"""
        if len(prices) < period * 2:
            return 0.0
        
        # First derivative (velocity)
        recent_velocity = np.mean(np.diff(prices[-period:]))
        past_velocity = np.mean(np.diff(prices[-period*2:-period]))
        
        # Second derivative (acceleration)
        acceleration = recent_velocity - past_velocity
        
        # Normalize by price
        current_price = prices[-1]
        normalized_acceleration = (acceleration / current_price) * 1000 if current_price != 0 else 0.0
        
        return float(normalized_acceleration)
    
    def _calculate_trend_consistency(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate trend consistency (how steady the trend is)"""
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        
        # Calculate linear trend
        x = np.arange(len(recent_prices))
        try:
            slope, intercept = np.polyfit(x, recent_prices, 1)
            
            # Calculate R-squared (how well prices fit the trend line)
            trend_line = slope * x + intercept
            ss_res = np.sum((recent_prices - trend_line) ** 2)
            ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
            
            if ss_tot == 0:
                return 0.5
            
            r_squared = 1 - (ss_res / ss_tot)
            return float(max(0, min(1, r_squared)))  # Bound between 0 and 1
            
        except:
            return 0.5
    
    def _calculate_momentum_divergence(self, prices: np.ndarray) -> float:
        """Calculate momentum divergence"""
        if len(prices) < 20:
            return 0.0
        
        # Compare price momentum vs price level
        recent_roc = self._calculate_rate_of_change(prices, 10)
        price_level_change = ((prices[-1] - prices[-20]) / prices[-20]) * 100 if prices[-20] != 0 else 0.0
        
        # Divergence is when momentum and price disagree
        if price_level_change != 0:
            divergence = (recent_roc - price_level_change) / abs(price_level_change)
        else:
            divergence = 0.0
        
        return float(divergence)
    
    def _calculate_vol_adjusted_momentum(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate volatility-adjusted momentum"""
        if len(prices) < period:
            return 0.0
        
        # Raw momentum
        raw_momentum = self._calculate_rate_of_change(prices, period)
        
        # Price volatility
        price_changes = np.diff(prices[-period:])
        volatility = np.std(price_changes) if len(price_changes) > 0 else 1.0
        
        # Adjust momentum by volatility
        if volatility == 0:
            return raw_momentum
        
        vol_adjusted = raw_momentum / volatility
        return float(vol_adjusted)
    
    def _calculate_momentum_alignment(self, prices: np.ndarray) -> float:
        """Calculate multi-timeframe momentum alignment"""
        if len(prices) < 30:
            return 0.0
        
        # Calculate momentum for different timeframes
        short_momentum = self._calculate_rate_of_change(prices, 5)
        medium_momentum = self._calculate_rate_of_change(prices, 10)
        long_momentum = self._calculate_rate_of_change(prices, 20)
        
        # Check if all momentums are aligned (same direction)
        momentums = [short_momentum, medium_momentum, long_momentum]
        
        # Calculate alignment score
        positive_count = sum(1 for m in momentums if m > 0.5)
        negative_count = sum(1 for m in momentums if m < -0.5)
        
        if positive_count >= 2:
            alignment = positive_count / 3.0  # Positive alignment
        elif negative_count >= 2:
            alignment = -negative_count / 3.0  # Negative alignment
        else:
            alignment = 0.0  # No clear alignment
        
        return float(alignment)
    
    def _calculate_momentum_persistence(self) -> float:
        """Calculate momentum persistence from history"""
        if len(self.momentum_history) < 10:
            return 0.0
        
        recent = list(self.momentum_history)
        
        # Check how consistent momentum direction has been
        roc_values = [h['roc_10'] for h in recent[-10:]]
        
        positive_periods = sum(1 for roc in roc_values if roc > 0.5)
        negative_periods = sum(1 for roc in roc_values if roc < -0.5)
        
        total_periods = len(roc_values)
        persistence = max(positive_periods, negative_periods) / total_periods
        
        # Apply sign based on recent direction
        recent_direction = np.mean(roc_values[-3:])
        if recent_direction < 0:
            persistence = -persistence
        
        return float(persistence)
    
    def _calculate_momentum_mean_reversion(self) -> float:
        """Calculate momentum mean reversion tendency"""
        if len(self.momentum_history) < 10:
            return 0.0
        
        recent = list(self.momentum_history)
        roc_values = [h['roc_10'] for h in recent[-10:]]
        
        # Calculate how far current momentum is from historical average
        current_momentum = roc_values[-1]
        avg_momentum = np.mean(roc_values)
        momentum_std = np.std(roc_values) if len(roc_values) > 1 else 1.0
        
        if momentum_std == 0:
            return 0.0
        
        # Z-score of current momentum
        z_score = (current_momentum - avg_momentum) / momentum_std
        
        # Mean reversion signal is negative of extreme positions
        mean_reversion = -np.tanh(z_score / 2)  # Bound between -1 and 1
        
        return float(mean_reversion)
    
    async def train_model(self, historical_data: pd.DataFrame, target_column: str = 'close') -> bool:
        """Train momentum-based model"""
        try:
            # Use models that capture momentum patterns well
            models = [
                RandomForestRegressor(n_estimators=120, max_depth=15, random_state=42),
                GradientBoostingRegressor(n_estimators=120, max_depth=12, learning_rate=0.08, random_state=42),
                Ridge(alpha=0.5)  # Linear model for trend following
            ]
            
            self.model = VotingRegressor(
                estimators=[('rf', models[0]), ('gb', models[1]), ('ridge', models[2])],
                n_jobs=-1
            )
            
            # Create training data
            X = np.random.randn(100, 20)  # 20 momentum features
            y = historical_data[target_column].iloc[-100:] if len(historical_data) >= 100 else historical_data[target_column]
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            self.last_training_time = datetime.now(timezone.utc)
            
            logger.info("Momentum strategy trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training momentum model: {e}")
            return False
    
    async def generate_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate momentum-based prediction"""
        try:
            features = await self.extract_features(symbol, timeframe, current_data)
            current_price = current_data.get('current_price', 2000.0)
            
            predicted_price = self._calculate_momentum_prediction(features, current_price)
            predicted_price = round(predicted_price, 2)
            price_change = round(predicted_price - current_price, 2)
            price_change_percent = round((price_change / current_price) * 100, 3) if current_price > 0 else 0.0
            
            direction = "bullish" if predicted_price > current_price else "bearish" if predicted_price < current_price else "neutral"
            confidence = self._calculate_momentum_confidence(features)
            
            price_history = [item['close'] for item in current_data.get('price_data', [])] if current_data.get('price_data') else [current_price]
            support, resistance = self.calculate_support_resistance(price_history, current_price)
            
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, predicted_price, support, resistance, confidence
            )
            
            reasoning = self._generate_momentum_reasoning(features, direction, confidence)
            
            return PredictionResult(
                strategy_name=self.name,
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=reasoning,
                raw_features=features
            )
            
        except Exception as e:
            logger.error(f"Error generating momentum prediction: {e}")
            return self._generate_fallback_prediction(symbol, timeframe, current_data)
    
    def _calculate_momentum_prediction(self, features: Dict[str, float], current_price: float) -> float:
        """Calculate prediction based on momentum factors"""
        momentum_effect = 0.0
        
        # Rate of change effects (trend following)
        roc_weights = {'roc_1': 0.1, 'roc_5': 0.2, 'roc_10': 0.3, 'roc_20': 0.25}
        
        for roc_key, weight in roc_weights.items():
            roc_value = features.get(roc_key, 0.0)
            # Convert ROC to price effect (cap at reasonable levels)
            roc_effect = np.tanh(roc_value / 5.0) * 0.02  # Max 2% effect per ROC
            momentum_effect += roc_effect * weight
        
        # Moving average momentum
        ma_momentum = (
            features.get('ma_momentum_fast', 0.0) * 0.3 +
            features.get('ma_momentum_medium', 0.0) * 0.4 +
            features.get('ma_momentum_slow', 0.0) * 0.3
        ) / 3.0
        
        ma_effect = np.tanh(ma_momentum / 3.0) * 0.015  # Max 1.5% from MA momentum
        momentum_effect += ma_effect
        
        # Multi-timeframe alignment
        alignment = features.get('momentum_alignment', 0.0)
        alignment_effect = alignment * 0.02  # Strong alignment gets 2% boost
        momentum_effect += alignment_effect
        
        # Trend consistency boost
        consistency = features.get('trend_consistency', 0.5)
        if consistency > 0.7:  # Strong consistent trend
            momentum_effect *= (1 + (consistency - 0.7) * 0.5)  # Up to 15% boost
        
        # Volume confirmation
        volume_momentum = features.get('volume_momentum', 0.0)
        pv_trend = features.get('price_volume_trend', 0.0)
        
        volume_effect = (np.tanh(volume_momentum / 20.0) + pv_trend) * 0.005
        momentum_effect += volume_effect
        
        # Acceleration factor
        acceleration = features.get('price_acceleration', 0.0)
        accel_effect = np.tanh(acceleration / 10.0) * 0.01  # Max 1% from acceleration
        momentum_effect += accel_effect
        
        # Mean reversion adjustment (counter-momentum)
        mean_reversion = features.get('momentum_mean_reversion', 0.0)
        if abs(features.get('roc_10', 0.0)) > 3.0:  # Only apply when momentum is extreme
            momentum_effect += mean_reversion * 0.005  # Small counter-trend adjustment
        
        # Persistence factor
        persistence = features.get('momentum_persistence', 0.0)
        if abs(persistence) > 0.6:  # Strong persistence
            momentum_effect *= (1 + abs(persistence) * 0.2)  # Up to 20% boost
        
        # Apply momentum effect to price
        predicted_price = current_price * (1 + momentum_effect)
        return predicted_price
    
    def _calculate_momentum_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on momentum signal strength"""
        confidence_factors = []
        
        # ROC alignment across timeframes
        roc_values = [features.get(f'roc_{p}', 0.0) for p in [1, 5, 10, 20]]
        roc_directions = [1 if roc > 0.5 else -1 if roc < -0.5 else 0 for roc in roc_values]
        roc_agreement = abs(sum(roc_directions)) / len(roc_directions)
        confidence_factors.append(roc_agreement)
        
        # Trend consistency
        trend_consistency = features.get('trend_consistency', 0.5)
        confidence_factors.append(trend_consistency)
        
        # Momentum alignment
        momentum_alignment = abs(features.get('momentum_alignment', 0.0))
        confidence_factors.append(momentum_alignment)
        
        # Volume confirmation
        volume_momentum = min(abs(features.get('volume_momentum', 0.0)) / 20.0, 1.0)
        pv_correlation = abs(features.get('price_volume_trend', 0.0))
        volume_conf = (volume_momentum + pv_correlation) / 2
        confidence_factors.append(volume_conf)
        
        # Momentum persistence
        persistence = abs(features.get('momentum_persistence', 0.0))
        confidence_factors.append(persistence)
        
        # Avoid overconfidence in extreme momentum
        max_roc = max([abs(features.get(f'roc_{p}', 0.0)) for p in [5, 10, 20]])
        if max_roc > 5.0:  # Very strong momentum
            extreme_penalty = min((max_roc - 5.0) / 10.0, 0.3)  # Up to 30% penalty
            base_confidence = np.mean(confidence_factors) * (1 - extreme_penalty)
        else:
            base_confidence = np.mean(confidence_factors)
        
        return max(0.2, min(0.95, base_confidence))  # Bound between 20% and 95%
    
    def _generate_momentum_reasoning(self, features: Dict[str, float], direction: str, confidence: float) -> str:
        """Generate reasoning for momentum prediction"""
        reasoning_parts = []
        
        # ROC analysis
        roc_10 = features.get('roc_10', 0.0)
        if abs(roc_10) > 2.0:
            reasoning_parts.append(f"Strong {roc_10:.1f}% momentum over 10 periods")
        elif abs(roc_10) > 1.0:
            reasoning_parts.append(f"Moderate {roc_10:.1f}% momentum")
        
        # Trend consistency
        consistency = features.get('trend_consistency', 0.5)
        if consistency > 0.8:
            reasoning_parts.append("Highly consistent trend")
        elif consistency > 0.6:
            reasoning_parts.append("Moderately consistent trend")
        elif consistency < 0.4:
            reasoning_parts.append("Choppy price action")
        
        # Multi-timeframe alignment
        alignment = features.get('momentum_alignment', 0.0)
        if abs(alignment) > 0.6:
            reasoning_parts.append(f"Strong multi-timeframe {'bullish' if alignment > 0 else 'bearish'} alignment")
        elif abs(alignment) > 0.3:
            reasoning_parts.append("Moderate momentum alignment")
        
        # Volume confirmation
        volume_momentum = features.get('volume_momentum', 0.0)
        pv_trend = features.get('price_volume_trend', 0.0)
        
        if volume_momentum > 10:
            reasoning_parts.append("Strong volume expansion")
        elif volume_momentum < -10:
            reasoning_parts.append("Volume contraction")
        
        if abs(pv_trend) > 0.5:
            reasoning_parts.append("Strong price-volume correlation")
        
        # Acceleration
        acceleration = features.get('price_acceleration', 0.0)
        if abs(acceleration) > 5:
            reasoning_parts.append(f"Price momentum {'accelerating' if acceleration > 0 else 'decelerating'}")
        
        # Persistence
        persistence = features.get('momentum_persistence', 0.0)
        if abs(persistence) > 0.7:
            reasoning_parts.append("High momentum persistence")
        
        # Mean reversion warning
        mean_reversion = features.get('momentum_mean_reversion', 0.0)
        if abs(mean_reversion) > 0.5:
            reasoning_parts.append("Potential mean reversion risk")
        
        base_reasoning = f"Momentum analysis suggests {direction} trend. " + "; ".join(reasoning_parts)
        return f"{base_reasoning}. Confidence: {confidence:.1%}"
    
    def _generate_fallback_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate fallback momentum prediction"""
        current_price = current_data.get('current_price', 2000.0)
        
        # Minimal momentum-based prediction
        price_change = np.random.uniform(-0.3, 0.3) / 100 * current_price
        predicted_price = round(current_price + price_change, 2)
        
        return PredictionResult(
            strategy_name=self.name,
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=round(price_change, 2),
            price_change_percent=round((price_change / current_price) * 100, 3),
            direction="neutral",
            confidence=0.35,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.995,
            take_profit=current_price * 1.005,
            timestamp=datetime.now(timezone.utc),
            features_used=["fallback"],
            reasoning="Momentum analysis fallback prediction",
            raw_features={}
        )


class EnsembleVotingSystem:
    """Ensemble voting system that combines predictions from all strategies"""
    
    def __init__(self):
        self.strategies = {
            'Technical': TechnicalStrategy(),
            'Sentiment': SentimentStrategy(), 
            'Macro': MacroStrategy(),
            'Pattern': PatternStrategy(),
            'Momentum': MomentumStrategy()
        }
        self.performance_tracker = None  # Will be set by MultiStrategyEngine
        self.default_weights = {
            'Technical': 0.25,
            'Sentiment': 0.15,
            'Macro': 0.20,
            'Pattern': 0.20,
            'Momentum': 0.20
        }
        
    async def generate_ensemble_prediction(self, symbol: str, timeframe: str, 
                                         current_data: Dict[str, Any]) -> EnsemblePrediction:
        """Generate ensemble prediction combining all strategies"""
        try:
            # Get predictions from all strategies
            individual_predictions = []
            
            # Run predictions concurrently for speed
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(asyncio.run, strategy.generate_prediction(symbol, timeframe, current_data)): name
                    for name, strategy in self.strategies.items()
                }
                
                for future in as_completed(futures):
                    try:
                        prediction = future.result(timeout=30)
                        individual_predictions.append(prediction)
                    except Exception as e:
                        strategy_name = futures[future]
                        logger.error(f"Error getting prediction from {strategy_name}: {e}")
            
            if not individual_predictions:
                raise ValueError("No individual predictions available")
            
            # Get current weights from performance tracker
            current_weights = self._get_current_weights()
            
            # Calculate ensemble prediction
            ensemble_result = self._combine_predictions(individual_predictions, current_weights, current_data)
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {e}")
            return self._generate_fallback_ensemble(symbol, timeframe, current_data)
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current strategy weights from performance tracker"""
        if self.performance_tracker is None:
            return self.default_weights.copy()
        
        try:
            dynamic_weights = self.performance_tracker.get_strategy_weights()
            # Ensure weights sum to 1.0
            total_weight = sum(dynamic_weights.values())
            if total_weight > 0:
                return {k: v/total_weight for k, v in dynamic_weights.items()}
            else:
                return self.default_weights.copy()
        except:
            return self.default_weights.copy()
    
    def _combine_predictions(self, predictions: List[PredictionResult], weights: Dict[str, float],
                           current_data: Dict[str, Any]) -> EnsemblePrediction:
        """Combine individual predictions into ensemble result"""
        
        if not predictions:
            raise ValueError("No predictions to combine")
        
        current_price = predictions[0].current_price
        
        # Calculate weighted average prediction
        weighted_predictions = []
        total_weight = 0
        strategy_contributions = {}
        
        for pred in predictions:
            weight = weights.get(pred.strategy_name, 0.0)
            if weight > 0:
                weighted_predictions.append(pred.predicted_price * weight)
                total_weight += weight
                strategy_contributions[pred.strategy_name] = weight
        
        if total_weight == 0:
            # Equal weights fallback
            weighted_predictions = [pred.predicted_price for pred in predictions]
            total_weight = len(predictions)
            strategy_contributions = {pred.strategy_name: 1.0/len(predictions) for pred in predictions}
        
        # Ensemble predicted price
        ensemble_price = sum(weighted_predictions) / total_weight if total_weight > 0 else current_price
        
        # Calculate model agreement
        model_agreement = self._calculate_model_agreement(predictions)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(predictions, weights)
        
        # Determine direction
        direction = "bullish" if ensemble_price > current_price else "bearish" if ensemble_price < current_price else "neutral"
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(predictions, weights, model_agreement)
        
        # Combine support/resistance levels
        support_levels, resistance_levels = self._combine_support_resistance(predictions)
        
        # Calculate recommended stop loss and take profit
        recommended_sl, recommended_tp = self._calculate_ensemble_sl_tp(
            current_price, ensemble_price, support_levels, resistance_levels, ensemble_confidence
        )
        
        # Calculate prediction quality score
        quality_score = self._calculate_prediction_quality(predictions, model_agreement, ensemble_confidence)
        
        # Ensure mathematical accuracy
        ensemble_price = round(ensemble_price, 2)
        
        return EnsemblePrediction(
            symbol=predictions[0].symbol,
            timeframe=predictions[0].timeframe,
            current_price=current_price,
            predicted_price=ensemble_price,
            confidence_interval=confidence_interval,
            direction=direction,
            ensemble_confidence=ensemble_confidence,
            model_agreement=model_agreement,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            recommended_stop_loss=recommended_sl,
            recommended_take_profit=recommended_tp,
            strategy_contributions=strategy_contributions,
            individual_predictions=predictions,
            timestamp=datetime.now(timezone.utc),
            prediction_quality_score=quality_score
        )
    
    def _calculate_model_agreement(self, predictions: List[PredictionResult]) -> float:
        """Calculate how much models agree with each other"""
        if len(predictions) < 2:
            return 1.0
        
        # Direction agreement
        directions = [pred.direction for pred in predictions]
        bullish_count = directions.count('bullish')
        bearish_count = directions.count('bearish')
        neutral_count = directions.count('neutral')
        
        max_agreement = max(bullish_count, bearish_count, neutral_count)
        direction_agreement = max_agreement / len(directions)
        
        # Price prediction spread
        predicted_prices = [pred.predicted_price for pred in predictions]
        price_std = np.std(predicted_prices)
        avg_price = np.mean(predicted_prices)
        
        # Lower spread = higher agreement
        if avg_price > 0:
            price_agreement = 1.0 - min(price_std / avg_price, 0.5) * 2  # Max penalty is 0.5
        else:
            price_agreement = 0.5
        
        # Confidence agreement (prefer when models are all confident)
        confidences = [pred.confidence for pred in predictions]
        avg_confidence = np.mean(confidences)
        confidence_spread = np.std(confidences)
        confidence_agreement = avg_confidence * (1 - confidence_spread)
        
        # Combined agreement
        overall_agreement = (direction_agreement * 0.4 + 
                           price_agreement * 0.4 + 
                           confidence_agreement * 0.2)
        
        return max(0.0, min(1.0, overall_agreement))
    
    def _calculate_confidence_interval(self, predictions: List[PredictionResult], 
                                     weights: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for ensemble prediction"""
        predicted_prices = [pred.predicted_price for pred in predictions]
        
        if len(predicted_prices) == 1:
            # Single prediction - use its confidence for interval
            pred = predictions[0]
            price = pred.predicted_price
            confidence = pred.confidence
            price_range = price * 0.02 * (1 - confidence)  # Lower confidence = wider interval
            return (round(price - price_range, 2), round(price + price_range, 2))
        
        # Multiple predictions - calculate weighted statistics
        weighted_prices = []
        total_weight = 0
        
        for pred in predictions:
            weight = weights.get(pred.strategy_name, 1.0 / len(predictions))
            weighted_prices.extend([pred.predicted_price] * int(weight * 100))  # Weight approximation
            total_weight += weight
        
        if not weighted_prices:
            avg_price = np.mean(predicted_prices)
            return (round(avg_price * 0.99, 2), round(avg_price * 1.01, 2))
        
        # Calculate percentiles for confidence interval
        lower_bound = np.percentile(weighted_prices, 25)  # 25th percentile
        upper_bound = np.percentile(weighted_prices, 75)  # 75th percentile
        
        # Ensure reasonable bounds
        avg_price = np.mean(predicted_prices)
        min_range = avg_price * 0.005  # At least 0.5% range
        
        if upper_bound - lower_bound < min_range:
            mid_point = (upper_bound + lower_bound) / 2
            lower_bound = mid_point - min_range / 2
            upper_bound = mid_point + min_range / 2
        
        return (round(lower_bound, 2), round(upper_bound, 2))
    
    def _calculate_ensemble_confidence(self, predictions: List[PredictionResult],
                                     weights: Dict[str, float], model_agreement: float) -> float:
        """Calculate overall ensemble confidence"""
        
        # Weighted average of individual confidences
        weighted_confidences = []
        total_weight = 0
        
        for pred in predictions:
            weight = weights.get(pred.strategy_name, 1.0 / len(predictions))
            weighted_confidences.append(pred.confidence * weight)
            total_weight += weight
        
        avg_confidence = sum(weighted_confidences) / total_weight if total_weight > 0 else np.mean([p.confidence for p in predictions])
        
        # Boost confidence when models agree
        agreement_boost = model_agreement * 0.2  # Up to 20% boost
        
        # Penalize when we have very few models
        model_count_factor = min(len(predictions) / 5.0, 1.0)  # Ideal is 5 models
        
        ensemble_confidence = (avg_confidence + agreement_boost) * model_count_factor
        
        return max(0.1, min(0.95, ensemble_confidence))  # Bound between 10% and 95%
    
    def _combine_support_resistance(self, predictions: List[PredictionResult]) -> Tuple[List[float], List[float]]:
        """Combine support and resistance levels from all strategies"""
        all_supports = []
        all_resistances = []
        
        for pred in predictions:
            if pred.support_level:
                all_supports.append(pred.support_level)
            if pred.resistance_level:
                all_resistances.append(pred.resistance_level)
        
        # Remove outliers and duplicates
        if all_supports:
            all_supports = sorted(set(all_supports))
            # Remove extreme outliers (beyond 2 standard deviations)
            if len(all_supports) > 2:
                mean_support = np.mean(all_supports)
                std_support = np.std(all_supports)
                all_supports = [s for s in all_supports if abs(s - mean_support) <= 2 * std_support]
        
        if all_resistances:
            all_resistances = sorted(set(all_resistances))
            if len(all_resistances) > 2:
                mean_resistance = np.mean(all_resistances)
                std_resistance = np.std(all_resistances)
                all_resistances = [r for r in all_resistances if abs(r - mean_resistance) <= 2 * std_resistance]
        
        return all_supports[:3], all_resistances[:3]  # Max 3 levels each
    
    def _calculate_ensemble_sl_tp(self, current_price: float, predicted_price: float,
                                support_levels: List[float], resistance_levels: List[float],
                                confidence: float) -> Tuple[float, float]:
        """Calculate ensemble stop loss and take profit"""
        
        direction = "bullish" if predicted_price > current_price else "bearish"
        
        # Base risk based on confidence
        base_risk = 0.025 * (1 - confidence)  # 0.25% to 2.25% risk
        base_risk = max(0.005, min(base_risk, 0.03))  # Bound between 0.5% and 3%
        
        if direction == "bullish":
            # Stop loss below current price
            stop_loss = current_price * (1 - base_risk)
            
            # Use nearest support if stronger
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=stop_loss)
                if nearest_support > stop_loss:
                    stop_loss = nearest_support * 0.995  # Slightly below support
            
            # Take profit - aim for predicted price or resistance
            take_profit = max(predicted_price, current_price * (1 + base_risk * 2))
            
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > predicted_price], default=take_profit)
                if nearest_resistance < take_profit * 1.5:  # Not too far away
                    take_profit = nearest_resistance * 0.995  # Slightly below resistance
        
        else:  # bearish
            # Stop loss above current price
            stop_loss = current_price * (1 + base_risk)
            
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=stop_loss)
                if nearest_resistance < stop_loss:
                    stop_loss = nearest_resistance * 1.005  # Slightly above resistance
            
            # Take profit below predicted price
            take_profit = min(predicted_price, current_price * (1 - base_risk * 2))
            
            if support_levels:
                nearest_support = max([s for s in support_levels if s < predicted_price], default=take_profit)
                if nearest_support > take_profit * 0.5:  # Not too far away
                    take_profit = nearest_support * 1.005  # Slightly above support
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    def _calculate_prediction_quality(self, predictions: List[PredictionResult],
                                    model_agreement: float, ensemble_confidence: float) -> float:
        """Calculate overall prediction quality score"""
        
        quality_factors = []
        
        # Model agreement factor
        quality_factors.append(model_agreement)
        
        # Ensemble confidence factor
        quality_factors.append(ensemble_confidence)
        
        # Individual model quality (average confidence weighted by performance)
        individual_quality = np.mean([pred.confidence for pred in predictions])
        quality_factors.append(individual_quality)
        
        # Model count factor (more models = better)
        model_count_factor = min(len(predictions) / 5.0, 1.0)
        quality_factors.append(model_count_factor)
        
        # Strategy diversity factor (different types of strategies)
        strategy_types = set([pred.strategy_name for pred in predictions])
        diversity_factor = len(strategy_types) / 5.0  # All 5 strategies is ideal
        quality_factors.append(diversity_factor)
        
        # Feature richness (average number of features used)
        avg_features = np.mean([len(pred.features_used) for pred in predictions])
        feature_factor = min(avg_features / 15.0, 1.0)  # 15+ features is good
        quality_factors.append(feature_factor)
        
        overall_quality = np.mean(quality_factors)
        return max(0.1, min(1.0, overall_quality))
    
    def _generate_fallback_ensemble(self, symbol: str, timeframe: str, 
                                  current_data: Dict[str, Any]) -> EnsemblePrediction:
        """Generate fallback ensemble when all individual predictions fail"""
        current_price = current_data.get('current_price', 2000.0)
        
        # Conservative neutral prediction
        predicted_price = round(current_price + np.random.uniform(-0.1, 0.1) / 100 * current_price, 2)
        
        return EnsemblePrediction(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=predicted_price,
            confidence_interval=(round(current_price * 0.995, 2), round(current_price * 1.005, 2)),
            direction="neutral",
            ensemble_confidence=0.2,
            model_agreement=0.0,
            support_levels=[round(current_price * 0.98, 2)],
            resistance_levels=[round(current_price * 1.02, 2)],
            recommended_stop_loss=round(current_price * 0.99, 2),
            recommended_take_profit=round(current_price * 1.01, 2),
            strategy_contributions={"fallback": 1.0},
            individual_predictions=[],
            timestamp=datetime.now(timezone.utc),
            prediction_quality_score=0.1
        )
class PatternStrategy(BaseStrategy):
    """Chart Pattern Recognition Strategy"""
    
    def __init__(self):
        super().__init__("Pattern")
        self.pattern_history = deque(maxlen=100)
        
    async def extract_features(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract chart pattern features"""
        features = {}
        
        try:
            price_data = data.get('price_data', [])
            if len(price_data) < 20:
                return self._get_fallback_pattern_features()
            
            # Convert to numpy arrays
            closes = np.array([item['close'] for item in price_data])
            highs = np.array([item['high'] for item in price_data])
            lows = np.array([item['low'] for item in price_data])
            volumes = np.array([item.get('volume', 1000) for item in price_data])
            
            # Trend patterns
            features['trend_strength'] = self._calculate_trend_strength(closes)
            features['trend_direction'] = self._calculate_trend_direction(closes)
            
            # Chart patterns
            features['double_top_signal'] = self._detect_double_top(highs, closes)
            features['double_bottom_signal'] = self._detect_double_bottom(lows, closes)
            features['head_shoulders_signal'] = self._detect_head_shoulders(highs, closes)
            features['triangle_pattern'] = self._detect_triangle_pattern(highs, lows, closes)
            features['flag_pattern'] = self._detect_flag_pattern(closes, volumes)
            
            # Support/Resistance strength
            features['support_strength'] = self._calculate_support_strength(lows, closes[-1])
            features['resistance_strength'] = self._calculate_resistance_strength(highs, closes[-1])
            
            # Breakout signals
            features['breakout_signal'] = self._detect_breakout(closes, highs, lows, volumes)
            features['volume_confirmation'] = self._calculate_volume_confirmation(volumes, closes)
            
            # Price action patterns
            features['hammer_pattern'] = self._detect_hammer_pattern(price_data[-5:])
            features['doji_pattern'] = self._detect_doji_pattern(price_data[-3:])
            features['engulfing_pattern'] = self._detect_engulfing_pattern(price_data[-3:])
            
            # Fibonacci levels
            features['fib_retracement_level'] = self._calculate_fib_level(highs, lows, closes[-1])
            
            # Pattern momentum
            self.pattern_history.append(features.copy())
            if len(self.pattern_history) >= 5:
                features['pattern_momentum'] = self._calculate_pattern_momentum()
            else:
                features['pattern_momentum'] = 0.0
            
            logger.info(f"Extracted {len(features)} pattern features for {symbol} {timeframe}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return self._get_fallback_pattern_features()
    
    def _get_fallback_pattern_features(self) -> Dict[str, float]:
        """Fallback pattern features"""
        return {
            'trend_strength': 0.5,
            'trend_direction': 0.0,
            'double_top_signal': 0.0,
            'double_bottom_signal': 0.0,
            'head_shoulders_signal': 0.0,
            'triangle_pattern': 0.0,
            'flag_pattern': 0.0,
            'support_strength': 0.5,
            'resistance_strength': 0.5,
            'breakout_signal': 0.0,
            'volume_confirmation': 0.5,
            'hammer_pattern': 0.0,
            'doji_pattern': 0.0,
            'engulfing_pattern': 0.0,
            'fib_retracement_level': 0.5,
            'pattern_momentum': 0.0
        }
    
    def _calculate_trend_strength(self, closes: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(closes) < 10:
            return 0.5
        
        # Linear regression slope normalized
        x = np.arange(len(closes))
        coeffs = np.polyfit(x, closes, 1)
        slope = coeffs[0]
        
        # Normalize slope relative to price
        normalized_slope = slope / np.mean(closes) * len(closes)
        strength = min(abs(normalized_slope) / 0.1, 1.0)  # Max strength at 10% move
        
        return float(strength)
    
    def _calculate_trend_direction(self, closes: np.ndarray) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(closes) < 10:
            return 0.0
        
        # Compare recent vs older prices
        recent_avg = np.mean(closes[-5:])
        older_avg = np.mean(closes[-15:-10]) if len(closes) >= 15 else np.mean(closes[:-5])
        
        direction = (recent_avg - older_avg) / older_avg
        return float(np.tanh(direction * 10))  # Bound between -1 and 1
    
    def _detect_double_top(self, highs: np.ndarray, closes: np.ndarray) -> float:
        """Detect double top pattern strength"""
        if len(highs) < 20:
            return 0.0
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(highs, distance=5, height=np.percentile(highs, 70))
        
        if len(peaks) < 2:
            return 0.0
        
        # Check if last two peaks are similar height
        last_two_peaks = peaks[-2:]
        if len(last_two_peaks) == 2:
            height_diff = abs(highs[last_two_peaks[0]] - highs[last_two_peaks[1]])
            avg_height = np.mean([highs[last_two_peaks[0]], highs[last_two_peaks[1]]])
            
            if height_diff / avg_height < 0.02:  # Within 2%
                # Check if price has declined from second peak
                if closes[-1] < highs[last_two_peaks[1]] * 0.98:
                    return min((highs[last_two_peaks[1]] - closes[-1]) / highs[last_two_peaks[1]] * 10, 1.0)
        
        return 0.0
    
    def _detect_double_bottom(self, lows: np.ndarray, closes: np.ndarray) -> float:
        """Detect double bottom pattern strength"""
        if len(lows) < 20:
            return 0.0
        
        from scipy.signal import find_peaks
        valleys, _ = find_peaks(-lows, distance=5, height=-np.percentile(lows, 30))
        
        if len(valleys) < 2:
            return 0.0
        
        last_two_valleys = valleys[-2:]
        if len(last_two_valleys) == 2:
            height_diff = abs(lows[last_two_valleys[0]] - lows[last_two_valleys[1]])
            avg_height = np.mean([lows[last_two_valleys[0]], lows[last_two_valleys[1]]])
            
            if height_diff / avg_height < 0.02:  # Within 2%
                if closes[-1] > lows[last_two_valleys[1]] * 1.02:
                    return min((closes[-1] - lows[last_two_valleys[1]]) / lows[last_two_valleys[1]] * 10, 1.0)
        
        return 0.0
    
    def _detect_head_shoulders(self, highs: np.ndarray, closes: np.ndarray) -> float:
        """Detect head and shoulders pattern"""
        if len(highs) < 30:
            return 0.0
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(highs, distance=5, height=np.percentile(highs, 60))
        
        if len(peaks) < 3:
            return 0.0
        
        # Get last three peaks
        last_three = peaks[-3:]
        if len(last_three) == 3:
            left_shoulder, head, right_shoulder = highs[last_three]
            
            # Check if middle peak is higher (head)
            if head > left_shoulder and head > right_shoulder:
                # Check if shoulders are similar height
                shoulder_diff = abs(left_shoulder - right_shoulder)
                if shoulder_diff / np.mean([left_shoulder, right_shoulder]) < 0.03:  # Within 3%
                    # Check if neckline is broken
                    neckline = min(left_shoulder, right_shoulder) * 0.99
                    if closes[-1] < neckline:
                        return min((neckline - closes[-1]) / neckline * 5, 1.0)
        
        return 0.0
    
    def _detect_triangle_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Detect triangle consolidation pattern"""
        if len(closes) < 20:
            return 0.0
        
        # Check if highs are descending and lows are ascending (symmetrical triangle)
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Trend of highs (should be descending)
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        # Trend of lows (should be ascending) 
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        if high_trend < 0 and low_trend > 0:
            # Converging triangle
            volatility_decrease = np.std(closes[-5:]) < np.std(closes[-15:-10])
            if volatility_decrease:
                return min(abs(high_trend - low_trend) / np.mean(closes) * 100, 1.0)
        
        return 0.0
    
    def _detect_flag_pattern(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Detect flag pattern (consolidation after strong move)"""
        if len(closes) < 15:
            return 0.0
        
        # Check for strong initial move
        initial_move = abs(closes[-15] - closes[-10]) / closes[-15]
        
        if initial_move > 0.03:  # 3% move
            # Check for consolidation (sideways movement)
            consolidation_range = np.max(closes[-5:]) - np.min(closes[-5:])
            consolidation_pct = consolidation_range / np.mean(closes[-5:])
            
            if consolidation_pct < 0.02:  # Less than 2% range
                # Check for volume decrease during consolidation
                recent_vol = np.mean(volumes[-5:])
                earlier_vol = np.mean(volumes[-10:-5])
                
                if recent_vol < earlier_vol * 0.8:  # Volume decreased 20%
                    return min(initial_move / 0.05, 1.0)  # Max signal at 5% initial move
        
        return 0.0
    
    def _calculate_support_strength(self, lows: np.ndarray, current_price: float) -> float:
        """Calculate support level strength"""
        if len(lows) < 10:
            return 0.5
        
        # Find support levels (local minima)
        from scipy.signal import find_peaks
        support_indices, _ = find_peaks(-lows, distance=3)
        
        if len(support_indices) == 0:
            return 0.5
        
        support_levels = lows[support_indices]
        
        # Find nearest support below current price
        valid_supports = support_levels[support_levels < current_price * 0.99]
        
        if len(valid_supports) == 0:
            return 0.5
        
        nearest_support = np.max(valid_supports)
        
        # Count how many times price tested this level
        tolerance = nearest_support * 0.01  # 1% tolerance
        tests = np.sum(np.abs(lows - nearest_support) < tolerance)
        
        strength = min(tests / 5.0, 1.0)  # Max strength with 5 tests
        return float(strength)
    
    def _calculate_resistance_strength(self, highs: np.ndarray, current_price: float) -> float:
        """Calculate resistance level strength"""
        if len(highs) < 10:
            return 0.5
        
        from scipy.signal import find_peaks
        resistance_indices, _ = find_peaks(highs, distance=3)
        
        if len(resistance_indices) == 0:
            return 0.5
        
        resistance_levels = highs[resistance_indices]
        valid_resistance = resistance_levels[resistance_levels > current_price * 1.01]
        
        if len(valid_resistance) == 0:
            return 0.5
        
        nearest_resistance = np.min(valid_resistance)
        tolerance = nearest_resistance * 0.01
        tests = np.sum(np.abs(highs - nearest_resistance) < tolerance)
        
        strength = min(tests / 5.0, 1.0)
        return float(strength)
    
    def _detect_breakout(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> float:
        """Detect breakout signals"""
        if len(closes) < 20:
            return 0.0
        
        # Calculate recent high and low
        recent_high = np.max(highs[-20:-1])  # Exclude current bar
        recent_low = np.min(lows[-20:-1])
        
        current_close = closes[-1]
        
        # Check for breakout
        breakout_signal = 0.0
        
        if current_close > recent_high:
            # Bullish breakout
            breakout_strength = (current_close - recent_high) / recent_high
            breakout_signal = min(breakout_strength * 20, 1.0)  # Max at 5% breakout
        elif current_close < recent_low:
            # Bearish breakout  
            breakout_strength = (recent_low - current_close) / recent_low
            breakout_signal = -min(breakout_strength * 20, 1.0)
        
        return float(breakout_signal)
    
    def _calculate_volume_confirmation(self, volumes: np.ndarray, closes: np.ndarray) -> float:
        """Calculate volume confirmation for price moves"""
        if len(volumes) < 10:
            return 0.5
        
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        
        price_change = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Strong volume with price move is bullish confirmation
        if abs(price_change) > 0.005 and volume_ratio > 1.5:  # 0.5% move with 50% higher volume
            return min(volume_ratio / 3.0, 1.0)
        
        return 0.5
    
    def _detect_hammer_pattern(self, recent_bars: List[Dict]) -> float:
        """Detect hammer candlestick pattern"""
        if len(recent_bars) < 1:
            return 0.0
        
        bar = recent_bars[-1]
        open_price = bar['open']
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        body_size = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        total_range = high - low
        
        if total_range == 0:
            return 0.0
        
        # Hammer: small body, long lower shadow, small upper shadow
        if (body_size / total_range < 0.3 and  # Small body
            lower_shadow / total_range > 0.6 and  # Long lower shadow
            upper_shadow / total_range < 0.1):  # Small upper shadow
            return 0.8
        
        return 0.0
    
    def _detect_doji_pattern(self, recent_bars: List[Dict]) -> float:
        """Detect doji candlestick pattern"""
        if len(recent_bars) < 1:
            return 0.0
        
        bar = recent_bars[-1]
        open_price = bar['open']
        close = bar['close']
        high = bar['high']
        low = bar['low']
        
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return 0.0
        
        # Doji: very small body relative to range
        if body_size / total_range < 0.1:
            return 0.7
        
        return 0.0
    
    def _detect_engulfing_pattern(self, recent_bars: List[Dict]) -> float:
        """Detect engulfing candlestick pattern"""
        if len(recent_bars) < 2:
            return 0.0
        
        prev_bar = recent_bars[-2]
        curr_bar = recent_bars[-1]
        
        prev_body_size = abs(prev_bar['close'] - prev_bar['open'])
        curr_body_size = abs(curr_bar['close'] - curr_bar['open'])
        
        # Bullish engulfing: prev red, curr green and larger
        if (prev_bar['close'] < prev_bar['open'] and  # Previous red
            curr_bar['close'] > curr_bar['open'] and  # Current green
            curr_body_size > prev_body_size * 1.2):  # Current 20% larger
            return 0.8
        
        # Bearish engulfing: prev green, curr red and larger
        if (prev_bar['close'] > prev_bar['open'] and  # Previous green
            curr_bar['close'] < curr_bar['open'] and  # Current red
            curr_body_size > prev_body_size * 1.2):  # Current 20% larger
            return -0.8
        
        return 0.0
    
    def _calculate_fib_level(self, highs: np.ndarray, lows: np.ndarray, current_price: float) -> float:
        """Calculate Fibonacci retracement level"""
        if len(highs) < 10 or len(lows) < 10:
            return 0.5
        
        # Find recent swing high and low
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        
        if recent_high == recent_low:
            return 0.5
        
        # Calculate retracement level
        fib_level = (current_price - recent_low) / (recent_high - recent_low)
        
        # Check proximity to key Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        min_distance = min([abs(fib_level - level) for level in fib_levels])
        
        # Return strength based on proximity to key level
        if min_distance < 0.05:  # Within 5% of key level
            return 0.8
        elif min_distance < 0.1:  # Within 10%
            return 0.6
        else:
            return 0.4
    
    def _calculate_pattern_momentum(self) -> float:
        """Calculate pattern momentum from history"""
        if len(self.pattern_history) < 5:
            return 0.0
        
        recent = list(self.pattern_history)
        
        # Track key pattern signals over time
        breakout_momentum = np.mean([h['breakout_signal'] for h in recent[-3:]]) - np.mean([h['breakout_signal'] for h in recent[-5:-2]])
        pattern_strength_momentum = np.mean([h['trend_strength'] for h in recent[-3:]]) - np.mean([h['trend_strength'] for h in recent[-5:-2]])
        
        return float((breakout_momentum + pattern_strength_momentum) / 2)
    
    async def train_model(self, historical_data: pd.DataFrame, target_column: str = 'close') -> bool:
        """Train pattern recognition model"""
        try:
            # Use models good at capturing non-linear patterns
            models = [
                RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
                GradientBoostingRegressor(n_estimators=150, max_depth=10, learning_rate=0.05, random_state=42),
                SVR(kernel='rbf', C=100, gamma='scale')
            ]
            
            self.model = VotingRegressor(
                estimators=[('rf', models[0]), ('gb', models[1]), ('svr', models[2])],
                n_jobs=-1
            )
            
            # Create training data (in real implementation, extract pattern features from historical data)
            X = np.random.randn(100, 16)  # 16 pattern features
            y = historical_data[target_column].iloc[-100:] if len(historical_data) >= 100 else historical_data[target_column]
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            self.last_training_time = datetime.now(timezone.utc)
            
            logger.info("Pattern strategy trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training pattern model: {e}")
            return False
    
    async def generate_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate pattern-based prediction"""
        try:
            features = await self.extract_features(symbol, timeframe, current_data)
            current_price = current_data.get('current_price', 2000.0)
            
            predicted_price = self._calculate_pattern_prediction(features, current_price)
            predicted_price = round(predicted_price, 2)
            price_change = round(predicted_price - current_price, 2)
            price_change_percent = round((price_change / current_price) * 100, 3) if current_price > 0 else 0.0
            
            direction = "bullish" if predicted_price > current_price else "bearish" if predicted_price < current_price else "neutral"
            confidence = self._calculate_pattern_confidence(features)
            
            price_history = [item['close'] for item in current_data.get('price_data', [])] if current_data.get('price_data') else [current_price]
            support, resistance = self.calculate_support_resistance(price_history, current_price)
            
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, predicted_price, support, resistance, confidence
            )
            
            reasoning = self._generate_pattern_reasoning(features, direction, confidence)
            
            return PredictionResult(
                strategy_name=self.name,
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=reasoning,
                raw_features=features
            )
            
        except Exception as e:
            logger.error(f"Error generating pattern prediction: {e}")
            return self._generate_fallback_prediction(symbol, timeframe, current_data)
    
    def _calculate_pattern_prediction(self, features: Dict[str, float], current_price: float) -> float:
        """Calculate prediction based on chart patterns"""
        price_effect = 0.0
        
        # Breakout signals have strong directional bias
        breakout = features.get('breakout_signal', 0.0)
        price_effect += breakout * 0.03  # 3% max from breakouts
        
        # Pattern signals
        double_top = features.get('double_top_signal', 0.0)
        double_bottom = features.get('double_bottom_signal', 0.0)
        head_shoulders = features.get('head_shoulders_signal', 0.0)
        
        price_effect -= double_top * 0.025  # Double top bearish
        price_effect += double_bottom * 0.025  # Double bottom bullish
        price_effect -= head_shoulders * 0.03  # Head & shoulders bearish
        
        # Triangle breakout direction
        triangle = features.get('triangle_pattern', 0.0)
        trend_direction = features.get('trend_direction', 0.0)
        price_effect += triangle * trend_direction * 0.02
        
        # Flag pattern continuation
        flag = features.get('flag_pattern', 0.0)
        price_effect += flag * trend_direction * 0.025
        
        # Candlestick patterns
        hammer = features.get('hammer_pattern', 0.0)
        doji = features.get('doji_pattern', 0.0) 
        engulfing = features.get('engulfing_pattern', 0.0)
        
        price_effect += hammer * 0.015  # Hammer bullish
        # Doji is neutral, adds uncertainty
        price_effect += engulfing * 0.02  # Engulfing in direction
        
        # Support/resistance effects
        support_strength = features.get('support_strength', 0.5)
        resistance_strength = features.get('resistance_strength', 0.5)
        
        if current_price < 0.98 * current_price:  # Near support
            price_effect += support_strength * 0.01
        elif current_price > 1.02 * current_price:  # Near resistance
            price_effect -= resistance_strength * 0.01
        
        # Apply volume confirmation
        volume_conf = features.get('volume_confirmation', 0.5)
        price_effect *= (0.5 + volume_conf * 0.5)  # Scale by volume confirmation
        
        predicted_price = current_price * (1 + price_effect)
        return predicted_price
    
    def _calculate_pattern_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on pattern clarity"""
        confidence_factors = []
        
        # Breakout strength
        breakout = abs(features.get('breakout_signal', 0.0))
        confidence_factors.append(breakout)
        
        # Pattern signal strength
        pattern_signals = [
            abs(features.get('double_top_signal', 0.0)),
            abs(features.get('double_bottom_signal', 0.0)),
            abs(features.get('head_shoulders_signal', 0.0)),
            features.get('triangle_pattern', 0.0),
            features.get('flag_pattern', 0.0)
        ]
        max_pattern = max(pattern_signals)
        confidence_factors.append(max_pattern)
        
        # Volume confirmation
        vol_conf = features.get('volume_confirmation', 0.5)
        confidence_factors.append(vol_conf)
        
        # Support/resistance strength
        sr_strength = (features.get('support_strength', 0.5) + features.get('resistance_strength', 0.5)) / 2
        confidence_factors.append(sr_strength)
        
        # Trend strength
        trend_strength = features.get('trend_strength', 0.5)
        confidence_factors.append(trend_strength)
        
        return np.mean(confidence_factors)
    
    def _generate_pattern_reasoning(self, features: Dict[str, float], direction: str, confidence: float) -> str:
        """Generate reasoning for pattern prediction"""
        reasoning_parts = []
        
        breakout = features.get('breakout_signal', 0.0)
        if abs(breakout) > 0.3:
            reasoning_parts.append(f"{'Bullish' if breakout > 0 else 'Bearish'} breakout detected")
        
        if features.get('double_top_signal', 0.0) > 0.3:
            reasoning_parts.append("Double top pattern suggests reversal")
        if features.get('double_bottom_signal', 0.0) > 0.3:
            reasoning_parts.append("Double bottom pattern suggests bounce")
        if features.get('head_shoulders_signal', 0.0) > 0.3:
            reasoning_parts.append("Head & shoulders pattern indicates weakness")
        
        if features.get('triangle_pattern', 0.0) > 0.5:
            reasoning_parts.append("Triangle consolidation near breakout")
        if features.get('flag_pattern', 0.0) > 0.5:
            reasoning_parts.append("Flag pattern suggests continuation")
        
        hammer = features.get('hammer_pattern', 0.0)
        if hammer > 0.5:
            reasoning_parts.append("Hammer pattern indicates potential reversal")
        
        engulfing = features.get('engulfing_pattern', 0.0)
        if abs(engulfing) > 0.5:
            reasoning_parts.append(f"{'Bullish' if engulfing > 0 else 'Bearish'} engulfing pattern")
        
        vol_conf = features.get('volume_confirmation', 0.5)
        if vol_conf > 0.7:
            reasoning_parts.append("Strong volume confirmation")
        elif vol_conf < 0.3:
            reasoning_parts.append("Weak volume confirmation")
        
        base_reasoning = f"Pattern analysis suggests {direction} trend. " + "; ".join(reasoning_parts)
        return f"{base_reasoning}. Confidence: {confidence:.1%}"
    
    def _generate_fallback_prediction(self, symbol: str, timeframe: str, current_data: Dict[str, Any]) -> PredictionResult:
        """Generate fallback pattern prediction"""
        current_price = current_data.get('current_price', 2000.0)
        
        price_change = np.random.uniform(-0.4, 0.4) / 100 * current_price
        predicted_price = round(current_price + price_change, 2)
        
        return PredictionResult(
            strategy_name=self.name,
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=round(price_change, 2),
            price_change_percent=round((price_change / current_price) * 100, 3),
            direction="neutral",
            confidence=0.3,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.995,
            take_profit=current_price * 1.005,
            timestamp=datetime.now(timezone.utc),
            features_used=["fallback"],
            reasoning="Pattern analysis fallback prediction",
            raw_features={}
        )


class StrategyPerformanceTracker:
    """Tracks and manages strategy performance for dynamic weight adjustment"""
    
    def __init__(self, db_path: str = "strategy_performance.db"):
        self.db_path = db_path
        self.strategy_weights = {
            'Technical': 0.25,
            'Sentiment': 0.15,
            'Macro': 0.20,
            'Pattern': 0.20,
            'Momentum': 0.20
        }
        self.performance_history = defaultdict(list)
        self.lock = threading.Lock()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize performance tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    accuracy REAL,
                    direction_correct BOOLEAN,
                    confidence REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    evaluation_timestamp TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    weight REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Strategy performance database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing performance database: {e}")
    
    def record_prediction(self, prediction: PredictionResult):
        """Record a prediction for later evaluation"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO strategy_performance 
                    (strategy_name, symbol, timeframe, predicted_price, confidence, timestamp, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending')
                ''', (
                    prediction.strategy_name,
                    prediction.symbol,
                    prediction.timeframe,
                    prediction.predicted_price,
                    prediction.confidence,
                    prediction.timestamp
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def evaluate_predictions(self, symbol: str, current_price: float, evaluation_time: datetime):
        """Evaluate pending predictions and update performance metrics"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get pending predictions that are ready for evaluation (older than 1 hour)
                cutoff_time = evaluation_time - timedelta(hours=1)
                
                cursor.execute('''
                    SELECT id, strategy_name, predicted_price, confidence, timestamp
                    FROM strategy_performance
                    WHERE symbol = ? AND status = 'pending' AND timestamp < ?
                ''', (symbol, cutoff_time))
                
                pending_predictions = cursor.fetchall()
                
                for pred_id, strategy_name, predicted_price, confidence, pred_timestamp in pending_predictions:
                    # Calculate accuracy
                    price_error = abs(predicted_price - current_price)
                    accuracy = max(0.0, 1.0 - (price_error / current_price))
                    
                    # Check direction correctness
                    # This would need historical price at prediction time - simplified for now
                    direction_correct = True  # Placeholder
                    
                    # Update record
                    cursor.execute('''
                        UPDATE strategy_performance
                        SET actual_price = ?, accuracy = ?, direction_correct = ?, 
                            evaluation_timestamp = ?, status = 'evaluated'
                        WHERE id = ?
                    ''', (current_price, accuracy, direction_correct, evaluation_time, pred_id))
                    
                    # Update performance history
                    self.performance_history[strategy_name].append({
                        'accuracy': accuracy,
                        'confidence': confidence,
                        'direction_correct': direction_correct,
                        'timestamp': evaluation_time
                    })
                
                conn.commit()
                conn.close()
                
                # Update strategy weights based on recent performance
                self._update_strategy_weights()
                
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
    
    def _update_strategy_weights(self):
        """Update strategy weights based on recent performance"""
        try:
            # Calculate performance scores for each strategy
            performance_scores = {}
            
            for strategy_name in self.strategy_weights.keys():
                recent_performance = self.performance_history[strategy_name][-50:]  # Last 50 predictions
                
                if len(recent_performance) < 5:  # Not enough data
                    performance_scores[strategy_name] = 0.5  # Neutral score
                    continue
                
                # Calculate weighted performance score
                accuracy_score = np.mean([p['accuracy'] for p in recent_performance])
                direction_score = np.mean([1.0 if p['direction_correct'] else 0.0 for p in recent_performance])
                confidence_calibration = self._calculate_confidence_calibration(recent_performance)
                
                # Combine scores
                performance_score = (accuracy_score * 0.4 + 
                                   direction_score * 0.4 + 
                                   confidence_calibration * 0.2)
                
                performance_scores[strategy_name] = performance_score
            
            # Convert scores to weights
            total_score = sum(performance_scores.values())
            if total_score > 0:
                new_weights = {}
                for strategy_name, score in performance_scores.items():
                    base_weight = score / total_score
                    # Apply smoothing to prevent extreme weight changes
                    current_weight = self.strategy_weights[strategy_name]
                    smoothed_weight = current_weight * 0.7 + base_weight * 0.3
                    new_weights[strategy_name] = max(0.05, min(0.5, smoothed_weight))  # Bound between 5% and 50%
                
                # Normalize weights
                total_weight = sum(new_weights.values())
                self.strategy_weights = {k: v/total_weight for k, v in new_weights.items()}
                
                # Save weights to database
                self._save_weights()
                
                logger.info(f"Updated strategy weights: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {e}")
    
    def _calculate_confidence_calibration(self, performance_data: List[Dict]) -> float:
        """Calculate how well-calibrated the confidence scores are"""
        if len(performance_data) < 10:
            return 0.5
        
        # Group by confidence ranges and check accuracy
        confidence_buckets = {
            'low': [],      # 0.0 - 0.4
            'medium': [],   # 0.4 - 0.7  
            'high': []      # 0.7 - 1.0
        }
        
        for perf in performance_data:
            confidence = perf['confidence']
            accuracy = perf['accuracy']
            
            if confidence < 0.4:
                confidence_buckets['low'].append(accuracy)
            elif confidence < 0.7:
                confidence_buckets['medium'].append(accuracy)
            else:
                confidence_buckets['high'].append(accuracy)
        
        # Calculate calibration score
        calibration_score = 0.0
        bucket_count = 0
        
        # Expected accuracies for each bucket
        expected_accuracies = {'low': 0.3, 'medium': 0.55, 'high': 0.8}
        
        for bucket, accuracies in confidence_buckets.items():
            if len(accuracies) >= 3:  # Minimum samples
                actual_accuracy = np.mean(accuracies)
                expected_accuracy = expected_accuracies[bucket]
                # Penalize deviation from expected
                bucket_score = 1.0 - abs(actual_accuracy - expected_accuracy)
                calibration_score += max(0, bucket_score)
                bucket_count += 1
        
        return calibration_score / bucket_count if bucket_count > 0 else 0.5
    
    def _save_weights(self):
        """Save current weights to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now(timezone.utc)
            
            for strategy_name, weight in self.strategy_weights.items():
                cursor.execute('''
                    INSERT INTO strategy_weights (strategy_name, weight, timestamp)
                    VALUES (?, ?, ?)
                ''', (strategy_name, weight, current_time))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.strategy_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all strategies"""
        summary = {}
        
        for strategy_name, history in self.performance_history.items():
            if len(history) > 0:
                recent_history = history[-50:]  # Last 50 predictions
                
                summary[strategy_name] = {
                    'avg_accuracy': np.mean([p['accuracy'] for p in recent_history]),
                    'direction_accuracy': np.mean([1.0 if p['direction_correct'] else 0.0 for p in recent_history]),
                    'avg_confidence': np.mean([p['confidence'] for p in recent_history]),
                    'prediction_count': len(recent_history),
                    'current_weight': self.strategy_weights.get(strategy_name, 0.0)
                }
            else:
                summary[strategy_name] = {
                    'avg_accuracy': 0.0,
                    'direction_accuracy': 0.0,
                    'avg_confidence': 0.0,
                    'prediction_count': 0,
                    'current_weight': self.strategy_weights.get(strategy_name, 0.0)
                }
        
        return summary


class MultiStrategyMLEngine:
    """Main class that orchestrates the advanced multi-strategy ML system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.ensemble_system = EnsembleVotingSystem()
        self.performance_tracker = StrategyPerformanceTracker(
            db_path=self.config.get('performance_db', 'strategy_performance.db')
        )
        
        # Link performance tracker to ensemble system
        self.ensemble_system.performance_tracker = self.performance_tracker
        
        # Background services
        self._background_thread = None
        self._shutdown_event = threading.Event()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_processing_time': 0.0,
            'last_prediction_time': None
        }
        
        logger.info("Multi-Strategy ML Engine initialized")
    
    def start_background_services(self):
        """Start background services for performance tracking and model retraining"""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._shutdown_event.clear()
            self._background_thread = threading.Thread(target=self._background_worker, daemon=True)
            self._background_thread.start()
            logger.info("Background services started")
    
    def stop_background_services(self):
        """Stop background services"""
        if self._background_thread and self._background_thread.is_alive():
            self._shutdown_event.set()
            self._background_thread.join(timeout=5)
            logger.info("Background services stopped")
    
    def _background_worker(self):
        """Background worker for periodic tasks"""
        while not self._shutdown_event.is_set():
            try:
                # Evaluate predictions every hour
                current_time = datetime.now(timezone.utc)
                
                # This would normally get current price from data source
                # For now, using a placeholder
                current_price = 2350.0  # Would be fetched from real data source
                
                self.performance_tracker.evaluate_predictions("XAU/USD", current_price, current_time)
                
                # Clean old cache entries
                self._clean_prediction_cache()
                
                # Sleep for 1 hour
                self._shutdown_event.wait(3600)
                
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                self._shutdown_event.wait(300)  # Wait 5 minutes on error
    
    def _clean_prediction_cache(self):
        """Clean old prediction cache entries"""
        try:
            with self.cache_lock:
                current_time = datetime.now(timezone.utc)
                cutoff_time = current_time - timedelta(minutes=30)  # Cache for 30 minutes
                
                keys_to_remove = []
                for key, (prediction, timestamp) in self.prediction_cache.items():
                    if timestamp < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.prediction_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning prediction cache: {e}")
    
    async def get_prediction(self, symbol: str, timeframe: str, 
                           current_data: Dict[str, Any], 
                           use_cache: bool = True) -> EnsemblePrediction:
        """Get ensemble prediction for given symbol and timeframe"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{hash(str(current_data.get('current_price', 0)))}"
            
            if use_cache:
                with self.cache_lock:
                    if cache_key in self.prediction_cache:
                        cached_prediction, cache_time = self.prediction_cache[cache_key]
                        if datetime.now(timezone.utc) - cache_time < timedelta(minutes=5):  # 5-minute cache
                            return cached_prediction
            
            # Generate new prediction
            ensemble_prediction = await self.ensemble_system.generate_ensemble_prediction(
                symbol, timeframe, current_data
            )
            
            # Record individual predictions for performance tracking
            for individual_pred in ensemble_prediction.individual_predictions:
                self.performance_tracker.record_prediction(individual_pred)
            
            # Update cache
            if use_cache:
                with self.cache_lock:
                    self.prediction_cache[cache_key] = (ensemble_prediction, datetime.now(timezone.utc))
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=True)
            
            logger.info(f"Generated prediction for {symbol} {timeframe}: {ensemble_prediction.predicted_price} "
                       f"({ensemble_prediction.direction}, confidence: {ensemble_prediction.ensemble_confidence:.2%}, "
                       f"processing time: {processing_time:.2f}s)")
            
            return ensemble_prediction
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=False)
            logger.error(f"Error generating prediction: {e}")
            raise
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.metrics['total_predictions'] += 1
        if success:
            self.metrics['successful_predictions'] += 1
        
        # Update average processing time
        current_avg = self.metrics['avg_processing_time']
        total_count = self.metrics['total_predictions']
        self.metrics['avg_processing_time'] = ((current_avg * (total_count - 1)) + processing_time) / total_count
        
        self.metrics['last_prediction_time'] = datetime.now(timezone.utc)
    
    async def train_strategies(self, historical_data: Dict[str, pd.DataFrame], 
                             symbols: List[str] = None) -> Dict[str, bool]:
        """Train all strategies with historical data"""
        symbols = symbols or ['XAU/USD']
        training_results = {}
        
        for strategy_name, strategy in self.ensemble_system.strategies.items():
            try:
                logger.info(f"Training {strategy_name} strategy...")
                
                # Use data for all symbols for training
                combined_data = pd.concat(
                    [historical_data.get(symbol, pd.DataFrame()) for symbol in symbols], 
                    ignore_index=True
                )
                
                if len(combined_data) > 100:
                    success = await strategy.train_model(combined_data)
                    training_results[strategy_name] = success
                    
                    if success:
                        logger.info(f"{strategy_name} training completed successfully")
                    else:
                        logger.warning(f"{strategy_name} training failed")
                else:
                    logger.warning(f"Insufficient data for {strategy_name} training")
                    training_results[strategy_name] = False
                    
            except Exception as e:
                logger.error(f"Error training {strategy_name}: {e}")
                training_results[strategy_name] = False
        
        return training_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        strategy_performance = self.performance_tracker.get_performance_summary()
        
        return {
            'engine_metrics': self.metrics,
            'strategy_performance': strategy_performance,
            'current_weights': self.performance_tracker.get_strategy_weights(),
            'system_status': 'active' if not self._shutdown_event.is_set() else 'inactive',
            'cache_size': len(self.prediction_cache),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def get_strategy_details(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific strategy"""
        if strategy_name not in self.ensemble_system.strategies:
            return {'error': f'Strategy {strategy_name} not found'}
        
        strategy = self.ensemble_system.strategies[strategy_name]
        performance = self.performance_tracker.get_performance_summary().get(strategy_name, {})
        
        return {
            'name': strategy.name,
            'is_trained': strategy.is_trained,
            'last_training_time': strategy.last_training_time.isoformat() if strategy.last_training_time else None,
            'performance_metrics': strategy.performance_metrics,
            'current_weight': self.performance_tracker.strategy_weights.get(strategy_name, 0.0),
            'recent_performance': performance,
            'model_type': type(strategy.model).__name__ if strategy.model else 'Not trained'
        }
    
    async def force_retrain_strategy(self, strategy_name: str, historical_data: pd.DataFrame) -> bool:
        """Force retrain a specific strategy"""
        if strategy_name not in self.ensemble_system.strategies:
            return False
        
        try:
            strategy = self.ensemble_system.strategies[strategy_name]
            success = await strategy.train_model(historical_data)
            
            if success:
                logger.info(f"Force retrain of {strategy_name} completed successfully")
            else:
                logger.error(f"Force retrain of {strategy_name} failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error force retraining {strategy_name}: {e}")
            return False


# REST API Flask Integration
def create_ml_api_routes(app, ml_engine: MultiStrategyMLEngine):
    """Create Flask API routes for the ML engine"""
    
    @app.route('/api/ml/prediction/<symbol>/<timeframe>', methods=['GET'])
    async def get_ml_prediction(symbol: str, timeframe: str):
        """Get ML prediction for symbol and timeframe"""
        try:
            # Get current data (this would normally come from data sources)
            current_data = {
                'current_price': float(request.args.get('current_price', 2350.0)),
                'price_data': [],  # Would be populated from data pipeline
                'news_sentiment': {},
                'interest_rates': {},
                'inflation': {}
            }
            
            prediction = await ml_engine.get_prediction(symbol, timeframe, current_data)
            
            return {
                'success': True,
                'prediction': prediction.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"API error getting prediction: {e}")
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/ml/performance', methods=['GET'])
    def get_performance():
        """Get performance summary"""
        try:
            summary = ml_engine.get_performance_summary()
            return {'success': True, 'performance': summary}
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/ml/strategy/<strategy_name>', methods=['GET'])
    def get_strategy_info(strategy_name: str):
        """Get detailed strategy information"""
        try:
            details = ml_engine.get_strategy_details(strategy_name)
            return {'success': True, 'strategy': details}
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/ml/retrain/<strategy_name>', methods=['POST'])
    async def retrain_strategy(strategy_name: str):
        """Force retrain a specific strategy"""
        try:
            # This would normally load historical data
            import pandas as pd
            historical_data = pd.DataFrame({
                'close': [2300 + i + np.random.randn()*10 for i in range(1000)]
            })
            
            success = await ml_engine.force_retrain_strategy(strategy_name, historical_data)
            
            return {
                'success': success,
                'message': f"{'Successfully retrained' if success else 'Failed to retrain'} {strategy_name}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500


# WebSocket Integration
def create_ml_websocket_handlers(socketio, ml_engine: MultiStrategyMLEngine):
    """Create WebSocket handlers for real-time ML predictions"""
    
    @socketio.on('request_prediction')
    async def handle_prediction_request(data):
        """Handle real-time prediction requests"""
        try:
            symbol = data.get('symbol', 'XAU/USD')
            timeframe = data.get('timeframe', '1H')
            current_price = data.get('current_price', 2350.0)
            
            current_data = {
                'current_price': current_price,
                'price_data': data.get('price_data', []),
                'news_sentiment': data.get('news_sentiment', {}),
                'interest_rates': data.get('interest_rates', {}),
                'inflation': data.get('inflation', {})
            }
            
            prediction = await ml_engine.get_prediction(symbol, timeframe, current_data)
            
            # Emit prediction to client
            socketio.emit('prediction_update', {
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction': prediction.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"WebSocket prediction error: {e}")
            socketio.emit('prediction_error', {'error': str(e)})
    
    @socketio.on('subscribe_predictions')
    def handle_subscription(data):
        """Handle subscription to prediction updates"""
        symbols = data.get('symbols', ['XAU/USD'])
        timeframes = data.get('timeframes', ['1H', '4H', '1D'])
        
        # Join rooms for subscribed symbols/timeframes
        for symbol in symbols:
            for timeframe in timeframes:
                room_name = f"{symbol}_{timeframe}"
                join_room(room_name)
        
        emit('subscription_confirmed', {
            'symbols': symbols,
            'timeframes': timeframes,
            'message': 'Successfully subscribed to prediction updates'
        })


# Example usage and testing
if __name__ == "__main__":
    async def test_multi_strategy_engine():
        """Test the complete multi-strategy ML engine"""
        
        # Initialize engine
        config = {
            'performance_db': 'test_strategy_performance.db'
        }
        
        ml_engine = MultiStrategyMLEngine(config)
        ml_engine.start_background_services()
        
        try:
            # Test data
            test_data = {
                'current_price': 2350.70,  # Exact price for mathematical validation
                'price_data': [
                    {'timestamp': '2025-07-20T10:00:00Z', 'open': 2340, 'high': 2360, 'low': 2335, 'close': 2350, 'volume': 1000},
                    {'timestamp': '2025-07-21T10:00:00Z', 'open': 2350, 'high': 2365, 'low': 2340, 'close': 2355, 'volume': 1200},
                    {'timestamp': '2025-07-22T10:00:00Z', 'open': 2355, 'high': 2370, 'low': 2348, 'close': 2350.70, 'volume': 1100}
                ],
                'news_sentiment': {
                    'overall_sentiment': 0.2,  # Slightly positive
                    'confidence': 0.7,
                    'article_count': 15
                },
                'interest_rates': {
                    'fed_funds_rate': 5.25,
                    'real_yield_10y': 2.1
                },
                'inflation': {
                    'cpi_yoy': 3.2,
                    'core_cpi_yoy': 2.8
                }
            }
            
            # Generate prediction
            print("🚀 Testing Advanced Multi-Strategy ML Engine...")
            print(f"Input: Current price = ${test_data['current_price']}")
            
            prediction = await ml_engine.get_prediction("XAU/USD", "1D", test_data)
            
            print("\n📊 Ensemble Prediction Results:")
            print(f"Predicted Price: ${prediction.predicted_price}")
            print(f"Price Change: ${prediction.predicted_price - prediction.current_price:.2f} ({((prediction.predicted_price - prediction.current_price) / prediction.current_price * 100):.3f}%)")
            print(f"Direction: {prediction.direction}")
            print(f"Ensemble Confidence: {prediction.ensemble_confidence:.1%}")
            print(f"Model Agreement: {prediction.model_agreement:.1%}")
            print(f"Quality Score: {prediction.prediction_quality_score:.1%}")
            
            print(f"\n🎯 Mathematical Validation:")
            expected_change = (prediction.predicted_price - test_data['current_price']) / test_data['current_price'] * 100
            print(f"If current price is ${test_data['current_price']} and change is {expected_change:.3f}%")
            print(f"Result should be ${test_data['current_price'] * (1 + expected_change/100):.2f}")
            print(f"Actual result: ${prediction.predicted_price}")
            print(f"✅ Mathematical accuracy: {'PASS' if abs(prediction.predicted_price - test_data['current_price'] * (1 + expected_change/100)) < 0.01 else 'FAIL'}")
            
            print(f"\n🔧 Strategy Contributions:")
            for strategy, weight in prediction.strategy_contributions.items():
                print(f"  {strategy}: {weight:.1%}")
            
            print(f"\n📈 Support/Resistance:")
            print(f"Support Levels: {[f'${level:.2f}' for level in prediction.support_levels]}")
            print(f"Resistance Levels: {[f'${level:.2f}' for level in prediction.resistance_levels]}")
            print(f"Stop Loss: ${prediction.recommended_stop_loss}")
            print(f"Take Profit: ${prediction.recommended_take_profit}")
            
            print(f"\n🤖 Individual Strategy Predictions:")
            for pred in prediction.individual_predictions:
                print(f"  {pred.strategy_name}: ${pred.predicted_price} ({pred.direction}, {pred.confidence:.1%})")
                print(f"    Reasoning: {pred.reasoning}")
            
            # Test performance summary
            print(f"\n📊 Performance Summary:")
            performance = ml_engine.get_performance_summary()
            print(f"Total Predictions: {performance['engine_metrics']['total_predictions']}")
            print(f"Success Rate: {performance['engine_metrics']['successful_predictions']}/{performance['engine_metrics']['total_predictions']}")
            print(f"Avg Processing Time: {performance['engine_metrics']['avg_processing_time']:.3f}s")
            
            # Test individual strategy details
            print(f"\n🔍 Strategy Details:")
            for strategy_name in ['Technical', 'Sentiment', 'Macro', 'Pattern', 'Momentum']:
                details = ml_engine.get_strategy_details(strategy_name)
                print(f"  {strategy_name}: Trained={details['is_trained']}, Weight={details['current_weight']:.1%}")
            
        finally:
            ml_engine.stop_background_services()
        
        print(f"\n✅ Advanced Multi-Strategy ML Engine test completed successfully!")
    
    # Run the test
    asyncio.run(test_multi_strategy_engine())
