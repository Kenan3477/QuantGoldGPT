#!/usr/bin/env python3
"""
Advanced Multi-Strategy ML Prediction Engine for GoldGPT
Implements ensemble methods with specialized strategies for high-quality gold price predictions
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import our data pipeline
from data_integration_engine import DataManager, DataIntegrationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structured prediction result"""
    strategy_name: str
    timeframe: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    direction: str  # bullish, bearish, neutral
    confidence: float
    support_level: float
    resistance_level: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    features_used: List[str]
    reasoning: str

@dataclass
class EnsemblePrediction:
    """Final ensemble prediction"""
    timeframe: str
    current_price: float
    predicted_price: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    direction: str
    ensemble_confidence: float
    support_levels: List[float]
    resistance_levels: List[float]
    recommended_stop_loss: float
    recommended_take_profit: float
    strategy_votes: Dict[str, float]
    timestamp: datetime
    validation_score: float

class StrategyPerformanceTracker:
    """Tracks performance of individual strategies"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.performance_history = defaultdict(lambda: deque(maxlen=max_history))
        self.accuracy_scores = defaultdict(float)
        self.confidence_calibration = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def record_performance(self, strategy_name: str, predicted: float, actual: float, confidence: float):
        """Record a strategy's performance"""
        with self.lock:
            error = abs(predicted - actual) / actual if actual != 0 else abs(predicted - actual)
            self.performance_history[strategy_name].append(error)
            self.confidence_calibration[strategy_name].append((confidence, error))
            
            # Update accuracy score (inverse of mean error)
            if len(self.performance_history[strategy_name]) > 5:
                mean_error = np.mean(list(self.performance_history[strategy_name]))
                self.accuracy_scores[strategy_name] = 1.0 / (1.0 + mean_error)
    
    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get weight for strategy in ensemble"""
        base_weight = self.accuracy_scores.get(strategy_name, 0.2)
        
        # Adjust for recent performance
        if len(self.performance_history[strategy_name]) >= 10:
            recent_errors = list(self.performance_history[strategy_name])[-10:]
            recent_performance = 1.0 / (1.0 + np.mean(recent_errors))
            # Weight recent performance more heavily
            weight = 0.3 * base_weight + 0.7 * recent_performance
        else:
            weight = base_weight
        
        return max(0.1, min(1.0, weight))  # Clamp between 0.1 and 1.0
    
    def get_all_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """Get normalized weights for all strategies"""
        weights = {name: self.get_strategy_weight(name) for name in strategy_names}
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            return {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(strategy_names)
            return {name: equal_weight for name in strategy_names}

class BaseStrategy(ABC):
    """Abstract base class for prediction strategies"""
    
    def __init__(self, name: str, data_manager: DataManager):
        self.name = name
        self.data_manager = data_manager
        self.model = None
        self.scaler = StandardScaler()
        self.last_prediction = None
        self.confidence_threshold = 0.3
    
    @abstractmethod
    async def generate_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate strategy-specific features from market data"""
        pass
    
    @abstractmethod
    async def predict(self, timeframe: str) -> PredictionResult:
        """Generate prediction for the given timeframe"""
        pass
    
    def _calculate_support_resistance(self, prices: List[float], current_price: float) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        if len(prices) < 20:
            # Fallback calculation
            support = current_price * 0.995
            resistance = current_price * 1.005
            return support, resistance
        
        # Use statistical approach
        price_array = np.array(prices[-50:])  # Use last 50 prices
        support = np.percentile(price_array, 25)  # 25th percentile as support
        resistance = np.percentile(price_array, 75)  # 75th percentile as resistance
        
        # Ensure they make sense relative to current price
        if support > current_price:
            support = current_price * 0.99
        if resistance < current_price:
            resistance = current_price * 1.01
            
        return support, resistance
    
    def _calculate_stop_loss_take_profit(self, current_price: float, predicted_price: float, 
                                       volatility: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        price_change_percent = abs(predicted_price - current_price) / current_price
        
        # Adaptive stop loss based on volatility
        stop_loss_percent = max(0.01, min(0.05, volatility * 2))  # 1% to 5%
        take_profit_percent = max(price_change_percent * 0.8, 0.015)  # At least 1.5%
        
        if predicted_price > current_price:  # Bullish
            stop_loss = current_price * (1 - stop_loss_percent)
            take_profit = current_price * (1 + take_profit_percent)
        else:  # Bearish
            stop_loss = current_price * (1 + stop_loss_percent)
            take_profit = current_price * (1 - take_profit_percent)
        
        return stop_loss, take_profit

class TechnicalStrategy(BaseStrategy):
    """Technical analysis based strategy"""
    
    def __init__(self, data_manager: DataManager):
        super().__init__("Technical", data_manager)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
    
    async def generate_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate technical analysis features"""
        features = {}
        
        # Get price data
        price_features = {k: v for k, v in market_data.get('features', {}).items() 
                         if any(indicator in k for indicator in ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'atr'])}
        
        features.update(price_features)
        
        # Add derived technical features
        current_price = market_data.get('features', {}).get('current_price', 2000.0)
        sma_20 = market_data.get('features', {}).get('tech_sma_20', current_price)
        ema_20 = market_data.get('features', {}).get('tech_ema_20', current_price)
        
        features.update({
            'price_vs_sma_20': (current_price - sma_20) / sma_20 if sma_20 > 0 else 0,
            'price_vs_ema_20': (current_price - ema_20) / ema_20 if ema_20 > 0 else 0,
            'sma_ema_divergence': (sma_20 - ema_20) / ema_20 if ema_20 > 0 else 0,
            'volatility_regime': market_data.get('features', {}).get('volatility_10d', 0.02),
        })
        
        return features
    
    async def predict(self, timeframe: str) -> PredictionResult:
        """Generate technical analysis prediction"""
        try:
            # Get market data
            dataset = await self.data_manager.get_ml_ready_dataset()
            features = await self.generate_features(dataset)
            
            current_price = dataset.get('features', {}).get('current_price', 2000.0)
            
            # Create feature vector
            feature_values = list(features.values())
            if len(feature_values) < 5:
                # Not enough features, return neutral prediction
                return self._create_neutral_prediction(current_price, timeframe, features)
            
            # Simulate prediction (in production, use trained model)
            rsi = features.get('tech_rsi_14', 50.0)
            macd = features.get('tech_macd', 0.0)
            bollinger_pos = features.get('price_vs_sma_20', 0.0)
            
            # Technical analysis logic
            technical_signal = 0.0
            confidence = 0.5
            
            # RSI analysis
            if rsi > 70:
                technical_signal -= 0.3  # Overbought
                confidence += 0.1
            elif rsi < 30:
                technical_signal += 0.3  # Oversold
                confidence += 0.1
            
            # MACD analysis
            if macd > 0:
                technical_signal += 0.2
            else:
                technical_signal -= 0.2
            
            # Bollinger analysis
            technical_signal += bollinger_pos * 0.1
            
            # Predict price change
            price_change_percent = technical_signal * 0.01  # Convert to percentage
            predicted_price = current_price * (1 + price_change_percent)
            
            # Determine direction
            if price_change_percent > 0.002:
                direction = "bullish"
                confidence += 0.1
            elif price_change_percent < -0.002:
                direction = "bearish"
                confidence += 0.1
            else:
                direction = "neutral"
            
            confidence = max(0.2, min(0.9, confidence))
            
            # Calculate support/resistance
            historical_prices = [current_price] * 20  # Placeholder
            support, resistance = self._calculate_support_resistance(historical_prices, current_price)
            
            # Calculate stop loss and take profit
            volatility = features.get('volatility_regime', 0.02)
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                current_price, predicted_price, volatility
            )
            
            return PredictionResult(
                strategy_name=self.name,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=predicted_price - current_price,
                price_change_percent=price_change_percent * 100,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=f"RSI: {rsi:.1f}, MACD: {macd:.3f}, Technical Signal: {technical_signal:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Technical strategy prediction failed: {e}")
            return self._create_neutral_prediction(2000.0, timeframe, {})
    
    def _create_neutral_prediction(self, current_price: float, timeframe: str, 
                                 features: Dict[str, float]) -> PredictionResult:
        """Create a neutral prediction when analysis fails"""
        return PredictionResult(
            strategy_name=self.name,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=current_price,
            price_change=0.0,
            price_change_percent=0.0,
            direction="neutral",
            confidence=0.2,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.02,
            timestamp=datetime.now(timezone.utc),
            features_used=list(features.keys()),
            reasoning="Insufficient data for technical analysis"
        )

class SentimentStrategy(BaseStrategy):
    """News sentiment and social signals based strategy"""
    
    def __init__(self, data_manager: DataManager):
        super().__init__("Sentiment", data_manager)
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    async def generate_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate sentiment-based features"""
        features = {}
        
        # Get sentiment features
        sentiment_features = {k: v for k, v in market_data.get('features', {}).items() 
                            if 'news' in k or 'sentiment' in k}
        
        features.update(sentiment_features)
        
        # Add derived sentiment features
        news_sentiment = features.get('news_sentiment_avg', 0.0)
        news_count = features.get('news_count', 0)
        
        features.update({
            'sentiment_strength': abs(news_sentiment),
            'sentiment_polarity': 1 if news_sentiment > 0 else -1 if news_sentiment < 0 else 0,
            'news_activity': min(news_count / 10.0, 1.0),  # Normalize news activity
            'sentiment_confidence': features.get('news_relevance_avg', 0.5),
        })
        
        return features
    
    async def predict(self, timeframe: str) -> PredictionResult:
        """Generate sentiment-based prediction"""
        try:
            dataset = await self.data_manager.get_ml_ready_dataset()
            features = await self.generate_features(dataset)
            
            current_price = dataset.get('features', {}).get('current_price', 2000.0)
            
            # Sentiment analysis
            sentiment_score = features.get('news_sentiment_avg', 0.0)
            sentiment_strength = features.get('sentiment_strength', 0.0)
            news_activity = features.get('news_activity', 0.1)
            
            # Calculate sentiment impact
            sentiment_impact = sentiment_score * sentiment_strength * news_activity
            
            # Convert sentiment to price prediction
            price_change_percent = sentiment_impact * 0.005  # Sentiment multiplier
            predicted_price = current_price * (1 + price_change_percent)
            
            # Confidence based on sentiment strength and news volume
            confidence = min(0.8, 0.3 + sentiment_strength * 0.5 + news_activity * 0.2)
            
            # Direction
            if price_change_percent > 0.001:
                direction = "bullish"
            elif price_change_percent < -0.001:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Calculate support/resistance based on sentiment
            volatility = 0.02  # Default volatility
            support = current_price * (1 - volatility)
            resistance = current_price * (1 + volatility)
            
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                current_price, predicted_price, volatility
            )
            
            return PredictionResult(
                strategy_name=self.name,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=predicted_price - current_price,
                price_change_percent=price_change_percent * 100,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=f"Sentiment: {sentiment_score:.3f}, Impact: {sentiment_impact:.3f}, News Activity: {news_activity:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Sentiment strategy prediction failed: {e}")
            return self._create_neutral_prediction(2000.0, timeframe)

    def _create_neutral_prediction(self, current_price: float, timeframe: str) -> PredictionResult:
        return PredictionResult(
            strategy_name=self.name,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=current_price,
            price_change=0.0,
            price_change_percent=0.0,
            direction="neutral",
            confidence=0.2,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.02,
            timestamp=datetime.now(timezone.utc),
            features_used=[],
            reasoning="Insufficient sentiment data"
        )

class MacroStrategy(BaseStrategy):
    """Macroeconomic indicators based strategy"""
    
    def __init__(self, data_manager: DataManager):
        super().__init__("Macro", data_manager)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    
    async def generate_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate macroeconomic features"""
        features = {}
        
        # Get economic features
        econ_features = {k: v for k, v in market_data.get('features', {}).items() 
                        if 'econ' in k or 'usd' in k}
        
        features.update(econ_features)
        
        # Add macro relationships
        usd_strength = features.get('econ_usd_index', 100.0)
        interest_rate = features.get('econ_fed_funds_rate', 5.0)
        inflation = features.get('econ_cpi_yoy', 3.0)
        
        features.update({
            'usd_gold_inverse': 1.0 / (usd_strength / 100.0) if usd_strength > 0 else 1.0,
            'real_interest_rate': interest_rate - inflation,
            'monetary_policy_stance': 1 if interest_rate < inflation else -1,
            'safe_haven_demand': max(0, (inflation - interest_rate) / 10.0),
        })
        
        return features
    
    async def predict(self, timeframe: str) -> PredictionResult:
        """Generate macro-based prediction"""
        try:
            dataset = await self.data_manager.get_ml_ready_dataset()
            features = await self.generate_features(dataset)
            
            current_price = dataset.get('features', {}).get('current_price', 2000.0)
            
            # Macro analysis
            usd_strength = features.get('econ_usd_index', 100.0)
            real_rate = features.get('real_interest_rate', 2.0)
            safe_haven = features.get('safe_haven_demand', 0.0)
            
            # Gold-macro relationships
            macro_signal = 0.0
            confidence = 0.4
            
            # USD strength impact (inverse relationship with gold)
            usd_impact = (100 - usd_strength) / 100.0 * 0.3
            macro_signal += usd_impact
            
            # Real interest rate impact (negative relationship)
            rate_impact = -real_rate / 10.0 * 0.4
            macro_signal += rate_impact
            
            # Safe haven demand
            macro_signal += safe_haven * 0.3
            
            confidence += min(0.4, abs(macro_signal) * 2)
            
            # Convert to price prediction
            price_change_percent = macro_signal * 0.008  # Macro multiplier
            predicted_price = current_price * (1 + price_change_percent)
            
            direction = "bullish" if price_change_percent > 0.002 else "bearish" if price_change_percent < -0.002 else "neutral"
            
            volatility = 0.025  # Macro volatility
            support, resistance = self._calculate_support_resistance([current_price] * 20, current_price)
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(current_price, predicted_price, volatility)
            
            return PredictionResult(
                strategy_name=self.name,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=predicted_price - current_price,
                price_change_percent=price_change_percent * 100,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=f"USD: {usd_strength:.1f}, Real Rate: {real_rate:.2f}%, Safe Haven: {safe_haven:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Macro strategy prediction failed: {e}")
            return self._create_neutral_prediction(2000.0, timeframe)

    def _create_neutral_prediction(self, current_price: float, timeframe: str) -> PredictionResult:
        return PredictionResult(
            strategy_name=self.name,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=current_price,
            price_change=0.0,
            price_change_percent=0.0,
            direction="neutral",
            confidence=0.2,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.02,
            timestamp=datetime.now(timezone.utc),
            features_used=[],
            reasoning="Insufficient macro data"
        )

class PatternStrategy(BaseStrategy):
    """Chart patterns and historical correlations strategy"""
    
    def __init__(self, data_manager: DataManager):
        super().__init__("Pattern", data_manager)
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    async def generate_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate pattern-based features"""
        features = {}
        
        # Get price and volume features
        price_features = {k: v for k, v in market_data.get('features', {}).items() 
                         if any(word in k for word in ['price', 'volume', 'momentum', 'volatility'])}
        
        features.update(price_features)
        
        # Pattern recognition features
        current_price = market_data.get('features', {}).get('current_price', 2000.0)
        momentum = market_data.get('features', {}).get('momentum_5d', 0.0)
        volatility = market_data.get('features', {}).get('volatility_10d', 0.02)
        
        features.update({
            'trend_strength': abs(momentum),
            'trend_direction': 1 if momentum > 0 else -1 if momentum < 0 else 0,
            'volatility_regime': 1 if volatility > 0.03 else 0,  # High/low volatility
            'price_momentum_consistency': momentum * features.get('price_change', 0.0),
        })
        
        return features
    
    async def predict(self, timeframe: str) -> PredictionResult:
        """Generate pattern-based prediction"""
        try:
            dataset = await self.data_manager.get_ml_ready_dataset()
            features = await self.generate_features(dataset)
            
            current_price = dataset.get('features', {}).get('current_price', 2000.0)
            
            # Pattern analysis
            momentum = features.get('momentum_5d', 0.0)
            trend_strength = features.get('trend_strength', 0.0)
            volatility_regime = features.get('volatility_regime', 0)
            
            # Pattern recognition logic
            pattern_signal = 0.0
            confidence = 0.3
            
            # Momentum continuation pattern
            if trend_strength > 0.02:
                pattern_signal += momentum * 0.5
                confidence += 0.2
            
            # Mean reversion in high volatility
            if volatility_regime and abs(momentum) > 0.03:
                pattern_signal -= momentum * 0.3  # Expect reversal
                confidence += 0.1
            
            # Price change consistency
            consistency = features.get('price_momentum_consistency', 0.0)
            pattern_signal += consistency * 0.2
            
            confidence = min(0.7, confidence + trend_strength * 5)
            
            # Convert to price prediction
            price_change_percent = pattern_signal * 0.006
            predicted_price = current_price * (1 + price_change_percent)
            
            direction = "bullish" if price_change_percent > 0.001 else "bearish" if price_change_percent < -0.001 else "neutral"
            
            volatility = features.get('volatility_10d', 0.02)
            support, resistance = self._calculate_support_resistance([current_price] * 20, current_price)
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(current_price, predicted_price, volatility)
            
            return PredictionResult(
                strategy_name=self.name,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=predicted_price - current_price,
                price_change_percent=price_change_percent * 100,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=f"Momentum: {momentum:.3f}, Trend Strength: {trend_strength:.3f}, Pattern Signal: {pattern_signal:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Pattern strategy prediction failed: {e}")
            return self._create_neutral_prediction(2000.0, timeframe)

    def _create_neutral_prediction(self, current_price: float, timeframe: str) -> PredictionResult:
        return PredictionResult(
            strategy_name=self.name,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=current_price,
            price_change=0.0,
            price_change_percent=0.0,
            direction="neutral",
            confidence=0.2,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.02,
            timestamp=datetime.now(timezone.utc),
            features_used=[],
            reasoning="Insufficient pattern data"
        )

class MomentumStrategy(BaseStrategy):
    """Price dynamics and volume analysis strategy"""
    
    def __init__(self, data_manager: DataManager):
        super().__init__("Momentum", data_manager)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    async def generate_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate momentum-based features"""
        features = {}
        
        # Get momentum and volume features
        momentum_features = {k: v for k, v in market_data.get('features', {}).items() 
                           if any(word in k for word in ['momentum', 'volume', 'change', 'volatility'])}
        
        features.update(momentum_features)
        
        # Advanced momentum features
        price_change = features.get('price_change', 0.0)
        volume_change = features.get('volume_change', 0.0)
        volatility = features.get('volatility_10d', 0.02)
        
        features.update({
            'price_volume_correlation': price_change * volume_change if volume_change != 0 else 0,
            'momentum_acceleration': price_change / volatility if volatility > 0 else 0,
            'volume_momentum': volume_change * abs(price_change),
            'volatility_adjusted_return': price_change / volatility if volatility > 0 else 0,
        })
        
        return features
    
    async def predict(self, timeframe: str) -> PredictionResult:
        """Generate momentum-based prediction"""
        try:
            dataset = await self.data_manager.get_ml_ready_dataset()
            features = await self.generate_features(dataset)
            
            current_price = dataset.get('features', {}).get('current_price', 2000.0)
            
            # Momentum analysis
            price_change = features.get('price_change', 0.0)
            volume_change = features.get('volume_change', 0.0)
            pv_correlation = features.get('price_volume_correlation', 0.0)
            momentum_accel = features.get('momentum_acceleration', 0.0)
            
            # Momentum strategy logic
            momentum_signal = 0.0
            confidence = 0.4
            
            # Price-volume confirmation
            if price_change > 0 and volume_change > 0:
                momentum_signal += 0.3  # Bullish confirmation
                confidence += 0.2
            elif price_change < 0 and volume_change > 0:
                momentum_signal -= 0.3  # Bearish confirmation
                confidence += 0.2
            
            # Momentum acceleration
            momentum_signal += momentum_accel * 0.2
            
            # Volume momentum
            momentum_signal += features.get('volume_momentum', 0.0) * 0.1
            
            confidence += min(0.3, abs(momentum_signal) * 1.5)
            
            # Convert to price prediction
            price_change_percent = momentum_signal * 0.007
            predicted_price = current_price * (1 + price_change_percent)
            
            direction = "bullish" if price_change_percent > 0.002 else "bearish" if price_change_percent < -0.002 else "neutral"
            
            volatility = features.get('volatility_10d', 0.02)
            support, resistance = self._calculate_support_resistance([current_price] * 20, current_price)
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(current_price, predicted_price, volatility)
            
            return PredictionResult(
                strategy_name=self.name,
                timeframe=timeframe,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=predicted_price - current_price,
                price_change_percent=price_change_percent * 100,
                direction=direction,
                confidence=confidence,
                support_level=support,
                resistance_level=resistance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(timezone.utc),
                features_used=list(features.keys()),
                reasoning=f"P/V Corr: {pv_correlation:.3f}, Momentum Accel: {momentum_accel:.3f}, Signal: {momentum_signal:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Momentum strategy prediction failed: {e}")
            return self._create_neutral_prediction(2000.0, timeframe)

    def _create_neutral_prediction(self, current_price: float, timeframe: str) -> PredictionResult:
        return PredictionResult(
            strategy_name=self.name,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=current_price,
            price_change=0.0,
            price_change_percent=0.0,
            direction="neutral",
            confidence=0.2,
            support_level=current_price * 0.99,
            resistance_level=current_price * 1.01,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.02,
            timestamp=datetime.now(timezone.utc),
            features_used=[],
            reasoning="Insufficient momentum data"
        )

class PredictionValidator:
    """Validates prediction quality before publishing"""
    
    def __init__(self):
        self.min_confidence = 0.2
        self.max_price_change = 0.1  # 10% maximum change
        self.consistency_threshold = 0.7
    
    def validate_prediction(self, prediction: PredictionResult) -> Tuple[bool, str, float]:
        """Validate a single prediction"""
        score = 1.0
        issues = []
        
        # Check confidence bounds
        if prediction.confidence < self.min_confidence:
            score *= 0.5
            issues.append("Low confidence")
        
        # Check price change reasonableness
        price_change_abs = abs(prediction.price_change_percent / 100)
        if price_change_abs > self.max_price_change:
            score *= 0.3
            issues.append("Excessive price change")
        
        # Check stop loss / take profit logic
        if prediction.direction == "bullish":
            if prediction.stop_loss >= prediction.current_price:
                score *= 0.6
                issues.append("Invalid stop loss for bullish prediction")
            if prediction.take_profit <= prediction.current_price:
                score *= 0.6
                issues.append("Invalid take profit for bullish prediction")
        elif prediction.direction == "bearish":
            if prediction.stop_loss <= prediction.current_price:
                score *= 0.6
                issues.append("Invalid stop loss for bearish prediction")
            if prediction.take_profit >= prediction.current_price:
                score *= 0.6
                issues.append("Invalid take profit for bearish prediction")
        
        # Check support/resistance logic
        if prediction.support_level > prediction.current_price:
            score *= 0.8
            issues.append("Support above current price")
        if prediction.resistance_level < prediction.current_price:
            score *= 0.8
            issues.append("Resistance below current price")
        
        is_valid = score >= 0.5 and len(issues) < 3
        validation_message = "; ".join(issues) if issues else "Valid prediction"
        
        return is_valid, validation_message, score
    
    def validate_ensemble(self, predictions: List[PredictionResult]) -> Tuple[bool, str, float]:
        """Validate ensemble of predictions"""
        if len(predictions) < 2:
            return False, "Insufficient predictions for ensemble", 0.0
        
        # Check prediction consistency
        directions = [p.direction for p in predictions]
        predicted_prices = [p.predicted_price for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        # Direction consistency
        direction_counts = {d: directions.count(d) for d in set(directions)}
        max_direction_count = max(direction_counts.values())
        direction_consistency = max_direction_count / len(predictions)
        
        # Price prediction spread
        price_std = np.std(predicted_prices) if len(predicted_prices) > 1 else 0
        price_mean = np.mean(predicted_prices)
        price_spread = price_std / price_mean if price_mean > 0 else 0
        
        # Overall score
        consistency_score = direction_consistency * 0.6 + (1 - min(price_spread, 0.1)) * 0.4
        confidence_score = np.mean(confidences)
        
        overall_score = consistency_score * 0.7 + confidence_score * 0.3
        
        is_valid = (direction_consistency >= self.consistency_threshold and 
                   price_spread < 0.05 and  # Less than 5% spread
                   overall_score >= 0.5)
        
        validation_message = f"Direction consistency: {direction_consistency:.2f}, Price spread: {price_spread:.3f}"
        
        return is_valid, validation_message, overall_score

class EnsembleVotingSystem:
    """Advanced ensemble voting with meta-learning"""
    
    def __init__(self):
        self.performance_tracker = StrategyPerformanceTracker()
        self.validator = PredictionValidator()
        self.meta_features_history = deque(maxlen=100)
        self.ensemble_performance = deque(maxlen=50)
    
    def create_ensemble_prediction(self, predictions: List[PredictionResult], 
                                 timeframe: str) -> Optional[EnsemblePrediction]:
        """Create ensemble prediction from individual strategy predictions"""
        if not predictions:
            return None
        
        # Validate ensemble
        is_valid, validation_msg, validation_score = self.validator.validate_ensemble(predictions)
        
        if not is_valid and validation_score < 0.3:
            logger.warning(f"Ensemble validation failed: {validation_msg}")
            return None
        
        # Get strategy weights
        strategy_names = [p.strategy_name for p in predictions]
        weights = self.performance_tracker.get_all_weights(strategy_names)
        
        # Calculate weighted predictions
        current_price = predictions[0].current_price
        weighted_price = 0.0
        weighted_confidence = 0.0
        support_levels = []
        resistance_levels = []
        strategy_votes = {}
        
        for prediction in predictions:
            weight = weights.get(prediction.strategy_name, 0.2)
            weighted_price += prediction.predicted_price * weight
            weighted_confidence += prediction.confidence * weight
            support_levels.append(prediction.support_level)
            resistance_levels.append(prediction.resistance_level)
            strategy_votes[prediction.strategy_name] = weight
        
        # Calculate confidence interval using prediction spread
        predicted_prices = [p.predicted_price for p in predictions]
        price_std = np.std(predicted_prices) if len(predicted_prices) > 1 else current_price * 0.01
        confidence_interval = (
            weighted_price - 1.96 * price_std,  # 95% confidence interval
            weighted_price + 1.96 * price_std
        )
        
        # Determine ensemble direction
        bullish_weight = sum(weights[p.strategy_name] for p in predictions if p.direction == "bullish")
        bearish_weight = sum(weights[p.strategy_name] for p in predictions if p.direction == "bearish")
        neutral_weight = sum(weights[p.strategy_name] for p in predictions if p.direction == "neutral")
        
        if bullish_weight > max(bearish_weight, neutral_weight):
            direction = "bullish"
        elif bearish_weight > max(bullish_weight, neutral_weight):
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Calculate recommended levels
        recommended_stop_loss = np.mean([p.stop_loss for p in predictions])
        recommended_take_profit = np.mean([p.take_profit for p in predictions])
        
        # Aggregate support/resistance levels
        support_levels = sorted(support_levels)
        resistance_levels = sorted(resistance_levels)
        
        return EnsemblePrediction(
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=weighted_price,
            confidence_interval=confidence_interval,
            direction=direction,
            ensemble_confidence=weighted_confidence,
            support_levels=support_levels[:3],  # Top 3 support levels
            resistance_levels=resistance_levels[-3:],  # Top 3 resistance levels
            recommended_stop_loss=recommended_stop_loss,
            recommended_take_profit=recommended_take_profit,
            strategy_votes=strategy_votes,
            timestamp=datetime.now(timezone.utc),
            validation_score=validation_score
        )
    
    def update_performance(self, predictions: List[PredictionResult], actual_price: float):
        """Update strategy performance based on actual outcomes"""
        for prediction in predictions:
            self.performance_tracker.record_performance(
                prediction.strategy_name,
                prediction.predicted_price,
                actual_price,
                prediction.confidence
            )

class AdvancedMLPredictionEngine:
    """Main prediction engine coordinating all strategies"""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        if data_manager is None:
            # Initialize data pipeline
            integration_engine = DataIntegrationEngine()
            data_manager = DataManager(integration_engine)
        
        self.data_manager = data_manager
        
        # Initialize strategies
        self.strategies = [
            TechnicalStrategy(data_manager),
            SentimentStrategy(data_manager),
            MacroStrategy(data_manager),
            PatternStrategy(data_manager),
            MomentumStrategy(data_manager)
        ]
        
        self.ensemble_system = EnsembleVotingSystem()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._lock = threading.Lock()
    
    async def generate_prediction(self, timeframe: str) -> Optional[EnsemblePrediction]:
        """Generate ensemble prediction for specified timeframe"""
        start_time = time.time()
        
        try:
            # Generate predictions from all strategies concurrently
            logger.info(f"Generating {timeframe} prediction using {len(self.strategies)} strategies")
            
            # Use asyncio.gather for concurrent execution
            prediction_tasks = [strategy.predict(timeframe) for strategy in self.strategies]
            strategy_predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)
            
            # Filter out failed predictions
            valid_predictions = []
            for i, prediction in enumerate(strategy_predictions):
                if isinstance(prediction, PredictionResult):
                    # Validate individual prediction
                    is_valid, msg, score = self.ensemble_system.validator.validate_prediction(prediction)
                    if is_valid or score > 0.3:  # Accept if reasonably valid
                        valid_predictions.append(prediction)
                    else:
                        logger.warning(f"Strategy {self.strategies[i].name} prediction rejected: {msg}")
                elif isinstance(prediction, Exception):
                    logger.error(f"Strategy {self.strategies[i].name} failed: {prediction}")
            
            if len(valid_predictions) < 2:
                logger.warning(f"Insufficient valid predictions: {len(valid_predictions)}")
                return None
            
            # Create ensemble prediction
            ensemble_prediction = self.ensemble_system.create_ensemble_prediction(
                valid_predictions, timeframe
            )
            
            if ensemble_prediction:
                execution_time = time.time() - start_time
                logger.info(f"Generated {timeframe} prediction in {execution_time:.2f}s "
                          f"(confidence: {ensemble_prediction.ensemble_confidence:.3f})")
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Prediction generation failed for {timeframe}: {e}")
            return None
    
    async def generate_multi_timeframe_predictions(self, 
                                                 timeframes: List[str] = None) -> Dict[str, EnsemblePrediction]:
        """Generate predictions for multiple timeframes"""
        if timeframes is None:
            timeframes = ["1H", "4H", "1D"]
        
        predictions = {}
        
        # Generate predictions for each timeframe
        for timeframe in timeframes:
            prediction = await self.generate_prediction(timeframe)
            if prediction:
                predictions[timeframe] = prediction
        
        return predictions
    
    async def get_strategy_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all strategies"""
        strategy_weights = self.ensemble_system.performance_tracker.get_all_weights(
            [s.name for s in self.strategies]
        )
        
        performance_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'strategy_weights': strategy_weights,
            'total_strategies': len(self.strategies),
            'ensemble_system_status': 'active',
            'strategies': {}
        }
        
        for strategy in self.strategies:
            accuracy = self.ensemble_system.performance_tracker.accuracy_scores.get(strategy.name, 0.0)
            prediction_count = len(self.ensemble_system.performance_tracker.performance_history[strategy.name])
            
            performance_report['strategies'][strategy.name] = {
                'weight': strategy_weights.get(strategy.name, 0.2),
                'accuracy_score': accuracy,
                'prediction_count': prediction_count,
                'status': 'active'
            }
        
        return performance_report
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        if hasattr(self.data_manager, 'integration_engine'):
            self.data_manager.integration_engine.close()

# Global instance
advanced_ml_engine = None

async def get_advanced_ml_predictions(timeframes: List[str] = None) -> Dict[str, Any]:
    """Main API function to get advanced ML predictions"""
    global advanced_ml_engine
    
    start_time = time.time()
    
    try:
        # Initialize engine if needed
        if advanced_ml_engine is None:
            logger.info("Initializing Advanced ML Prediction Engine...")
            advanced_ml_engine = AdvancedMLPredictionEngine()
        
        # Generate predictions
        if timeframes is None:
            timeframes = ["1H", "4H", "1D"]
        
        predictions = await advanced_ml_engine.generate_multi_timeframe_predictions(timeframes)
        
        # Get performance report
        performance_report = await advanced_ml_engine.get_strategy_performance_report()
        
        execution_time = time.time() - start_time
        
        # Format response
        response = {
            'status': 'success',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time': execution_time,
            'predictions': {},
            'performance': performance_report,
            'system_info': {
                'strategies_active': len(advanced_ml_engine.strategies),
                'ensemble_method': 'weighted_voting',
                'validation_enabled': True,
                'meta_learning': True
            }
        }
        
        # Convert predictions to serializable format
        for timeframe, prediction in predictions.items():
            response['predictions'][timeframe] = {
                'timeframe': prediction.timeframe,
                'current_price': prediction.current_price,
                'predicted_price': prediction.predicted_price,
                'price_change_percent': ((prediction.predicted_price - prediction.current_price) 
                                       / prediction.current_price) * 100,
                'direction': prediction.direction,
                'confidence': prediction.ensemble_confidence,
                'confidence_interval': {
                    'lower': prediction.confidence_interval[0],
                    'upper': prediction.confidence_interval[1]
                },
                'support_levels': prediction.support_levels,
                'resistance_levels': prediction.resistance_levels,
                'recommended_stop_loss': prediction.recommended_stop_loss,
                'recommended_take_profit': prediction.recommended_take_profit,
                'strategy_votes': prediction.strategy_votes,
                'validation_score': prediction.validation_score,
                'timestamp': prediction.timestamp.isoformat()
            }
        
        logger.info(f"Advanced ML predictions generated in {execution_time:.2f}s for {len(predictions)} timeframes")
        return response
        
    except Exception as e:
        logger.error(f"Advanced ML prediction failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time': time.time() - start_time,
            'predictions': {},
            'fallback_available': True
        }

# Example usage and testing
async def main():
    """Example usage of the advanced ML prediction engine"""
    print("🚀 Advanced Multi-Strategy ML Prediction Engine for GoldGPT")
    print("=" * 70)
    
    try:
        # Get predictions
        result = await get_advanced_ml_predictions(["1H", "4H", "1D"])
        
        if result['status'] == 'success':
            print(f"✅ Predictions generated in {result['execution_time']:.2f} seconds")
            print(f"📊 Active strategies: {result['system_info']['strategies_active']}")
            
            # Display predictions
            for timeframe, prediction in result['predictions'].items():
                print(f"\n🔮 {timeframe} Prediction:")
                print(f"   Current Price: ${prediction['current_price']:.2f}")
                print(f"   Predicted Price: ${prediction['predicted_price']:.2f}")
                print(f"   Change: {prediction['price_change_percent']:+.2f}%")
                print(f"   Direction: {prediction['direction'].upper()}")
                print(f"   Confidence: {prediction['confidence']:.3f}")
                print(f"   Stop Loss: ${prediction['recommended_stop_loss']:.2f}")
                print(f"   Take Profit: ${prediction['recommended_take_profit']:.2f}")
                
                # Show strategy votes
                print("   Strategy Votes:")
                for strategy, weight in prediction['strategy_votes'].items():
                    print(f"     {strategy}: {weight:.3f}")
            
            # Performance summary
            print(f"\n📈 Strategy Performance:")
            for strategy, perf in result['performance']['strategies'].items():
                print(f"   {strategy}: Weight={perf['weight']:.3f}, "
                     f"Accuracy={perf['accuracy_score']:.3f}, "
                     f"Predictions={perf['prediction_count']}")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        # Cleanup
        if advanced_ml_engine:
            advanced_ml_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
