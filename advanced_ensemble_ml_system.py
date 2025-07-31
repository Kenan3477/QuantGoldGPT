"""
🏛️ ADVANCED MULTI-STRATEGY ML ARCHITECTURE - PHASE 2
=======================================================

Institutional-grade ensemble system with specialized prediction strategies
Implements sophisticated voting mechanism and meta-learning optimization

Architecture:
- BaseStrategy abstract class with standardized interface
- 5 specialized strategy implementations
- EnsembleVotingSystem with dynamic weighting
- MetaLearningEngine for continuous optimization
- Real-time performance monitoring and adaptation

Created: July 23, 2025
Author: GoldGPT AI System
"""

import asyncio
import numpy as np
import pandas as pd
import sqlite3
import logging
import json
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
from scipy.optimize import minimize
import talib
import requests

warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('advanced_ensemble_ml')

@dataclass
class StrategyResult:
    """Standardized strategy prediction result"""
    strategy_name: str
    predicted_price: float
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    timeframe: str
    technical_indicators: Dict[str, float]
    reasoning: str
    risk_assessment: Dict[str, float]
    timestamp: datetime

@dataclass 
class MarketConditions:
    """Current market conditions for strategy selection"""
    volatility: float
    trend_strength: float
    volume_profile: float
    sentiment_score: float
    macro_environment: Dict[str, float]
    market_regime: str  # 'trending', 'ranging', 'volatile', 'crisis'

@dataclass
class EnsemblePrediction:
    """Final ensemble prediction result"""
    predicted_price: float
    confidence: float
    direction: str
    contributing_strategies: List[Dict[str, Any]]
    ensemble_weights: Dict[str, float]
    risk_metrics: Dict[str, float]
    market_conditions: MarketConditions
    timestamp: datetime

class BaseStrategy(ABC):
    """
    🏛️ Abstract base class for all prediction strategies
    Provides standardized interface for institutional-grade ML strategies
    """
    
    def __init__(self, name: str, db_path: str = "advanced_ensemble_performance.db"):
        self.name = name
        self.db_path = db_path
        self.performance_history = []
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.confidence_threshold = 0.6
        self.last_update = datetime.now()
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize strategy performance tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        prediction_timestamp DATETIME NOT NULL,
                        predicted_price REAL NOT NULL,
                        actual_price REAL,
                        confidence REAL NOT NULL,
                        accuracy_score REAL,
                        timeframe TEXT NOT NULL,
                        market_conditions TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_weights (
                        strategy_name TEXT PRIMARY KEY,
                        current_weight REAL NOT NULL,
                        performance_score REAL NOT NULL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Initialize strategy weight
                cursor.execute("""
                    INSERT OR REPLACE INTO strategy_weights 
                    (strategy_name, current_weight, performance_score)
                    VALUES (?, 0.2, 0.5)
                """, (self.name,))
                
                logger.info(f"✅ {self.name} strategy database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed for {self.name}: {e}")
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, timeframe: str) -> StrategyResult:
        """Generate prediction for given data and timeframe"""
        pass
    
    @abstractmethod
    def get_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence based on market conditions"""
        pass
    
    def update_performance(self, actual_outcome: float, prediction_result: StrategyResult) -> None:
        """Update strategy performance with actual outcome"""
        try:
            accuracy = 1.0 - abs(actual_outcome - prediction_result.predicted_price) / actual_outcome
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE strategy_performance 
                    SET actual_price = ?, accuracy_score = ?
                    WHERE strategy_name = ? AND prediction_timestamp = ?
                """, (actual_outcome, accuracy, self.name, prediction_result.timestamp))
                
                # Update strategy weight based on performance
                self._update_strategy_weight(accuracy)
                
            logger.info(f"📊 {self.name} performance updated: {accuracy:.3f} accuracy")
            
        except Exception as e:
            logger.error(f"❌ Performance update failed for {self.name}: {e}")
    
    def _update_strategy_weight(self, latest_accuracy: float):
        """Update strategy weight based on recent performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent performance
                cursor.execute("""
                    SELECT accuracy_score FROM strategy_performance 
                    WHERE strategy_name = ? AND accuracy_score IS NOT NULL
                    ORDER BY prediction_timestamp DESC LIMIT 10
                """, (self.name,))
                
                recent_scores = [row[0] for row in cursor.fetchall()]
                
                if recent_scores:
                    avg_performance = np.mean(recent_scores)
                    trend = np.mean(recent_scores[:5]) - np.mean(recent_scores[5:]) if len(recent_scores) >= 5 else 0
                    
                    # Dynamic weight calculation
                    base_weight = 0.2
                    performance_factor = avg_performance - 0.5  # Center around 0.5
                    trend_factor = trend * 2  # Amplify trend impact
                    
                    new_weight = max(0.05, min(0.5, base_weight + performance_factor + trend_factor))
                    
                    cursor.execute("""
                        UPDATE strategy_weights 
                        SET current_weight = ?, performance_score = ?
                        WHERE strategy_name = ?
                    """, (new_weight, avg_performance, self.name))
                    
                    logger.info(f"🎯 {self.name} weight updated: {new_weight:.3f} (perf: {avg_performance:.3f})")
                    
        except Exception as e:
            logger.error(f"❌ Weight update failed for {self.name}: {e}")
    
    def get_current_weight(self) -> float:
        """Get current strategy weight"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT current_weight FROM strategy_weights WHERE strategy_name = ?", (self.name,))
                result = cursor.fetchone()
                return result[0] if result else 0.2
        except:
            return 0.2

class TechnicalStrategy(BaseStrategy):
    """
    📈 Technical Analysis Strategy
    Implements comprehensive technical indicator analysis
    """
    
    def __init__(self):
        super().__init__("TechnicalStrategy")
        self.indicators = ['RSI', 'MACD', 'BOLLINGER', 'STOCHASTIC', 'WILLIAMS']
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def predict(self, data: pd.DataFrame, timeframe: str) -> StrategyResult:
        """Generate technical analysis prediction"""
        try:
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(data)
            
            # Prepare features
            features = self._prepare_technical_features(data, indicators)
            
            if len(features) < 50:  # Need sufficient data
                raise ValueError("Insufficient data for technical analysis")
            
            # Train/update model
            X, y = self._prepare_training_data(features)
            if len(X) > 0:
                self.model.fit(X, y)
            
            # Generate prediction
            latest_features = features.iloc[-1:].values
            predicted_price = self.model.predict(latest_features)[0]
            
            # Calculate confidence
            confidence = self._calculate_technical_confidence(indicators, data)
            
            # Determine direction
            current_price = data['Close'].iloc[-1]
            direction = self._determine_direction(predicted_price, current_price, indicators)
            
            # Risk assessment
            risk_metrics = self._assess_technical_risk(data, indicators)
            
            result = StrategyResult(
                strategy_name=self.name,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                timeframe=timeframe,
                technical_indicators=indicators,
                reasoning=self._generate_technical_reasoning(indicators, direction),
                risk_assessment=risk_metrics,
                timestamp=datetime.now()
            )
            
            # Store prediction
            self._store_prediction(result)
            
            logger.info(f"📈 Technical prediction: ${predicted_price:.2f} ({direction}, {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Technical strategy prediction failed: {e}")
            # Return neutral prediction as fallback
            current_price = data['Close'].iloc[-1]
            return StrategyResult(
                strategy_name=self.name,
                predicted_price=current_price,
                confidence=0.1,
                direction='neutral',
                timeframe=timeframe,
                technical_indicators={},
                reasoning="Technical analysis failed - insufficient data",
                risk_assessment={'overall_risk': 0.8},
                timestamp=datetime.now()
            )
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(close))
            
            indicators = {}
            
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            indicators['RSI'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['MACD'] = macd[-1] if not np.isnan(macd[-1]) else 0.0
            indicators['MACD_Signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
            indicators['MACD_Histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['BB_Upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else close[-1]
            indicators['BB_Lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else close[-1]
            indicators['BB_Position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['Stoch_K'] = slowk[-1] if not np.isnan(slowk[-1]) else 50.0
            indicators['Stoch_D'] = slowd[-1] if not np.isnan(slowd[-1]) else 50.0
            
            # Williams %R
            willr = talib.WILLR(high, low, close)
            indicators['Williams_R'] = willr[-1] if not np.isnan(willr[-1]) else -50.0
            
            # Additional indicators
            indicators['EMA_12'] = talib.EMA(close, timeperiod=12)[-1]
            indicators['EMA_26'] = talib.EMA(close, timeperiod=26)[-1]
            indicators['SMA_50'] = talib.SMA(close, timeperiod=50)[-1]
            indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Technical indicators calculation failed: {e}")
            return {}
    
    def _prepare_technical_features(self, data: pd.DataFrame, indicators: Dict[str, float]) -> pd.DataFrame:
        """Prepare feature matrix for technical analysis"""
        try:
            features = data.copy()
            
            # Add technical indicators as features
            for indicator, value in indicators.items():
                features[indicator] = value
            
            # Add price-based features
            features['Price_Change'] = features['Close'].pct_change()
            features['Volume_MA'] = features['Volume'].rolling(20).mean() if 'Volume' in features.columns else 1
            features['High_Low_Ratio'] = features['High'] / features['Low']
            features['Close_Position'] = (features['Close'] - features['Low']) / (features['High'] - features['Low'])
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Technical features preparation failed: {e}")
            return data
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for technical model"""
        try:
            # Use technical indicators as features
            feature_columns = [col for col in features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            X = features[feature_columns].values
            
            # Target is next period's closing price
            y = features['Close'].shift(-1).dropna().values
            X = X[:-1]  # Remove last row to match y length
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Training data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def _calculate_technical_confidence(self, indicators: Dict[str, float], data: pd.DataFrame) -> float:
        """Calculate confidence based on technical signal strength"""
        try:
            confidence_factors = []
            
            # RSI confidence
            rsi = indicators.get('RSI', 50)
            if rsi < 30 or rsi > 70:
                confidence_factors.append(0.8)  # Strong signal
            elif 40 <= rsi <= 60:
                confidence_factors.append(0.3)  # Weak signal
            else:
                confidence_factors.append(0.6)  # Moderate signal
            
            # MACD confirmation
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_Signal', 0)
            if abs(macd - macd_signal) > 0.01:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
            
            # Bollinger Bands position
            bb_position = indicators.get('BB_Position', 0.5)
            if bb_position < 0.2 or bb_position > 0.8:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Trend consistency
            ema_12 = indicators.get('EMA_12', 0)
            ema_26 = indicators.get('EMA_26', 0)
            if abs(ema_12 - ema_26) / ema_26 > 0.01:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
            
            return min(0.95, max(0.1, np.mean(confidence_factors)))
            
        except Exception as e:
            logger.error(f"❌ Technical confidence calculation failed: {e}")
            return 0.5
    
    def _determine_direction(self, predicted_price: float, current_price: float, indicators: Dict[str, float]) -> str:
        """Determine market direction based on technical analysis"""
        try:
            price_change = (predicted_price - current_price) / current_price
            
            # Technical confirmation
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI signals
            rsi = indicators.get('RSI', 50)
            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1
            
            # MACD signals
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_Signal', 0)
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Price vs EMA
            ema_12 = indicators.get('EMA_12', current_price)
            if current_price > ema_12:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Final direction
            if price_change > 0.005 and bullish_signals > bearish_signals:
                return 'bullish'
            elif price_change < -0.005 and bearish_signals > bullish_signals:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _assess_technical_risk(self, data: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, float]:
        """Assess technical risk factors"""
        try:
            volatility = data['Close'].pct_change().std() * 100
            atr = indicators.get('ATR', 0)
            
            return {
                'volatility_risk': min(1.0, volatility / 5.0),
                'atr_normalized': min(1.0, atr / data['Close'].iloc[-1]),
                'overall_risk': min(1.0, (volatility / 5.0 + atr / data['Close'].iloc[-1]) / 2)
            }
        except Exception:
            return {'overall_risk': 0.5}
    
    def _generate_technical_reasoning(self, indicators: Dict[str, float], direction: str) -> str:
        """Generate human-readable reasoning for technical prediction"""
        try:
            rsi = indicators.get('RSI', 50)
            macd = indicators.get('MACD', 0)
            bb_position = indicators.get('BB_Position', 0.5)
            
            reasoning = f"Technical analysis shows {direction} bias. "
            
            if rsi < 30:
                reasoning += "RSI indicates oversold conditions. "
            elif rsi > 70:
                reasoning += "RSI indicates overbought conditions. "
            
            if macd > 0:
                reasoning += "MACD suggests positive momentum. "
            else:
                reasoning += "MACD suggests negative momentum. "
            
            if bb_position > 0.8:
                reasoning += "Price near upper Bollinger Band resistance."
            elif bb_position < 0.2:
                reasoning += "Price near lower Bollinger Band support."
            
            return reasoning
            
        except Exception:
            return f"Technical analysis suggests {direction} market direction."
    
    def get_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence based on market conditions"""
        try:
            base_confidence = 0.7
            
            # Technical strategies work better in trending markets
            if market_conditions.market_regime == 'trending':
                base_confidence += 0.2
            elif market_conditions.market_regime == 'ranging':
                base_confidence -= 0.1
            elif market_conditions.market_regime == 'volatile':
                base_confidence -= 0.3
            
            # Adjust for trend strength
            trend_factor = market_conditions.trend_strength * 0.2
            base_confidence += trend_factor
            
            return min(0.95, max(0.1, base_confidence))
            
        except Exception:
            return 0.7
    
    def _store_prediction(self, result: StrategyResult):
        """Store prediction in database"""
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
            logger.error(f"❌ Failed to store prediction: {e}")

class SentimentStrategy(BaseStrategy):
    """
    📰 Sentiment Analysis Strategy
    Real-time news sentiment and social media analysis
    """
    
    def __init__(self):
        super().__init__("SentimentStrategy")
        self.sentiment_sources = ['news', 'social', 'vix']
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    def predict(self, data: pd.DataFrame, timeframe: str) -> StrategyResult:
        """Generate sentiment-based prediction"""
        try:
            # Gather sentiment data
            sentiment_data = self._gather_sentiment_data()
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(sentiment_data)
            
            # Prepare features
            features = self._prepare_sentiment_features(data, sentiment_data)
            
            # Generate prediction
            current_price = data['Close'].iloc[-1]
            sentiment_impact = self._calculate_sentiment_impact(sentiment_score, timeframe)
            predicted_price = current_price * (1 + sentiment_impact)
            
            # Calculate confidence
            confidence = self._calculate_sentiment_confidence(sentiment_data)
            
            # Determine direction
            direction = self._determine_sentiment_direction(sentiment_score)
            
            result = StrategyResult(
                strategy_name=self.name,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                timeframe=timeframe,
                technical_indicators={'sentiment_score': sentiment_score},
                reasoning=self._generate_sentiment_reasoning(sentiment_data, direction),
                risk_assessment={'sentiment_volatility': abs(sentiment_score)},
                timestamp=datetime.now()
            )
            
            self._store_prediction(result)
            
            logger.info(f"📰 Sentiment prediction: ${predicted_price:.2f} ({direction}, {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Sentiment strategy prediction failed: {e}")
            current_price = data['Close'].iloc[-1]
            return StrategyResult(
                strategy_name=self.name,
                predicted_price=current_price,
                confidence=0.1,
                direction='neutral',
                timeframe=timeframe,
                technical_indicators={},
                reasoning="Sentiment analysis unavailable",
                risk_assessment={'overall_risk': 0.7},
                timestamp=datetime.now()
            )
    
    def _gather_sentiment_data(self) -> Dict[str, float]:
        """Gather sentiment data from multiple sources"""
        try:
            sentiment_data = {}
            
            # Mock sentiment data (replace with real APIs)
            sentiment_data['news_sentiment'] = np.random.normal(0, 0.3)  # -1 to 1
            sentiment_data['social_sentiment'] = np.random.normal(0, 0.2)
            sentiment_data['fear_greed_index'] = np.random.uniform(20, 80)
            sentiment_data['vix_level'] = np.random.uniform(15, 35)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Sentiment data gathering failed: {e}")
            return {'news_sentiment': 0, 'social_sentiment': 0}
    
    def _calculate_sentiment_score(self, sentiment_data: Dict[str, float]) -> float:
        """Calculate overall sentiment score"""
        try:
            news_weight = 0.4
            social_weight = 0.3
            fear_greed_weight = 0.2
            vix_weight = 0.1
            
            news_sentiment = sentiment_data.get('news_sentiment', 0)
            social_sentiment = sentiment_data.get('social_sentiment', 0)
            fear_greed = (sentiment_data.get('fear_greed_index', 50) - 50) / 50  # Normalize to -1 to 1
            vix_sentiment = -(sentiment_data.get('vix_level', 25) - 25) / 25  # High VIX = negative sentiment
            
            overall_sentiment = (
                news_sentiment * news_weight +
                social_sentiment * social_weight +
                fear_greed * fear_greed_weight +
                vix_sentiment * vix_weight
            )
            
            return max(-1.0, min(1.0, overall_sentiment))
            
        except Exception:
            return 0.0
    
    def _prepare_sentiment_features(self, data: pd.DataFrame, sentiment_data: Dict[str, float]) -> pd.DataFrame:
        """Prepare sentiment-based features"""
        try:
            features = data.copy()
            
            # Add sentiment features
            for key, value in sentiment_data.items():
                features[key] = value
            
            # Add price momentum features
            features['price_momentum_5'] = features['Close'].pct_change(5)
            features['price_momentum_20'] = features['Close'].pct_change(20)
            
            return features.dropna()
            
        except Exception:
            return data
    
    def _calculate_sentiment_impact(self, sentiment_score: float, timeframe: str) -> float:
        """Calculate sentiment impact on price"""
        try:
            # Timeframe multipliers
            timeframe_multipliers = {
                '1h': 0.001,
                '4h': 0.003,
                '1d': 0.01,
                '1w': 0.03
            }
            
            base_multiplier = timeframe_multipliers.get(timeframe, 0.01)
            
            # Sentiment impact with diminishing returns
            impact = sentiment_score * base_multiplier * (1 + abs(sentiment_score) * 0.5)
            
            return max(-0.05, min(0.05, impact))  # Limit to ±5%
            
        except Exception:
            return 0.0
    
    def _calculate_sentiment_confidence(self, sentiment_data: Dict[str, float]) -> float:
        """Calculate confidence based on sentiment data quality"""
        try:
            # Check data availability
            available_sources = len([v for v in sentiment_data.values() if v is not None])
            max_sources = len(self.sentiment_sources)
            
            data_quality = available_sources / max_sources
            
            # Check sentiment strength
            overall_sentiment = abs(self._calculate_sentiment_score(sentiment_data))
            sentiment_strength = min(1.0, overall_sentiment * 2)
            
            confidence = (data_quality * 0.6 + sentiment_strength * 0.4)
            
            return max(0.1, min(0.9, confidence))
            
        except Exception:
            return 0.5
    
    def _determine_sentiment_direction(self, sentiment_score: float) -> str:
        """Determine direction based on sentiment score"""
        if sentiment_score > 0.1:
            return 'bullish'
        elif sentiment_score < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def _generate_sentiment_reasoning(self, sentiment_data: Dict[str, float], direction: str) -> str:
        """Generate reasoning for sentiment prediction"""
        try:
            news_sentiment = sentiment_data.get('news_sentiment', 0)
            social_sentiment = sentiment_data.get('social_sentiment', 0)
            
            reasoning = f"Sentiment analysis indicates {direction} market bias. "
            
            if news_sentiment > 0.2:
                reasoning += "News sentiment is positive. "
            elif news_sentiment < -0.2:
                reasoning += "News sentiment is negative. "
            
            if social_sentiment > 0.2:
                reasoning += "Social media sentiment is bullish. "
            elif social_sentiment < -0.2:
                reasoning += "Social media sentiment is bearish. "
            
            return reasoning
            
        except Exception:
            return f"Sentiment analysis suggests {direction} market direction."
    
    def get_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence based on market conditions"""
        try:
            base_confidence = 0.6
            
            # Sentiment works better in crisis or volatile markets
            if market_conditions.market_regime == 'crisis':
                base_confidence += 0.3
            elif market_conditions.market_regime == 'volatile':
                base_confidence += 0.2
            elif market_conditions.market_regime == 'trending':
                base_confidence -= 0.1
            
            # Adjust for sentiment score strength
            sentiment_strength = abs(market_conditions.sentiment_score)
            base_confidence += sentiment_strength * 0.2
            
            return min(0.95, max(0.1, base_confidence))
            
        except Exception:
            return 0.6
    
    def _store_prediction(self, result: StrategyResult):
        """Store sentiment prediction in database"""
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
            logger.error(f"❌ Failed to store sentiment prediction: {e}")

# Continue with remaining strategies and ensemble system in next part...
