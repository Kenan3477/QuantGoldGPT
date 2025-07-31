"""
🏛️ ENSEMBLE VOTING SYSTEM & META-LEARNING ENGINE
================================================

Advanced ensemble system with dynamic weighting and meta-learning optimization
Implements institutional-grade ML prediction aggregation and continuous optimization

Components:
- EnsembleVotingSystem: Dynamic strategy weighting and prediction aggregation
- MetaLearningEngine: Continuous optimization and adaptation
- MarketRegimeDetector: Market condition analysis
- PerformanceMonitor: Real-time strategy performance tracking

Author: GoldGPT AI System  
Created: July 23, 2025
"""

import asyncio
import numpy as np
import pandas as pd
import sqlite3
import logging
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings

# Import strategy classes
from advanced_ensemble_ml_system import (
    BaseStrategy, TechnicalStrategy, SentimentStrategy, StrategyResult, 
    MarketConditions, EnsemblePrediction, logger
)
from advanced_strategies_part2 import MacroStrategy, PatternStrategy, MomentumStrategy

warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    🎯 Market Regime Detection System
    Analyzes market conditions to optimize strategy selection
    """
    
    def __init__(self):
        self.volatility_threshold_low = 0.01
        self.volatility_threshold_high = 0.04
        self.trend_threshold = 0.02
        
    def detect_market_regime(self, data: pd.DataFrame) -> MarketConditions:
        """Detect current market regime and conditions"""
        try:
            close = data['Close'].values
            volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(close))
            
            # Calculate market metrics
            volatility = self._calculate_volatility(close)
            trend_strength = self._calculate_trend_strength(close)
            volume_profile = self._calculate_volume_profile(volume)
            sentiment_score = self._estimate_sentiment(close)
            
            # Determine market regime
            regime = self._classify_regime(volatility, trend_strength, volume_profile)
            
            # Gather macro environment (simplified)
            macro_environment = {
                'interest_rate_environment': 'high',  # Current environment
                'inflation_regime': 'moderate',
                'economic_cycle': 'expansion'
            }
            
            market_conditions = MarketConditions(
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                sentiment_score=sentiment_score,
                macro_environment=macro_environment,
                market_regime=regime
            )
            
            logger.info(f"🎯 Market regime detected: {regime} (vol: {volatility:.3f}, trend: {trend_strength:.3f})")
            return market_conditions
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            # Return default conditions
            return MarketConditions(
                volatility=0.02,
                trend_strength=0.5,
                volume_profile=1.0,
                sentiment_score=0.0,
                macro_environment={},
                market_regime='ranging'
            )
    
    def _calculate_volatility(self, close: np.ndarray) -> float:
        """Calculate price volatility"""
        try:
            if len(close) < 20:
                return 0.02
            
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return min(1.0, max(0.001, volatility))
            
        except Exception:
            return 0.02
    
    def _calculate_trend_strength(self, close: np.ndarray) -> float:
        """Calculate trend strength"""
        try:
            if len(close) < 20:
                return 0.5
            
            # Linear regression slope
            x = np.arange(len(close[-20:]))
            slope, _, r_value, _, _ = stats.linregress(x, close[-20:])
            
            # Normalize slope
            normalized_slope = slope / close[-1]
            trend_strength = abs(normalized_slope) * 100
            
            return min(1.0, max(0.0, trend_strength))
            
        except Exception:
            return 0.5
    
    def _calculate_volume_profile(self, volume: np.ndarray) -> float:
        """Calculate volume profile"""
        try:
            if len(volume) < 20:
                return 1.0
            
            recent_avg = np.mean(volume[-10:])
            historical_avg = np.mean(volume[-30:-10]) if len(volume) >= 30 else np.mean(volume[:-10])
            
            if historical_avg > 0:
                volume_ratio = recent_avg / historical_avg
                return min(3.0, max(0.1, volume_ratio))
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _estimate_sentiment(self, close: np.ndarray) -> float:
        """Estimate market sentiment from price action"""
        try:
            if len(close) < 10:
                return 0.0
            
            # Recent price momentum
            recent_change = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
            
            # Price position relative to recent range
            recent_high = np.max(close[-20:]) if len(close) >= 20 else close[-1]
            recent_low = np.min(close[-20:]) if len(close) >= 20 else close[-1]
            
            if recent_high != recent_low:
                price_position = (close[-1] - recent_low) / (recent_high - recent_low)
                position_sentiment = (price_position - 0.5) * 2  # -1 to 1
            else:
                position_sentiment = 0
            
            # Combine momentum and position
            sentiment = (recent_change * 5 + position_sentiment) / 2
            
            return max(-1.0, min(1.0, sentiment))
            
        except Exception:
            return 0.0
    
    def _classify_regime(self, volatility: float, trend_strength: float, volume_profile: float) -> str:
        """Classify market regime based on metrics"""
        try:
            # Crisis conditions
            if volatility > 0.06 and volume_profile > 2.0:
                return 'crisis'
            
            # High volatility
            elif volatility > self.volatility_threshold_high:
                return 'volatile'
            
            # Strong trend
            elif trend_strength > 0.7:
                return 'trending'
            
            # Low volatility, weak trend
            elif volatility < self.volatility_threshold_low and trend_strength < 0.3:
                return 'ranging'
            
            # Default
            else:
                return 'mixed'
                
        except Exception:
            return 'ranging'

class PerformanceMonitor:
    """
    📊 Real-time Strategy Performance Monitor
    Tracks and analyzes strategy performance metrics
    """
    
    def __init__(self, db_path: str = "advanced_ensemble_performance.db"):
        self.db_path = db_path
        self.performance_cache = {}
        self.initialize_monitoring()
    
    def initialize_monitoring(self):
        """Initialize performance monitoring database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ensemble_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        ensemble_prediction REAL NOT NULL,
                        actual_price REAL,
                        accuracy_score REAL,
                        strategy_weights TEXT NOT NULL,
                        market_regime TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_rankings (
                        strategy_name TEXT PRIMARY KEY,
                        total_predictions INTEGER DEFAULT 0,
                        accuracy_score REAL DEFAULT 0.5,
                        sharpe_ratio REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.5,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                logger.info("📊 Performance monitoring initialized")
                
        except Exception as e:
            logger.error(f"❌ Performance monitoring initialization failed: {e}")
    
    def update_strategy_performance(self, strategy_name: str, accuracy: float, confidence: float):
        """Update individual strategy performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current stats
                cursor.execute("SELECT * FROM strategy_rankings WHERE strategy_name = ?", (strategy_name,))
                current = cursor.fetchone()
                
                if current:
                    # Update existing record
                    total_predictions = current[1] + 1
                    # Exponential moving average for accuracy
                    alpha = 0.1  # Learning rate
                    new_accuracy = (1 - alpha) * current[2] + alpha * accuracy
                    new_confidence = (1 - alpha) * current[5] + alpha * confidence
                    
                    cursor.execute("""
                        UPDATE strategy_rankings 
                        SET total_predictions = ?, accuracy_score = ?, avg_confidence = ?, last_updated = ?
                        WHERE strategy_name = ?
                    """, (total_predictions, new_accuracy, new_confidence, datetime.now(), strategy_name))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO strategy_rankings 
                        (strategy_name, total_predictions, accuracy_score, avg_confidence)
                        VALUES (?, 1, ?, ?)
                    """, (strategy_name, accuracy, confidence))
                
                logger.info(f"📊 {strategy_name} performance updated: {accuracy:.3f} accuracy")
                
        except Exception as e:
            logger.error(f"❌ Strategy performance update failed: {e}")
    
    def get_strategy_rankings(self) -> Dict[str, Dict[str, float]]:
        """Get current strategy performance rankings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT strategy_name, accuracy_score, avg_confidence, total_predictions 
                    FROM strategy_rankings 
                    ORDER BY accuracy_score DESC
                """)
                
                rankings = {}
                for row in cursor.fetchall():
                    rankings[row[0]] = {
                        'accuracy': row[1],
                        'confidence': row[2],
                        'predictions': row[3],
                        'score': row[1] * row[2]  # Combined score
                    }
                
                return rankings
                
        except Exception as e:
            logger.error(f"❌ Failed to get strategy rankings: {e}")
            return {}

class EnsembleVotingSystem:
    """
    🗳️ Advanced Ensemble Voting System
    Dynamic strategy weighting and prediction aggregation with performance-based optimization
    """
    
    def __init__(self, db_path: str = "advanced_ensemble_performance.db"):
        self.db_path = db_path
        self.strategies = self._initialize_strategies()
        self.regime_detector = MarketRegimeDetector()
        self.performance_monitor = PerformanceMonitor(db_path)
        self.meta_learner = None  # Will be set by MetaLearningEngine
        
        # Ensemble parameters
        self.min_confidence_threshold = 0.3
        self.diversity_bonus = 0.1
        self.performance_window = 50
        
        logger.info("🗳️ Ensemble Voting System initialized")
    
    def _initialize_strategies(self) -> Dict[str, BaseStrategy]:
        """Initialize all prediction strategies"""
        try:
            strategies = {
                'technical': TechnicalStrategy(),
                'sentiment': SentimentStrategy(), 
                'macro': MacroStrategy(),
                'pattern': PatternStrategy(),
                'momentum': MomentumStrategy()
            }
            
            logger.info(f"✅ Initialized {len(strategies)} strategies")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ Strategy initialization failed: {e}")
            return {}
    
    async def generate_ensemble_prediction(self, data: pd.DataFrame, timeframe: str) -> EnsemblePrediction:
        """Generate ensemble prediction from all strategies"""
        try:
            logger.info(f"🗳️ Generating ensemble prediction for {timeframe}")
            
            # Detect market conditions
            market_conditions = self.regime_detector.detect_market_regime(data)
            
            # Generate predictions from all strategies
            strategy_results = await self._generate_strategy_predictions(data, timeframe, market_conditions)
            
            # Calculate dynamic weights
            strategy_weights = self._calculate_dynamic_weights(strategy_results, market_conditions)
            
            # Aggregate predictions
            ensemble_prediction = self._aggregate_predictions(strategy_results, strategy_weights, market_conditions)
            
            # Store ensemble prediction
            self._store_ensemble_prediction(ensemble_prediction)
            
            logger.info(f"🎯 Ensemble prediction: ${ensemble_prediction.predicted_price:.2f} "
                       f"({ensemble_prediction.direction}, {ensemble_prediction.confidence:.2%})")
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            # Return fallback prediction
            current_price = data['Close'].iloc[-1]
            return EnsemblePrediction(
                predicted_price=current_price,
                confidence=0.1,
                direction='neutral',
                contributing_strategies=[],
                ensemble_weights={},
                risk_metrics={'overall_risk': 0.8},
                market_conditions=MarketConditions(0.02, 0.5, 1.0, 0.0, {}, 'ranging'),
                timestamp=datetime.now()
            )
    
    async def _generate_strategy_predictions(self, data: pd.DataFrame, timeframe: str, 
                                           market_conditions: MarketConditions) -> List[StrategyResult]:
        """Generate predictions from all strategies concurrently"""
        try:
            strategy_results = []
            
            # Use ThreadPoolExecutor for CPU-bound strategy computations
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                
                for name, strategy in self.strategies.items():
                    future = executor.submit(strategy.predict, data, timeframe)
                    futures[future] = name
                
                # Collect results with timeout
                for future in as_completed(futures, timeout=30):
                    strategy_name = futures[future]
                    try:
                        result = future.result()
                        if result.confidence >= self.min_confidence_threshold:
                            strategy_results.append(result)
                            logger.info(f"✅ {strategy_name}: ${result.predicted_price:.2f} ({result.confidence:.2%})")
                        else:
                            logger.warning(f"⚠️ {strategy_name}: Low confidence ({result.confidence:.2%})")
                    except Exception as e:
                        logger.error(f"❌ {strategy_name} prediction failed: {e}")
            
            return strategy_results
            
        except Exception as e:
            logger.error(f"❌ Strategy prediction generation failed: {e}")
            return []
    
    def _calculate_dynamic_weights(self, strategy_results: List[StrategyResult], 
                                 market_conditions: MarketConditions) -> Dict[str, float]:
        """Calculate dynamic weights based on performance and market conditions"""
        try:
            if not strategy_results:
                return {}
            
            weights = {}
            total_weight = 0.0
            
            # Get strategy performance rankings
            rankings = self.performance_monitor.get_strategy_rankings()
            
            for result in strategy_results:
                strategy_name = result.strategy_name
                
                # Base weight from current performance
                base_weight = 0.2  # Default equal weight
                
                # Performance-based adjustment
                if strategy_name in rankings:
                    performance_score = rankings[strategy_name]['score']
                    base_weight = max(0.05, min(0.5, performance_score))
                
                # Market condition adjustment
                strategy_obj = self.strategies.get(strategy_name.lower().replace('strategy', ''))
                if strategy_obj:
                    condition_confidence = strategy_obj.get_confidence(market_conditions)
                    base_weight *= condition_confidence
                
                # Confidence adjustment
                confidence_factor = result.confidence ** 0.5  # Square root to reduce extreme values
                base_weight *= confidence_factor
                
                # Diversity bonus (encourage diverse predictions)
                diversity_factor = self._calculate_diversity_bonus(result, strategy_results)
                base_weight *= (1 + diversity_factor * self.diversity_bonus)
                
                weights[strategy_name] = base_weight
                total_weight += base_weight
            
            # Normalize weights
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}
            
            logger.info(f"🎯 Dynamic weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"❌ Dynamic weight calculation failed: {e}")
            # Return equal weights as fallback
            num_strategies = len(strategy_results)
            return {result.strategy_name: 1.0 / num_strategies for result in strategy_results}
    
    def _calculate_diversity_bonus(self, target_result: StrategyResult, all_results: List[StrategyResult]) -> float:
        """Calculate diversity bonus for unique predictions"""
        try:
            if len(all_results) <= 1:
                return 0.0
            
            target_price = target_result.predicted_price
            other_prices = [r.predicted_price for r in all_results if r.strategy_name != target_result.strategy_name]
            
            if not other_prices:
                return 0.0
            
            # Calculate how different this prediction is from others
            avg_other_price = np.mean(other_prices)
            price_difference = abs(target_price - avg_other_price) / avg_other_price
            
            # Bonus for being different (but not too different)
            diversity_bonus = min(0.5, price_difference * 10)  # Cap at 50% bonus
            
            return diversity_bonus
            
        except Exception:
            return 0.0
    
    def _aggregate_predictions(self, strategy_results: List[StrategyResult], 
                             strategy_weights: Dict[str, float], 
                             market_conditions: MarketConditions) -> EnsemblePrediction:
        """Aggregate strategy predictions into ensemble prediction"""
        try:
            if not strategy_results:
                raise ValueError("No strategy results to aggregate")
            
            # Weighted price prediction
            weighted_prices = []
            weighted_confidences = []
            total_weight = 0.0
            
            contributing_strategies = []
            
            for result in strategy_results:
                weight = strategy_weights.get(result.strategy_name, 0.0)
                if weight > 0:
                    weighted_prices.append(result.predicted_price * weight)
                    weighted_confidences.append(result.confidence * weight)
                    total_weight += weight
                    
                    contributing_strategies.append({
                        'strategy': result.strategy_name,
                        'prediction': result.predicted_price,
                        'confidence': result.confidence,
                        'weight': weight,
                        'direction': result.direction
                    })
            
            if not weighted_prices:
                raise ValueError("No valid weighted predictions")
            
            # Calculate ensemble prediction
            ensemble_price = sum(weighted_prices)
            ensemble_confidence = sum(weighted_confidences)
            
            # Determine ensemble direction
            bullish_weight = sum(w['weight'] for w in contributing_strategies if w['direction'] == 'bullish')
            bearish_weight = sum(w['weight'] for w in contributing_strategies if w['direction'] == 'bearish')
            
            if bullish_weight > bearish_weight:
                ensemble_direction = 'bullish'
            elif bearish_weight > bullish_weight:
                ensemble_direction = 'bearish'
            else:
                ensemble_direction = 'neutral'
            
            # Calculate risk metrics
            risk_metrics = self._calculate_ensemble_risk(strategy_results, strategy_weights, market_conditions)
            
            ensemble_prediction = EnsemblePrediction(
                predicted_price=ensemble_price,
                confidence=min(0.95, ensemble_confidence),
                direction=ensemble_direction,
                contributing_strategies=contributing_strategies,
                ensemble_weights=strategy_weights,
                risk_metrics=risk_metrics,
                market_conditions=market_conditions,
                timestamp=datetime.now()
            )
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction aggregation failed: {e}")
            # Return fallback prediction
            avg_price = np.mean([r.predicted_price for r in strategy_results])
            return EnsemblePrediction(
                predicted_price=avg_price,
                confidence=0.3,
                direction='neutral',
                contributing_strategies=[],
                ensemble_weights={},
                risk_metrics={'overall_risk': 0.7},
                market_conditions=market_conditions,
                timestamp=datetime.now()
            )
    
    def _calculate_ensemble_risk(self, strategy_results: List[StrategyResult], 
                               strategy_weights: Dict[str, float], 
                               market_conditions: MarketConditions) -> Dict[str, float]:
        """Calculate ensemble risk metrics"""
        try:
            risk_metrics = {}
            
            # Prediction variance risk
            predictions = [r.predicted_price for r in strategy_results]
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                avg_prediction = np.mean(predictions)
                variance_risk = prediction_std / avg_prediction if avg_prediction > 0 else 0.1
                risk_metrics['prediction_variance'] = min(1.0, variance_risk * 10)
            else:
                risk_metrics['prediction_variance'] = 0.5
            
            # Confidence risk (inverse of ensemble confidence)
            avg_confidence = np.mean([r.confidence for r in strategy_results])
            risk_metrics['confidence_risk'] = 1.0 - avg_confidence
            
            # Market regime risk
            regime_risk_map = {
                'crisis': 0.9,
                'volatile': 0.7,
                'trending': 0.3,
                'ranging': 0.4,
                'mixed': 0.5
            }
            risk_metrics['regime_risk'] = regime_risk_map.get(market_conditions.market_regime, 0.5)
            
            # Overall risk (weighted average)
            risk_metrics['overall_risk'] = (
                risk_metrics['prediction_variance'] * 0.4 +
                risk_metrics['confidence_risk'] * 0.3 +
                risk_metrics['regime_risk'] * 0.3
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ Risk calculation failed: {e}")
            return {'overall_risk': 0.6}
    
    def _store_ensemble_prediction(self, prediction: EnsemblePrediction):
        """Store ensemble prediction in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ensemble_performance 
                    (timestamp, ensemble_prediction, strategy_weights, market_regime, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    prediction.timestamp,
                    prediction.predicted_price,
                    json.dumps(prediction.ensemble_weights),
                    prediction.market_conditions.market_regime,
                    prediction.confidence
                ))
                
        except Exception as e:
            logger.error(f"❌ Failed to store ensemble prediction: {e}")
    
    def update_ensemble_performance(self, prediction_timestamp: datetime, actual_price: float):
        """Update ensemble performance with actual outcome"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find the prediction
                cursor.execute("""
                    SELECT ensemble_prediction, confidence FROM ensemble_performance 
                    WHERE timestamp = ?
                """, (prediction_timestamp,))
                
                result = cursor.fetchone()
                if result:
                    predicted_price, confidence = result
                    accuracy = 1.0 - abs(actual_price - predicted_price) / actual_price
                    
                    # Update ensemble record
                    cursor.execute("""
                        UPDATE ensemble_performance 
                        SET actual_price = ?, accuracy_score = ?
                        WHERE timestamp = ?
                    """, (actual_price, accuracy, prediction_timestamp))
                    
                    logger.info(f"📊 Ensemble performance updated: {accuracy:.3f} accuracy")
                    
        except Exception as e:
            logger.error(f"❌ Ensemble performance update failed: {e}")

class MetaLearningEngine:
    """
    🧠 Meta-Learning Engine for Continuous Optimization
    Implements hyperparameter tuning, strategy optimization, and system evolution
    """
    
    def __init__(self, ensemble_system: EnsembleVotingSystem):
        self.ensemble_system = ensemble_system
        self.optimization_history = []
        self.current_parameters = self._get_default_parameters()
        
        # Set reference in ensemble system
        ensemble_system.meta_learner = self
        
        logger.info("🧠 Meta-Learning Engine initialized")
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default system parameters"""
        return {
            'min_confidence_threshold': 0.3,
            'diversity_bonus': 0.1,
            'performance_window': 50,
            'learning_rate': 0.1,
            'weight_decay': 0.95,
            'risk_tolerance': 0.7
        }
    
    async def optimize_system(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize ensemble system parameters"""
        try:
            logger.info("🧠 Starting meta-learning optimization...")
            
            # Define parameter search space
            param_bounds = {
                'min_confidence_threshold': (0.1, 0.6),
                'diversity_bonus': (0.0, 0.3),
                'learning_rate': (0.05, 0.2),
                'weight_decay': (0.9, 0.99)
            }
            
            # Objective function for optimization
            def objective_function(params):
                return self._evaluate_parameter_set(params, historical_data)
            
            # Use differential evolution for global optimization
            param_values = list(param_bounds.values())
            
            result = differential_evolution(
                lambda x: objective_function(dict(zip(param_bounds.keys(), x))),
                param_values,
                maxiter=20,
                popsize=10,
                seed=42
            )
            
            # Update system parameters
            if result.success:
                new_params = dict(zip(param_bounds.keys(), result.x))
                self._update_system_parameters(new_params)
                
                logger.info(f"✅ Optimization completed: {new_params}")
                return new_params
            else:
                logger.warning("⚠️ Optimization failed, keeping current parameters")
                return self.current_parameters
                
        except Exception as e:
            logger.error(f"❌ Meta-learning optimization failed: {e}")
            return self.current_parameters
    
    def _evaluate_parameter_set(self, params: Dict[str, float], data: pd.DataFrame) -> float:
        """Evaluate parameter set using cross-validation"""
        try:
            # Temporarily update system parameters
            original_params = self.current_parameters.copy()
            self._update_system_parameters(params)
            
            # Evaluate on historical data
            scores = []
            window_size = 100
            
            for i in range(window_size, len(data), 20):  # Every 20 periods
                train_data = data.iloc[i-window_size:i]
                
                # Generate prediction (simplified for optimization)
                try:
                    # This would be a simplified version of ensemble prediction
                    # for optimization purposes
                    score = self._quick_evaluation(train_data)
                    scores.append(score)
                except:
                    scores.append(0.0)
            
            # Restore original parameters
            self._update_system_parameters(original_params)
            
            # Return negative average score (minimize)
            return -np.mean(scores) if scores else 1.0
            
        except Exception as e:
            logger.error(f"❌ Parameter evaluation failed: {e}")
            return 1.0  # Poor score
    
    def _quick_evaluation(self, data: pd.DataFrame) -> float:
        """Quick evaluation for optimization (simplified)"""
        try:
            # Simplified evaluation metric
            # In practice, this would run a fast version of the ensemble
            
            volatility = np.std(data['Close'].pct_change()) * 100
            trend_strength = abs(np.polyfit(range(len(data)), data['Close'].values, 1)[0])
            
            # Simple score based on volatility and trend detection
            score = min(1.0, trend_strength / volatility) if volatility > 0 else 0.5
            
            return score
            
        except Exception:
            return 0.0
    
    def _update_system_parameters(self, new_params: Dict[str, Any]):
        """Update ensemble system parameters"""
        try:
            self.current_parameters.update(new_params)
            
            # Update ensemble system
            if hasattr(self.ensemble_system, 'min_confidence_threshold'):
                self.ensemble_system.min_confidence_threshold = new_params.get(
                    'min_confidence_threshold', 0.3
                )
            if hasattr(self.ensemble_system, 'diversity_bonus'):
                self.ensemble_system.diversity_bonus = new_params.get('diversity_bonus', 0.1)
            
            logger.info(f"🎯 System parameters updated: {new_params}")
            
        except Exception as e:
            logger.error(f"❌ Parameter update failed: {e}")
    
    def analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze individual strategy performance for optimization"""
        try:
            rankings = self.ensemble_system.performance_monitor.get_strategy_rankings()
            
            analysis = {
                'top_performer': max(rankings.items(), key=lambda x: x[1]['accuracy'])[0] if rankings else None,
                'worst_performer': min(rankings.items(), key=lambda x: x[1]['accuracy'])[0] if rankings else None,
                'avg_accuracy': np.mean([v['accuracy'] for v in rankings.values()]) if rankings else 0.5,
                'strategy_diversity': len(rankings),
                'recommendations': self._generate_recommendations(rankings)
            }
            
            logger.info(f"📊 Strategy analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Strategy performance analysis failed: {e}")
            return {}
    
    def _generate_recommendations(self, rankings: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        try:
            if not rankings:
                recommendations.append("Insufficient data for recommendations")
                return recommendations
            
            # Analyze performance patterns
            accuracies = [v['accuracy'] for v in rankings.values()]
            avg_accuracy = np.mean(accuracies)
            
            if avg_accuracy < 0.6:
                recommendations.append("Overall accuracy below target - consider parameter tuning")
            
            # Check for underperforming strategies
            for strategy, metrics in rankings.items():
                if metrics['accuracy'] < 0.4:
                    recommendations.append(f"Consider retraining {strategy} - low accuracy")
                if metrics['predictions'] < 10:
                    recommendations.append(f"{strategy} needs more data for reliable assessment")
            
            # Check diversity
            if len(rankings) < 3:
                recommendations.append("Consider adding more strategy diversity")
            
            return recommendations
            
        except Exception:
            return ["Analysis failed - manual review recommended"]

# Advanced ML System Integration Function
async def create_advanced_ml_system(data_source_path: str = None) -> EnsembleVotingSystem:
    """
    🚀 Create and initialize the complete advanced ML system
    """
    try:
        logger.info("🚀 Creating Advanced Multi-Strategy ML System...")
        
        # Create ensemble system
        ensemble_system = EnsembleVotingSystem()
        
        # Create meta-learning engine
        meta_learner = MetaLearningEngine(ensemble_system)
        
        # If historical data available, run initial optimization
        if data_source_path:
            try:
                historical_data = pd.read_csv(data_source_path)
                await meta_learner.optimize_system(historical_data)
            except Exception as e:
                logger.warning(f"⚠️ Initial optimization skipped: {e}")
        
        logger.info("✅ Advanced ML System created successfully")
        return ensemble_system
        
    except Exception as e:
        logger.error(f"❌ Advanced ML System creation failed: {e}")
        raise

# Export main classes and functions
__all__ = [
    'EnsembleVotingSystem', 'MetaLearningEngine', 'MarketRegimeDetector', 
    'PerformanceMonitor', 'create_advanced_ml_system'
]
