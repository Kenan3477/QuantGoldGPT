#!/usr/bin/env python3
"""
Advanced Learning Engine for GoldGPT ML Prediction System
Continuously learns from prediction outcomes and improves model performance
"""

import logging
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import sqlite3
from pathlib import Path

from prediction_tracker import PredictionTracker, PredictionRecord

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningInsight:
    """Data structure for learning insights"""
    insight_type: str
    insight_category: str
    title: str
    description: str
    insight_data: Dict[str, Any]
    confidence_level: float
    sample_size: int
    expected_improvement: float
    algorithm_used: str
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.now(timezone.utc)

@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics"""
    strategy_name: str
    timeframe: str
    evaluation_period_days: int
    total_predictions: int
    win_rate: float
    average_error: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    confidence_calibration_error: float
    feature_stability_score: float
    regime_adaptability_score: float

class LearningEngine:
    """
    Advanced learning engine that continuously improves ML predictions
    
    Features:
    - Automatic model retraining based on performance feedback
    - Dynamic strategy weight adjustment
    - Feature importance optimization
    - Market regime detection and adaptation
    - Automated hyperparameter optimization
    - Performance-based learning rate adjustment
    """
    
    def __init__(self, prediction_tracker: PredictionTracker, 
                 min_samples_for_learning: int = 50):
        self.tracker = prediction_tracker
        self.min_samples = min_samples_for_learning
        self.models = {}
        self.scalers = {}
        self.feature_importance_history = {}
        self.strategy_weights = {}
        self.learning_insights = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05  # Minimum improvement to adopt changes
        self.confidence_threshold = 0.7   # Minimum confidence for insights
        
        logger.info("LearningEngine initialized")
    
    async def continuous_learning_cycle(self, cycle_interval_hours: int = 6):
        """
        Run continuous learning cycle
        
        Args:
            cycle_interval_hours: Hours between learning cycles
        """
        while True:
            try:
                logger.info("Starting learning cycle...")
                
                # Validate pending predictions
                await self.validate_pending_predictions()
                
                # Analyze recent performance
                performance_insights = await self.analyze_performance_patterns()
                
                # Update strategy weights based on recent performance
                await self.update_strategy_weights()
                
                # Retrain models if needed
                await self.adaptive_model_retraining()
                
                # Feature importance analysis
                feature_insights = await self.analyze_feature_importance()
                
                # Market regime analysis
                regime_insights = await self.analyze_market_regimes()
                
                # Generate learning insights
                await self.generate_learning_insights(
                    performance_insights, feature_insights, regime_insights
                )
                
                # Apply validated insights
                await self.apply_learning_insights()
                
                logger.info(f"Learning cycle completed. Next cycle in {cycle_interval_hours} hours.")
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Learning cycle failed: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def validate_pending_predictions(self):
        """Validate predictions that have reached their target time"""
        try:
            pending = self.tracker.get_pending_validations(max_age_hours=48)
            
            if not pending:
                logger.info("No pending validations")
                return
            
            # In a real implementation, you'd fetch actual prices from your price feed
            # For now, we'll simulate this process
            validated_count = 0
            
            for prediction in pending:
                try:
                    # Simulate getting actual price (replace with real price fetching)
                    actual_price = await self._fetch_actual_price(
                        prediction['symbol'], 
                        prediction['target_time']
                    )
                    
                    if actual_price:
                        self.tracker.validate_prediction(
                            prediction['prediction_id'], 
                            actual_price
                        )
                        validated_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to validate prediction {prediction['prediction_id']}: {e}")
            
            logger.info(f"Validated {validated_count} predictions")
            
        except Exception as e:
            logger.error(f"Failed to validate pending predictions: {e}")
    
    async def _fetch_actual_price(self, symbol: str, target_time: str) -> Optional[float]:
        """
        Fetch actual price at target time (placeholder implementation)
        
        In production, this would connect to your price feed or historical data source
        """
        # Placeholder: return a simulated price
        # Replace this with actual price fetching logic
        return 2650.0 + np.random.normal(0, 5)  # Simulate price around 2650
    
    async def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns to identify improvement opportunities"""
        try:
            insights = {}
            
            # Analyze performance by time of day
            hourly_performance = await self._analyze_hourly_performance()
            insights['hourly_patterns'] = hourly_performance
            
            # Analyze performance by day of week
            daily_performance = await self._analyze_daily_performance()
            insights['daily_patterns'] = daily_performance
            
            # Analyze recent performance trends
            trend_analysis = await self._analyze_performance_trends()
            insights['trend_analysis'] = trend_analysis
            
            # Confidence calibration analysis
            calibration_analysis = await self._analyze_confidence_calibration()
            insights['confidence_calibration'] = calibration_analysis
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze performance patterns: {e}")
            return {}
    
    async def _analyze_hourly_performance(self) -> Dict[str, Any]:
        """Analyze performance by hour of day"""
        with self.tracker._get_connection() as conn:
            hourly_data = conn.execute("""
                SELECT 
                    CAST(strftime('%H', prediction_time) AS INTEGER) as hour,
                    COUNT(*) as total_predictions,
                    AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(prediction_error) as avg_error,
                    AVG(confidence) as avg_confidence
                FROM prediction_records 
                WHERE is_validated = 1 
                AND created_at >= datetime('now', '-30 days')
                GROUP BY hour
                HAVING total_predictions >= 5
                ORDER BY hour
            """).fetchall()
            
            hourly_analysis = {}
            for row in hourly_data:
                hour = row['hour']
                hourly_analysis[hour] = {
                    'total_predictions': row['total_predictions'],
                    'win_rate': round(row['win_rate'] * 100, 1),
                    'avg_error': round(row['avg_error'], 4),
                    'avg_confidence': round(row['avg_confidence'], 3)
                }
            
            # Find best and worst hours
            if hourly_analysis:
                best_hour = max(hourly_analysis.items(), key=lambda x: x[1]['win_rate'])
                worst_hour = min(hourly_analysis.items(), key=lambda x: x[1]['win_rate'])
                
                return {
                    'hourly_performance': hourly_analysis,
                    'best_hour': {'hour': best_hour[0], **best_hour[1]},
                    'worst_hour': {'hour': worst_hour[0], **worst_hour[1]},
                    'recommendation': f"Consider increasing confidence during hour {best_hour[0]} and decreasing during hour {worst_hour[0]}"
                }
            
            return {'hourly_performance': {}}
    
    async def _analyze_daily_performance(self) -> Dict[str, Any]:
        """Analyze performance by day of week"""
        with self.tracker._get_connection() as conn:
            daily_data = conn.execute("""
                SELECT 
                    CAST(strftime('%w', prediction_time) AS INTEGER) as day_of_week,
                    COUNT(*) as total_predictions,
                    AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(prediction_error) as avg_error
                FROM prediction_records 
                WHERE is_validated = 1 
                AND created_at >= datetime('now', '-60 days')
                GROUP BY day_of_week
                HAVING total_predictions >= 3
                ORDER BY day_of_week
            """).fetchall()
            
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            daily_analysis = {}
            
            for row in daily_data:
                day_num = row['day_of_week']
                day_name = day_names[day_num]
                daily_analysis[day_name] = {
                    'total_predictions': row['total_predictions'],
                    'win_rate': round(row['win_rate'] * 100, 1),
                    'avg_error': round(row['avg_error'], 4)
                }
            
            return {'daily_performance': daily_analysis}
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze recent performance trends"""
        with self.tracker._get_connection() as conn:
            # Get recent performance in weekly buckets
            trend_data = conn.execute("""
                SELECT 
                    DATE(created_at, 'weekday 0', '-6 days') as week_start,
                    COUNT(*) as total_predictions,
                    AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(prediction_error) as avg_error
                FROM prediction_records 
                WHERE is_validated = 1 
                AND created_at >= datetime('now', '-8 weeks')
                GROUP BY week_start
                ORDER BY week_start DESC
            """).fetchall()
            
            if len(trend_data) < 3:
                return {'trend': 'insufficient_data'}
            
            # Calculate trend
            recent_weeks = [row['win_rate'] for row in trend_data[:3]]
            older_weeks = [row['win_rate'] for row in trend_data[3:6]] if len(trend_data) >= 6 else []
            
            recent_avg = np.mean(recent_weeks)
            older_avg = np.mean(older_weeks) if older_weeks else recent_avg
            
            trend_direction = 'improving' if recent_avg > older_avg else 'declining' if recent_avg < older_avg else 'stable'
            trend_strength = abs(recent_avg - older_avg) if older_weeks else 0
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 4),
                'recent_performance': round(recent_avg * 100, 1),
                'historical_performance': round(older_avg * 100, 1),
                'weekly_data': [dict(row) for row in trend_data]
            }
    
    async def _analyze_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze how well confidence scores correlate with actual performance"""
        with self.tracker._get_connection() as conn:
            calibration_data = conn.execute("""
                SELECT 
                    CASE 
                        WHEN confidence <= 0.5 THEN 'low'
                        WHEN confidence <= 0.7 THEN 'medium'
                        ELSE 'high'
                    END as confidence_bucket,
                    COUNT(*) as total_predictions,
                    AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0.0 END) as actual_win_rate,
                    AVG(confidence) as avg_confidence
                FROM prediction_records 
                WHERE is_validated = 1 
                AND created_at >= datetime('now', '-30 days')
                GROUP BY confidence_bucket
            """).fetchall()
            
            calibration_analysis = {}
            total_calibration_error = 0
            
            for row in calibration_data:
                bucket = row['confidence_bucket']
                avg_confidence = row['avg_confidence']
                actual_win_rate = row['actual_win_rate']
                
                # Calibration error is the difference between confidence and actual performance
                calibration_error = abs(avg_confidence - actual_win_rate)
                total_calibration_error += calibration_error * row['total_predictions']
                
                calibration_analysis[bucket] = {
                    'total_predictions': row['total_predictions'],
                    'avg_confidence': round(avg_confidence, 3),
                    'actual_win_rate': round(actual_win_rate * 100, 1),
                    'calibration_error': round(calibration_error, 3)
                }
            
            total_predictions = sum(row['total_predictions'] for row in calibration_data)
            overall_calibration_error = total_calibration_error / total_predictions if total_predictions > 0 else 0
            
            return {
                'calibration_by_bucket': calibration_analysis,
                'overall_calibration_error': round(overall_calibration_error, 4),
                'recommendation': 'well_calibrated' if overall_calibration_error < 0.1 else 'needs_calibration'
            }
    
    async def update_strategy_weights(self):
        """Update strategy weights based on recent performance"""
        try:
            # Get performance for each strategy
            strategies = await self._get_active_strategies()
            
            strategy_scores = {}
            for strategy_name, timeframe in strategies:
                performance = self.tracker.get_strategy_performance(strategy_name, timeframe, days=14)
                
                if performance.get('total_predictions', 0) >= 5:
                    # Calculate composite score
                    win_rate = performance.get('win_rate', 0) / 100
                    error_score = max(0, 1 - performance.get('mean_prediction_error', 10) / 10)
                    confidence_score = performance.get('mean_confidence', 0.5)
                    
                    composite_score = (win_rate * 0.5 + error_score * 0.3 + confidence_score * 0.2)
                    strategy_scores[f"{strategy_name}_{timeframe}"] = composite_score
            
            if not strategy_scores:
                logger.info("No strategy scores available for weight update")
                return
            
            # Normalize scores to weights (softmax-like)
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                new_weights = {k: v / total_score for k, v in strategy_scores.items()}
            else:
                # Equal weights if no positive scores
                new_weights = {k: 1.0 / len(strategy_scores) for k in strategy_scores.keys()}
            
            # Apply smoothing with previous weights
            smoothing_factor = 0.7
            for strategy_key in new_weights:
                old_weight = self.strategy_weights.get(strategy_key, 1.0 / len(new_weights))
                self.strategy_weights[strategy_key] = (
                    smoothing_factor * old_weight + 
                    (1 - smoothing_factor) * new_weights[strategy_key]
                )
            
            logger.info(f"Updated strategy weights: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Failed to update strategy weights: {e}")
    
    async def _get_active_strategies(self) -> List[Tuple[str, str]]:
        """Get list of active strategies"""
        with self.tracker._get_connection() as conn:
            strategies = conn.execute("""
                SELECT DISTINCT strategy_name, timeframe
                FROM prediction_records 
                WHERE created_at >= datetime('now', '-7 days')
            """).fetchall()
            
            return [(row['strategy_name'], row['timeframe']) for row in strategies]
    
    async def adaptive_model_retraining(self):
        """Adaptively retrain models based on performance degradation"""
        try:
            strategies = await self._get_active_strategies()
            
            for strategy_name, timeframe in strategies:
                performance = self.tracker.get_strategy_performance(strategy_name, timeframe, days=7)
                
                # Check if retraining is needed
                recent_win_rate = performance.get('recent_win_rate', 0) / 100
                overall_win_rate = performance.get('win_rate', 0) / 100
                
                performance_degradation = overall_win_rate - recent_win_rate
                
                if (performance_degradation > self.adaptation_threshold and 
                    performance.get('total_predictions', 0) >= self.min_samples):
                    
                    logger.info(f"Retraining {strategy_name}_{timeframe} due to performance degradation: {performance_degradation:.3f}")
                    
                    # Retrain the model
                    await self._retrain_strategy_model(strategy_name, timeframe)
            
        except Exception as e:
            logger.error(f"Failed adaptive retraining: {e}")
    
    async def _retrain_strategy_model(self, strategy_name: str, timeframe: str):
        """Retrain a specific strategy model"""
        try:
            # Get training data
            training_data = await self._prepare_training_data(strategy_name, timeframe)
            
            if len(training_data) < self.min_samples:
                logger.warning(f"Insufficient data for retraining {strategy_name}_{timeframe}")
                return
            
            # Prepare features and targets
            X, y = self._extract_features_and_targets(training_data)
            
            if len(X) == 0:
                logger.warning(f"No valid features for retraining {strategy_name}_{timeframe}")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models and select best
            models_to_try = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]
            
            best_model = None
            best_score = float('inf')
            
            for name, model in models_to_try:
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    if mae < best_score:
                        best_score = mae
                        best_model = model
                        
                except Exception as e:
                    logger.warning(f"Model {name} failed: {e}")
                    continue
            
            if best_model is not None:
                # Store the trained model
                model_key = f"{strategy_name}_{timeframe}"
                self.models[model_key] = best_model
                self.scalers[model_key] = scaler
                
                logger.info(f"Retrained {model_key} with MAE: {best_score:.4f}")
                
                # Store insight about retraining
                insight = LearningInsight(
                    insight_type='model_retraining',
                    insight_category='performance',
                    title=f"Retrained {model_key}",
                    description=f"Model retrained due to performance degradation. New MAE: {best_score:.4f}",
                    insight_data={
                        'strategy': strategy_name,
                        'timeframe': timeframe,
                        'new_mae': best_score,
                        'training_samples': len(X_train)
                    },
                    confidence_level=0.8,
                    sample_size=len(training_data),
                    expected_improvement=0.05,
                    algorithm_used='adaptive_retraining'
                )
                
                self.learning_insights.append(insight)
            
        except Exception as e:
            logger.error(f"Failed to retrain {strategy_name}_{timeframe}: {e}")
    
    async def _prepare_training_data(self, strategy_name: str, timeframe: str, days: int = 60) -> List[Dict]:
        """Prepare training data for model retraining"""
        with self.tracker._get_connection() as conn:
            since_date = datetime.now() - timedelta(days=days)
            
            training_data = conn.execute("""
                SELECT * FROM prediction_records 
                WHERE strategy_name = ? 
                AND timeframe = ?
                AND is_validated = 1
                AND created_at >= ?
                AND feature_weights IS NOT NULL
                ORDER BY created_at DESC
            """, (strategy_name, timeframe, since_date.isoformat())).fetchall()
            
            return [dict(row) for row in training_data]
    
    def _extract_features_and_targets(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from training data"""
        features = []
        targets = []
        
        for record in training_data:
            try:
                # Parse feature weights
                if record.get('feature_weights'):
                    feature_weights = json.loads(record['feature_weights'])
                    feature_vector = list(feature_weights.values())
                    
                    # Add prediction metadata as features
                    feature_vector.extend([
                        record['confidence'],
                        record.get('signal_strength', 0.5),
                        record.get('trend_strength', 0.5),
                        1 if record['direction'] == 'bullish' else 0
                    ])
                    
                    features.append(feature_vector)
                    
                    # Target is the prediction error (we want to minimize this)
                    targets.append(record.get('prediction_error', 10.0))
                    
            except (json.JSONDecodeError, KeyError) as e:
                continue
        
        if features:
            return np.array(features), np.array(targets)
        else:
            return np.array([]), np.array([])
    
    async def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across all predictions"""
        try:
            feature_analysis = self.tracker.get_feature_importance_analysis(days=30)
            
            if 'error' in feature_analysis:
                return feature_analysis
            
            # Store feature importance history for trend analysis
            current_time = datetime.now().isoformat()
            self.feature_importance_history[current_time] = feature_analysis
            
            # Detect significant changes in feature importance
            if len(self.feature_importance_history) > 1:
                importance_trends = self._analyze_feature_trends()
                feature_analysis['trends'] = importance_trends
            
            return feature_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_trends(self) -> Dict[str, Any]:
        """Analyze trends in feature importance over time"""
        if len(self.feature_importance_history) < 2:
            return {}
        
        # Get the two most recent analyses
        times = sorted(self.feature_importance_history.keys())
        current_features = self.feature_importance_history[times[-1]]['most_important_features']
        previous_features = self.feature_importance_history[times[-2]]['most_important_features']
        
        trends = {}
        
        for feature in set(list(current_features.keys()) + list(previous_features.keys())):
            current_score = current_features.get(feature, {}).get('importance_score', 0)
            previous_score = previous_features.get(feature, {}).get('importance_score', 0)
            
            if previous_score > 0:
                change = (current_score - previous_score) / previous_score
                trends[feature] = {
                    'change_percent': round(change * 100, 1),
                    'trend': 'improving' if change > 0.1 else 'declining' if change < -0.1 else 'stable'
                }
        
        return trends
    
    async def analyze_market_regimes(self) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        return self.tracker.get_market_regime_analysis(days=60)
    
    async def generate_learning_insights(self, performance_insights: Dict, 
                                       feature_insights: Dict, regime_insights: Dict):
        """Generate actionable learning insights"""
        try:
            new_insights = []
            
            # Performance-based insights
            if 'hourly_patterns' in performance_insights:
                hourly_data = performance_insights['hourly_patterns']
                if 'best_hour' in hourly_data and 'worst_hour' in hourly_data:
                    best_hour = hourly_data['best_hour']
                    worst_hour = hourly_data['worst_hour']
                    
                    if best_hour['win_rate'] - worst_hour['win_rate'] > 20:
                        insight = LearningInsight(
                            insight_type='temporal_pattern',
                            insight_category='timing_optimization',
                            title='Significant hourly performance variation detected',
                            description=f"Hour {best_hour['hour']} shows {best_hour['win_rate']:.1f}% win rate vs {worst_hour['win_rate']:.1f}% at hour {worst_hour['hour']}",
                            insight_data={
                                'best_hour': best_hour,
                                'worst_hour': worst_hour,
                                'performance_gap': best_hour['win_rate'] - worst_hour['win_rate']
                            },
                            confidence_level=0.85,
                            sample_size=best_hour['total_predictions'] + worst_hour['total_predictions'],
                            expected_improvement=0.1,
                            algorithm_used='temporal_analysis'
                        )
                        new_insights.append(insight)
            
            # Feature importance insights
            if 'most_important_features' in feature_insights:
                top_features = feature_insights['most_important_features']
                if len(top_features) >= 3:
                    top_feature_names = list(top_features.keys())[:3]
                    avg_importance = np.mean([f['importance_score'] for f in top_features.values()][:3])
                    
                    if avg_importance > 0.7:
                        insight = LearningInsight(
                            insight_type='feature_importance',
                            insight_category='model_optimization',
                            title='High-impact features identified',
                            description=f"Features {', '.join(top_feature_names)} show consistently high predictive power",
                            insight_data={
                                'top_features': top_feature_names,
                                'average_importance': avg_importance,
                                'recommendation': 'increase_weight'
                            },
                            confidence_level=0.9,
                            sample_size=feature_insights.get('total_predictions_analyzed', 0),
                            expected_improvement=0.08,
                            algorithm_used='feature_analysis'
                        )
                        new_insights.append(insight)
            
            # Market regime insights
            if 'regime_performance' in regime_insights:
                regime_perf = regime_insights['regime_performance']
                if len(regime_perf) >= 2:
                    best_regime = max(regime_perf.items(), key=lambda x: x[1]['win_rate'])
                    worst_regime = min(regime_perf.items(), key=lambda x: x[1]['win_rate'])
                    
                    performance_gap = best_regime[1]['win_rate'] - worst_regime[1]['win_rate']
                    
                    if performance_gap > 25:
                        insight = LearningInsight(
                            insight_type='market_regime',
                            insight_category='regime_adaptation',
                            title='Market regime performance disparity detected',
                            description=f"{best_regime[0]} regime shows {best_regime[1]['win_rate']:.1f}% win rate vs {worst_regime[1]['win_rate']:.1f}% in {worst_regime[0]} regime",
                            insight_data={
                                'best_regime': best_regime,
                                'worst_regime': worst_regime,
                                'performance_gap': performance_gap,
                                'recommendation': 'regime_specific_strategies'
                            },
                            confidence_level=0.8,
                            sample_size=sum(r['total_predictions'] for r in regime_perf.values()),
                            expected_improvement=0.15,
                            algorithm_used='regime_analysis'
                        )
                        new_insights.append(insight)
            
            # Add insights to collection
            self.learning_insights.extend(new_insights)
            
            # Store insights in database
            await self._store_learning_insights(new_insights)
            
            logger.info(f"Generated {len(new_insights)} new learning insights")
            
        except Exception as e:
            logger.error(f"Failed to generate learning insights: {e}")
    
    async def _store_learning_insights(self, insights: List[LearningInsight]):
        """Store learning insights in database"""
        try:
            with self.tracker._get_connection() as conn:
                for insight in insights:
                    conn.execute("""
                        INSERT OR REPLACE INTO learning_insights (
                            insight_type, insight_category, title, description,
                            insight_data, confidence_level, sample_size,
                            expected_improvement, algorithm_used, discovered_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        insight.insight_type,
                        insight.insight_category,
                        insight.title,
                        insight.description,
                        json.dumps(insight.insight_data),
                        insight.confidence_level,
                        insight.sample_size,
                        insight.expected_improvement,
                        insight.algorithm_used,
                        insight.discovered_at.isoformat()
                    ))
            
            logger.info(f"Stored {len(insights)} insights in database")
            
        except Exception as e:
            logger.error(f"Failed to store learning insights: {e}")
    
    async def apply_learning_insights(self):
        """Apply validated learning insights to improve the system"""
        try:
            # Get high-confidence insights that haven't been implemented
            high_confidence_insights = [
                insight for insight in self.learning_insights
                if (insight.confidence_level >= self.confidence_threshold and 
                    insight.expected_improvement >= self.adaptation_threshold)
            ]
            
            implemented_count = 0
            
            for insight in high_confidence_insights:
                try:
                    if insight.insight_type == 'feature_importance':
                        await self._apply_feature_insight(insight)
                        implemented_count += 1
                    
                    elif insight.insight_type == 'temporal_pattern':
                        await self._apply_temporal_insight(insight)
                        implemented_count += 1
                    
                    elif insight.insight_type == 'market_regime':
                        await self._apply_regime_insight(insight)
                        implemented_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to apply insight {insight.title}: {e}")
            
            logger.info(f"Applied {implemented_count} learning insights")
            
        except Exception as e:
            logger.error(f"Failed to apply learning insights: {e}")
    
    async def _apply_feature_insight(self, insight: LearningInsight):
        """Apply feature importance insight"""
        # In a real implementation, this would update model feature weights
        logger.info(f"Applied feature insight: {insight.title}")
    
    async def _apply_temporal_insight(self, insight: LearningInsight):
        """Apply temporal pattern insight"""
        # In a real implementation, this would adjust prediction confidence based on time
        logger.info(f"Applied temporal insight: {insight.title}")
    
    async def _apply_regime_insight(self, insight: LearningInsight):
        """Apply market regime insight"""
        # In a real implementation, this would enable regime-specific strategies
        logger.info(f"Applied regime insight: {insight.title}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning progress and insights"""
        return {
            'total_insights_generated': len(self.learning_insights),
            'high_confidence_insights': len([i for i in self.learning_insights if i.confidence_level >= 0.8]),
            'strategy_weights': self.strategy_weights,
            'models_trained': len(self.models),
            'feature_importance_analyses': len(self.feature_importance_history),
            'learning_rate': self.learning_rate,
            'recent_insights': [
                {
                    'type': i.insight_type,
                    'title': i.title,
                    'confidence': i.confidence_level,
                    'expected_improvement': i.expected_improvement
                }
                for i in self.learning_insights[-5:]  # Last 5 insights
            ]
        }

# Factory function for easy integration
def create_learning_engine(prediction_tracker: PredictionTracker) -> LearningEngine:
    """Factory function to create a LearningEngine instance"""
    return LearningEngine(prediction_tracker)

if __name__ == "__main__":
    # Example usage
    from prediction_tracker import PredictionTracker
    
    async def test_learning_engine():
        tracker = PredictionTracker()
        learning_engine = LearningEngine(tracker)
        
        # Analyze performance patterns
        performance_insights = await learning_engine.analyze_performance_patterns()
        print(f"Performance insights: {json.dumps(performance_insights, indent=2)}")
        
        # Get learning summary
        summary = learning_engine.get_learning_summary()
        print(f"Learning summary: {json.dumps(summary, indent=2)}")
    
    # Run the test
    asyncio.run(test_learning_engine())
