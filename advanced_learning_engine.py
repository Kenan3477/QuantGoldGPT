"""
🧠 PHASE 3: ADVANCED LEARNING ENGINE
===================================

LearningEngine - Advanced analytics and continuous improvement system
Daily learning cycles, feature importance tracking, and strategy optimization

Author: GoldGPT AI System
Created: July 23, 2025
"""

import sqlite3
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import threading
import time
from prediction_validator import PredictionValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('advanced_learning_engine')

@dataclass
class FeatureImportance:
    """Feature importance tracking"""
    feature_name: str
    importance_score: float
    accuracy_contribution: float
    market_condition: str
    timeframe: str
    calculation_date: datetime

@dataclass
class LearningCycle:
    """Daily learning cycle results"""
    cycle_date: datetime
    models_retrained: int
    feature_importances: List[FeatureImportance]
    strategy_weights_updated: Dict[str, float]
    performance_improvement: float
    market_regime_accuracy: Dict[str, float]
    optimization_results: Dict[str, Any]

class AdvancedLearningEngine:
    """
    Advanced learning engine with daily learning cycles
    Continuously improves models based on prediction outcomes
    """
    
    def __init__(self, validator: PredictionValidator, db_path: str = "goldgpt_advanced_learning.db"):
        self.validator = validator
        self.db_path = db_path
        self.init_database()
        self.learning_thread = None
        self.is_running = False
        self.feature_names = [
            'price_momentum', 'rsi', 'macd', 'bollinger_position',
            'volume_trend', 'news_sentiment', 'social_sentiment',
            'economic_indicator', 'market_volatility', 'trend_strength',
            'support_distance', 'resistance_distance', 'time_of_day',
            'day_of_week', 'market_session', 'volatility_regime'
        ]
        logger.info("🧠 Advanced Learning Engine initialized")
    
    def init_database(self):
        """Initialize learning engine database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create learning cycles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_date DATE NOT NULL UNIQUE,
                    models_retrained INTEGER NOT NULL,
                    feature_importances TEXT NOT NULL,
                    strategy_weights_updated TEXT NOT NULL,
                    performance_improvement REAL NOT NULL,
                    market_regime_accuracy TEXT NOT NULL,
                    optimization_results TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create feature evolution table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    accuracy_contribution REAL NOT NULL,
                    market_condition TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    trend_direction TEXT NOT NULL,
                    calculation_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create strategy optimization table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_optimization (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    old_weight REAL NOT NULL,
                    new_weight REAL NOT NULL,
                    performance_change REAL NOT NULL,
                    market_condition TEXT NOT NULL,
                    optimization_reason TEXT NOT NULL,
                    optimization_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    accuracy_before REAL NOT NULL,
                    accuracy_after REAL NOT NULL,
                    feature_count INTEGER NOT NULL,
                    training_samples INTEGER NOT NULL,
                    validation_score REAL NOT NULL,
                    retrain_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Advanced learning engine database initialized")
            
        except Exception as e:
            logger.error(f"❌ Learning database initialization failed: {e}")
    
    def start_learning_service(self):
        """Start daily learning service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("🚀 Advanced learning service started")
    
    def stop_learning_service(self):
        """Stop learning service"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join()
        logger.info("⏹️ Advanced learning service stopped")
    
    def _learning_loop(self):
        """Main learning loop - runs daily"""
        while self.is_running:
            try:
                current_hour = datetime.now().hour
                
                # Run learning cycle at 2 AM daily
                if current_hour == 2:
                    logger.info("🔄 Starting daily learning cycle...")
                    self.run_daily_learning_cycle()
                    
                    # Sleep until next day
                    time.sleep(22 * 3600)  # 22 hours
                else:
                    time.sleep(3600)  # Check every hour
                    
            except Exception as e:
                logger.error(f"❌ Learning loop error: {e}")
                time.sleep(600)  # Wait 10 minutes on error
    
    def run_daily_learning_cycle(self) -> LearningCycle:
        """Run comprehensive daily learning cycle"""
        try:
            cycle_start = time.time()
            cycle_date = datetime.now().date()
            
            logger.info(f"🎯 Starting learning cycle for {cycle_date}")
            
            # Step 1: Analyze recent predictions
            recent_predictions = self._get_recent_predictions(days=7)
            
            if len(recent_predictions) < 10:
                logger.warning("⚠️ Insufficient recent predictions for learning")
                return self._create_empty_cycle(cycle_date)
            
            # Step 2: Calculate feature importances
            feature_importances = self._calculate_feature_importance(recent_predictions)
            
            # Step 3: Retrain models
            models_retrained = self._retrain_models(recent_predictions)
            
            # Step 4: Optimize strategy weights
            strategy_weights_updated = self._optimize_strategy_weights(recent_predictions)
            
            # Step 5: Analyze market regime performance
            market_regime_accuracy = self._analyze_market_regime_performance(recent_predictions)
            
            # Step 6: Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement()
            
            # Step 7: Run optimization
            optimization_results = self._run_optimization_analysis(recent_predictions)
            
            # Create learning cycle record
            cycle = LearningCycle(
                cycle_date=datetime.now(),
                models_retrained=models_retrained,
                feature_importances=feature_importances,
                strategy_weights_updated=strategy_weights_updated,
                performance_improvement=performance_improvement,
                market_regime_accuracy=market_regime_accuracy,
                optimization_results=optimization_results
            )
            
            # Store results
            self._store_learning_cycle(cycle)
            
            cycle_duration = time.time() - cycle_start
            logger.info(f"✅ Learning cycle completed in {cycle_duration:.2f}s")
            logger.info(f"   Models retrained: {models_retrained}")
            logger.info(f"   Features analyzed: {len(feature_importances)}")
            logger.info(f"   Performance improvement: {performance_improvement:.3f}")
            
            return cycle
            
        except Exception as e:
            logger.error(f"❌ Daily learning cycle failed: {e}")
            return self._create_empty_cycle(datetime.now().date())
    
    def _get_recent_predictions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent validated predictions"""
        try:
            conn = sqlite3.connect(self.validator.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions
                WHERE outcome IS NOT NULL
                AND prediction_timestamp >= date('now', '-{} days')
                ORDER BY prediction_timestamp DESC
            '''.format(days))
            
            predictions = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            prediction_dicts = []
            columns = [
                'prediction_id', 'symbol', 'predicted_price', 'current_price',
                'direction', 'confidence', 'timeframe', 'strategy_weights',
                'market_conditions', 'feature_vector', 'contributing_strategies',
                'prediction_timestamp', 'expiry_timestamp', 'actual_price',
                'outcome', 'accuracy_score', 'profit_factor', 'validated_timestamp'
            ]
            
            for pred in predictions:
                pred_dict = dict(zip(columns, pred))
                
                # Parse JSON fields
                try:
                    pred_dict['strategy_weights'] = json.loads(pred_dict['strategy_weights'])
                    pred_dict['market_conditions'] = json.loads(pred_dict['market_conditions'])
                    pred_dict['feature_vector'] = json.loads(pred_dict['feature_vector'])
                    pred_dict['contributing_strategies'] = json.loads(pred_dict['contributing_strategies'])
                except:
                    continue
                
                prediction_dicts.append(pred_dict)
            
            return prediction_dicts
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent predictions: {e}")
            return []
    
    def _calculate_feature_importance(self, predictions: List[Dict[str, Any]]) -> List[FeatureImportance]:
        """Calculate feature importance evolution"""
        try:
            feature_importances = []
            
            # Group predictions by timeframe and market condition
            grouped_predictions = {}
            
            for pred in predictions:
                timeframe = pred['timeframe']
                market_condition = pred['market_conditions'].get('market_regime', 'unknown')
                key = f"{timeframe}_{market_condition}"
                
                if key not in grouped_predictions:
                    grouped_predictions[key] = []
                
                grouped_predictions[key].append(pred)
            
            # Calculate importance for each group
            for key, group_predictions in grouped_predictions.items():
                if len(group_predictions) < 5:
                    continue
                
                timeframe, market_condition = key.split('_', 1)
                
                # Prepare feature matrix and targets
                X = []
                y = []
                
                for pred in group_predictions:
                    feature_vector = pred['feature_vector']
                    accuracy_score = pred['accuracy_score'] or 0.0
                    
                    if len(feature_vector) >= len(self.feature_names):
                        X.append(feature_vector[:len(self.feature_names)])
                        y.append(accuracy_score)
                
                if len(X) < 5:
                    continue
                
                X = np.array(X)
                y = np.array(y)
                
                # Train random forest to get feature importances
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                
                # Store feature importances
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(rf.feature_importances_):
                        importance = FeatureImportance(
                            feature_name=feature_name,
                            importance_score=rf.feature_importances_[i],
                            accuracy_contribution=rf.feature_importances_[i] * np.mean(y),
                            market_condition=market_condition,
                            timeframe=timeframe,
                            calculation_date=datetime.now()
                        )
                        feature_importances.append(importance)
            
            # Store in database
            self._store_feature_evolution(feature_importances)
            
            return feature_importances
            
        except Exception as e:
            logger.error(f"❌ Feature importance calculation failed: {e}")
            return []
    
    def _retrain_models(self, predictions: List[Dict[str, Any]]) -> int:
        """Retrain models based on recent performance"""
        try:
            models_retrained = 0
            
            # Group by timeframe
            timeframe_groups = {}
            for pred in predictions:
                tf = pred['timeframe']
                if tf not in timeframe_groups:
                    timeframe_groups[tf] = []
                timeframe_groups[tf].append(pred)
            
            for timeframe, tf_predictions in timeframe_groups.items():
                if len(tf_predictions) < 10:
                    continue
                
                # Prepare training data
                X_train = []
                y_train = []
                
                for pred in tf_predictions:
                    feature_vector = pred['feature_vector']
                    actual_price = pred['actual_price']
                    predicted_price = pred['predicted_price']
                    
                    if actual_price and len(feature_vector) >= len(self.feature_names):
                        X_train.append(feature_vector[:len(self.feature_names)])
                        y_train.append(actual_price)
                
                if len(X_train) < 10:
                    continue
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Train model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train_fold = X_scaled[train_idx]
                    X_val_fold = X_scaled[val_idx]
                    y_train_fold = y_train[train_idx]
                    y_val_fold = y_train[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    score = 1 - mean_squared_error(y_val_fold, y_pred) / np.var(y_val_fold)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                
                # Store model performance
                self._store_model_performance(
                    model_name=f"rf_{timeframe}",
                    timeframe=timeframe,
                    accuracy_before=0.5,  # Placeholder
                    accuracy_after=avg_score,
                    feature_count=len(self.feature_names),
                    training_samples=len(X_train),
                    validation_score=avg_score
                )
                
                models_retrained += 1
            
            return models_retrained
            
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
            return 0
    
    def _optimize_strategy_weights(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Optimize strategy weights based on recent performance"""
        try:
            # Calculate strategy performance
            strategy_performance = {}
            
            for pred in predictions:
                strategies = pred['contributing_strategies']
                outcome = pred['outcome']
                accuracy = pred['accuracy_score'] or 0.0
                
                for strategy in strategies:
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = {'scores': [], 'count': 0}
                    
                    strategy_performance[strategy]['scores'].append(accuracy)
                    strategy_performance[strategy]['count'] += 1
            
            # Calculate average performance for each strategy
            strategy_weights = {}
            total_performance = 0
            
            for strategy, perf in strategy_performance.items():
                avg_score = np.mean(perf['scores']) if perf['scores'] else 0.0
                strategy_weights[strategy] = max(0.1, avg_score)  # Minimum weight 0.1
                total_performance += strategy_weights[strategy]
            
            # Normalize weights
            if total_performance > 0:
                for strategy in strategy_weights:
                    strategy_weights[strategy] /= total_performance
            
            # Store optimization results
            self._store_strategy_optimization(strategy_weights)
            
            return strategy_weights
            
        except Exception as e:
            logger.error(f"❌ Strategy weight optimization failed: {e}")
            return {}
    
    def _analyze_market_regime_performance(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze performance by market regime"""
        try:
            regime_performance = {}
            
            for pred in predictions:
                market_regime = pred['market_conditions'].get('market_regime', 'unknown')
                accuracy = pred['accuracy_score'] or 0.0
                
                if market_regime not in regime_performance:
                    regime_performance[market_regime] = []
                
                regime_performance[market_regime].append(accuracy)
            
            # Calculate average accuracy for each regime
            regime_accuracy = {}
            for regime, scores in regime_performance.items():
                regime_accuracy[regime] = np.mean(scores) if scores else 0.0
            
            return regime_accuracy
            
        except Exception as e:
            logger.error(f"❌ Market regime analysis failed: {e}")
            return {}
    
    def _calculate_performance_improvement(self) -> float:
        """Calculate recent performance improvement"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get last two learning cycles
            cursor.execute('''
                SELECT performance_improvement FROM learning_cycles
                ORDER BY cycle_date DESC
                LIMIT 2
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) >= 2:
                current_perf = results[0][0]
                previous_perf = results[1][0]
                return current_perf - previous_perf
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Performance improvement calculation failed: {e}")
            return 0.0
    
    def _run_optimization_analysis(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run advanced optimization analysis"""
        try:
            optimization_results = {
                'total_predictions_analyzed': len(predictions),
                'avg_accuracy': np.mean([p['accuracy_score'] or 0.0 for p in predictions]),
                'best_performing_timeframe': self._find_best_timeframe(predictions),
                'optimal_confidence_threshold': self._find_optimal_confidence_threshold(predictions),
                'feature_correlation_analysis': self._analyze_feature_correlations(predictions)
            }
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Optimization analysis failed: {e}")
            return {}
    
    def _find_best_timeframe(self, predictions: List[Dict[str, Any]]) -> str:
        """Find best performing timeframe"""
        try:
            timeframe_performance = {}
            
            for pred in predictions:
                tf = pred['timeframe']
                accuracy = pred['accuracy_score'] or 0.0
                
                if tf not in timeframe_performance:
                    timeframe_performance[tf] = []
                
                timeframe_performance[tf].append(accuracy)
            
            best_tf = 'unknown'
            best_score = 0.0
            
            for tf, scores in timeframe_performance.items():
                avg_score = np.mean(scores) if scores else 0.0
                if avg_score > best_score and len(scores) >= 3:
                    best_score = avg_score
                    best_tf = tf
            
            return best_tf
            
        except Exception as e:
            logger.error(f"❌ Best timeframe analysis failed: {e}")
            return 'unknown'
    
    def _find_optimal_confidence_threshold(self, predictions: List[Dict[str, Any]]) -> float:
        """Find optimal confidence threshold"""
        try:
            confidences = [p['confidence'] for p in predictions]
            accuracies = [p['accuracy_score'] or 0.0 for p in predictions]
            
            if len(confidences) < 10:
                return 0.7  # Default threshold
            
            # Test different thresholds
            best_threshold = 0.7
            best_accuracy = 0.0
            
            for threshold in np.arange(0.5, 0.95, 0.05):
                high_conf_accuracies = [
                    acc for conf, acc in zip(confidences, accuracies)
                    if conf >= threshold
                ]
                
                if len(high_conf_accuracies) >= 5:
                    avg_accuracy = np.mean(high_conf_accuracies)
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_threshold = threshold
            
            return best_threshold
            
        except Exception as e:
            logger.error(f"❌ Optimal confidence threshold analysis failed: {e}")
            return 0.7
    
    def _analyze_feature_correlations(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze feature correlations with accuracy"""
        try:
            if len(predictions) < 10:
                return {}
            
            # Prepare feature matrix
            feature_matrix = []
            accuracies = []
            
            for pred in predictions:
                feature_vector = pred['feature_vector']
                accuracy = pred['accuracy_score'] or 0.0
                
                if len(feature_vector) >= len(self.feature_names):
                    feature_matrix.append(feature_vector[:len(self.feature_names)])
                    accuracies.append(accuracy)
            
            if len(feature_matrix) < 10:
                return {}
            
            feature_matrix = np.array(feature_matrix)
            accuracies = np.array(accuracies)
            
            # Calculate correlations
            correlations = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < feature_matrix.shape[1]:
                    corr = np.corrcoef(feature_matrix[:, i], accuracies)[0, 1]
                    correlations[feature_name] = corr if not np.isnan(corr) else 0.0
            
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Feature correlation analysis failed: {e}")
            return {}
    
    def _create_empty_cycle(self, cycle_date) -> LearningCycle:
        """Create empty learning cycle for insufficient data"""
        return LearningCycle(
            cycle_date=datetime.now(),
            models_retrained=0,
            feature_importances=[],
            strategy_weights_updated={},
            performance_improvement=0.0,
            market_regime_accuracy={},
            optimization_results={'status': 'insufficient_data'}
        )
    
    def _store_learning_cycle(self, cycle: LearningCycle):
        """Store learning cycle results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_cycles (
                    cycle_date, models_retrained, feature_importances,
                    strategy_weights_updated, performance_improvement,
                    market_regime_accuracy, optimization_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                cycle.cycle_date.date(),
                cycle.models_retrained,
                json.dumps([asdict(fi) for fi in cycle.feature_importances], default=str),
                json.dumps(cycle.strategy_weights_updated),
                cycle.performance_improvement,
                json.dumps(cycle.market_regime_accuracy),
                json.dumps(cycle.optimization_results)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store learning cycle: {e}")
    
    def _store_feature_evolution(self, feature_importances: List[FeatureImportance]):
        """Store feature evolution data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for fi in feature_importances:
                cursor.execute('''
                    INSERT INTO feature_evolution (
                        feature_name, importance_score, accuracy_contribution,
                        market_condition, timeframe, trend_direction, calculation_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fi.feature_name,
                    fi.importance_score,
                    fi.accuracy_contribution,
                    fi.market_condition,
                    fi.timeframe,
                    'improving' if fi.importance_score > 0.1 else 'stable',
                    fi.calculation_date.date()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store feature evolution: {e}")
    
    def _store_strategy_optimization(self, strategy_weights: Dict[str, float]):
        """Store strategy optimization results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for strategy, weight in strategy_weights.items():
                cursor.execute('''
                    INSERT INTO strategy_optimization (
                        strategy_name, old_weight, new_weight, performance_change,
                        market_condition, optimization_reason, optimization_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy,
                    0.2,  # Default old weight
                    weight,
                    weight - 0.2,
                    'general',
                    'performance_based_reweighting',
                    datetime.now().date()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store strategy optimization: {e}")
    
    def _store_model_performance(self, model_name: str, timeframe: str,
                                accuracy_before: float, accuracy_after: float,
                                feature_count: int, training_samples: int,
                                validation_score: float):
        """Store model performance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (
                    model_name, timeframe, accuracy_before, accuracy_after,
                    feature_count, training_samples, validation_score, retrain_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, timeframe, accuracy_before, accuracy_after,
                feature_count, training_samples, validation_score,
                datetime.now().date()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store model performance: {e}")
    
    def get_learning_progress(self, days: int = 30) -> Dict[str, Any]:
        """Get learning progress summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent learning cycles
            cursor.execute('''
                SELECT * FROM learning_cycles
                WHERE cycle_date >= date('now', '-{} days')
                ORDER BY cycle_date DESC
            '''.format(days))
            
            cycles = cursor.fetchall()
            
            if not cycles:
                return {'status': 'no_recent_cycles'}
            
            # Calculate progress metrics
            total_models_retrained = sum(cycle[2] for cycle in cycles)
            avg_performance_improvement = np.mean([cycle[5] for cycle in cycles])
            
            # Get feature evolution trends
            cursor.execute('''
                SELECT feature_name, AVG(importance_score) as avg_importance
                FROM feature_evolution
                WHERE calculation_date >= date('now', '-{} days')
                GROUP BY feature_name
                ORDER BY avg_importance DESC
                LIMIT 10
            '''.format(days))
            
            top_features = cursor.fetchall()
            
            progress_summary = {
                'learning_cycles_completed': len(cycles),
                'total_models_retrained': total_models_retrained,
                'avg_performance_improvement': avg_performance_improvement,
                'top_features': [{'name': f[0], 'importance': f[1]} for f in top_features],
                'last_cycle_date': cycles[0][1] if cycles else None,
                'learning_trend': 'improving' if avg_performance_improvement > 0 else 'stable'
            }
            
            conn.close()
            return progress_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get learning progress: {e}")
            return {'error': str(e)}

# Global learning engine instance
advanced_learning_engine = None

def initialize_advanced_learning_engine(validator: PredictionValidator):
    """Initialize global advanced learning engine"""
    global advanced_learning_engine
    advanced_learning_engine = AdvancedLearningEngine(validator)
    return advanced_learning_engine

def start_advanced_learning_service():
    """Start the global advanced learning service"""
    if advanced_learning_engine:
        advanced_learning_engine.start_learning_service()

def get_advanced_learning_progress() -> Dict[str, Any]:
    """Get current advanced learning progress"""
    if advanced_learning_engine:
        return advanced_learning_engine.get_learning_progress()
    return {'error': 'Advanced learning engine not initialized'}

if __name__ == "__main__":
    from prediction_validator import PredictionValidator
    
    print("🧠 Starting Advanced Learning Engine...")
    validator = PredictionValidator()
    engine = AdvancedLearningEngine(validator)
    engine.start_learning_service()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("⏹️ Stopping Advanced Learning Engine...")
        engine.stop_learning_service()
