"""
📊 PHASE 3: OUTCOME TRACKER - COMPREHENSIVE ANALYSIS
====================================================

OutcomeTracker - Tracks prediction vs actual price movements
Identifies best performing features and analyzes accuracy by market conditions

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from prediction_validator import PredictionValidator
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('outcome_tracker')

@dataclass
class PredictionOutcome:
    """Individual prediction outcome analysis"""
    prediction_id: str
    symbol: str
    timeframe: str
    predicted_price: float
    actual_price: float
    price_difference: float
    price_difference_percent: float
    direction_predicted: str
    direction_actual: str
    direction_correct: bool
    confidence: float
    market_volatility: float
    market_regime: str
    contributing_strategies: List[str]
    feature_performance: Dict[str, float]
    outcome_timestamp: datetime

@dataclass
class FeaturePerformanceMetrics:
    """Feature performance analysis"""
    feature_name: str
    accuracy_contribution: float
    directional_accuracy: float
    price_accuracy: float
    volatility_performance: Dict[str, float]
    timeframe_performance: Dict[str, float]
    market_regime_performance: Dict[str, float]
    improvement_trend: str
    
@dataclass
class MarketConditionAnalysis:
    """Market condition specific analysis"""
    condition_type: str
    condition_value: str
    total_predictions: int
    correct_predictions: int
    accuracy_rate: float
    avg_price_error: float
    best_timeframe: str
    best_strategy: str
    volatility_impact: float

class OutcomeTracker:
    """
    Comprehensive outcome tracking and analysis system
    Tracks prediction vs actual movements and identifies performance patterns
    """
    
    def __init__(self, validator: PredictionValidator, db_path: str = "goldgpt_outcome_tracker.db"):
        self.validator = validator
        self.db_path = db_path
        self.init_database()
        self.analysis_thread = None
        self.is_running = False
        logger.info("📊 Outcome Tracker initialized")
    
    def init_database(self):
        """Initialize outcome tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create outcome analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS outcome_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL NOT NULL,
                    price_difference REAL NOT NULL,
                    price_difference_percent REAL NOT NULL,
                    direction_predicted TEXT NOT NULL,
                    direction_actual TEXT NOT NULL,
                    direction_correct BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    market_volatility REAL NOT NULL,
                    market_regime TEXT NOT NULL,
                    contributing_strategies TEXT NOT NULL,
                    feature_performance TEXT NOT NULL,
                    outcome_timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create feature performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    accuracy_contribution REAL NOT NULL,
                    directional_accuracy REAL NOT NULL,
                    price_accuracy REAL NOT NULL,
                    volatility_performance TEXT NOT NULL,
                    timeframe_performance TEXT NOT NULL,
                    market_regime_performance TEXT NOT NULL,
                    improvement_trend TEXT NOT NULL,
                    analysis_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create market condition analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_condition_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_type TEXT NOT NULL,
                    condition_value TEXT NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_rate REAL NOT NULL,
                    avg_price_error REAL NOT NULL,
                    best_timeframe TEXT NOT NULL,
                    best_strategy TEXT NOT NULL,
                    volatility_impact REAL NOT NULL,
                    analysis_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create strategy insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    total_contributions INTEGER NOT NULL,
                    successful_contributions INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_confidence_when_correct REAL NOT NULL,
                    avg_confidence_when_incorrect REAL NOT NULL,
                    best_market_conditions TEXT NOT NULL,
                    worst_market_conditions TEXT NOT NULL,
                    optimal_timeframes TEXT NOT NULL,
                    insights_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Outcome tracker database initialized")
            
        except Exception as e:
            logger.error(f"❌ Outcome tracker database initialization failed: {e}")
    
    def start_tracking_service(self):
        """Start outcome tracking service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.analysis_thread.start()
        logger.info("🚀 Outcome tracking service started")
    
    def stop_tracking_service(self):
        """Stop tracking service"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join()
        logger.info("⏹️ Outcome tracking service stopped")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.is_running:
            try:
                # Run analysis every hour
                self.analyze_recent_outcomes()
                self.update_feature_performance()
                self.analyze_market_conditions()
                self.generate_strategy_insights()
                
                time.sleep(3600)  # Wait 1 hour
                
            except Exception as e:
                logger.error(f"❌ Tracking loop error: {e}")
                time.sleep(600)  # Wait 10 minutes on error
    
    def analyze_recent_outcomes(self, days: int = 1):
        """Analyze recent prediction outcomes"""
        try:
            # Get recent validated predictions from validator
            recent_predictions = self._get_recent_validated_predictions(days)
            
            for pred_data in recent_predictions:
                if pred_data['outcome'] and pred_data['actual_price']:
                    outcome = self._create_prediction_outcome(pred_data)
                    self._store_outcome_analysis(outcome)
            
            logger.info(f"✅ Analyzed {len(recent_predictions)} recent outcomes")
            
        except Exception as e:
            logger.error(f"❌ Recent outcome analysis failed: {e}")
    
    def _get_recent_validated_predictions(self, days: int) -> List[Dict[str, Any]]:
        """Get recent validated predictions from validator"""
        try:
            conn = sqlite3.connect(self.validator.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions
                WHERE outcome IS NOT NULL
                AND actual_price IS NOT NULL
                AND validated_timestamp >= datetime('now', '-{} days')
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
            logger.error(f"❌ Failed to get recent validated predictions: {e}")
            return []
    
    def _create_prediction_outcome(self, pred_data: Dict[str, Any]) -> PredictionOutcome:
        """Create prediction outcome analysis from prediction data"""
        try:
            predicted_price = pred_data['predicted_price']
            actual_price = pred_data['actual_price']
            
            # Calculate price differences
            price_difference = actual_price - predicted_price
            price_difference_percent = (price_difference / predicted_price) * 100
            
            # Determine actual direction
            if price_difference > 0:
                direction_actual = 'bullish'
            elif price_difference < 0:
                direction_actual = 'bearish'
            else:
                direction_actual = 'neutral'
            
            # Check if direction was correct
            direction_predicted = pred_data['direction']
            direction_correct = direction_predicted == direction_actual
            
            # Calculate feature performance
            feature_performance = self._calculate_feature_performance(pred_data)
            
            # Get market conditions
            market_conditions = pred_data['market_conditions']
            market_volatility = market_conditions.get('volatility', 0.0)
            market_regime = market_conditions.get('market_regime', 'unknown')
            
            outcome = PredictionOutcome(
                prediction_id=pred_data['prediction_id'],
                symbol=pred_data['symbol'],
                timeframe=pred_data['timeframe'],
                predicted_price=predicted_price,
                actual_price=actual_price,
                price_difference=price_difference,
                price_difference_percent=price_difference_percent,
                direction_predicted=direction_predicted,
                direction_actual=direction_actual,
                direction_correct=direction_correct,
                confidence=pred_data['confidence'],
                market_volatility=market_volatility,
                market_regime=market_regime,
                contributing_strategies=pred_data['contributing_strategies'],
                feature_performance=feature_performance,
                outcome_timestamp=datetime.now()
            )
            
            return outcome
            
        except Exception as e:
            logger.error(f"❌ Failed to create prediction outcome: {e}")
            return None
    
    def _calculate_feature_performance(self, pred_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual feature performance"""
        try:
            feature_vector = pred_data['feature_vector']
            accuracy_score = pred_data['accuracy_score'] or 0.0
            
            feature_names = [
                'price_momentum', 'rsi', 'macd', 'bollinger_position',
                'volume_trend', 'news_sentiment', 'social_sentiment',
                'economic_indicator', 'market_volatility', 'trend_strength',
                'support_distance', 'resistance_distance', 'time_of_day',
                'day_of_week', 'market_session', 'volatility_regime'
            ]
            
            feature_performance = {}
            
            for i, feature_name in enumerate(feature_names):
                if i < len(feature_vector):
                    # Simple performance calculation based on feature value and accuracy
                    feature_value = feature_vector[i]
                    performance = feature_value * accuracy_score
                    feature_performance[feature_name] = performance
                else:
                    feature_performance[feature_name] = 0.0
            
            return feature_performance
            
        except Exception as e:
            logger.error(f"❌ Feature performance calculation failed: {e}")
            return {}
    
    def _store_outcome_analysis(self, outcome: PredictionOutcome):
        """Store outcome analysis in database"""
        try:
            if not outcome:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO outcome_analysis (
                    prediction_id, symbol, timeframe, predicted_price, actual_price,
                    price_difference, price_difference_percent, direction_predicted,
                    direction_actual, direction_correct, confidence, market_volatility,
                    market_regime, contributing_strategies, feature_performance,
                    outcome_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                outcome.prediction_id,
                outcome.symbol,
                outcome.timeframe,
                outcome.predicted_price,
                outcome.actual_price,
                outcome.price_difference,
                outcome.price_difference_percent,
                outcome.direction_predicted,
                outcome.direction_actual,
                outcome.direction_correct,
                outcome.confidence,
                outcome.market_volatility,
                outcome.market_regime,
                json.dumps(outcome.contributing_strategies),
                json.dumps(outcome.feature_performance),
                outcome.outcome_timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store outcome analysis: {e}")
    
    def update_feature_performance(self):
        """Update comprehensive feature performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent outcomes for analysis
            cursor.execute('''
                SELECT * FROM outcome_analysis
                WHERE outcome_timestamp >= date('now', '-7 days')
            ''')
            
            outcomes = cursor.fetchall()
            
            if len(outcomes) < 5:
                logger.warning("⚠️ Insufficient outcomes for feature performance analysis")
                return
            
            # Analyze each feature
            feature_names = [
                'price_momentum', 'rsi', 'macd', 'bollinger_position',
                'volume_trend', 'news_sentiment', 'social_sentiment',
                'economic_indicator', 'market_volatility', 'trend_strength',
                'support_distance', 'resistance_distance', 'time_of_day',
                'day_of_week', 'market_session', 'volatility_regime'
            ]
            
            for feature_name in feature_names:
                feature_metrics = self._analyze_feature_performance(feature_name, outcomes)
                self._store_feature_performance(feature_metrics)
            
            conn.close()
            logger.info("✅ Feature performance updated")
            
        except Exception as e:
            logger.error(f"❌ Feature performance update failed: {e}")
    
    def _analyze_feature_performance(self, feature_name: str, outcomes: List) -> FeaturePerformanceMetrics:
        """Analyze performance of a specific feature"""
        try:
            # Extract feature performance data
            feature_accuracies = []
            directional_accuracies = []
            price_accuracies = []
            volatility_performance = {'low': [], 'medium': [], 'high': []}
            timeframe_performance = {}
            regime_performance = {}
            
            for outcome in outcomes:
                # Parse feature performance JSON
                try:
                    feature_perf = json.loads(outcome[15])  # feature_performance column
                    if feature_name in feature_perf:
                        feature_value = feature_perf[feature_name]
                        
                        # Calculate accuracies
                        direction_correct = outcome[10]  # direction_correct column
                        price_error = abs(outcome[6])  # price_difference_percent column
                        volatility = outcome[12]  # market_volatility column
                        timeframe = outcome[3]  # timeframe column
                        regime = outcome[13]  # market_regime column
                        
                        feature_accuracies.append(feature_value)
                        directional_accuracies.append(1.0 if direction_correct else 0.0)
                        price_accuracies.append(max(0, 1 - price_error / 100))
                        
                        # Categorize volatility
                        vol_category = 'low' if volatility < 0.02 else 'high' if volatility > 0.05 else 'medium'
                        volatility_performance[vol_category].append(feature_value)
                        
                        # Track timeframe performance
                        if timeframe not in timeframe_performance:
                            timeframe_performance[timeframe] = []
                        timeframe_performance[timeframe].append(feature_value)
                        
                        # Track regime performance
                        if regime not in regime_performance:
                            regime_performance[regime] = []
                        regime_performance[regime].append(feature_value)
                        
                except:
                    continue
            
            # Calculate averages
            avg_accuracy_contribution = np.mean(feature_accuracies) if feature_accuracies else 0.0
            avg_directional_accuracy = np.mean(directional_accuracies) if directional_accuracies else 0.0
            avg_price_accuracy = np.mean(price_accuracies) if price_accuracies else 0.0
            
            # Calculate volatility performance
            vol_perf = {}
            for vol_level, values in volatility_performance.items():
                vol_perf[vol_level] = np.mean(values) if values else 0.0
            
            # Calculate timeframe performance
            tf_perf = {}
            for tf, values in timeframe_performance.items():
                tf_perf[tf] = np.mean(values) if values else 0.0
            
            # Calculate regime performance
            regime_perf = {}
            for regime, values in regime_performance.items():
                regime_perf[regime] = np.mean(values) if values else 0.0
            
            # Determine improvement trend
            improvement_trend = 'improving' if avg_accuracy_contribution > 0.5 else 'stable'
            
            metrics = FeaturePerformanceMetrics(
                feature_name=feature_name,
                accuracy_contribution=avg_accuracy_contribution,
                directional_accuracy=avg_directional_accuracy,
                price_accuracy=avg_price_accuracy,
                volatility_performance=vol_perf,
                timeframe_performance=tf_perf,
                market_regime_performance=regime_perf,
                improvement_trend=improvement_trend
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Feature {feature_name} analysis failed: {e}")
            return FeaturePerformanceMetrics(
                feature_name=feature_name,
                accuracy_contribution=0.0,
                directional_accuracy=0.0,
                price_accuracy=0.0,
                volatility_performance={},
                timeframe_performance={},
                market_regime_performance={},
                improvement_trend='unknown'
            )
    
    def _store_feature_performance(self, metrics: FeaturePerformanceMetrics):
        """Store feature performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO feature_performance_tracking (
                    feature_name, accuracy_contribution, directional_accuracy,
                    price_accuracy, volatility_performance, timeframe_performance,
                    market_regime_performance, improvement_trend, analysis_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.feature_name,
                metrics.accuracy_contribution,
                metrics.directional_accuracy,
                metrics.price_accuracy,
                json.dumps(metrics.volatility_performance),
                json.dumps(metrics.timeframe_performance),
                json.dumps(metrics.market_regime_performance),
                metrics.improvement_trend,
                datetime.now().date()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store feature performance: {e}")
    
    def analyze_market_conditions(self):
        """Analyze prediction accuracy by market conditions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent outcomes
            cursor.execute('''
                SELECT * FROM outcome_analysis
                WHERE outcome_timestamp >= date('now', '-14 days')
            ''')
            
            outcomes = cursor.fetchall()
            
            if len(outcomes) < 10:
                logger.warning("⚠️ Insufficient outcomes for market condition analysis")
                return
            
            # Analyze by market regime
            self._analyze_by_condition('market_regime', 13, outcomes)
            
            # Analyze by volatility levels
            self._analyze_volatility_conditions(outcomes)
            
            # Analyze by timeframe
            self._analyze_by_condition('timeframe', 3, outcomes)
            
            conn.close()
            logger.info("✅ Market condition analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Market condition analysis failed: {e}")
    
    def _analyze_by_condition(self, condition_type: str, column_index: int, outcomes: List):
        """Analyze outcomes by specific condition"""
        try:
            condition_analysis = {}
            
            for outcome in outcomes:
                condition_value = outcome[column_index]
                direction_correct = outcome[10]
                price_error = abs(outcome[6])
                timeframe = outcome[3]
                strategies = json.loads(outcome[14])  # contributing_strategies
                volatility = outcome[12]
                
                if condition_value not in condition_analysis:
                    condition_analysis[condition_value] = {
                        'total': 0,
                        'correct': 0,
                        'price_errors': [],
                        'timeframes': {},
                        'strategies': {},
                        'volatilities': []
                    }
                
                analysis = condition_analysis[condition_value]
                analysis['total'] += 1
                analysis['price_errors'].append(price_error)
                analysis['volatilities'].append(volatility)
                
                if direction_correct:
                    analysis['correct'] += 1
                
                # Track timeframes
                if timeframe not in analysis['timeframes']:
                    analysis['timeframes'][timeframe] = {'total': 0, 'correct': 0}
                analysis['timeframes'][timeframe]['total'] += 1
                if direction_correct:
                    analysis['timeframes'][timeframe]['correct'] += 1
                
                # Track strategies
                for strategy in strategies:
                    if strategy not in analysis['strategies']:
                        analysis['strategies'][strategy] = {'total': 0, 'correct': 0}
                    analysis['strategies'][strategy]['total'] += 1
                    if direction_correct:
                        analysis['strategies'][strategy]['correct'] += 1
            
            # Store analysis results
            for condition_value, analysis in condition_analysis.items():
                if analysis['total'] >= 3:  # Minimum sample size
                    self._store_market_condition_analysis(
                        condition_type, condition_value, analysis
                    )
                    
        except Exception as e:
            logger.error(f"❌ Condition analysis failed for {condition_type}: {e}")
    
    def _analyze_volatility_conditions(self, outcomes: List):
        """Analyze outcomes by volatility levels"""
        try:
            volatility_analysis = {'low': [], 'medium': [], 'high': []}
            
            for outcome in outcomes:
                volatility = outcome[12]
                direction_correct = outcome[10]
                
                if volatility < 0.02:
                    vol_level = 'low'
                elif volatility > 0.05:
                    vol_level = 'high'
                else:
                    vol_level = 'medium'
                
                volatility_analysis[vol_level].append({
                    'correct': direction_correct,
                    'price_error': abs(outcome[6]),
                    'timeframe': outcome[3],
                    'strategies': json.loads(outcome[14])
                })
            
            # Analyze each volatility level
            for vol_level, data in volatility_analysis.items():
                if len(data) >= 3:
                    analysis = {
                        'total': len(data),
                        'correct': sum(1 for d in data if d['correct']),
                        'price_errors': [d['price_error'] for d in data],
                        'timeframes': {},
                        'strategies': {},
                        'volatilities': []
                    }
                    
                    self._store_market_condition_analysis(
                        'volatility_level', vol_level, analysis
                    )
                    
        except Exception as e:
            logger.error(f"❌ Volatility condition analysis failed: {e}")
    
    def _store_market_condition_analysis(self, condition_type: str, condition_value: str, analysis: Dict):
        """Store market condition analysis"""
        try:
            # Calculate metrics
            accuracy_rate = analysis['correct'] / analysis['total']
            avg_price_error = np.mean(analysis['price_errors'])
            volatility_impact = np.mean(analysis['volatilities']) if analysis['volatilities'] else 0.0
            
            # Find best timeframe
            best_timeframe = 'unknown'
            best_tf_accuracy = 0.0
            for tf, tf_data in analysis['timeframes'].items():
                if tf_data['total'] >= 2:
                    tf_accuracy = tf_data['correct'] / tf_data['total']
                    if tf_accuracy > best_tf_accuracy:
                        best_tf_accuracy = tf_accuracy
                        best_timeframe = tf
            
            # Find best strategy
            best_strategy = 'unknown'
            best_strategy_accuracy = 0.0
            for strategy, strategy_data in analysis['strategies'].items():
                if strategy_data['total'] >= 2:
                    strategy_accuracy = strategy_data['correct'] / strategy_data['total']
                    if strategy_accuracy > best_strategy_accuracy:
                        best_strategy_accuracy = strategy_accuracy
                        best_strategy = strategy
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO market_condition_analysis (
                    condition_type, condition_value, total_predictions,
                    correct_predictions, accuracy_rate, avg_price_error,
                    best_timeframe, best_strategy, volatility_impact, analysis_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                condition_type, condition_value, analysis['total'],
                analysis['correct'], accuracy_rate, avg_price_error,
                best_timeframe, best_strategy, volatility_impact,
                datetime.now().date()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store market condition analysis: {e}")
    
    def generate_strategy_insights(self):
        """Generate comprehensive strategy insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent outcomes with strategy data
            cursor.execute('''
                SELECT contributing_strategies, direction_correct, confidence,
                       market_regime, timeframe FROM outcome_analysis
                WHERE outcome_timestamp >= date('now', '-14 days')
            ''')
            
            outcomes = cursor.fetchall()
            
            if len(outcomes) < 10:
                logger.warning("⚠️ Insufficient outcomes for strategy insights")
                return
            
            strategy_data = {}
            
            for outcome in outcomes:
                strategies = json.loads(outcome[0])
                correct = outcome[1]
                confidence = outcome[2]
                market_regime = outcome[3]
                timeframe = outcome[4]
                
                for strategy in strategies:
                    if strategy not in strategy_data:
                        strategy_data[strategy] = {
                            'total': 0,
                            'correct': 0,
                            'confidences_correct': [],
                            'confidences_incorrect': [],
                            'market_conditions': {},
                            'timeframes': {}
                        }
                    
                    data = strategy_data[strategy]
                    data['total'] += 1
                    
                    if correct:
                        data['correct'] += 1
                        data['confidences_correct'].append(confidence)
                    else:
                        data['confidences_incorrect'].append(confidence)
                    
                    # Track market conditions
                    if market_regime not in data['market_conditions']:
                        data['market_conditions'][market_regime] = {'total': 0, 'correct': 0}
                    data['market_conditions'][market_regime]['total'] += 1
                    if correct:
                        data['market_conditions'][market_regime]['correct'] += 1
                    
                    # Track timeframes
                    if timeframe not in data['timeframes']:
                        data['timeframes'][timeframe] = {'total': 0, 'correct': 0}
                    data['timeframes'][timeframe]['total'] += 1
                    if correct:
                        data['timeframes'][timeframe]['correct'] += 1
            
            # Store insights for each strategy
            for strategy_name, data in strategy_data.items():
                if data['total'] >= 5:  # Minimum sample size
                    self._store_strategy_insights(strategy_name, data)
            
            conn.close()
            logger.info("✅ Strategy insights generated")
            
        except Exception as e:
            logger.error(f"❌ Strategy insights generation failed: {e}")
    
    def _store_strategy_insights(self, strategy_name: str, data: Dict):
        """Store strategy insights"""
        try:
            success_rate = data['correct'] / data['total']
            avg_conf_correct = np.mean(data['confidences_correct']) if data['confidences_correct'] else 0.0
            avg_conf_incorrect = np.mean(data['confidences_incorrect']) if data['confidences_incorrect'] else 0.0
            
            # Find best and worst market conditions
            best_conditions = []
            worst_conditions = []
            
            for condition, cond_data in data['market_conditions'].items():
                if cond_data['total'] >= 2:
                    accuracy = cond_data['correct'] / cond_data['total']
                    if accuracy >= 0.7:
                        best_conditions.append(condition)
                    elif accuracy <= 0.3:
                        worst_conditions.append(condition)
            
            # Find optimal timeframes
            optimal_timeframes = []
            for tf, tf_data in data['timeframes'].items():
                if tf_data['total'] >= 2:
                    accuracy = tf_data['correct'] / tf_data['total']
                    if accuracy >= 0.6:
                        optimal_timeframes.append(tf)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO strategy_insights (
                    strategy_name, total_contributions, successful_contributions,
                    success_rate, avg_confidence_when_correct, avg_confidence_when_incorrect,
                    best_market_conditions, worst_market_conditions, optimal_timeframes,
                    insights_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_name,
                data['total'],
                data['correct'],
                success_rate,
                avg_conf_correct,
                avg_conf_incorrect,
                json.dumps(best_conditions),
                json.dumps(worst_conditions),
                json.dumps(optimal_timeframes),
                datetime.now().date()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store strategy insights: {e}")
    
    def get_comprehensive_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive outcome analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get summary statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_outcomes,
                    AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) as directional_accuracy,
                    AVG(ABS(price_difference_percent)) as avg_price_error,
                    AVG(confidence) as avg_confidence
                FROM outcome_analysis
                WHERE outcome_timestamp >= date('now', '-{} days')
            '''.format(days))
            
            summary_stats = cursor.fetchone()
            
            # Get best performing features
            cursor.execute('''
                SELECT feature_name, accuracy_contribution, directional_accuracy
                FROM feature_performance_tracking
                WHERE analysis_date >= date('now', '-{} days')
                ORDER BY accuracy_contribution DESC
                LIMIT 10
            '''.format(days))
            
            top_features = cursor.fetchall()
            
            # Get market condition performance
            cursor.execute('''
                SELECT condition_type, condition_value, accuracy_rate, total_predictions
                FROM market_condition_analysis
                WHERE analysis_date >= date('now', '-{} days')
                ORDER BY accuracy_rate DESC
                LIMIT 10
            '''.format(days))
            
            market_performance = cursor.fetchall()
            
            # Get strategy insights
            cursor.execute('''
                SELECT strategy_name, success_rate, total_contributions
                FROM strategy_insights
                WHERE insights_date >= date('now', '-{} days')
                ORDER BY success_rate DESC
            '''.format(days))
            
            strategy_performance = cursor.fetchall()
            
            analysis_report = {
                'summary_statistics': {
                    'total_outcomes': summary_stats[0] if summary_stats else 0,
                    'directional_accuracy': summary_stats[1] if summary_stats else 0.0,
                    'avg_price_error': summary_stats[2] if summary_stats else 0.0,
                    'avg_confidence': summary_stats[3] if summary_stats else 0.0
                },
                'top_performing_features': [
                    {
                        'name': f[0],
                        'accuracy_contribution': f[1],
                        'directional_accuracy': f[2]
                    } for f in top_features
                ],
                'market_condition_performance': [
                    {
                        'condition_type': m[0],
                        'condition_value': m[1],
                        'accuracy_rate': m[2],
                        'sample_size': m[3]
                    } for m in market_performance
                ],
                'strategy_performance': [
                    {
                        'strategy': s[0],
                        'success_rate': s[1],
                        'contributions': s[2]
                    } for s in strategy_performance
                ],
                'analysis_date': datetime.now().isoformat()
            }
            
            conn.close()
            return analysis_report
            
        except Exception as e:
            logger.error(f"❌ Failed to get comprehensive analysis: {e}")
            return {'error': str(e)}

# Global outcome tracker instance
outcome_tracker = None

def initialize_outcome_tracker(validator: PredictionValidator):
    """Initialize global outcome tracker"""
    global outcome_tracker
    outcome_tracker = OutcomeTracker(validator)
    return outcome_tracker

def start_outcome_tracking():
    """Start the global outcome tracking service"""
    if outcome_tracker:
        outcome_tracker.start_tracking_service()

def get_outcome_analysis(days: int = 7) -> Dict[str, Any]:
    """Get comprehensive outcome analysis"""
    if outcome_tracker:
        return outcome_tracker.get_comprehensive_analysis(days)
    return {'error': 'Outcome tracker not initialized'}

if __name__ == "__main__":
    from prediction_validator import PredictionValidator
    
    print("📊 Starting Outcome Tracker...")
    validator = PredictionValidator()
    tracker = OutcomeTracker(validator)
    tracker.start_tracking_service()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("⏹️ Stopping Outcome Tracker...")
        tracker.stop_tracking_service()
