"""
📈 PHASE 3: PERFORMANCE ANALYTICS DASHBOARD
===========================================

PerformanceAnalytics - Real-time accuracy visualization and analytics
Strategy contribution analysis, learning progress metrics, and confidence calibration

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
from flask import Blueprint, request, jsonify, render_template
from prediction_validator import PredictionValidator
from advanced_learning_engine import AdvancedLearningEngine
from outcome_tracker import OutcomeTracker
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('performance_analytics')

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    overall_accuracy: float
    directional_accuracy: float
    price_accuracy: float
    confidence_calibration: float
    prediction_volume: int
    strategy_contributions: Dict[str, float]
    learning_progress: Dict[str, Any]
    feature_importance_trends: Dict[str, float]

class PerformanceAnalytics:
    """
    Real-time performance analytics dashboard
    Provides comprehensive visualization and analysis of prediction performance
    """
    
    def __init__(self, validator: PredictionValidator, learning_engine: AdvancedLearningEngine, 
                 outcome_tracker: OutcomeTracker, db_path: str = "goldgpt_performance_analytics.db"):
        self.validator = validator
        self.learning_engine = learning_engine
        self.outcome_tracker = outcome_tracker
        self.db_path = db_path
        self.init_database()
        self.analytics_thread = None
        self.is_running = False
        self.current_metrics = None
        logger.info("📈 Performance Analytics initialized")
    
    def init_database(self):
        """Initialize analytics database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create real-time metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    overall_accuracy REAL NOT NULL,
                    directional_accuracy REAL NOT NULL,
                    price_accuracy REAL NOT NULL,
                    confidence_calibration REAL NOT NULL,
                    prediction_volume INTEGER NOT NULL,
                    strategy_contributions TEXT NOT NULL,
                    learning_progress TEXT NOT NULL,
                    feature_importance_trends TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create accuracy trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accuracy_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    hour INTEGER NOT NULL,
                    accuracy_1h REAL NOT NULL,
                    accuracy_4h REAL NOT NULL,
                    accuracy_1d REAL NOT NULL,
                    volume_1h INTEGER NOT NULL,
                    volume_4h INTEGER NOT NULL,
                    volume_1d INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create confidence calibration table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS confidence_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    confidence_bucket TEXT NOT NULL,
                    predicted_accuracy REAL NOT NULL,
                    actual_accuracy REAL NOT NULL,
                    prediction_count INTEGER NOT NULL,
                    calibration_score REAL NOT NULL,
                    analysis_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    contribution_percentage REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    predictions_count INTEGER NOT NULL,
                    profit_contribution REAL NOT NULL,
                    improvement_rate REAL NOT NULL,
                    analysis_timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Performance analytics database initialized")
            
        except Exception as e:
            logger.error(f"❌ Analytics database initialization failed: {e}")
    
    def start_analytics_service(self):
        """Start real-time analytics service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
        self.analytics_thread.start()
        logger.info("🚀 Performance analytics service started")
    
    def stop_analytics_service(self):
        """Stop analytics service"""
        self.is_running = False
        if self.analytics_thread:
            self.analytics_thread.join()
        logger.info("⏹️ Performance analytics service stopped")
    
    def _analytics_loop(self):
        """Main analytics loop"""
        while self.is_running:
            try:
                # Update metrics every 5 minutes
                self.update_realtime_metrics()
                self.calculate_accuracy_trends()
                self.analyze_confidence_calibration()
                self.analyze_strategy_performance()
                
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Analytics loop error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def update_realtime_metrics(self):
        """Update real-time performance metrics"""
        try:
            # Get recent performance data
            validation_metrics = self.validator.get_recent_performance(days=1)
            learning_progress = self.learning_engine.get_learning_progress(days=7)
            outcome_analysis = self.outcome_tracker.get_comprehensive_analysis(days=1)
            
            # Calculate overall metrics
            overall_accuracy = validation_metrics.get('current_win_rate', 0.0)
            directional_accuracy = outcome_analysis.get('summary_statistics', {}).get('directional_accuracy', 0.0)
            price_accuracy = 1.0 - (outcome_analysis.get('summary_statistics', {}).get('avg_price_error', 0.0) / 100)
            price_accuracy = max(0, min(1, price_accuracy))
            
            # Calculate confidence calibration
            confidence_calibration = self._calculate_confidence_calibration()
            
            # Get prediction volume
            prediction_volume = validation_metrics.get('total_predictions', 0)
            
            # Get strategy contributions
            strategy_contributions = validation_metrics.get('strategy_performance', {})
            
            # Get feature importance trends
            feature_importance_trends = self._get_feature_importance_trends()
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                overall_accuracy=overall_accuracy,
                directional_accuracy=directional_accuracy,
                price_accuracy=price_accuracy,
                confidence_calibration=confidence_calibration,
                prediction_volume=prediction_volume,
                strategy_contributions=strategy_contributions,
                learning_progress=learning_progress,
                feature_importance_trends=feature_importance_trends
            )
            
            # Store metrics
            self._store_realtime_metrics(metrics)
            self.current_metrics = metrics
            
            logger.info(f"✅ Real-time metrics updated: {overall_accuracy:.3f} accuracy")
            
        except Exception as e:
            logger.error(f"❌ Real-time metrics update failed: {e}")
    
    def _calculate_confidence_calibration(self) -> float:
        """Calculate confidence calibration score"""
        try:
            conn = sqlite3.connect(self.validator.db_path)
            cursor = conn.cursor()
            
            # Get recent predictions with confidence and outcome
            cursor.execute('''
                SELECT confidence, 
                       CASE WHEN outcome = 'correct' THEN 1.0 ELSE 0.0 END as actual_outcome
                FROM predictions
                WHERE outcome IS NOT NULL
                AND prediction_timestamp >= date('now', '-7 days')
            ''')
            
            predictions = cursor.fetchall()
            conn.close()
            
            if len(predictions) < 10:
                return 0.5  # Default calibration score
            
            # Group by confidence buckets
            confidence_buckets = {}
            for conf, outcome in predictions:
                bucket = int(conf * 10) / 10  # Round to nearest 0.1
                if bucket not in confidence_buckets:
                    confidence_buckets[bucket] = {'predictions': [], 'outcomes': []}
                
                confidence_buckets[bucket]['predictions'].append(conf)
                confidence_buckets[bucket]['outcomes'].append(outcome)
            
            # Calculate calibration score
            calibration_error = 0.0
            total_predictions = 0
            
            for bucket, data in confidence_buckets.items():
                if len(data['outcomes']) >= 3:  # Minimum sample size
                    predicted_accuracy = np.mean(data['predictions'])
                    actual_accuracy = np.mean(data['outcomes'])
                    bucket_size = len(data['outcomes'])
                    
                    # Weight by bucket size
                    calibration_error += abs(predicted_accuracy - actual_accuracy) * bucket_size
                    total_predictions += bucket_size
            
            if total_predictions > 0:
                calibration_error /= total_predictions
                calibration_score = max(0, 1 - calibration_error)
            else:
                calibration_score = 0.5
            
            return calibration_score
            
        except Exception as e:
            logger.error(f"❌ Confidence calibration calculation failed: {e}")
            return 0.5
    
    def _get_feature_importance_trends(self) -> Dict[str, float]:
        """Get feature importance trends"""
        try:
            if not self.learning_engine:
                return {}
            
            conn = sqlite3.connect(self.learning_engine.db_path)
            cursor = conn.cursor()
            
            # Get recent feature importance data
            cursor.execute('''
                SELECT feature_name, AVG(importance_score) as avg_importance
                FROM feature_evolution
                WHERE calculation_date >= date('now', '-7 days')
                GROUP BY feature_name
                ORDER BY avg_importance DESC
                LIMIT 10
            ''')
            
            features = cursor.fetchall()
            conn.close()
            
            feature_trends = {}
            for feature_name, importance in features:
                feature_trends[feature_name] = importance
            
            return feature_trends
            
        except Exception as e:
            logger.error(f"❌ Feature importance trends failed: {e}")
            return {}
    
    def _store_realtime_metrics(self, metrics: PerformanceMetrics):
        """Store real-time metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_metrics (
                    timestamp, overall_accuracy, directional_accuracy, price_accuracy,
                    confidence_calibration, prediction_volume, strategy_contributions,
                    learning_progress, feature_importance_trends
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp,
                metrics.overall_accuracy,
                metrics.directional_accuracy,
                metrics.price_accuracy,
                metrics.confidence_calibration,
                metrics.prediction_volume,
                json.dumps(metrics.strategy_contributions),
                json.dumps(metrics.learning_progress),
                json.dumps(metrics.feature_importance_trends)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store real-time metrics: {e}")
    
    def calculate_accuracy_trends(self):
        """Calculate accuracy trends by timeframe"""
        try:
            conn = sqlite3.connect(self.validator.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now()
            current_date = current_time.date()
            current_hour = current_time.hour
            
            # Calculate accuracy for different timeframes
            timeframes = ['1H', '4H', '1D']
            timeframe_hours = {'1H': 1, '4H': 4, '1D': 24}
            
            accuracies = {}
            volumes = {}
            
            for tf in timeframes:
                hours = timeframe_hours[tf]
                
                cursor.execute('''
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) as correct
                    FROM predictions
                    WHERE timeframe = ?
                    AND outcome IS NOT NULL
                    AND prediction_timestamp >= datetime('now', '-{} hours')
                '''.format(hours), (tf,))
                
                result = cursor.fetchone()
                total = result[0] if result else 0
                correct = result[1] if result else 0
                
                accuracy = correct / total if total > 0 else 0.0
                accuracies[tf] = accuracy
                volumes[tf] = total
            
            # Store trend data
            conn_analytics = sqlite3.connect(self.db_path)
            cursor_analytics = conn_analytics.cursor()
            
            cursor_analytics.execute('''
                INSERT OR REPLACE INTO accuracy_trends (
                    date, hour, accuracy_1h, accuracy_4h, accuracy_1d,
                    volume_1h, volume_4h, volume_1d
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_date, current_hour,
                accuracies.get('1H', 0.0),
                accuracies.get('4H', 0.0),
                accuracies.get('1D', 0.0),
                volumes.get('1H', 0),
                volumes.get('4H', 0),
                volumes.get('1D', 0)
            ))
            
            conn_analytics.commit()
            conn_analytics.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Accuracy trends calculation failed: {e}")
    
    def analyze_confidence_calibration(self):
        """Analyze confidence calibration in detail"""
        try:
            conn = sqlite3.connect(self.validator.db_path)
            cursor = conn.cursor()
            
            # Get predictions from last 7 days
            cursor.execute('''
                SELECT confidence, 
                       CASE WHEN outcome = 'correct' THEN 1.0 ELSE 0.0 END as actual_outcome
                FROM predictions
                WHERE outcome IS NOT NULL
                AND prediction_timestamp >= date('now', '-7 days')
            ''')
            
            predictions = cursor.fetchall()
            conn.close()
            
            if len(predictions) < 20:
                return
            
            # Create confidence buckets
            bucket_ranges = [
                (0.0, 0.5, 'low'),
                (0.5, 0.7, 'medium'),
                (0.7, 0.85, 'high'),
                (0.85, 1.0, 'very_high')
            ]
            
            calibration_data = []
            
            for min_conf, max_conf, bucket_name in bucket_ranges:
                bucket_predictions = [
                    (conf, outcome) for conf, outcome in predictions
                    if min_conf <= conf < max_conf
                ]
                
                if len(bucket_predictions) >= 5:
                    confidences = [p[0] for p in bucket_predictions]
                    outcomes = [p[1] for p in bucket_predictions]
                    
                    predicted_accuracy = np.mean(confidences)
                    actual_accuracy = np.mean(outcomes)
                    calibration_score = 1 - abs(predicted_accuracy - actual_accuracy)
                    
                    calibration_data.append({
                        'bucket': bucket_name,
                        'predicted_accuracy': predicted_accuracy,
                        'actual_accuracy': actual_accuracy,
                        'count': len(bucket_predictions),
                        'calibration_score': calibration_score
                    })
            
            # Store calibration analysis
            conn_analytics = sqlite3.connect(self.db_path)
            cursor_analytics = conn_analytics.cursor()
            
            for data in calibration_data:
                cursor_analytics.execute('''
                    INSERT OR REPLACE INTO confidence_calibration (
                        confidence_bucket, predicted_accuracy, actual_accuracy,
                        prediction_count, calibration_score, analysis_date
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['bucket'],
                    data['predicted_accuracy'],
                    data['actual_accuracy'],
                    data['count'],
                    data['calibration_score'],
                    datetime.now().date()
                ))
            
            conn_analytics.commit()
            conn_analytics.close()
            
        except Exception as e:
            logger.error(f"❌ Confidence calibration analysis failed: {e}")
    
    def analyze_strategy_performance(self):
        """Analyze detailed strategy performance"""
        try:
            conn = sqlite3.connect(self.validator.db_path)
            cursor = conn.cursor()
            
            # Get strategy performance data
            cursor.execute('''
                SELECT contributing_strategies, outcome, confidence, profit_factor
                FROM predictions
                WHERE outcome IS NOT NULL
                AND prediction_timestamp >= date('now', '-7 days')
            ''')
            
            predictions = cursor.fetchall()
            conn.close()
            
            if len(predictions) < 10:
                return
            
            strategy_stats = {}
            
            for pred in predictions:
                strategies = json.loads(pred[0])
                outcome = pred[1]
                confidence = pred[2] or 0.0
                profit_factor = pred[3] or 0.0
                
                for strategy in strategies:
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = {
                            'total': 0,
                            'successful': 0,
                            'confidences': [],
                            'profit_factors': []
                        }
                    
                    stats = strategy_stats[strategy]
                    stats['total'] += 1
                    stats['confidences'].append(confidence)
                    stats['profit_factors'].append(profit_factor)
                    
                    if outcome == 'correct':
                        stats['successful'] += 1
            
            # Calculate performance metrics and store
            conn_analytics = sqlite3.connect(self.db_path)
            cursor_analytics = conn_analytics.cursor()
            
            total_contributions = sum(stats['total'] for stats in strategy_stats.values())
            
            for strategy_name, stats in strategy_stats.items():
                if stats['total'] >= 5:  # Minimum sample size
                    contribution_percentage = (stats['total'] / total_contributions) * 100
                    success_rate = stats['successful'] / stats['total']
                    avg_confidence = np.mean(stats['confidences'])
                    profit_contribution = np.mean(stats['profit_factors'])
                    
                    # Calculate improvement rate (simplified)
                    improvement_rate = success_rate - 0.5  # Compared to random
                    
                    cursor_analytics.execute('''
                        INSERT OR REPLACE INTO strategy_performance_analytics (
                            strategy_name, contribution_percentage, success_rate,
                            avg_confidence, predictions_count, profit_contribution,
                            improvement_rate, analysis_timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        strategy_name,
                        contribution_percentage,
                        success_rate,
                        avg_confidence,
                        stats['total'],
                        profit_contribution,
                        improvement_rate,
                        datetime.now()
                    ))
            
            conn_analytics.commit()
            conn_analytics.close()
            
        except Exception as e:
            logger.error(f"❌ Strategy performance analysis failed: {e}")
    
    def get_dashboard_data(self, timeframe: str = '24h') -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            timeframe_hours = {
                '1h': 1, '4h': 4, '12h': 12, '24h': 24, 
                '7d': 168, '30d': 720
            }
            
            hours = timeframe_hours.get(timeframe, 24)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent metrics
            cursor.execute('''
                SELECT * FROM realtime_metrics
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
                LIMIT 100
            '''.format(hours))
            
            metrics_data = cursor.fetchall()
            
            # Get accuracy trends
            cursor.execute('''
                SELECT * FROM accuracy_trends
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC, hour DESC
                LIMIT 50
            '''.format(hours // 24 + 1))
            
            trends_data = cursor.fetchall()
            
            # Get confidence calibration
            cursor.execute('''
                SELECT * FROM confidence_calibration
                WHERE analysis_date >= date('now', '-7 days')
                ORDER BY analysis_date DESC
            ''')
            
            calibration_data = cursor.fetchall()
            
            # Get strategy performance
            cursor.execute('''
                SELECT * FROM strategy_performance_analytics
                WHERE analysis_timestamp >= datetime('now', '-{} hours')
                ORDER BY analysis_timestamp DESC
            '''.format(hours))
            
            strategy_data = cursor.fetchall()
            
            conn.close()
            
            # Format dashboard data
            dashboard_data = {
                'current_metrics': self._format_current_metrics(),
                'accuracy_trends': self._format_accuracy_trends(trends_data),
                'confidence_calibration': self._format_calibration_data(calibration_data),
                'strategy_performance': self._format_strategy_data(strategy_data),
                'real_time_metrics': self._format_realtime_data(metrics_data),
                'timeframe': timeframe,
                'last_updated': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Dashboard data retrieval failed: {e}")
            return {'error': str(e)}
    
    def _format_current_metrics(self) -> Dict[str, Any]:
        """Format current metrics for dashboard"""
        if not self.current_metrics:
            return {
                'overall_accuracy': 0.0,
                'directional_accuracy': 0.0,
                'price_accuracy': 0.0,
                'confidence_calibration': 0.0,
                'prediction_volume': 0
            }
        
        return {
            'overall_accuracy': self.current_metrics.overall_accuracy,
            'directional_accuracy': self.current_metrics.directional_accuracy,
            'price_accuracy': self.current_metrics.price_accuracy,
            'confidence_calibration': self.current_metrics.confidence_calibration,
            'prediction_volume': self.current_metrics.prediction_volume,
            'strategy_contributions': self.current_metrics.strategy_contributions,
            'top_features': self.current_metrics.feature_importance_trends
        }
    
    def _format_accuracy_trends(self, trends_data: List) -> List[Dict[str, Any]]:
        """Format accuracy trends for dashboard"""
        formatted_trends = []
        
        for trend in trends_data:
            formatted_trends.append({
                'timestamp': f"{trend[1]}-{trend[2]:02d}:00",  # date-hour
                '1h_accuracy': trend[3],
                '4h_accuracy': trend[4],
                '1d_accuracy': trend[5],
                '1h_volume': trend[6],
                '4h_volume': trend[7],
                '1d_volume': trend[8]
            })
        
        return formatted_trends
    
    def _format_calibration_data(self, calibration_data: List) -> List[Dict[str, Any]]:
        """Format confidence calibration data"""
        formatted_calibration = []
        
        for cal in calibration_data:
            formatted_calibration.append({
                'confidence_bucket': cal[1],
                'predicted_accuracy': cal[2],
                'actual_accuracy': cal[3],
                'prediction_count': cal[4],
                'calibration_score': cal[5]
            })
        
        return formatted_calibration
    
    def _format_strategy_data(self, strategy_data: List) -> List[Dict[str, Any]]:
        """Format strategy performance data"""
        formatted_strategies = []
        
        for strategy in strategy_data:
            formatted_strategies.append({
                'strategy_name': strategy[1],
                'contribution_percentage': strategy[2],
                'success_rate': strategy[3],
                'avg_confidence': strategy[4],
                'predictions_count': strategy[5],
                'profit_contribution': strategy[6],
                'improvement_rate': strategy[7]
            })
        
        return formatted_strategies
    
    def _format_realtime_data(self, metrics_data: List) -> List[Dict[str, Any]]:
        """Format real-time metrics data"""
        formatted_metrics = []
        
        for metric in metrics_data:
            formatted_metrics.append({
                'timestamp': metric[1],
                'overall_accuracy': metric[2],
                'directional_accuracy': metric[3],
                'price_accuracy': metric[4],
                'confidence_calibration': metric[5],
                'prediction_volume': metric[6]
            })
        
        return formatted_metrics

# Create Flask Blueprint for analytics API
analytics_bp = Blueprint('analytics', __name__)

# Global analytics instance
performance_analytics = None

def initialize_performance_analytics(validator: PredictionValidator, 
                                   learning_engine: AdvancedLearningEngine,
                                   outcome_tracker: OutcomeTracker):
    """Initialize global performance analytics"""
    global performance_analytics
    performance_analytics = PerformanceAnalytics(validator, learning_engine, outcome_tracker)
    return performance_analytics

def start_performance_analytics():
    """Start the global performance analytics service"""
    if performance_analytics:
        performance_analytics.start_analytics_service()

@analytics_bp.route('/api/analytics/dashboard')
def get_analytics_dashboard():
    """Get analytics dashboard data"""
    try:
        timeframe = request.args.get('timeframe', '24h')
        
        if performance_analytics:
            dashboard_data = performance_analytics.get_dashboard_data(timeframe)
            return jsonify(dashboard_data)
        else:
            return jsonify({'error': 'Performance analytics not initialized'}), 500
            
    except Exception as e:
        logger.error(f"❌ Dashboard API error: {e}")
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/api/analytics/current-metrics')
def get_current_metrics():
    """Get current performance metrics"""
    try:
        if performance_analytics and performance_analytics.current_metrics:
            return jsonify(asdict(performance_analytics.current_metrics))
        else:
            return jsonify({'error': 'No current metrics available'}), 404
            
    except Exception as e:
        logger.error(f"❌ Current metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/analytics')
def analytics_dashboard():
    """Render analytics dashboard"""
    return render_template('performance_analytics_dashboard.html')

if __name__ == "__main__":
    print("📈 Starting Performance Analytics...")
    
    # This would normally be initialized with actual components
    print("⚠️ Performance Analytics requires validator, learning engine, and outcome tracker")
    print("   Initialize with: initialize_performance_analytics(validator, learning_engine, outcome_tracker)")
    print("   Start with: start_performance_analytics()")
