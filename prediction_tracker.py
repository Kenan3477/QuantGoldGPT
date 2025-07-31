#!/usr/bin/env python3
"""
Advanced Prediction Tracking System for GoldGPT
Stores, validates, and analyzes ML predictions for continuous learning
"""

import sqlite3
import json
import uuid
import logging
import asyncio
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    """Data structure for a complete prediction record"""
    prediction_id: str
    created_at: datetime
    prediction_time: datetime
    target_time: datetime
    symbol: str
    timeframe: str
    strategy_name: str
    engine_version: str
    current_price: float
    predicted_price: float
    price_change_percent: float
    direction: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    position_size_suggested: Optional[float] = None
    confidence: float = 0.5
    signal_strength: Optional[float] = None
    model_uncertainty: Optional[float] = None
    ensemble_agreement: Optional[float] = None
    features_used: Optional[List[str]] = None
    feature_weights: Optional[Dict[str, float]] = None
    technical_indicators: Optional[Dict[str, float]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    strategy_votes: Optional[Dict[str, float]] = None
    strategy_weights: Optional[Dict[str, float]] = None
    market_regime: Optional[str] = None
    news_impact_score: Optional[float] = None
    volatility_regime: Optional[str] = None
    trend_strength: Optional[float] = None
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

@dataclass
class ValidationResult:
    """Data structure for prediction validation results"""
    prediction_id: str
    validation_time: datetime
    actual_price: float
    prediction_error_percent: float
    direction_correct: bool
    magnitude_accuracy_score: float
    hit_stop_loss: bool = False
    hit_take_profit: bool = False
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    simulated_pnl_pips: Optional[float] = None
    simulated_pnl_percent: Optional[float] = None
    volatility_realized: Optional[float] = None
    trend_strength_realized: Optional[float] = None
    news_events_count: int = 0
    prediction_quality_score: float = 0.0
    confidence_calibration_error: Optional[float] = None

class PredictionTracker:
    """
    Advanced prediction tracking and validation system
    
    Features:
    - Stores all prediction metadata with comprehensive context
    - Validates predictions against actual market outcomes
    - Calculates detailed accuracy and performance metrics
    - Tracks feature importance and strategy effectiveness
    - Provides insights for model improvement
    """
    
    def __init__(self, db_path: str = "goldgpt_prediction_learning.db"):
        self.db_path = Path(db_path)
        self.schema_path = Path(__file__).parent / "prediction_learning_schema.sql"
        self._initialize_database()
        logger.info(f"PredictionTracker initialized with database: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize the database with the required schema"""
        try:
            with self._get_connection() as conn:
                # Load and execute schema
                if self.schema_path.exists():
                    with open(self.schema_path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    
                    # Execute schema creation (handle multiple statements)
                    for statement in schema_sql.split(';'):
                        statement = statement.strip()
                        if statement:
                            conn.execute(statement)
                    
                    logger.info("Database schema initialized successfully")
                else:
                    logger.warning(f"Schema file not found: {self.schema_path}")
                    self._create_basic_tables(conn)
                    
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_basic_tables(self, conn: sqlite3.Connection):
        """Create basic tables if schema file is not available"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction_time TIMESTAMP NOT NULL,
                target_time TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL DEFAULT 'XAUUSD',
                timeframe TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                price_change_percent REAL NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_validated BOOLEAN DEFAULT FALSE,
                is_winner BOOLEAN NULL,
                actual_price REAL NULL,
                prediction_error REAL NULL,
                direction_correct BOOLEAN NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                validation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                actual_price REAL NOT NULL,
                prediction_error_percent REAL NOT NULL,
                direction_correct BOOLEAN NOT NULL,
                prediction_quality_score REAL NOT NULL
            )
        """)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            conn.close()
    
    def store_prediction(self, prediction: Union[PredictionRecord, Dict[str, Any]]) -> str:
        """
        Store a new prediction record
        
        Args:
            prediction: PredictionRecord object or dictionary with prediction data
            
        Returns:
            str: The prediction ID
        """
        try:
            if isinstance(prediction, dict):
                # Convert dict to PredictionRecord
                prediction_id = prediction.get('prediction_id', str(uuid.uuid4()))
                prediction = PredictionRecord(
                    prediction_id=prediction_id,
                    **prediction
                )
            
            with self._get_connection() as conn:
                # Convert dataclass to dict for database insertion
                data = asdict(prediction)
                
                # Convert datetime objects to ISO strings
                for key, value in data.items():
                    if isinstance(value, datetime):
                        data[key] = value.isoformat()
                    elif isinstance(value, (dict, list)) and value is not None:
                        data[key] = json.dumps(value)
                
                # Insert prediction record
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['?' for _ in data])
                
                conn.execute(f"""
                    INSERT OR REPLACE INTO prediction_records ({columns})
                    VALUES ({placeholders})
                """, list(data.values()))
                
                logger.info(f"Stored prediction: {prediction.prediction_id}")
                return prediction.prediction_id
                
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            raise
    
    def validate_prediction(self, prediction_id: str, actual_price: float, 
                          price_path: Optional[List[float]] = None) -> ValidationResult:
        """
        Validate a prediction against actual market outcome
        
        Args:
            prediction_id: The prediction to validate
            actual_price: The actual price at target time
            price_path: Optional list of prices during the prediction period
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        try:
            with self._get_connection() as conn:
                # Get the original prediction
                prediction_row = conn.execute("""
                    SELECT * FROM prediction_records 
                    WHERE prediction_id = ?
                """, (prediction_id,)).fetchone()
                
                if not prediction_row:
                    raise ValueError(f"Prediction not found: {prediction_id}")
                
                prediction = dict(prediction_row)
                
                # Calculate validation metrics
                predicted_price = prediction['predicted_price']
                current_price = prediction['current_price']
                
                # Prediction error
                prediction_error_percent = abs(actual_price - predicted_price) / current_price * 100
                
                # Direction accuracy
                predicted_direction = prediction['direction']
                actual_direction = 'bullish' if actual_price > current_price else 'bearish'
                direction_correct = predicted_direction == actual_direction
                
                # Magnitude accuracy (how close was the predicted change)
                predicted_change = abs(predicted_price - current_price) / current_price
                actual_change = abs(actual_price - current_price) / current_price
                magnitude_accuracy = 1 - abs(predicted_change - actual_change) / max(predicted_change, actual_change, 0.001)
                
                # Calculate excursions if price path provided
                mfe = mae = 0.0
                hit_sl = hit_tp = False
                
                if price_path:
                    mfe, mae = self._calculate_excursions(current_price, price_path, predicted_direction)
                    
                    # Check if stop loss or take profit were hit
                    if prediction.get('stop_loss'):
                        hit_sl = any(p <= prediction['stop_loss'] for p in price_path) if predicted_direction == 'bullish' else any(p >= prediction['stop_loss'] for p in price_path)
                    
                    if prediction.get('take_profit'):
                        hit_tp = any(p >= prediction['take_profit'] for p in price_path) if predicted_direction == 'bullish' else any(p <= prediction['take_profit'] for p in price_path)
                
                # Calculate overall quality score
                quality_score = self._calculate_quality_score(
                    direction_correct, magnitude_accuracy, 
                    prediction['confidence'], prediction_error_percent
                )
                
                # Create validation result
                validation_result = ValidationResult(
                    prediction_id=prediction_id,
                    validation_time=datetime.now(timezone.utc),
                    actual_price=actual_price,
                    prediction_error_percent=prediction_error_percent,
                    direction_correct=direction_correct,
                    magnitude_accuracy_score=magnitude_accuracy,
                    hit_stop_loss=hit_sl,
                    hit_take_profit=hit_tp,
                    max_favorable_excursion=mfe,
                    max_adverse_excursion=mae,
                    prediction_quality_score=quality_score
                )
                
                # Store validation result
                self._store_validation_result(conn, validation_result)
                
                # Update prediction record with validation data
                is_winner = direction_correct and (not hit_sl)
                
                conn.execute("""
                    UPDATE prediction_records 
                    SET is_validated = 1,
                        actual_price = ?,
                        prediction_error = ?,
                        direction_correct = ?,
                        is_winner = ?,
                        hit_stop_loss = ?,
                        hit_take_profit = ?,
                        max_favorable_excursion = ?,
                        max_adverse_excursion = ?,
                        validation_time = ?
                    WHERE prediction_id = ?
                """, (actual_price, prediction_error_percent, direction_correct,
                     is_winner, hit_sl, hit_tp, mfe, mae,
                     validation_result.validation_time.isoformat(), prediction_id))
                
                logger.info(f"Validated prediction {prediction_id}: {'WIN' if is_winner else 'LOSS'}")
                return validation_result
                
        except Exception as e:
            logger.error(f"Failed to validate prediction {prediction_id}: {e}")
            raise
    
    def _calculate_excursions(self, entry_price: float, price_path: List[float], 
                             direction: str) -> Tuple[float, float]:
        """Calculate maximum favorable and adverse excursions"""
        if not price_path:
            return 0.0, 0.0
        
        if direction == 'bullish':
            mfe = max(p - entry_price for p in price_path)
            mae = min(p - entry_price for p in price_path)
        else:
            mfe = max(entry_price - p for p in price_path)
            mae = min(entry_price - p for p in price_path)
        
        return max(mfe, 0.0), abs(min(mae, 0.0))
    
    def _calculate_quality_score(self, direction_correct: bool, magnitude_accuracy: float,
                               confidence: float, error_percent: float) -> float:
        """Calculate overall prediction quality score (0-1)"""
        direction_score = 1.0 if direction_correct else 0.0
        magnitude_score = max(0, min(1, magnitude_accuracy))
        
        # Confidence calibration (penalize overconfidence on wrong predictions)
        confidence_penalty = 0
        if not direction_correct and confidence > 0.7:
            confidence_penalty = (confidence - 0.7) * 0.5
        
        # Error penalty (lower scores for higher errors)
        error_score = max(0, 1 - error_percent / 10.0)  # 10% error = 0 score
        
        # Weighted combination
        quality_score = (
            direction_score * 0.4 +
            magnitude_score * 0.3 +
            error_score * 0.2 +
            (confidence * direction_score) * 0.1 -
            confidence_penalty
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _store_validation_result(self, conn: sqlite3.Connection, result: ValidationResult):
        """Store validation result in database"""
        data = asdict(result)
        
        # Convert datetime to ISO string
        data['validation_time'] = result.validation_time.isoformat()
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        
        conn.execute(f"""
            INSERT OR REPLACE INTO validation_results ({columns})
            VALUES ({placeholders})
        """, list(data.values()))
    
    def get_strategy_performance(self, strategy_name: str, timeframe: str,
                               days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for a strategy
        
        Args:
            strategy_name: Name of the strategy
            timeframe: Timeframe to analyze
            days: Number of days to look back
            
        Returns:
            Dict with comprehensive performance metrics
        """
        try:
            with self._get_connection() as conn:
                since_date = datetime.now() - timedelta(days=days)
                
                # Get all validated predictions for the strategy
                predictions = conn.execute("""
                    SELECT * FROM prediction_records 
                    WHERE strategy_name = ? 
                    AND timeframe = ?
                    AND is_validated = 1
                    AND created_at >= ?
                    ORDER BY created_at DESC
                """, (strategy_name, timeframe, since_date.isoformat())).fetchall()
                
                if not predictions:
                    return {
                        'strategy_name': strategy_name,
                        'timeframe': timeframe,
                        'total_predictions': 0,
                        'error': 'No validated predictions found'
                    }
                
                # Convert to list of dicts for easier processing
                predictions = [dict(row) for row in predictions]
                
                # Calculate basic metrics
                total_predictions = len(predictions)
                wins = sum(1 for p in predictions if p['is_winner'])
                losses = total_predictions - wins
                win_rate = wins / total_predictions if total_predictions > 0 else 0
                
                # Calculate error metrics
                errors = [p['prediction_error'] for p in predictions if p['prediction_error'] is not None]
                mean_error = statistics.mean(errors) if errors else 0
                std_error = statistics.stdev(errors) if len(errors) > 1 else 0
                
                # Calculate confidence metrics
                confidences = [p['confidence'] for p in predictions]
                mean_confidence = statistics.mean(confidences)
                
                # Direction accuracy
                direction_correct = sum(1 for p in predictions if p['direction_correct'])
                direction_accuracy = direction_correct / total_predictions
                
                # P&L metrics (if available)
                pnl_data = [p for p in predictions if p.get('profit_loss_percent')]
                total_pnl = sum(p['profit_loss_percent'] for p in pnl_data) if pnl_data else 0
                
                # Recent performance trend
                recent_predictions = predictions[:10]  # Last 10 predictions
                recent_win_rate = sum(1 for p in recent_predictions if p['is_winner']) / len(recent_predictions) if recent_predictions else 0
                
                # Calculate additional metrics
                max_consecutive_wins = self._calculate_max_consecutive(predictions, True)
                max_consecutive_losses = self._calculate_max_consecutive(predictions, False)
                
                performance_metrics = {
                    'strategy_name': strategy_name,
                    'timeframe': timeframe,
                    'analysis_period_days': days,
                    'total_predictions': total_predictions,
                    'validated_predictions': total_predictions,
                    'winning_predictions': wins,
                    'losing_predictions': losses,
                    'win_rate': round(win_rate * 100, 2),
                    'loss_rate': round((1 - win_rate) * 100, 2),
                    'direction_accuracy': round(direction_accuracy * 100, 2),
                    'mean_prediction_error': round(mean_error, 4),
                    'std_prediction_error': round(std_error, 4),
                    'mean_confidence': round(mean_confidence, 3),
                    'total_pnl_percent': round(total_pnl, 2),
                    'recent_win_rate': round(recent_win_rate * 100, 2),
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses,
                    'first_prediction': predictions[-1]['created_at'] if predictions else None,
                    'last_prediction': predictions[0]['created_at'] if predictions else None,
                    'performance_trend': 'improving' if recent_win_rate > win_rate else 'declining' if recent_win_rate < win_rate else 'stable'
                }
                
                return performance_metrics
                
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return {'error': str(e)}
    
    def _calculate_max_consecutive(self, predictions: List[Dict], winners: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for prediction in predictions:
            if (prediction['is_winner'] and winners) or (not prediction['is_winner'] and not winners):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def get_feature_importance_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze which features are most predictive of successful outcomes
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with feature importance analysis
        """
        try:
            with self._get_connection() as conn:
                since_date = datetime.now() - timedelta(days=days)
                
                predictions = conn.execute("""
                    SELECT feature_weights, features_used, is_winner, prediction_error
                    FROM prediction_records 
                    WHERE is_validated = 1 
                    AND created_at >= ?
                    AND feature_weights IS NOT NULL
                """, (since_date.isoformat(),)).fetchall()
                
                if not predictions:
                    return {'error': 'No predictions with feature data found'}
                
                # Aggregate feature performance
                feature_performance = {}
                feature_counts = {}
                
                for row in predictions:
                    try:
                        feature_weights = json.loads(row['feature_weights']) if row['feature_weights'] else {}
                        is_winner = bool(row['is_winner'])
                        error = float(row['prediction_error']) if row['prediction_error'] else 0
                        
                        for feature, weight in feature_weights.items():
                            if feature not in feature_performance:
                                feature_performance[feature] = {'wins': 0, 'total': 0, 'weight_sum': 0, 'error_sum': 0}
                            
                            feature_performance[feature]['total'] += 1
                            feature_performance[feature]['weight_sum'] += weight
                            feature_performance[feature]['error_sum'] += error
                            
                            if is_winner:
                                feature_performance[feature]['wins'] += 1
                                
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse feature data: {e}")
                        continue
                
                # Calculate importance scores
                importance_analysis = {}
                for feature, stats in feature_performance.items():
                    if stats['total'] > 0:
                        win_rate = stats['wins'] / stats['total']
                        avg_weight = stats['weight_sum'] / stats['total']
                        avg_error = stats['error_sum'] / stats['total']
                        
                        # Importance score combines win rate, usage frequency, and low error
                        importance_score = (win_rate * 0.5 + (1 - avg_error/10) * 0.3 + avg_weight * 0.2)
                        
                        importance_analysis[feature] = {
                            'importance_score': round(importance_score, 3),
                            'win_rate': round(win_rate * 100, 1),
                            'usage_count': stats['total'],
                            'average_weight': round(avg_weight, 3),
                            'average_error': round(avg_error, 4)
                        }
                
                # Sort by importance score
                sorted_features = sorted(importance_analysis.items(), 
                                       key=lambda x: x[1]['importance_score'], reverse=True)
                
                return {
                    'analysis_period_days': days,
                    'total_predictions_analyzed': len(predictions),
                    'features_analyzed': len(importance_analysis),
                    'most_important_features': dict(sorted_features[:10]),
                    'least_important_features': dict(sorted_features[-5:]),
                    'feature_rankings': [{'feature': k, **v} for k, v in sorted_features]
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {e}")
            return {'error': str(e)}
    
    def get_market_regime_analysis(self, days: int = 90) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with market regime analysis
        """
        try:
            with self._get_connection() as conn:
                since_date = datetime.now() - timedelta(days=days)
                
                regime_performance = conn.execute("""
                    SELECT 
                        market_regime,
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                        AVG(prediction_error) as avg_error,
                        AVG(confidence) as avg_confidence
                    FROM prediction_records 
                    WHERE is_validated = 1 
                    AND created_at >= ?
                    AND market_regime IS NOT NULL
                    GROUP BY market_regime
                """, (since_date.isoformat(),)).fetchall()
                
                regime_analysis = {}
                for row in regime_performance:
                    regime = row['market_regime']
                    total = row['total_predictions']
                    wins = row['wins']
                    
                    regime_analysis[regime] = {
                        'total_predictions': total,
                        'wins': wins,
                        'losses': total - wins,
                        'win_rate': round((wins / total) * 100, 1) if total > 0 else 0,
                        'average_error': round(row['avg_error'], 4) if row['avg_error'] else 0,
                        'average_confidence': round(row['avg_confidence'], 3) if row['avg_confidence'] else 0
                    }
                
                return {
                    'analysis_period_days': days,
                    'regimes_analyzed': len(regime_analysis),
                    'regime_performance': regime_analysis,
                    'best_regime': max(regime_analysis.items(), key=lambda x: x[1]['win_rate'])[0] if regime_analysis else None,
                    'worst_regime': min(regime_analysis.items(), key=lambda x: x[1]['win_rate'])[0] if regime_analysis else None
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze market regimes: {e}")
            return {'error': str(e)}
    
    def get_pending_validations(self, max_age_hours: int = 72) -> List[Dict[str, Any]]:
        """
        Get predictions that are ready for validation
        
        Args:
            max_age_hours: Maximum age in hours to consider for validation
            
        Returns:
            List of predictions ready for validation
        """
        try:
            with self._get_connection() as conn:
                current_time = datetime.now(timezone.utc)
                min_time = current_time - timedelta(hours=max_age_hours)
                
                pending = conn.execute("""
                    SELECT * FROM prediction_records 
                    WHERE is_validated = 0 
                    AND target_time <= ?
                    AND target_time >= ?
                    ORDER BY target_time ASC
                """, (current_time.isoformat(), min_time.isoformat())).fetchall()
                
                return [dict(row) for row in pending]
                
        except Exception as e:
            logger.error(f"Failed to get pending validations: {e}")
            return []
    
    def export_performance_data(self, filepath: str, days: int = 30) -> bool:
        """
        Export performance data to CSV for external analysis
        
        Args:
            filepath: Path to export CSV file
            days: Number of days of data to export
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                since_date = datetime.now() - timedelta(days=days)
                
                predictions = conn.execute("""
                    SELECT * FROM prediction_records 
                    WHERE is_validated = 1 
                    AND created_at >= ?
                    ORDER BY created_at DESC
                """, (since_date.isoformat(),)).fetchall()
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in predictions])
                
                # Export to CSV
                df.to_csv(filepath, index=False)
                logger.info(f"Exported {len(predictions)} records to {filepath}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Clean up old prediction data to maintain database performance
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            with self._get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Delete old predictions
                result = conn.execute("""
                    DELETE FROM prediction_records 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = result.rowcount
                
                # Clean up orphaned validation results
                conn.execute("""
                    DELETE FROM validation_results 
                    WHERE prediction_id NOT IN (
                        SELECT prediction_id FROM prediction_records
                    )
                """)
                
                logger.info(f"Cleaned up {deleted_count} old prediction records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

# Utility functions for easy integration
def create_prediction_tracker(db_path: str = None) -> PredictionTracker:
    """Factory function to create a PredictionTracker instance"""
    return PredictionTracker(db_path or "goldgpt_prediction_learning.db")

def quick_validation(tracker: PredictionTracker, prediction_id: str, actual_price: float) -> Dict[str, Any]:
    """Quick validation helper function"""
    try:
        result = tracker.validate_prediction(prediction_id, actual_price)
        return {
            'success': True,
            'prediction_id': prediction_id,
            'direction_correct': result.direction_correct,
            'prediction_error': result.prediction_error_percent,
            'quality_score': result.prediction_quality_score
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Example usage and testing
    tracker = PredictionTracker()
    
    # Create a sample prediction
    sample_prediction = PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        prediction_time=datetime.now(timezone.utc),
        target_time=datetime.now(timezone.utc) + timedelta(hours=1),
        symbol="XAUUSD",
        timeframe="1H",
        strategy_name="technical_ensemble",
        engine_version="v2.1",
        current_price=2650.50,
        predicted_price=2655.75,
        price_change_percent=0.2,
        direction="bullish",
        confidence=0.75,
        features_used=["rsi", "macd", "sma_20"],
        feature_weights={"rsi": 0.4, "macd": 0.35, "sma_20": 0.25}
    )
    
    # Store the prediction
    pred_id = tracker.store_prediction(sample_prediction)
    print(f"Stored prediction: {pred_id}")
    
    # Simulate validation (in real use, this would happen after target_time)
    validation_result = tracker.validate_prediction(pred_id, 2654.25)
    print(f"Validation result: Direction correct: {validation_result.direction_correct}")
    
    # Get performance metrics
    performance = tracker.get_strategy_performance("technical_ensemble", "1H", 30)
    print(f"Strategy performance: {performance}")
