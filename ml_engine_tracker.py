#!/usr/bin/env python3
"""
ML Engine Tracking Manager
Stores predictions from multiple ML engines and tracks their accuracy over time
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MLPrediction:
    """Structure for an ML prediction"""
    engine_name: str
    symbol: str
    current_price: float
    timeframe: str
    predicted_price: float
    change_percent: float
    direction: str
    confidence: float
    market_conditions: Dict
    prediction_factors: Dict
    prediction_time: datetime = None
    target_validation_time: datetime = None
    
    def __post_init__(self):
        if self.prediction_time is None:
            self.prediction_time = datetime.now(timezone.utc)
        
        if self.target_validation_time is None:
            # Calculate target validation time based on timeframe
            hours_map = {'1H': 1, '4H': 4, '1D': 24, '1W': 168}
            hours = hours_map.get(self.timeframe, 1)
            self.target_validation_time = self.prediction_time + timedelta(hours=hours)

@dataclass
class ValidationResult:
    """Structure for prediction validation results"""
    prediction_id: int
    actual_price: float
    actual_change_percent: float
    actual_direction: str
    price_accuracy: float
    direction_correct: bool
    accuracy_score: float

class MLEngineTracker:
    """Manages tracking and accuracy analysis for multiple ML engines"""
    
    def __init__(self, db_path: str = "goldgpt_ml_tracking.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize the tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Read and execute schema
                with open('ml_engine_tracking_schema.sql', 'r') as f:
                    schema = f.read()
                conn.executescript(schema)
                conn.commit()
                logger.info("✅ ML engine tracking database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize tracking database: {e}")
    
    def store_prediction(self, prediction: MLPrediction) -> int:
        """Store a new prediction in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ml_engine_predictions (
                        engine_name, symbol, prediction_date, prediction_time,
                        current_price, timeframe, predicted_price, change_percent,
                        direction, confidence, target_validation_time,
                        market_conditions, prediction_factors
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.engine_name,
                    prediction.symbol,
                    prediction.prediction_time.strftime('%Y-%m-%d'),
                    prediction.prediction_time.strftime('%H:%M:%S'),
                    prediction.current_price,
                    prediction.timeframe,
                    prediction.predicted_price,
                    prediction.change_percent,
                    prediction.direction,
                    prediction.confidence,
                    prediction.target_validation_time.strftime('%Y-%m-%d %H:%M:%S'),
                    json.dumps(prediction.market_conditions),
                    json.dumps(prediction.prediction_factors)
                ))
                
                prediction_id = cursor.lastrowid
                
                # Update engine statistics
                self._update_engine_stats(prediction.engine_name, 'prediction_added')
                
                logger.info(f"✅ Stored prediction {prediction_id} from {prediction.engine_name}")
                return prediction_id
                
        except Exception as e:
            logger.error(f"❌ Failed to store prediction: {e}")
            return None
    
    def validate_predictions(self, current_price: float, symbol: str = 'XAUUSD') -> List[ValidationResult]:
        """Validate predictions that have reached their target time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find predictions ready for validation (timezone aware)
                cursor.execute("""
                    SELECT id, engine_name, current_price, predicted_price, 
                           change_percent, direction, confidence, timeframe
                    FROM ml_engine_predictions 
                    WHERE status = 'pending' 
                    AND datetime(target_validation_time) <= datetime('now')
                    AND symbol = ?
                """, (symbol,))
                
                pending_predictions = cursor.fetchall()
                results = []
                
                for pred in pending_predictions:
                    pred_id, engine_name, original_price, predicted_price, predicted_change, predicted_direction, confidence, timeframe = pred
                    
                    # Calculate actual metrics
                    actual_change_percent = ((current_price - original_price) / original_price) * 100
                    
                    if actual_change_percent > 0.1:
                        actual_direction = 'bullish'
                    elif actual_change_percent < -0.1:
                        actual_direction = 'bearish'
                    else:
                        actual_direction = 'neutral'
                    
                    # Calculate accuracy metrics
                    price_error = abs(predicted_price - current_price)
                    price_accuracy = max(0, 100 - ((price_error / current_price) * 100))
                    direction_correct = predicted_direction == actual_direction
                    
                    # Overall accuracy score (weighted)
                    accuracy_score = (price_accuracy * 0.7) + (100 if direction_correct else 0) * 0.3
                    
                    # Update prediction with validation results
                    cursor.execute("""
                        UPDATE ml_engine_predictions 
                        SET actual_price = ?, actual_change_percent = ?, actual_direction = ?,
                            price_accuracy = ?, direction_correct = ?, accuracy_score = ?,
                            status = 'validated', validation_date = ?
                        WHERE id = ?
                    """, (
                        current_price, actual_change_percent, actual_direction,
                        price_accuracy, direction_correct, accuracy_score,
                        datetime.now(timezone.utc), pred_id
                    ))
                    
                    result = ValidationResult(
                        prediction_id=pred_id,
                        actual_price=current_price,
                        actual_change_percent=actual_change_percent,
                        actual_direction=actual_direction,
                        price_accuracy=price_accuracy,
                        direction_correct=direction_correct,
                        accuracy_score=accuracy_score
                    )
                    results.append(result)
                    
                    # Update engine performance stats
                    self._update_engine_stats(engine_name, 'prediction_validated')
                    
                    logger.info(f"✅ Validated prediction {pred_id}: {accuracy_score:.1f}% accuracy")
                
                conn.commit()
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to validate predictions: {e}")
            return []
    
    def get_engine_accuracy_stats(self) -> Dict[str, Dict]:
        """Get accuracy statistics for all ML engines"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get overall stats for each engine
                cursor.execute("""
                    SELECT 
                        engine_name,
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN status = 'validated' THEN 1 END) as validated,
                        AVG(CASE WHEN status = 'validated' THEN accuracy_score END) as avg_accuracy,
                        AVG(CASE WHEN status = 'validated' AND direction_correct = 1 THEN 100 ELSE 0 END) as direction_accuracy,
                        AVG(CASE WHEN status = 'validated' AND timeframe = '1H' THEN accuracy_score END) as accuracy_1h,
                        AVG(CASE WHEN status = 'validated' AND timeframe = '4H' THEN accuracy_score END) as accuracy_4h,
                        AVG(CASE WHEN status = 'validated' AND timeframe = '1D' THEN accuracy_score END) as accuracy_1d
                    FROM ml_engine_predictions 
                    GROUP BY engine_name
                """)
                
                results = {}
                for row in cursor.fetchall():
                    engine_name, total, validated, avg_acc, dir_acc, acc_1h, acc_4h, acc_1d = row
                    results[engine_name] = {
                        'total_predictions': total,
                        'validated_predictions': validated,
                        'overall_accuracy': round(avg_acc or 0, 1),
                        'direction_accuracy': round(dir_acc or 0, 1),
                        'accuracy_1h': round(acc_1h or 0, 1),
                        'accuracy_4h': round(acc_4h or 0, 1),
                        'accuracy_1d': round(acc_1d or 0, 1),
                        'success_rate': round((validated / total * 100) if total > 0 else 0, 1)
                    }
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to get accuracy stats: {e}")
            return {}
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Dict]:
        """Get recent performance trends for engines"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                cursor.execute("""
                    SELECT 
                        engine_name,
                        COUNT(*) as recent_predictions,
                        AVG(CASE WHEN status = 'validated' THEN accuracy_score END) as recent_accuracy,
                        COUNT(CASE WHEN status = 'validated' AND direction_correct = 1 THEN 1 END) * 100.0 / 
                        COUNT(CASE WHEN status = 'validated' THEN 1 END) as recent_direction_accuracy
                    FROM ml_engine_predictions 
                    WHERE created_at >= ?
                    GROUP BY engine_name
                """, (cutoff_date,))
                
                results = {}
                for row in cursor.fetchall():
                    engine_name, recent_preds, recent_acc, recent_dir_acc = row
                    results[engine_name] = {
                        'recent_predictions': recent_preds,
                        'recent_accuracy': round(recent_acc or 0, 1),
                        'recent_direction_accuracy': round(recent_dir_acc or 0, 1)
                    }
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to get recent performance: {e}")
            return {}
    
    def _update_engine_stats(self, engine_name: str, action: str):
        """Update engine performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current stats
                stats = self.get_engine_accuracy_stats().get(engine_name, {})
                
                # Update or insert performance record
                cursor.execute("""
                    INSERT OR REPLACE INTO ml_engine_performance 
                    (engine_name, total_predictions, validated_predictions, 
                     overall_accuracy, direction_accuracy, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    engine_name,
                    stats.get('total_predictions', 0),
                    stats.get('validated_predictions', 0),
                    stats.get('overall_accuracy', 0),
                    stats.get('direction_accuracy', 0),
                    datetime.now(timezone.utc)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to update engine stats: {e}")
    
    def _recalculate_all_engine_stats(self):
        """Recalculate performance stats for all engines"""
        try:
            engines = self.get_engine_accuracy_stats()
            for engine_name in engines.keys():
                self._update_engine_stats(engine_name, 'recalculate')
            logger.info("✅ Recalculated all engine stats")
        except Exception as e:
            logger.error(f"❌ Failed to recalculate engine stats: {e}")
    
    def get_dashboard_stats(self) -> Dict:
        """Get formatted stats for dashboard display"""
        accuracy_stats = self.get_engine_accuracy_stats()
        recent_stats = self.get_recent_performance()
        
        dashboard_data = {
            'engines': [],
            'best_performer': None,
            'total_predictions': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        for engine_name, stats in accuracy_stats.items():
            recent = recent_stats.get(engine_name, {})
            
            engine_data = {
                'name': engine_name,
                'display_name': engine_name.replace('_', ' ').title(),
                'overall_accuracy': stats['overall_accuracy'],
                'direction_accuracy': stats['direction_accuracy'],
                'total_predictions': stats['total_predictions'],
                'recent_accuracy': recent.get('recent_accuracy', 0),
                'trend': 'improving' if recent.get('recent_accuracy', 0) > stats['overall_accuracy'] else 'declining',
                'badge': self._get_performance_badge(stats['overall_accuracy']),
                'timeframe_accuracy': {
                    '1H': stats['accuracy_1h'],
                    '4H': stats['accuracy_4h'],
                    '1D': stats['accuracy_1d']
                }
            }
            
            dashboard_data['engines'].append(engine_data)
            dashboard_data['total_predictions'] += stats['total_predictions']
        
        # Sort by accuracy and set best performer
        dashboard_data['engines'].sort(key=lambda x: x['overall_accuracy'], reverse=True)
        if dashboard_data['engines']:
            dashboard_data['best_performer'] = dashboard_data['engines'][0]['name']
        
        return dashboard_data
    
    def _get_performance_badge(self, accuracy: float) -> Dict:
        """Get performance badge based on accuracy"""
        if accuracy >= 75:
            return {'label': 'Excellent', 'color': '#10b981', 'icon': '🏆'}
        elif accuracy >= 65:
            return {'label': 'Good', 'color': '#3b82f6', 'icon': '🎯'}
        elif accuracy >= 55:
            return {'label': 'Average', 'color': '#f59e0b', 'icon': '📊'}
        else:
            return {'label': 'Poor', 'color': '#ef4444', 'icon': '📉'}

# Global instance
ml_tracker = MLEngineTracker()

async def track_prediction_from_engine(engine_name: str, prediction_data: Dict, market_data: Dict = None) -> int:
    """Helper function to track a prediction from any ML engine"""
    try:
        prediction = MLPrediction(
            engine_name=engine_name,
            symbol=prediction_data.get('symbol', 'XAUUSD'),
            current_price=prediction_data['current_price'],
            timeframe=prediction_data['timeframe'],
            predicted_price=prediction_data['predicted_price'],
            change_percent=prediction_data['change_percent'],
            direction=prediction_data['direction'],
            confidence=prediction_data.get('confidence', 0.5),
            market_conditions=market_data or {},
            prediction_factors=prediction_data.get('factors', {})
        )
        
        prediction_id = ml_tracker.store_prediction(prediction)
        logger.info(f"📊 Tracked prediction {prediction_id} from {engine_name}")
        return prediction_id
        
    except Exception as e:
        logger.error(f"❌ Failed to track prediction from {engine_name}: {e}")
        return None

def get_ml_accuracy_dashboard_data() -> Dict:
    """Get ML accuracy data for dashboard display"""
    return ml_tracker.get_dashboard_stats()

# Background validation task
async def run_prediction_validation():
    """Background task to validate predictions"""
    try:
        # Import here to avoid circular imports
        from price_storage_manager import get_current_gold_price
        
        current_price = get_current_gold_price()
        if current_price:
            results = ml_tracker.validate_predictions(current_price)
            if results:
                logger.info(f"✅ Validated {len(results)} predictions")
    except Exception as e:
        logger.error(f"❌ Prediction validation failed: {e}")

if __name__ == "__main__":
    # Test the tracking system
    print("🎯 Testing ML Engine Tracking System")
    print("=" * 50)
    
    # Test storing predictions
    test_prediction = MLPrediction(
        engine_name="test_engine",
        symbol="XAUUSD",
        current_price=3350.70,
        timeframe="1H",
        predicted_price=3355.20,
        change_percent=0.13,
        direction="bullish",
        confidence=0.75,
        market_conditions={"dxy": 102.5, "vix": 16.2},
        prediction_factors={"rsi": 58, "macd": "bullish"}
    )
    
    pred_id = ml_tracker.store_prediction(test_prediction)
    print(f"✅ Stored test prediction: {pred_id}")
    
    # Test getting stats
    stats = ml_tracker.get_dashboard_stats()
    print(f"📊 Dashboard stats: {json.dumps(stats, indent=2)}")
    
    print("✅ ML Engine Tracking System ready!")
