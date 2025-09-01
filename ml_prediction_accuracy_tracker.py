"""
ML Prediction Accuracy Tracker
==============================
Tracks the accuracy of ML predictions across different timeframes to improve prediction quality.
Monitors hits/misses for each timeframe prediction and calculates accuracy metrics.
"""

import sqlite3
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from contextlib import contextmanager
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictionTracker:
    """Tracks ML prediction accuracy across timeframes"""
    
    def __init__(self, db_path: str = "ml_prediction_accuracy.db"):
        self.db_path = db_path
        self.gold_api_url = "https://api.gold-api.com/price/XAU"
        self._init_database()
        logger.info("✅ ML Prediction Accuracy Tracker initialized")
    
    def _init_database(self):
        """Initialize ML prediction tracking database"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,  -- BULLISH, BEARISH, NEUTRAL
                    target_price REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_timestamp TIMESTAMP NOT NULL,
                    actual_price REAL,
                    actual_high REAL,
                    actual_low REAL,
                    result TEXT,  -- HIT, MISS, PENDING, EXPIRED
                    accuracy_score REAL,
                    price_deviation REAL,
                    evaluation_timestamp TIMESTAMP,
                    reasoning TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS timeframe_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy_percentage REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    avg_price_deviation REAL DEFAULT 0,
                    bullish_accuracy REAL DEFAULT 0,
                    bearish_accuracy REAL DEFAULT 0,
                    neutral_accuracy REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timeframe ON ml_predictions(timeframe);
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_predictions(prediction_timestamp);
                CREATE INDEX IF NOT EXISTS idx_predictions_result ON ml_predictions(result);
            """)
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def add_prediction(self, timeframe: str, prediction_type: str, target_price: float, 
                      entry_price: float, confidence: float, reasoning: str = "") -> Dict[str, Any]:
        """Add a new ML prediction to track"""
        try:
            # Generate unique prediction ID
            prediction_id = f"PRED_{timeframe}_{int(time.time())}_{hash(f'{prediction_type}{target_price}') % 10000}"
            
            # Calculate expiry based on timeframe
            expiry_minutes = self._get_timeframe_minutes(timeframe)
            expiry_timestamp = datetime.now() + timedelta(minutes=expiry_minutes)
            
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO ml_predictions 
                    (prediction_id, timeframe, prediction_type, target_price, entry_price, 
                     confidence, expiry_timestamp, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (prediction_id, timeframe, prediction_type, target_price, entry_price,
                     confidence, expiry_timestamp, reasoning))
            
            logger.info(f"📊 Added ML prediction {prediction_id}: {timeframe} {prediction_type} target ${target_price:.2f}")
            
            return {
                'success': True,
                'prediction_id': prediction_id,
                'expiry_timestamp': expiry_timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error adding ML prediction: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes for expiry calculation"""
        timeframe_map = {
            '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '24h': 1440, '1w': 10080
        }
        return timeframe_map.get(timeframe, 60)  # Default to 1 hour
    
    def evaluate_predictions(self) -> Dict[str, Any]:
        """Evaluate pending predictions and update accuracy metrics"""
        try:
            current_price = self._get_current_gold_price()
            if not current_price:
                return {'success': False, 'error': 'Could not fetch current price'}
            
            evaluated_count = 0
            
            with self._get_db_connection() as conn:
                # Get pending predictions that need evaluation
                cursor = conn.execute("""
                    SELECT * FROM ml_predictions 
                    WHERE result IS NULL OR result = 'PENDING'
                    ORDER BY prediction_timestamp ASC
                """)
                
                pending_predictions = cursor.fetchall()
                
                for prediction in pending_predictions:
                    result = self._evaluate_single_prediction(prediction, current_price)
                    
                    if result['status'] != 'PENDING':
                        # Update prediction with result
                        conn.execute("""
                            UPDATE ml_predictions 
                            SET actual_price = ?, result = ?, accuracy_score = ?, 
                                price_deviation = ?, evaluation_timestamp = ?
                            WHERE prediction_id = ?
                        """, (current_price, result['status'], result['accuracy_score'],
                             result['price_deviation'], datetime.now(), prediction['prediction_id']))
                        
                        evaluated_count += 1
                        logger.info(f"📈 Evaluated prediction {prediction['prediction_id']}: {result['status']}")
            
            # Update accuracy metrics after evaluation
            if evaluated_count > 0:
                self._update_accuracy_metrics()
            
            return {
                'success': True,
                'evaluated_count': evaluated_count,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating predictions: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_single_prediction(self, prediction: sqlite3.Row, current_price: float) -> Dict[str, Any]:
        """Evaluate a single prediction"""
        try:
            timeframe = prediction['timeframe']
            prediction_type = prediction['prediction_type']
            target_price = prediction['target_price']
            entry_price = prediction['entry_price']
            expiry_timestamp = datetime.fromisoformat(prediction['expiry_timestamp'])
            
            # Check if prediction has expired
            if datetime.now() > expiry_timestamp:
                # Calculate final accuracy for expired prediction
                price_deviation = abs(current_price - target_price) / target_price
                
                if prediction_type == 'BULLISH':
                    # For bullish predictions, check if price moved up
                    accuracy_score = max(0, (current_price - entry_price) / (target_price - entry_price))
                    status = 'HIT' if current_price > entry_price else 'MISS'
                elif prediction_type == 'BEARISH':
                    # For bearish predictions, check if price moved down
                    accuracy_score = max(0, (entry_price - current_price) / (entry_price - target_price))
                    status = 'HIT' if current_price < entry_price else 'MISS'
                else:  # NEUTRAL
                    # For neutral predictions, check if price stayed within reasonable range
                    neutral_range = abs(entry_price * 0.01)  # 1% range
                    price_deviation_abs = abs(current_price - entry_price)
                    accuracy_score = max(0, 1 - (price_deviation_abs / neutral_range))
                    status = 'HIT' if price_deviation_abs <= neutral_range else 'MISS'
                
                return {
                    'status': status,
                    'accuracy_score': min(1.0, accuracy_score),
                    'price_deviation': price_deviation
                }
            
            # Check if target has been hit during timeframe
            elif prediction_type == 'BULLISH' and current_price >= target_price:
                return {
                    'status': 'HIT',
                    'accuracy_score': 1.0,
                    'price_deviation': 0.0
                }
            elif prediction_type == 'BEARISH' and current_price <= target_price:
                return {
                    'status': 'HIT',
                    'accuracy_score': 1.0,
                    'price_deviation': 0.0
                }
            else:
                # Still pending
                return {
                    'status': 'PENDING',
                    'accuracy_score': 0.0,
                    'price_deviation': abs(current_price - target_price) / target_price
                }
                
        except Exception as e:
            logger.error(f"❌ Error evaluating single prediction: {e}")
            return {
                'status': 'ERROR',
                'accuracy_score': 0.0,
                'price_deviation': 1.0
            }
    
    def _update_accuracy_metrics(self):
        """Update timeframe accuracy statistics"""
        try:
            with self._get_db_connection() as conn:
                timeframes = ['5m', '15m', '30m', '1h', '4h', '24h', '1w']
                
                for timeframe in timeframes:
                    # Get last 100 predictions for this timeframe
                    cursor = conn.execute("""
                        SELECT * FROM ml_predictions 
                        WHERE timeframe = ? AND result IN ('HIT', 'MISS')
                        ORDER BY evaluation_timestamp DESC
                        LIMIT 100
                    """, (timeframe,))
                    
                    predictions = cursor.fetchall()
                    
                    if not predictions:
                        continue
                    
                    # Calculate metrics
                    total_predictions = len(predictions)
                    correct_predictions = sum(1 for p in predictions if p['result'] == 'HIT')
                    accuracy_percentage = (correct_predictions / total_predictions) * 100
                    
                    avg_confidence = np.mean([p['confidence'] for p in predictions])
                    avg_price_deviation = np.mean([p['price_deviation'] for p in predictions])
                    
                    # Calculate accuracy by prediction type
                    bullish_preds = [p for p in predictions if p['prediction_type'] == 'BULLISH']
                    bearish_preds = [p for p in predictions if p['prediction_type'] == 'BEARISH']
                    neutral_preds = [p for p in predictions if p['prediction_type'] == 'NEUTRAL']
                    
                    bullish_accuracy = (sum(1 for p in bullish_preds if p['result'] == 'HIT') / len(bullish_preds) * 100) if bullish_preds else 0
                    bearish_accuracy = (sum(1 for p in bearish_preds if p['result'] == 'HIT') / len(bearish_preds) * 100) if bearish_preds else 0
                    neutral_accuracy = (sum(1 for p in neutral_preds if p['result'] == 'HIT') / len(neutral_preds) * 100) if neutral_preds else 0
                    
                    # Insert or update accuracy record
                    period_start = datetime.now() - timedelta(days=7)  # Last week
                    period_end = datetime.now()
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO timeframe_accuracy 
                        (timeframe, period_start, period_end, total_predictions, correct_predictions,
                         accuracy_percentage, avg_confidence, avg_price_deviation, 
                         bullish_accuracy, bearish_accuracy, neutral_accuracy, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (timeframe, period_start, period_end, total_predictions, correct_predictions,
                         accuracy_percentage, avg_confidence, avg_price_deviation,
                         bullish_accuracy, bearish_accuracy, neutral_accuracy, datetime.now()))
                    
                    logger.info(f"📊 Updated {timeframe} accuracy: {accuracy_percentage:.1f}% ({correct_predictions}/{total_predictions})")
                    
        except Exception as e:
            logger.error(f"❌ Error updating accuracy metrics: {e}")
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get comprehensive accuracy statistics"""
        try:
            with self._get_db_connection() as conn:
                # Get recent accuracy stats
                cursor = conn.execute("""
                    SELECT * FROM timeframe_accuracy 
                    ORDER BY updated_at DESC
                """)
                
                accuracy_stats = {}
                for row in cursor.fetchall():
                    accuracy_stats[row['timeframe']] = {
                        'accuracy_percentage': row['accuracy_percentage'],
                        'total_predictions': row['total_predictions'],
                        'correct_predictions': row['correct_predictions'],
                        'avg_confidence': row['avg_confidence'],
                        'bullish_accuracy': row['bullish_accuracy'],
                        'bearish_accuracy': row['bearish_accuracy'],
                        'neutral_accuracy': row['neutral_accuracy'],
                        'updated_at': row['updated_at']
                    }
                
                # Get pending predictions count
                cursor = conn.execute("""
                    SELECT timeframe, COUNT(*) as pending_count
                    FROM ml_predictions 
                    WHERE result = 'PENDING' OR result IS NULL
                    GROUP BY timeframe
                """)
                
                pending_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'success': True,
                    'accuracy_stats': accuracy_stats,
                    'pending_predictions': pending_stats,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting accuracy stats: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_current_gold_price(self) -> Optional[float]:
        """Get current gold price from API"""
        try:
            response = requests.get(self.gold_api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            logger.error(f"❌ Error fetching gold price: {e}")
        return None
    
    def get_prediction_insights(self, timeframe: str = None) -> Dict[str, Any]:
        """Get insights about prediction performance"""
        try:
            with self._get_db_connection() as conn:
                where_clause = "WHERE timeframe = ?" if timeframe else ""
                params = (timeframe,) if timeframe else ()
                
                # Get recent predictions performance
                cursor = conn.execute(f"""
                    SELECT 
                        prediction_type,
                        AVG(accuracy_score) as avg_accuracy,
                        COUNT(*) as total_count,
                        SUM(CASE WHEN result = 'HIT' THEN 1 ELSE 0 END) as hit_count,
                        AVG(price_deviation) as avg_deviation,
                        AVG(confidence) as avg_confidence
                    FROM ml_predictions 
                    {where_clause}
                    AND result IN ('HIT', 'MISS')
                    AND evaluation_timestamp > datetime('now', '-7 days')
                    GROUP BY prediction_type
                """, params)
                
                insights = {}
                for row in cursor.fetchall():
                    pred_type = row['prediction_type']
                    insights[pred_type] = {
                        'accuracy_rate': (row['hit_count'] / row['total_count']) * 100,
                        'avg_accuracy_score': row['avg_accuracy'],
                        'total_predictions': row['total_count'],
                        'hits': row['hit_count'],
                        'avg_price_deviation': row['avg_deviation'],
                        'avg_confidence': row['avg_confidence']
                    }
                
                return {
                    'success': True,
                    'insights': insights,
                    'timeframe': timeframe or 'all',
                    'period': 'last_7_days'
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting prediction insights: {e}")
            return {'success': False, 'error': str(e)}

# Global instance
ml_prediction_tracker = MLPredictionTracker()
