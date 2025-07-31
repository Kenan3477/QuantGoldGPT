"""
🎯 PHASE 3: SELF-LEARNING PREDICTION VALIDATION SYSTEM
=====================================================

PredictionValidator - Autonomous system that learns from every prediction
Tracks, validates, and analyzes all predictions with complete metadata

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
from price_storage_manager import PriceStorageManager
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('prediction_validator')

@dataclass
class PredictionRecord:
    """Complete prediction record with metadata"""
    prediction_id: str
    symbol: str
    predicted_price: float
    current_price: float
    direction: str
    confidence: float
    timeframe: str
    strategy_weights: Dict[str, float]
    market_conditions: Dict[str, Any]
    feature_vector: List[float]
    contributing_strategies: List[str]
    prediction_timestamp: datetime
    expiry_timestamp: datetime
    actual_price: Optional[float] = None
    outcome: Optional[str] = None  # 'correct', 'incorrect', 'pending'
    accuracy_score: Optional[float] = None
    profit_factor: Optional[float] = None
    validated_timestamp: Optional[datetime] = None

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    total_predictions: int
    correct_predictions: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_confidence: float
    strategy_performance: Dict[str, float]
    market_condition_performance: Dict[str, float]
    timeframe_performance: Dict[str, float]

class PredictionValidator:
    """
    Autonomous prediction validation system
    Stores, tracks, and validates all predictions with complete analytics
    """
    
    def __init__(self, db_path: str = "goldgpt_prediction_validation.db"):
        self.db_path = db_path
        self.price_manager = PriceStorageManager()
        self.init_database()
        self.validation_thread = None
        self.is_running = False
        logger.info("🎯 Prediction Validator initialized")
    
    def init_database(self):
        """Initialize prediction validation database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_weights TEXT NOT NULL,
                    market_conditions TEXT NOT NULL,
                    feature_vector TEXT NOT NULL,
                    contributing_strategies TEXT NOT NULL,
                    prediction_timestamp DATETIME NOT NULL,
                    expiry_timestamp DATETIME NOT NULL,
                    actual_price REAL,
                    outcome TEXT,
                    accuracy_score REAL,
                    profit_factor REAL,
                    validated_timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create validation metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculation_date DATE NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    strategy_performance TEXT NOT NULL,
                    market_condition_performance TEXT NOT NULL,
                    timeframe_performance TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create feature performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    accuracy_contribution REAL NOT NULL,
                    market_condition TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    calculation_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create market regime analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regime_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_type TEXT NOT NULL,
                    volatility_level TEXT NOT NULL,
                    trend_strength REAL NOT NULL,
                    prediction_accuracy REAL NOT NULL,
                    best_strategy TEXT NOT NULL,
                    optimal_timeframe TEXT NOT NULL,
                    analysis_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Prediction validation database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def store_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Store a new prediction with complete metadata"""
        try:
            prediction_id = f"pred_{int(time.time() * 1000)}"
            
            # Parse timeframe to calculate expiry
            timeframe_minutes = self._parse_timeframe(prediction_data.get('timeframe', '1H'))
            expiry_timestamp = datetime.now() + timedelta(minutes=timeframe_minutes)
            
            record = PredictionRecord(
                prediction_id=prediction_id,
                symbol=prediction_data.get('symbol', 'XAUUSD'),
                predicted_price=prediction_data.get('predicted_price', 0.0),
                current_price=prediction_data.get('current_price', 0.0),
                direction=prediction_data.get('direction', 'neutral'),
                confidence=prediction_data.get('confidence', 0.0),
                timeframe=prediction_data.get('timeframe', '1H'),
                strategy_weights=prediction_data.get('strategy_weights', {}),
                market_conditions=prediction_data.get('market_conditions', {}),
                feature_vector=prediction_data.get('feature_vector', []),
                contributing_strategies=prediction_data.get('contributing_strategies', []),
                prediction_timestamp=datetime.now(),
                expiry_timestamp=expiry_timestamp
            )
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (
                    prediction_id, symbol, predicted_price, current_price, direction,
                    confidence, timeframe, strategy_weights, market_conditions,
                    feature_vector, contributing_strategies, prediction_timestamp,
                    expiry_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.prediction_id,
                record.symbol,
                record.predicted_price,
                record.current_price,
                record.direction,
                record.confidence,
                record.timeframe,
                json.dumps(record.strategy_weights),
                json.dumps(record.market_conditions),
                json.dumps(record.feature_vector),
                json.dumps(record.contributing_strategies),
                record.prediction_timestamp,
                record.expiry_timestamp
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Prediction stored: {prediction_id} ({record.timeframe}, {record.direction})")
            return prediction_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store prediction: {e}")
            return ""
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes"""
        timeframe_map = {
            '1M': 1, '5M': 5, '15M': 15, '30M': 30,
            '1H': 60, '4H': 240, '1D': 1440, '1W': 10080
        }
        return timeframe_map.get(timeframe, 60)
    
    def start_validation_service(self):
        """Start autonomous validation service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.validation_thread = threading.Thread(target=self._validation_loop, daemon=True)
        self.validation_thread.start()
        logger.info("🚀 Autonomous validation service started")
    
    def stop_validation_service(self):
        """Stop validation service"""
        self.is_running = False
        if self.validation_thread:
            self.validation_thread.join()
        logger.info("⏹️ Validation service stopped")
    
    def _validation_loop(self):
        """Main validation loop"""
        while self.is_running:
            try:
                self.validate_expired_predictions()
                self.calculate_daily_metrics()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Validation loop error: {e}")
                time.sleep(10)
    
    def validate_expired_predictions(self):
        """Validate predictions that have reached their expiry time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get expired predictions that haven't been validated
            cursor.execute('''
                SELECT * FROM predictions
                WHERE expiry_timestamp <= ?
                AND outcome IS NULL
            ''', (datetime.now(),))
            
            expired_predictions = cursor.fetchall()
            
            for pred_row in expired_predictions:
                prediction_id = pred_row[0]
                symbol = pred_row[1]
                predicted_price = pred_row[2]
                direction = pred_row[4]
                expiry_timestamp = datetime.fromisoformat(pred_row[12])
                
                # Get actual price at expiry time
                actual_price = self._get_price_at_time(symbol, expiry_timestamp)
                
                if actual_price is not None:
                    # Calculate validation metrics
                    outcome, accuracy_score, profit_factor = self._calculate_outcome(
                        predicted_price, actual_price, direction
                    )
                    
                    # Update prediction record
                    cursor.execute('''
                        UPDATE predictions
                        SET actual_price = ?, outcome = ?, accuracy_score = ?,
                            profit_factor = ?, validated_timestamp = ?
                        WHERE prediction_id = ?
                    ''', (
                        actual_price, outcome, accuracy_score,
                        profit_factor, datetime.now(), prediction_id
                    ))
                    
                    logger.info(f"✅ Validated prediction {prediction_id}: {outcome} ({accuracy_score:.3f})")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Prediction validation failed: {e}")
    
    def _get_price_at_time(self, symbol: str, target_time: datetime) -> Optional[float]:
        """Get price closest to target time"""
        try:
            # Get historical prices around target time
            historical_prices = self.price_manager.get_historical_prices(
                symbol, hours=2
            )
            
            if not historical_prices:
                return None
            
            # Find closest price to target time
            min_diff = float('inf')
            closest_price = None
            
            for price_record in historical_prices:
                price_time = datetime.fromisoformat(price_record['timestamp'])
                time_diff = abs((target_time - price_time).total_seconds())
                
                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_price = price_record['price']
            
            return closest_price
            
        except Exception as e:
            logger.error(f"❌ Failed to get price at time: {e}")
            return None
    
    def _calculate_outcome(self, predicted_price: float, actual_price: float, 
                          direction: str) -> Tuple[str, float, float]:
        """Calculate prediction outcome and metrics"""
        try:
            price_change = actual_price - predicted_price
            price_change_percent = (price_change / predicted_price) * 100
            
            # Determine if prediction was correct
            if direction == 'bullish' and actual_price > predicted_price:
                outcome = 'correct'
            elif direction == 'bearish' and actual_price < predicted_price:
                outcome = 'correct'
            elif direction == 'neutral' and abs(price_change_percent) < 0.5:
                outcome = 'correct'
            else:
                outcome = 'incorrect'
            
            # Calculate accuracy score (0-1)
            max_error = abs(predicted_price * 0.05)  # 5% max error for perfect score
            actual_error = abs(price_change)
            accuracy_score = max(0, 1 - (actual_error / max_error))
            
            # Calculate profit factor
            if direction == 'bullish':
                profit_factor = max(-1, price_change_percent / 100)
            elif direction == 'bearish':
                profit_factor = max(-1, -price_change_percent / 100)
            else:
                profit_factor = max(-1, 1 - abs(price_change_percent) / 100)
            
            return outcome, accuracy_score, profit_factor
            
        except Exception as e:
            logger.error(f"❌ Outcome calculation failed: {e}")
            return 'error', 0.0, 0.0
    
    def calculate_daily_metrics(self) -> ValidationMetrics:
        """Calculate comprehensive daily validation metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all validated predictions from last 30 days
            cursor.execute('''
                SELECT * FROM predictions
                WHERE outcome IS NOT NULL
                AND prediction_timestamp >= date('now', '-30 days')
            ''')
            
            predictions = cursor.fetchall()
            
            if not predictions:
                return ValidationMetrics(
                    total_predictions=0, correct_predictions=0, win_rate=0.0,
                    profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                    avg_confidence=0.0, strategy_performance={},
                    market_condition_performance={}, timeframe_performance={}
                )
            
            # Calculate basic metrics
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p[14] == 'correct')
            win_rate = correct_predictions / total_predictions
            
            # Calculate profit metrics
            profit_factors = [p[16] for p in predictions if p[16] is not None]
            avg_profit_factor = np.mean(profit_factors) if profit_factors else 0.0
            
            # Calculate Sharpe ratio
            returns = np.array(profit_factors) if profit_factors else np.array([0])
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max)
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            # Calculate average confidence
            confidences = [p[5] for p in predictions]
            avg_confidence = np.mean(confidences)
            
            # Calculate strategy performance
            strategy_performance = self._calculate_strategy_performance(predictions)
            
            # Calculate market condition performance
            market_condition_performance = self._calculate_market_condition_performance(predictions)
            
            # Calculate timeframe performance
            timeframe_performance = self._calculate_timeframe_performance(predictions)
            
            metrics = ValidationMetrics(
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                win_rate=win_rate,
                profit_factor=avg_profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_confidence=avg_confidence,
                strategy_performance=strategy_performance,
                market_condition_performance=market_condition_performance,
                timeframe_performance=timeframe_performance
            )
            
            # Store metrics in database
            self._store_validation_metrics(metrics)
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Daily metrics calculation failed: {e}")
            return ValidationMetrics(
                total_predictions=0, correct_predictions=0, win_rate=0.0,
                profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                avg_confidence=0.0, strategy_performance={},
                market_condition_performance={}, timeframe_performance={}
            )
    
    def _calculate_strategy_performance(self, predictions: List) -> Dict[str, float]:
        """Calculate performance by strategy"""
        strategy_performance = {}
        
        try:
            for prediction in predictions:
                contributing_strategies = json.loads(prediction[10])
                outcome = prediction[14]
                
                for strategy in contributing_strategies:
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = {'correct': 0, 'total': 0}
                    
                    strategy_performance[strategy]['total'] += 1
                    if outcome == 'correct':
                        strategy_performance[strategy]['correct'] += 1
            
            # Convert to win rates
            for strategy in strategy_performance:
                total = strategy_performance[strategy]['total']
                correct = strategy_performance[strategy]['correct']
                strategy_performance[strategy] = correct / total if total > 0 else 0.0
                
        except Exception as e:
            logger.error(f"❌ Strategy performance calculation failed: {e}")
        
        return strategy_performance
    
    def _calculate_market_condition_performance(self, predictions: List) -> Dict[str, float]:
        """Calculate performance by market conditions"""
        condition_performance = {}
        
        try:
            for prediction in predictions:
                market_conditions = json.loads(prediction[8])
                outcome = prediction[14]
                
                market_regime = market_conditions.get('market_regime', 'unknown')
                
                if market_regime not in condition_performance:
                    condition_performance[market_regime] = {'correct': 0, 'total': 0}
                
                condition_performance[market_regime]['total'] += 1
                if outcome == 'correct':
                    condition_performance[market_regime]['correct'] += 1
            
            # Convert to win rates
            for condition in condition_performance:
                total = condition_performance[condition]['total']
                correct = condition_performance[condition]['correct']
                condition_performance[condition] = correct / total if total > 0 else 0.0
                
        except Exception as e:
            logger.error(f"❌ Market condition performance calculation failed: {e}")
        
        return condition_performance
    
    def _calculate_timeframe_performance(self, predictions: List) -> Dict[str, float]:
        """Calculate performance by timeframe"""
        timeframe_performance = {}
        
        try:
            for prediction in predictions:
                timeframe = prediction[6]
                outcome = prediction[14]
                
                if timeframe not in timeframe_performance:
                    timeframe_performance[timeframe] = {'correct': 0, 'total': 0}
                
                timeframe_performance[timeframe]['total'] += 1
                if outcome == 'correct':
                    timeframe_performance[timeframe]['correct'] += 1
            
            # Convert to win rates
            for tf in timeframe_performance:
                total = timeframe_performance[tf]['total']
                correct = timeframe_performance[tf]['correct']
                timeframe_performance[tf] = correct / total if total > 0 else 0.0
                
        except Exception as e:
            logger.error(f"❌ Timeframe performance calculation failed: {e}")
        
        return timeframe_performance
    
    def _store_validation_metrics(self, metrics: ValidationMetrics):
        """Store validation metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO validation_metrics (
                    calculation_date, total_predictions, correct_predictions,
                    win_rate, profit_factor, sharpe_ratio, max_drawdown,
                    avg_confidence, strategy_performance,
                    market_condition_performance, timeframe_performance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().date(),
                metrics.total_predictions,
                metrics.correct_predictions,
                metrics.win_rate,
                metrics.profit_factor,
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.avg_confidence,
                json.dumps(metrics.strategy_performance),
                json.dumps(metrics.market_condition_performance),
                json.dumps(metrics.timeframe_performance)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store validation metrics: {e}")
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get recent performance summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM validation_metrics
                WHERE calculation_date >= date('now', '-{} days')
                ORDER BY calculation_date DESC
            '''.format(days))
            
            metrics_rows = cursor.fetchall()
            
            if not metrics_rows:
                return {'error': 'No recent metrics found'}
            
            latest_metrics = metrics_rows[0]
            
            performance_summary = {
                'current_win_rate': latest_metrics[3],
                'current_profit_factor': latest_metrics[4],
                'current_sharpe_ratio': latest_metrics[5],
                'total_predictions': latest_metrics[1],
                'correct_predictions': latest_metrics[2],
                'avg_confidence': latest_metrics[7],
                'strategy_performance': json.loads(latest_metrics[8]),
                'timeframe_performance': json.loads(latest_metrics[10]),
                'last_updated': latest_metrics[0]
            }
            
            conn.close()
            return performance_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent performance: {e}")
            return {'error': str(e)}

# Global validator instance
validator = PredictionValidator()

def start_validation_service():
    """Start the global validation service"""
    validator.start_validation_service()

def store_prediction(prediction_data: Dict[str, Any]) -> str:
    """Store a prediction for validation"""
    return validator.store_prediction(prediction_data)

def get_validation_metrics() -> Dict[str, Any]:
    """Get current validation metrics"""
    return validator.get_recent_performance()

if __name__ == "__main__":
    print("🎯 Starting Prediction Validator...")
    validator = PredictionValidator()
    validator.start_validation_service()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("⏹️ Stopping Prediction Validator...")
        validator.stop_validation_service()
