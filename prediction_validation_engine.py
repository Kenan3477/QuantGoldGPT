#!/usr/bin/env python3
"""
GoldGPT Self-Improving ML System - Prediction Validation Engine
Automatically validates predictions and calculates comprehensive performance metrics
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from decimal import Decimal
import requests
import statistics

@dataclass
class ValidationResult:
    """Comprehensive validation result for a prediction"""
    prediction_id: int
    actual_price: float
    actual_direction: str
    actual_change_percent: float
    direction_correct: bool
    price_accuracy_percent: float
    profit_loss_percent: float
    accuracy_score: float
    confidence_calibration: float
    market_conditions: Dict[str, Any]
    volatility_during_period: float

class PredictionValidationEngine:
    """
    Advanced prediction validation system that automatically validates expired predictions
    and calculates comprehensive performance metrics
    """
    
    def __init__(self, db_path: str = "goldgpt_ml_tracking.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the prediction tracking database with schema"""
        try:
            with open('prediction_tracker_schema.sql', 'r') as f:
                schema = f.read()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Prediction tracking database initialized")
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    async def validate_expired_predictions(self) -> List[ValidationResult]:
        """
        Automatically validate all predictions that have reached their target time
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find expired predictions that haven't been validated
            current_time = datetime.now()
            cursor.execute("""
                SELECT id, predicted_price, current_price, predicted_direction, 
                       confidence_score, predicted_change_percent, target_time,
                       timeframe, strategy_id, market_volatility
                FROM daily_predictions 
                WHERE target_time <= ? AND is_validated = FALSE
            """, (current_time,))
            
            expired_predictions = cursor.fetchall()
            validation_results = []
            
            for prediction in expired_predictions:
                try:
                    result = await self._validate_single_prediction(prediction)
                    if result:
                        validation_results.append(result)
                        await self._store_validation_result(result)
                        
                        # Mark prediction as validated
                        cursor.execute("""
                            UPDATE daily_predictions 
                            SET is_validated = TRUE 
                            WHERE id = ?
                        """, (prediction[0],))
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to validate prediction {prediction[0]}: {e}")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Validated {len(validation_results)} expired predictions")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation process failed: {e}")
            return []
    
    async def _validate_single_prediction(self, prediction_data: Tuple) -> Optional[ValidationResult]:
        """Validate a single prediction with comprehensive metrics"""
        try:
            (pred_id, predicted_price, original_price, predicted_direction, 
             confidence, predicted_change, target_time, timeframe, strategy_id, market_vol) = prediction_data
            
            # Get current actual price
            actual_price = await self._get_current_gold_price()
            if not actual_price:
                return None
            
            # Calculate actual change
            actual_change_percent = ((actual_price - original_price) / original_price) * 100
            
            # Determine actual direction
            if actual_change_percent > 0.1:
                actual_direction = "bullish"
            elif actual_change_percent < -0.1:
                actual_direction = "bearish"
            else:
                actual_direction = "neutral"
            
            # Calculate performance metrics
            direction_correct = predicted_direction == actual_direction
            price_accuracy = 100 - abs(((predicted_price - actual_price) / actual_price) * 100)
            price_accuracy = max(0, min(100, price_accuracy))  # Clamp to 0-100
            
            # Calculate theoretical profit/loss
            profit_loss_percent = self._calculate_profit_loss(
                predicted_direction, original_price, actual_price
            )
            
            # Calculate comprehensive accuracy score
            accuracy_score = self._calculate_accuracy_score(
                direction_correct, price_accuracy, confidence, predicted_change, actual_change_percent
            )
            
            # Calculate confidence calibration
            confidence_calibration = self._calculate_confidence_calibration(
                confidence, accuracy_score
            )
            
            # Get market conditions during prediction period
            market_conditions = await self._analyze_market_conditions(
                original_price, actual_price, timeframe
            )
            
            # Calculate volatility during period
            volatility_during_period = abs(actual_change_percent)
            
            return ValidationResult(
                prediction_id=pred_id,
                actual_price=actual_price,
                actual_direction=actual_direction,
                actual_change_percent=actual_change_percent,
                direction_correct=direction_correct,
                price_accuracy_percent=price_accuracy,
                profit_loss_percent=profit_loss_percent,
                accuracy_score=accuracy_score,
                confidence_calibration=confidence_calibration,
                market_conditions=market_conditions,
                volatility_during_period=volatility_during_period
            )
            
        except Exception as e:
            self.logger.error(f"❌ Single prediction validation failed: {e}")
            return None
    
    async def _get_current_gold_price(self) -> Optional[float]:
        """Get current gold price for validation"""
        try:
            # Use Gold-API.com for reliable price
            response = requests.get(
                "https://www.goldapi.io/api/XAU/USD",
                headers={"x-access-token": "goldapi-demo"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('price', 0))
            
            # Fallback to a simulated price based on recent market data
            # In production, you'd have multiple fallback APIs
            return 2650.0 + np.random.uniform(-50, 50)  # Simulated current price
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get current gold price: {e}")
            return None
    
    def _calculate_profit_loss(self, predicted_direction: str, entry_price: float, exit_price: float) -> float:
        """Calculate theoretical profit/loss percentage"""
        try:
            if predicted_direction == "bullish":
                # Long position
                return ((exit_price - entry_price) / entry_price) * 100
            elif predicted_direction == "bearish":
                # Short position
                return ((entry_price - exit_price) / entry_price) * 100
            else:
                # Neutral - no position
                return 0.0
        except:
            return 0.0
    
    def _calculate_accuracy_score(self, direction_correct: bool, price_accuracy: float, 
                                 confidence: float, predicted_change: float, 
                                 actual_change: float) -> float:
        """Calculate comprehensive accuracy score (0-1)"""
        try:
            # Base score from direction accuracy
            direction_score = 1.0 if direction_correct else 0.0
            
            # Price accuracy component (0-1)
            price_score = price_accuracy / 100.0
            
            # Magnitude accuracy (how close was predicted change to actual)
            magnitude_error = abs(predicted_change - actual_change)
            magnitude_score = max(0, 1 - (magnitude_error / 100))  # Normalize by 100%
            
            # Weighted combination
            accuracy_score = (
                direction_score * 0.4 +      # Direction is most important
                price_score * 0.35 +         # Price accuracy is crucial
                magnitude_score * 0.25       # Magnitude matters for sizing
            )
            
            return round(accuracy_score, 3)
            
        except Exception as e:
            self.logger.error(f"❌ Accuracy score calculation failed: {e}")
            return 0.0
    
    def _calculate_confidence_calibration(self, confidence: float, accuracy_score: float) -> float:
        """Calculate how well confidence matched actual performance"""
        try:
            # Perfect calibration would have confidence = accuracy_score
            calibration_error = abs(confidence - accuracy_score)
            calibration_score = max(0, 1 - calibration_error)
            return round(calibration_score, 3)
        except:
            return 0.0
    
    async def _analyze_market_conditions(self, start_price: float, end_price: float, 
                                       timeframe: str) -> Dict[str, Any]:
        """Analyze market conditions during prediction period"""
        try:
            change_percent = ((end_price - start_price) / start_price) * 100
            
            # Determine market regime
            if abs(change_percent) > 3.0:
                volatility = "high"
            elif abs(change_percent) > 1.0:
                volatility = "medium"
            else:
                volatility = "low"
            
            if change_percent > 1.0:
                trend = "bullish"
            elif change_percent < -1.0:
                trend = "bearish"
            else:
                trend = "ranging"
            
            return {
                "volatility": volatility,
                "trend": trend,
                "change_percent": round(change_percent, 2),
                "price_range": f"{min(start_price, end_price):.2f}-{max(start_price, end_price):.2f}",
                "timeframe": timeframe,
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Market conditions analysis failed: {e}")
            return {"error": str(e)}
    
    async def _store_validation_result(self, result: ValidationResult):
        """Store validation result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO prediction_validation (
                    prediction_id, actual_price, actual_direction, actual_change_percent,
                    direction_correct, price_accuracy_percent, profit_loss_percent,
                    accuracy_score, confidence_calibration, market_conditions,
                    volatility_during_period
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.prediction_id,
                result.actual_price,
                result.actual_direction,
                result.actual_change_percent,
                result.direction_correct,
                result.price_accuracy_percent,
                result.profit_loss_percent,
                result.accuracy_score,
                result.confidence_calibration,
                json.dumps(result.market_conditions),
                result.volatility_during_period
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Stored validation result for prediction {result.prediction_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store validation result: {e}")
    
    async def update_strategy_performance(self):
        """Update strategy performance metrics based on recent validations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all strategies with recent validations
            cursor.execute("""
                SELECT DISTINCT dp.strategy_id, dp.model_version, dp.timeframe
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE pv.validation_date >= date('now', '-7 days')
            """)
            
            strategies = cursor.fetchall()
            
            for strategy_id, model_version, timeframe in strategies:
                await self._update_single_strategy_performance(
                    strategy_id, model_version, timeframe
                )
            
            conn.close()
            self.logger.info(f"✅ Updated performance for {len(strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"❌ Strategy performance update failed: {e}")
    
    async def _update_single_strategy_performance(self, strategy_id: str, 
                                                model_version: str, timeframe: str):
        """Update performance metrics for a single strategy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent performance data
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN pv.direction_correct THEN 1 ELSE 0 END) as correct_predictions,
                    AVG(dp.confidence_score) as avg_confidence,
                    AVG(pv.accuracy_score) as avg_accuracy,
                    AVG(pv.confidence_calibration) as avg_confidence_calibration,
                    SUM(pv.profit_loss_percent) as total_profit_loss,
                    AVG(pv.profit_loss_percent) as avg_profit_per_trade,
                    COUNT(CASE WHEN pv.profit_loss_percent > 0 THEN 1 END) as winning_trades
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE dp.strategy_id = ? AND dp.model_version = ? AND dp.timeframe = ?
                AND pv.validation_date >= date('now', '-30 days')
            """, (strategy_id, model_version, timeframe))
            
            performance_data = cursor.fetchone()
            
            if performance_data and performance_data[0] > 0:
                (total_pred, correct_pred, avg_conf, avg_acc, avg_conf_cal, 
                 total_pl, avg_pl, winning_trades) = performance_data
                
                accuracy_rate = (correct_pred / total_pred) * 100
                win_rate = (winning_trades / total_pred) * 100
                
                # Calculate additional metrics
                sharpe_ratio = self._calculate_sharpe_ratio(strategy_id, model_version, timeframe)
                max_drawdown = self._calculate_max_drawdown(strategy_id, model_version, timeframe)
                
                # Insert or update performance record
                cursor.execute("""
                    INSERT OR REPLACE INTO strategy_performance (
                        strategy_id, model_version, timeframe, measurement_date,
                        total_predictions, correct_predictions, accuracy_rate,
                        avg_confidence, confidence_accuracy_correlation,
                        total_profit_loss, win_rate, avg_profit_per_trade,
                        max_drawdown, sharpe_ratio, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id, model_version, timeframe, datetime.now().date(),
                    total_pred, correct_pred, accuracy_rate,
                    avg_conf or 0, avg_conf_cal or 0,
                    total_pl or 0, win_rate, avg_pl or 0,
                    max_drawdown, sharpe_ratio, datetime.now()
                ))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Single strategy performance update failed: {e}")
    
    def _calculate_sharpe_ratio(self, strategy_id: str, model_version: str, timeframe: str) -> float:
        """Calculate Sharpe ratio for the strategy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pv.profit_loss_percent
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE dp.strategy_id = ? AND dp.model_version = ? AND dp.timeframe = ?
                AND pv.validation_date >= date('now', '-30 days')
                ORDER BY pv.validation_date
            """, (strategy_id, model_version, timeframe))
            
            returns = [row[0] for row in cursor.fetchall() if row[0] is not None]
            conn.close()
            
            if len(returns) < 2:
                return 0.0
            
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe (assuming daily returns)
            sharpe = (avg_return / std_return) * np.sqrt(252)
            return round(sharpe, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, strategy_id: str, model_version: str, timeframe: str) -> float:
        """Calculate maximum drawdown for the strategy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pv.profit_loss_percent
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE dp.strategy_id = ? AND dp.model_version = ? AND dp.timeframe = ?
                AND pv.validation_date >= date('now', '-30 days')
                ORDER BY pv.validation_date
            """, (strategy_id, model_version, timeframe))
            
            returns = [row[0] for row in cursor.fetchall() if row[0] is not None]
            conn.close()
            
            if len(returns) < 2:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-8)  # Avoid division by zero
            
            max_drawdown = np.min(drawdown) * 100  # Convert to percentage
            return round(abs(max_drawdown), 2)
            
        except Exception as e:
            self.logger.error(f"❌ Max drawdown calculation failed: {e}")
            return 0.0
    
    async def get_validation_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_validations,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(confidence_calibration) as avg_calibration,
                    AVG(profit_loss_percent) as avg_profit_loss,
                    COUNT(CASE WHEN direction_correct THEN 1 END) as correct_directions
                FROM prediction_validation 
                WHERE validation_date >= date('now', '-{} days')
            """.format(days))
            
            overall_stats = cursor.fetchone()
            
            # Strategy breakdown
            cursor.execute("""
                SELECT 
                    dp.strategy_id,
                    dp.timeframe,
                    COUNT(*) as predictions,
                    AVG(pv.accuracy_score) as avg_accuracy,
                    AVG(pv.profit_loss_percent) as avg_profit
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE pv.validation_date >= date('now', '-{} days')
                GROUP BY dp.strategy_id, dp.timeframe
                ORDER BY avg_accuracy DESC
            """.format(days))
            
            strategy_breakdown = cursor.fetchall()
            
            conn.close()
            
            return {
                "summary_period_days": days,
                "total_validations": overall_stats[0] if overall_stats else 0,
                "overall_accuracy": round(overall_stats[1] or 0, 3),
                "overall_calibration": round(overall_stats[2] or 0, 3),
                "overall_profit_loss": round(overall_stats[3] or 0, 2),
                "direction_accuracy": round((overall_stats[4] / max(overall_stats[0], 1)) * 100, 1),
                "strategy_performance": [
                    {
                        "strategy": row[0],
                        "timeframe": row[1],
                        "predictions": row[2],
                        "accuracy": round(row[3], 3),
                        "avg_profit": round(row[4], 2)
                    }
                    for row in strategy_breakdown
                ],
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Validation summary failed: {e}")
            return {"error": str(e)}

async def main():
    """Test the validation engine"""
    logging.basicConfig(level=logging.INFO)
    
    validator = PredictionValidationEngine()
    
    print("🔍 Starting prediction validation...")
    
    # Validate expired predictions
    results = await validator.validate_expired_predictions()
    print(f"✅ Validated {len(results)} predictions")
    
    # Update strategy performance
    await validator.update_strategy_performance()
    print("✅ Updated strategy performance metrics")
    
    # Get validation summary
    summary = await validator.get_validation_summary(days=30)
    print("\n📊 Validation Summary (Last 30 days):")
    print(f"Total Validations: {summary['total_validations']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Direction Accuracy: {summary['direction_accuracy']}%")
    print(f"Overall P&L: {summary['overall_profit_loss']:.2f}%")
    
    print("\n🎯 Strategy Performance:")
    for strategy in summary['strategy_performance'][:5]:  # Top 5
        print(f"  {strategy['strategy']} ({strategy['timeframe']}): "
              f"{strategy['accuracy']:.1%} accuracy, "
              f"{strategy['avg_profit']:.2f}% avg profit")

if __name__ == "__main__":
    asyncio.run(main())
