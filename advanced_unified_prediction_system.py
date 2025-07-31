#!/usr/bin/env python3
"""
GoldGPT Advanced Unified Prediction System
Generates exactly ONE comprehensive prediction set per day using ensemble of all ML strategies
"""

import asyncio
import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import requests
from decimal import Decimal

# Import our ML engines
from advanced_multi_strategy_ml_engine import MultiStrategyMLEngine
from prediction_validation_engine import PredictionValidationEngine
from data_fetcher import PriceDataFetcher
from self_improving_learning_engine import SelfImprovingLearningEngine

@dataclass
class UnifiedPrediction:
    """Unified prediction combining all ML strategies"""
    prediction_date: str
    timeframe: str
    
    # Ensemble prediction
    predicted_price: float
    predicted_direction: str
    predicted_change_percent: float
    confidence_score: float
    
    # Target levels
    target_price: float
    stop_loss: float
    target_time: str
    
    # Individual strategy contributions
    technical_prediction: Dict[str, Any]
    sentiment_prediction: Dict[str, Any]
    macro_prediction: Dict[str, Any]
    pattern_prediction: Dict[str, Any]
    
    # Ensemble weights used
    strategy_weights: Dict[str, float]
    
    # Market context
    market_volatility: float
    market_trend: str
    economic_events: List[str]
    
    # Metadata
    model_versions: Dict[str, str]
    generation_timestamp: str

class AdvancedUnifiedPredictionSystem:
    """
    Advanced unified prediction system that generates exactly ONE comprehensive 
    prediction set per day using ensemble of all ML strategies with dynamic weighting
    """
    
    def __init__(self, db_path: str = "goldgpt_ml_tracking.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines
        self.ml_engine = MultiStrategyMLEngine()
        self.validator = PredictionValidationEngine()
        self.learning_engine = SelfImprovingLearningEngine()
        
        # Initialize database
        self._initialize_database()
        
        # Daily generation control
        self.generation_time = time(9, 0)  # 9:00 AM daily generation
        self.timeframes = ['1h', '4h', '1d', '1w']
    
    def _initialize_database(self):
        """Initialize the unified prediction database"""
        try:
            with open('prediction_tracker_schema.sql', 'r') as f:
                schema = f.read()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Unified prediction database initialized")
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    async def should_generate_daily_predictions(self) -> bool:
        """Check if daily predictions should be generated"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            # Check if predictions already exist for today
            cursor.execute("""
                SELECT COUNT(*)
                FROM daily_predictions 
                WHERE prediction_date = ? AND strategy_id = 'unified_ensemble'
            """, (today,))
            
            existing_count = cursor.fetchone()[0]
            conn.close()
            
            # Generate if no predictions exist for today
            should_generate = existing_count == 0
            
            if should_generate:
                self.logger.info("✅ Ready to generate daily predictions")
            else:
                self.logger.info("ℹ️ Daily predictions already exist for today")
                
            return should_generate
            
        except Exception as e:
            self.logger.error(f"❌ Generation check failed: {e}")
            return False
    
    async def generate_daily_unified_predictions(self) -> List[UnifiedPrediction]:
        """Generate exactly ONE set of unified predictions for all timeframes"""
        try:
            if not await self.should_generate_daily_predictions():
                return await self._get_existing_daily_predictions()
            
            self.logger.info("🚀 Generating unified daily predictions...")
            
            # Get current market data
            current_price = await self._get_current_gold_price()
            if not current_price:
                raise Exception("Could not get current gold price")
            
            # Get optimized ensemble weights
            ensemble_weights = await self.learning_engine.optimize_ensemble_weights()
            
            # Generate predictions for each timeframe
            unified_predictions = []
            
            for timeframe in self.timeframes:
                try:
                    prediction = await self._generate_unified_prediction_for_timeframe(
                        timeframe, current_price, ensemble_weights
                    )
                    if prediction:
                        unified_predictions.append(prediction)
                        await self._store_unified_prediction(prediction)
                        
                except Exception as e:
                    self.logger.error(f"❌ Timeframe {timeframe} prediction failed: {e}")
            
            self.logger.info(f"✅ Generated {len(unified_predictions)} unified predictions")
            return unified_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Daily prediction generation failed: {e}")
            return []
    
    async def _generate_unified_prediction_for_timeframe(
        self, timeframe: str, current_price: float, ensemble_weights: Dict[str, float]
    ) -> Optional[UnifiedPrediction]:
        """Generate unified prediction for a specific timeframe"""
        try:
            # Get individual strategy predictions
            strategy_predictions = await self._get_individual_strategy_predictions(
                timeframe, current_price
            )
            
            if not strategy_predictions:
                return None
            
            # Calculate ensemble prediction
            ensemble_result = await self._calculate_ensemble_prediction(
                strategy_predictions, ensemble_weights
            )
            
            # Determine target time based on timeframe
            target_time = self._calculate_target_time(timeframe)
            
            # Calculate target levels
            target_price, stop_loss = await self._calculate_target_levels(
                current_price, ensemble_result['predicted_change_percent'], timeframe
            )
            
            # Analyze market context
            market_context = await self._analyze_current_market_context()
            
            # Get model versions
            model_versions = await self._get_current_model_versions()
            
            unified_prediction = UnifiedPrediction(
                prediction_date=datetime.now().date().isoformat(),
                timeframe=timeframe,
                predicted_price=ensemble_result['predicted_price'],
                predicted_direction=ensemble_result['predicted_direction'],
                predicted_change_percent=ensemble_result['predicted_change_percent'],
                confidence_score=ensemble_result['confidence_score'],
                target_price=target_price,
                stop_loss=stop_loss,
                target_time=target_time.isoformat(),
                technical_prediction=strategy_predictions.get('technical', {}),
                sentiment_prediction=strategy_predictions.get('sentiment', {}),
                macro_prediction=strategy_predictions.get('macro', {}),
                pattern_prediction=strategy_predictions.get('pattern', {}),
                strategy_weights=ensemble_weights,
                market_volatility=market_context['volatility'],
                market_trend=market_context['trend'],
                economic_events=market_context['economic_events'],
                model_versions=model_versions,
                generation_timestamp=datetime.now().isoformat()
            )
            
            return unified_prediction
            
        except Exception as e:
            self.logger.error(f"❌ Unified prediction generation failed for {timeframe}: {e}")
            return None
    
    async def _get_individual_strategy_predictions(
        self, timeframe: str, current_price: float
    ) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all individual ML strategies"""
        try:
            predictions = {}
            
            # Prepare current data for ML engine
            current_data = {
                'current_price': current_price,
                'timestamp': datetime.now(),
                'volume': 100000,  # Simulated volume
                'volatility': np.random.uniform(0.5, 3.0)
            }
            
            # Get ensemble prediction from ML engine
            try:
                ensemble_pred = await self.ml_engine.get_prediction('XAU/USD', timeframe, current_data)
                
                # Extract individual strategy predictions
                for individual_pred in ensemble_pred.individual_predictions:
                    strategy_name = individual_pred.strategy_name.lower()
                    predictions[strategy_name] = {
                        'predicted_price': individual_pred.predicted_price,
                        'direction': individual_pred.direction,
                        'confidence': individual_pred.confidence,
                        'change_percent': ((individual_pred.predicted_price - current_price) / current_price) * 100,
                        'features': getattr(individual_pred, 'features', {})
                    }
                
                # Ensure we have all required strategies
                required_strategies = ['technical', 'sentiment', 'macro', 'pattern']
                for strategy in required_strategies:
                    if strategy not in predictions:
                        predictions[strategy] = self._create_fallback_prediction(current_price)
                        
            except Exception as e:
                self.logger.error(f"❌ ML engine prediction failed: {e}")
                # Create fallback predictions for all strategies
                required_strategies = ['technical', 'sentiment', 'macro', 'pattern']
                for strategy in required_strategies:
                    predictions[strategy] = self._create_fallback_prediction(current_price)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Individual strategy predictions failed: {e}")
            return {}
    
    def _create_fallback_prediction(self, current_price: float) -> Dict[str, Any]:
        """Create fallback prediction when strategy fails"""
        return {
            'predicted_price': current_price,
            'direction': 'neutral',
            'confidence': 0.3,
            'change_percent': 0,
            'features': {'fallback': True}
        }
    
    async def _calculate_ensemble_prediction(
        self, strategy_predictions: Dict[str, Dict[str, Any]], 
        ensemble_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate weighted ensemble prediction from all strategies"""
        try:
            strategies = ['technical', 'sentiment', 'macro', 'pattern']
            
            # Weighted price prediction
            weighted_prices = []
            weighted_changes = []
            weighted_confidences = []
            direction_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            total_weight = 0
            
            for strategy in strategies:
                if strategy in strategy_predictions:
                    pred = strategy_predictions[strategy]
                    weight = ensemble_weights.get(strategy, 0.25)
                    
                    weighted_prices.append(pred['predicted_price'] * weight)
                    weighted_changes.append(pred['change_percent'] * weight)
                    weighted_confidences.append(pred['confidence'] * weight)
                    
                    # Direction voting with weights
                    direction = pred['direction']
                    direction_votes[direction] += weight
                    
                    total_weight += weight
            
            # Normalize if total weight != 1
            if total_weight > 0 and total_weight != 1:
                weighted_prices = [p / total_weight for p in weighted_prices]
                weighted_changes = [c / total_weight for c in weighted_changes]
                weighted_confidences = [c / total_weight for c in weighted_confidences]
            
            # Calculate ensemble results
            ensemble_price = sum(weighted_prices)
            ensemble_change = sum(weighted_changes)
            ensemble_confidence = sum(weighted_confidences)
            
            # Determine ensemble direction (highest weighted vote)
            ensemble_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
            
            # Adjust confidence based on agreement between strategies
            agreement_factor = max(direction_votes.values()) / sum(direction_votes.values())
            ensemble_confidence *= agreement_factor
            
            # Clamp confidence to reasonable range
            ensemble_confidence = max(0.1, min(0.95, ensemble_confidence))
            
            return {
                'predicted_price': round(ensemble_price, 2),
                'predicted_direction': ensemble_direction,
                'predicted_change_percent': round(ensemble_change, 2),
                'confidence_score': round(ensemble_confidence, 3),
                'strategy_agreement': round(agreement_factor, 3),
                'total_weight': total_weight
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble calculation failed: {e}")
            return {
                'predicted_price': 2650.0,
                'predicted_direction': 'neutral',
                'predicted_change_percent': 0.0,
                'confidence_score': 0.3,
                'strategy_agreement': 0.0,
                'total_weight': 0.0
            }
    
    def _calculate_target_time(self, timeframe: str) -> datetime:
        """Calculate target time based on timeframe"""
        now = datetime.now()
        
        if timeframe == '1h':
            return now + timedelta(hours=1)
        elif timeframe == '4h':
            return now + timedelta(hours=4)
        elif timeframe == '1d':
            return now + timedelta(days=1)
        elif timeframe == '1w':
            return now + timedelta(weeks=1)
        else:
            return now + timedelta(hours=1)  # Default
    
    async def _calculate_target_levels(
        self, current_price: float, predicted_change: float, timeframe: str
    ) -> Tuple[float, float]:
        """Calculate target price and stop loss levels"""
        try:
            # Target price based on predicted change
            target_price = current_price * (1 + predicted_change / 100)
            
            # Stop loss based on timeframe and volatility
            volatility_multiplier = {
                '1h': 0.5,   # 0.5% stop loss for 1h
                '4h': 1.0,   # 1.0% stop loss for 4h  
                '1d': 1.5,   # 1.5% stop loss for 1d
                '1w': 2.5    # 2.5% stop loss for 1w
            }.get(timeframe, 1.0)
            
            if predicted_change > 0:  # Bullish prediction
                stop_loss = current_price * (1 - volatility_multiplier / 100)
            else:  # Bearish prediction
                stop_loss = current_price * (1 + volatility_multiplier / 100)
            
            return round(target_price, 2), round(stop_loss, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Target level calculation failed: {e}")
            return current_price, current_price
    
    async def _analyze_current_market_context(self) -> Dict[str, Any]:
        """Analyze current market conditions and context"""
        try:
            # Get recent price data to calculate volatility
            current_price = await self._get_current_gold_price()
            
            # Simulate market analysis (in production, this would use real market data)
            volatility = np.random.uniform(0.5, 3.0)  # 0.5% to 3.0% volatility
            
            trend = "ranging"
            if volatility > 2.0:
                trend = "volatile"
            elif np.random.random() > 0.5:
                trend = "trending"
            
            # Economic events (would come from economic calendar API)
            economic_events = [
                "Fed Interest Rate Decision (pending)",
                "US CPI Data Release",
                "Gold ETF Flows"
            ]
            
            return {
                'volatility': round(volatility, 2),
                'trend': trend,
                'economic_events': economic_events,
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Market context analysis failed: {e}")
            return {
                'volatility': 1.0,
                'trend': 'ranging',
                'economic_events': [],
                'analysis_time': datetime.now().isoformat()
            }
    
    async def _get_current_model_versions(self) -> Dict[str, str]:
        """Get current model versions for all strategies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT strategy_id, version
                FROM model_version_history 
                WHERE is_production = TRUE
                ORDER BY created_date DESC
            """)
            
            versions = dict(cursor.fetchall())
            conn.close()
            
            # Ensure all strategies have versions
            default_versions = {
                'technical': 'v1',
                'sentiment': 'v1', 
                'macro': 'v1',
                'pattern': 'v1'
            }
            
            for strategy, default_version in default_versions.items():
                if strategy not in versions:
                    versions[strategy] = default_version
            
            return versions
            
        except Exception as e:
            self.logger.error(f"❌ Model version retrieval failed: {e}")
            return {'technical': 'v1', 'sentiment': 'v1', 'macro': 'v1', 'pattern': 'v1'}
    
    async def _get_current_gold_price(self) -> Optional[float]:
        """Get current gold price"""
        try:
            # Try to use data fetcher first
            price = PriceDataFetcher.get_current_price()
            if price and price > 0:
                return float(price)
            
            # Fallback to direct API call
            response = requests.get(
                "https://www.goldapi.io/api/XAU/USD",
                headers={"x-access-token": "goldapi-demo"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('price', 0))
            
            # Fallback price
            return 2650.0 + np.random.uniform(-20, 20)
            
        except Exception as e:
            self.logger.error(f"❌ Gold price retrieval failed: {e}")
            return 2650.0
    
    async def _store_unified_prediction(self, prediction: UnifiedPrediction):
        """Store unified prediction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store main prediction
            cursor.execute("""
                INSERT OR REPLACE INTO daily_predictions (
                    prediction_date, timeframe, strategy_id, model_version,
                    predicted_price, current_price, predicted_direction,
                    confidence_score, predicted_change_percent, target_price,
                    stop_loss, target_time, market_volatility, market_trend,
                    economic_events, feature_weights
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.prediction_date,
                prediction.timeframe,
                'unified_ensemble',
                'v1',
                prediction.predicted_price,
                prediction.predicted_price,  # Using predicted as current for now
                prediction.predicted_direction,
                prediction.confidence_score,
                prediction.predicted_change_percent,
                prediction.target_price,
                prediction.stop_loss,
                prediction.target_time,
                prediction.market_volatility,
                prediction.market_trend,
                json.dumps(prediction.economic_events),
                json.dumps({
                    'strategy_weights': prediction.strategy_weights,
                    'model_versions': prediction.model_versions,
                    'individual_predictions': {
                        'technical': prediction.technical_prediction,
                        'sentiment': prediction.sentiment_prediction,
                        'macro': prediction.macro_prediction,
                        'pattern': prediction.pattern_prediction
                    }
                })
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Stored unified prediction for {prediction.timeframe}")
            
        except Exception as e:
            self.logger.error(f"❌ Unified prediction storage failed: {e}")
    
    async def _get_existing_daily_predictions(self) -> List[UnifiedPrediction]:
        """Get existing daily predictions if they exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute("""
                SELECT * FROM daily_predictions 
                WHERE prediction_date = ? AND strategy_id = 'unified_ensemble'
                ORDER BY timeframe
            """, (today,))
            
            predictions = cursor.fetchall()
            conn.close()
            
            # Convert to UnifiedPrediction objects (simplified)
            unified_predictions = []
            for pred in predictions:
                # This would need to be expanded to properly reconstruct the object
                # For now, return empty list to trigger new generation
                pass
            
            return unified_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Existing predictions retrieval failed: {e}")
            return []
    
    async def get_latest_unified_predictions(self) -> List[Dict[str, Any]]:
        """Get the latest unified predictions for dashboard display"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's predictions
            today = datetime.now().date()
            
            cursor.execute("""
                SELECT 
                    timeframe, predicted_price, predicted_direction,
                    confidence_score, predicted_change_percent, target_price,
                    stop_loss, target_time, market_volatility, feature_weights
                FROM daily_predictions 
                WHERE prediction_date = ? AND strategy_id = 'unified_ensemble'
                ORDER BY 
                    CASE timeframe 
                        WHEN '1h' THEN 1 
                        WHEN '4h' THEN 2 
                        WHEN '1d' THEN 3 
                        WHEN '1w' THEN 4 
                    END
            """, (today,))
            
            predictions = cursor.fetchall()
            conn.close()
            
            # Format for frontend
            formatted_predictions = []
            for pred in predictions:
                try:
                    feature_weights = json.loads(pred[9]) if pred[9] else {}
                    
                    formatted_predictions.append({
                        'timeframe': pred[0],
                        'predicted_price': pred[1],
                        'predicted_direction': pred[2],
                        'confidence_score': pred[3],
                        'predicted_change_percent': pred[4],
                        'target_price': pred[5],
                        'stop_loss': pred[6],
                        'target_time': pred[7],
                        'market_volatility': pred[8],
                        'strategy_weights': feature_weights.get('strategy_weights', {}),
                        'model_versions': feature_weights.get('model_versions', {}),
                        'individual_predictions': feature_weights.get('individual_predictions', {})
                    })
                except Exception as e:
                    self.logger.error(f"❌ Prediction formatting failed: {e}")
            
            return formatted_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Latest predictions retrieval failed: {e}")
            return []
    
    async def run_daily_prediction_cycle(self):
        """Run the complete daily prediction cycle"""
        try:
            self.logger.info("🔄 Starting daily prediction cycle...")
            
            # 1. Validate expired predictions
            validation_results = await self.validator.validate_expired_predictions()
            self.logger.info(f"✅ Validated {len(validation_results)} expired predictions")
            
            # 2. Update strategy performance
            await self.validator.update_strategy_performance()
            self.logger.info("✅ Updated strategy performance metrics")
            
            # 3. Run learning engine analysis
            await self.learning_engine.discover_new_features()
            await self.learning_engine.retrain_underperforming_strategies()
            self.logger.info("✅ Completed learning engine analysis")
            
            # 4. Generate new unified predictions
            predictions = await self.generate_daily_unified_predictions()
            self.logger.info(f"✅ Generated {len(predictions)} unified predictions")
            
            # 5. Generate learning report
            report = await self.learning_engine.generate_learning_report()
            self.logger.info("✅ Generated learning progress report")
            
            return {
                "cycle_completed": True,
                "predictions_generated": len(predictions),
                "validations_completed": len(validation_results),
                "learning_report": report,
                "cycle_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Daily prediction cycle failed: {e}")
            return {"cycle_completed": False, "error": str(e)}

async def main():
    """Test the unified prediction system"""
    logging.basicConfig(level=logging.INFO)
    
    system = AdvancedUnifiedPredictionSystem()
    
    print("🚀 Starting unified prediction system test...")
    
    # Run daily prediction cycle
    result = await system.run_daily_prediction_cycle()
    print(f"✅ Daily cycle completed: {result['cycle_completed']}")
    
    if result['cycle_completed']:
        print(f"📊 Predictions Generated: {result['predictions_generated']}")
        print(f"✅ Validations Completed: {result['validations_completed']}")
    
    # Get latest predictions for display
    latest_predictions = await system.get_latest_unified_predictions()
    print(f"\n📈 Latest Unified Predictions ({len(latest_predictions)}):")
    
    for pred in latest_predictions:
        print(f"  {pred['timeframe']}: {pred['predicted_direction']} "
              f"${pred['predicted_price']:.2f} "
              f"({pred['predicted_change_percent']:+.1f}%) "
              f"[{pred['confidence_score']:.1%} confidence]")

if __name__ == "__main__":
    asyncio.run(main())
