"""
Daily ML Prediction Scheduler
Integrates with Flask app to provide one prediction per day with learning capability
"""

from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import datetime
import logging
from self_improving_ml_engine import SelfImprovingMLEngine
from dynamic_prediction_engine import dynamic_prediction_engine

logger = logging.getLogger(__name__)

class DailyPredictionScheduler:
    def __init__(self, app: Flask = None):
        self.app = app
        self.ml_engine = SelfImprovingMLEngine()
        self.scheduler = BackgroundScheduler()
        self.current_prediction = None
        self.last_prediction_date = None
        
    def init_app(self, app: Flask):
        """Initialize with Flask app"""
        self.app = app
        self.setup_scheduler()
        
    def setup_scheduler(self):
        """Setup the daily prediction scheduler"""
        
        # Schedule daily prediction at 6:00 AM UTC
        self.scheduler.add_job(
            func=self.generate_daily_prediction,
            trigger=CronTrigger(hour=6, minute=0),
            id='daily_prediction',
            name='Generate Daily ML Prediction',
            replace_existing=True
        )
        
        # Schedule validation checks every hour
        self.scheduler.add_job(
            func=self.validate_predictions,
            trigger=CronTrigger(minute=0),
            id='hourly_validation',
            name='Validate Previous Predictions',
            replace_existing=True
        )
        
        # Schedule strategy evaluation every week
        self.scheduler.add_job(
            func=self.weekly_strategy_review,
            trigger=CronTrigger(day_of_week=0, hour=7, minute=0),
            id='weekly_review',
            name='Weekly Strategy Performance Review',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("🕐 Daily prediction scheduler started")
        
        # Generate initial prediction if none exists for today
        self.ensure_daily_prediction()
        
    def generate_daily_prediction(self):
        """Generate the single daily prediction with dynamic monitoring"""
        try:
            logger.info("🎯 Generating daily ML prediction...")
            
            prediction = self.ml_engine.generate_daily_prediction("XAUUSD")
            self.current_prediction = prediction
            self.last_prediction_date = datetime.date.today()
            
            # Register prediction with dynamic monitoring engine
            dynamic_prediction_engine.set_current_prediction("XAUUSD", prediction)
            
            # Emit to connected clients via WebSocket
            if self.app:
                from app import socketio
                socketio.emit('daily_prediction_update', {
                    'success': True,
                    'prediction_date': str(prediction.prediction_date),
                    'current_price': prediction.current_price,
                    'predictions': {
                        '1h': {
                            'change_percent': prediction.predictions['1h'],
                            'predicted_price': prediction.predicted_prices['1h'],
                            'confidence': prediction.confidence_scores['1h'],
                            'timeframe': '1H'
                        },
                        '4h': {
                            'change_percent': prediction.predictions['4h'],
                            'predicted_price': prediction.predicted_prices['4h'],
                            'confidence': prediction.confidence_scores['4h'],
                            'timeframe': '4H'
                        },
                        '1d': {
                            'change_percent': prediction.predictions['1d'],
                            'predicted_price': prediction.predicted_prices['1d'],
                            'confidence': prediction.confidence_scores['1d'],
                            'timeframe': '1D'
                        },
                        '3d': {
                            'change_percent': prediction.predictions['3d'],
                            'predicted_price': prediction.predicted_prices['3d'],
                            'confidence': prediction.confidence_scores['3d'],
                            'timeframe': '3D'
                        },
                        '7d': {
                            'change_percent': prediction.predictions['7d'],
                            'predicted_price': prediction.predicted_prices['7d'],
                            'confidence': prediction.confidence_scores['7d'],
                            'timeframe': '7D'
                        }
                    },
                    'strategy_info': {
                        'strategy_id': prediction.strategy_id,
                        'reasoning': prediction.reasoning
                    },
                    'generated_at': datetime.datetime.now().isoformat(),
                    'source': 'daily_ml_predictor',
                    'dynamic_monitoring': True
                })
                
            logger.info(f"✅ Daily prediction generated successfully for {prediction.prediction_date}")
            logger.info("🔄 Dynamic market monitoring activated")
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily prediction: {e}")
            return None
    
    def validate_predictions(self):
        """Validate previous predictions"""
        try:
            logger.info("🔍 Validating previous predictions...")
            self.ml_engine.validate_predictions("XAUUSD")
            logger.info("✅ Prediction validation completed")
        except Exception as e:
            logger.error(f"❌ Prediction validation failed: {e}")
    
    def weekly_strategy_review(self):
        """Weekly strategy performance review and optimization"""
        try:
            logger.info("📊 Conducting weekly strategy review...")
            
            # Get performance metrics
            performance_report = self.ml_engine.get_strategy_performance_report()
            
            # Check if strategy change is needed
            if self.ml_engine.should_change_strategy():
                old_strategy = self.ml_engine.current_strategy_id
                new_strategy = self.ml_engine.select_best_strategy()
                self.ml_engine.current_strategy_id = new_strategy
                
                logger.info(f"🔄 Strategy changed from {old_strategy} to {new_strategy}")
                
                # Emit strategy change notification
                if self.app:
                    from app import socketio
                    socketio.emit('strategy_change', {
                        'old_strategy_id': old_strategy,
                        'new_strategy_id': new_strategy,
                        'reason': 'Performance optimization',
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            
            logger.info("✅ Weekly strategy review completed")
            
        except Exception as e:
            logger.error(f"❌ Weekly strategy review failed: {e}")
    
    def ensure_daily_prediction(self):
        """Ensure there's a prediction for today"""
        today = datetime.date.today()
        
        if (self.last_prediction_date != today or 
            self.current_prediction is None):
            
            # Check if prediction exists in database
            if not self.ml_engine.has_prediction_for_date(today, "XAUUSD"):
                logger.info("📅 No prediction for today, generating...")
                self.generate_daily_prediction()
            else:
                # Load existing prediction
                self.current_prediction = self.ml_engine.get_prediction_for_date(today, "XAUUSD")
                self.last_prediction_date = today
                logger.info("📅 Loaded existing prediction for today")
    
    def get_current_prediction(self) -> dict:
        """Get the current daily prediction for API endpoints"""
        self.ensure_daily_prediction()
        
        if not self.current_prediction:
            return {
                'success': False,
                'error': 'No prediction available'
            }
        
        return {
            'success': True,
            'prediction_date': str(self.current_prediction.prediction_date),
            'current_price': self.current_prediction.current_price,
            'symbol': self.current_prediction.symbol,
            'predictions': [
                {
                    'timeframe': '1H',
                    'change_percent': self.current_prediction.predictions['1h'],
                    'predicted_price': self.current_prediction.predicted_prices['1h'],
                    'confidence': self.current_prediction.confidence_scores['1h'],
                    'direction': 'bullish' if self.current_prediction.predictions['1h'] > 0 else 'bearish'
                },
                {
                    'timeframe': '4H',
                    'change_percent': self.current_prediction.predictions['4h'],
                    'predicted_price': self.current_prediction.predicted_prices['4h'],
                    'confidence': self.current_prediction.confidence_scores['4h'],
                    'direction': 'bullish' if self.current_prediction.predictions['4h'] > 0 else 'bearish'
                },
                {
                    'timeframe': '1D',
                    'change_percent': self.current_prediction.predictions['1d'],
                    'predicted_price': self.current_prediction.predicted_prices['1d'],
                    'confidence': self.current_prediction.confidence_scores['1d'],
                    'direction': 'bullish' if self.current_prediction.predictions['1d'] > 0 else 'bearish'
                },
                {
                    'timeframe': '3D',
                    'change_percent': self.current_prediction.predictions['3d'],
                    'predicted_price': self.current_prediction.predicted_prices['3d'],
                    'confidence': self.current_prediction.confidence_scores['3d'],
                    'direction': 'bullish' if self.current_prediction.predictions['3d'] > 0 else 'bearish'
                },
                {
                    'timeframe': '7D',
                    'change_percent': self.current_prediction.predictions['7d'],
                    'predicted_price': self.current_prediction.predicted_prices['7d'],
                    'confidence': self.current_prediction.confidence_scores['7d'],
                    'direction': 'bullish' if self.current_prediction.predictions['7d'] > 0 else 'bearish'
                }
            ],
            'strategy_info': {
                'strategy_id': self.current_prediction.strategy_id,
                'reasoning': self.current_prediction.reasoning
            },
            'technical_analysis': self.current_prediction.technical_indicators,
            'sentiment_analysis': self.current_prediction.sentiment_data,
            'market_conditions': self.current_prediction.market_conditions,
            'generated_at': datetime.datetime.now().isoformat(),
            'source': 'daily_self_improving_ml'
        }
    
    def get_dynamic_prediction_data(self, symbol: str = "XAUUSD"):
        """Get current prediction with dynamic updates information"""
        try:
            # Get current prediction from dynamic engine
            dynamic_prediction = dynamic_prediction_engine.get_current_prediction(symbol)
            
            if not dynamic_prediction:
                return self.get_current_prediction()
            
            prediction_obj = dynamic_prediction['prediction']
            
            # Get update history
            update_history = dynamic_prediction_engine.get_prediction_history(symbol, limit=5)
            
            return {
                'success': True,
                'prediction_date': str(prediction_obj.prediction_date),
                'current_price': prediction_obj.current_price,
                'symbol': prediction_obj.symbol,
                'predictions': [
                    {
                        'timeframe': '1H',
                        'change_percent': prediction_obj.predictions['1h'],
                        'predicted_price': prediction_obj.predicted_prices['1h'],
                        'confidence': prediction_obj.confidence_scores['1h'],
                        'direction': 'bullish' if prediction_obj.predictions['1h'] > 0 else 'bearish'
                    },
                    {
                        'timeframe': '4H',
                        'change_percent': prediction_obj.predictions['4h'],
                        'predicted_price': prediction_obj.predicted_prices['4h'],
                        'confidence': prediction_obj.confidence_scores['4h'],
                        'direction': 'bullish' if prediction_obj.predictions['4h'] > 0 else 'bearish'
                    },
                    {
                        'timeframe': '1D',
                        'change_percent': prediction_obj.predictions['1d'],
                        'predicted_price': prediction_obj.predicted_prices['1d'],
                        'confidence': prediction_obj.confidence_scores['1d'],
                        'direction': 'bullish' if prediction_obj.predictions['1d'] > 0 else 'bearish'
                    },
                    {
                        'timeframe': '3D',
                        'change_percent': prediction_obj.predictions['3d'],
                        'predicted_price': prediction_obj.predicted_prices['3d'],
                        'confidence': prediction_obj.confidence_scores['3d'],
                        'direction': 'bullish' if prediction_obj.predictions['3d'] > 0 else 'bearish'
                    },
                    {
                        'timeframe': '7D',
                        'change_percent': prediction_obj.predictions['7d'],
                        'predicted_price': prediction_obj.predicted_prices['7d'],
                        'confidence': prediction_obj.confidence_scores['7d'],
                        'direction': 'bullish' if prediction_obj.predictions['7d'] > 0 else 'bearish'
                    }
                ],
                'strategy_info': {
                    'strategy_id': prediction_obj.strategy_id,
                    'reasoning': prediction_obj.reasoning
                },
                'technical_analysis': prediction_obj.technical_indicators,
                'sentiment_analysis': prediction_obj.sentiment_data,
                'market_conditions': prediction_obj.market_conditions,
                'dynamic_info': {
                    'is_dynamic': True,
                    'created_at': dynamic_prediction['created_at'].isoformat(),
                    'last_updated': dynamic_prediction['last_updated'].isoformat(),
                    'update_count': dynamic_prediction['update_count'],
                    'monitoring_active': dynamic_prediction_engine.monitoring
                },
                'update_history': update_history,
                'generated_at': datetime.datetime.now().isoformat(),
                'source': 'dynamic_ml_predictor'
            }
            
        except Exception as e:
            logger.error(f"Error getting dynamic prediction data: {e}")
            # Fallback to regular prediction
            return self.get_current_prediction()
    
    def get_performance_dashboard(self) -> dict:
        """Get performance dashboard data"""
        return self.ml_engine.get_performance_dashboard()
    
    def force_new_prediction(self):
        """Force generate a new prediction (for testing/manual override)"""
        logger.info("🔄 Force generating new prediction...")
        return self.generate_daily_prediction()

# Global instance
daily_predictor = DailyPredictionScheduler()
