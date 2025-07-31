#!/usr/bin/env python3
"""
GoldGPT Learning System Integration
Integrates advanced prediction tracking and learning system with the existing Flask application
"""

import logging
import asyncio
import os
import sqlite3
import threading
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from flask import Flask

# Import the learning system components
from prediction_tracker import PredictionTracker
from learning_engine import LearningEngine
from backtesting_framework import BacktestEngine, HistoricalDataManager
from dashboard_api import dashboard_bp, initialize_dashboard_services

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningSystemIntegration:
    """
    Integrates the advanced learning system with the existing GoldGPT Flask application
    """
    
    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.prediction_tracker = None
        self.learning_engine = None
        self.backtest_engine = None
        self.data_manager = None
        self.is_initialized = False
        
        # Database paths
        self.learning_db_path = "goldgpt_learning_system.db"
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize the learning system with the Flask app"""
        self.app = app
        
        # Initialize database
        self._init_database()
        
        # Initialize components
        self._init_components()
        
        # Register blueprints
        self._register_blueprints()
        
        # Register background tasks
        self._register_background_tasks()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        self.is_initialized = True
        logger.info("✅ Learning system integration completed successfully")
    
    def _init_database(self):
        """Initialize the learning system database"""
        try:
            # Create database if it doesn't exist
            if not os.path.exists(self.learning_db_path):
                # Read and execute schema
                schema_path = "prediction_learning_schema.sql"
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()
                    
                    with sqlite3.connect(self.learning_db_path) as conn:
                        # Split schema into individual statements
                        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                        
                        for statement in statements:
                            if statement and not statement.startswith('--'):
                                try:
                                    conn.execute(statement)
                                except sqlite3.Error as e:
                                    if "already exists" not in str(e):
                                        logger.warning(f"Schema execution warning: {e}")
                        
                        conn.commit()
                    logger.info("✅ Learning system database initialized")
                else:
                    logger.warning("⚠️ Schema file not found, using simplified initialization")
                    # Create basic tables if schema file is missing
                    self._create_basic_schema()
            else:
                logger.info("✅ Learning system database found")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _create_basic_schema(self):
        """Create basic schema if full schema file is not available"""
        basic_schema = """
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            strategy_name TEXT NOT NULL,
            symbol TEXT NOT NULL DEFAULT 'XAUUSD',
            timeframe TEXT NOT NULL,
            confidence REAL NOT NULL,
            direction TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            current_price REAL NOT NULL,
            is_validated BOOLEAN DEFAULT FALSE,
            actual_price REAL NULL,
            is_winner BOOLEAN NULL
        );
        
        CREATE TABLE IF NOT EXISTS learning_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_level REAL NOT NULL
        );
        """
        
        with sqlite3.connect(self.learning_db_path) as conn:
            conn.executescript(basic_schema)
            logger.info("✅ Basic schema created")
    
    def _init_components(self):
        """Initialize learning system components"""
        try:
            # Initialize prediction tracker
            self.prediction_tracker = PredictionTracker(db_path=self.learning_db_path)
            
            # Initialize learning engine
            self.learning_engine = LearningEngine(
                prediction_tracker=self.prediction_tracker
            )
            
            # Initialize historical data manager
            self.data_manager = HistoricalDataManager()
            
            # Initialize backtest engine
            self.backtest_engine = BacktestEngine(
                prediction_tracker=self.prediction_tracker,
                data_manager=self.data_manager
            )
            
            logger.info("✅ Learning system components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            # Create fallback components
            self._create_fallback_components()
    
    def _create_fallback_components(self):
        """Create simplified fallback components if full initialization fails"""
        class FallbackTracker:
            def store_prediction(self, *args, **kwargs):
                return "fallback_prediction_id"
            def validate_prediction(self, *args, **kwargs):
                return {"status": "fallback"}
            def get_performance_summary(self, *args, **kwargs):
                return {"total_predictions": 0, "accuracy_rate": 0.0}
        
        class FallbackEngine:
            def __init__(self, tracker):
                self.tracker = tracker
            def get_recent_insights(self, *args, **kwargs):
                return []
            def generate_learning_insights(self, *args, **kwargs):
                return {"insights": []}
        
        class FallbackBacktest:
            def __init__(self, tracker, data_manager):
                self.tracker = tracker
                self.data_manager = data_manager
            async def run_backtest(self, *args, **kwargs):
                return {"total_trades": 0, "win_rate": 0.0}
        
        self.prediction_tracker = FallbackTracker()
        self.learning_engine = FallbackEngine(self.prediction_tracker)
        self.backtest_engine = FallbackBacktest(self.prediction_tracker, None)
        
        logger.warning("⚠️ Using fallback components")
    
    def _register_blueprints(self):
        """Register the dashboard blueprint"""
        try:
            # Initialize dashboard services
            initialize_dashboard_services(
                self.prediction_tracker,
                self.learning_engine,
                self.backtest_engine
            )
            
            # Register the dashboard blueprint
            self.app.register_blueprint(dashboard_bp)
            
            logger.info("✅ Dashboard blueprint registered")
            
        except Exception as e:
            logger.error(f"❌ Failed to register blueprints: {e}")
    
    def _register_background_tasks(self):
        """Register background learning tasks"""
        try:
            # Start continuous learning in background
            if hasattr(self.learning_engine, 'start_continuous_learning'):
                threading.Thread(
                    target=self._run_continuous_learning,
                    daemon=True
                ).start()
                logger.info("✅ Background learning tasks started")
            
        except Exception as e:
            logger.error(f"❌ Failed to register background tasks: {e}")
    
    def _run_continuous_learning(self):
        """Run continuous learning in background thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while True:
                try:
                    # Run learning cycle every hour
                    if hasattr(self.learning_engine, 'continuous_learning_cycle'):
                        loop.run_until_complete(
                            self.learning_engine.continuous_learning_cycle()
                        )
                    
                    # Sleep for 1 hour
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Learning cycle error: {e}")
                    time.sleep(300)  # Wait 5 minutes before retry
                    
        except Exception as e:
            logger.error(f"Continuous learning failed: {e}")
    
    def _setup_event_handlers(self):
        """Set up event handlers for prediction tracking"""
        # This would integrate with existing WebSocket events
        pass
    
    # Public API methods for integration with existing prediction systems
    
    def track_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """
        Track a prediction made by the existing ML system
        
        Args:
            prediction_data: Dictionary containing prediction details
            
        Returns:
            Prediction ID for tracking
        """
        try:
            if not self.prediction_tracker:
                return "tracker_not_available"
            
            # Convert existing prediction format to tracker format
            tracker_data = self._convert_prediction_format(prediction_data)
            
            # Store prediction
            prediction_id = self.prediction_tracker.store_prediction(tracker_data)
            
            logger.info(f"✅ Prediction tracked: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Failed to track prediction: {e}")
            return f"error_{datetime.now().timestamp()}"
    
    def validate_prediction(self, prediction_id: str, actual_price: float, 
                          validation_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Validate a tracked prediction with actual outcome
        
        Args:
            prediction_id: ID of the prediction to validate
            actual_price: Actual market price at validation time
            validation_time: Time of validation (defaults to now)
            
        Returns:
            Validation result
        """
        try:
            if not self.prediction_tracker:
                return {"status": "tracker_not_available"}
            
            result = self.prediction_tracker.validate_prediction(
                prediction_id=prediction_id,
                actual_price=actual_price,
                validation_time=validation_time or datetime.now(timezone.utc)
            )
            
            logger.info(f"✅ Prediction validated: {prediction_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate prediction: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_learning_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning insights"""
        try:
            if not self.learning_engine:
                return []
            
            insights = self.learning_engine.get_recent_insights(limit=limit)
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return []
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the dashboard"""
        try:
            if not self.prediction_tracker:
                return {"error": "tracker_not_available"}
            
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            summary = self.prediction_tracker.get_performance_summary(
                start_date=start_date,
                end_date=end_date
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def _convert_prediction_format(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert existing prediction format to tracker format"""
        return {
            'symbol': prediction_data.get('symbol', 'XAUUSD'),
            'timeframe': prediction_data.get('timeframe', '1H'),
            'strategy_name': prediction_data.get('strategy', 'advanced_ml'),
            'direction': prediction_data.get('direction', 'neutral'),
            'confidence': prediction_data.get('confidence', 0.5),
            'predicted_price': prediction_data.get('predicted_price', 0.0),
            'current_price': prediction_data.get('current_price', 0.0),
            'stop_loss': prediction_data.get('stop_loss'),
            'take_profit': prediction_data.get('take_profit'),
            'features_used': json.dumps(prediction_data.get('features', [])),
            'technical_indicators': json.dumps(prediction_data.get('indicators', {})),
            'market_conditions': json.dumps(prediction_data.get('market_context', {}))
        }
    
    # Integration helper methods
    
    def enhance_existing_prediction_endpoint(self, original_prediction_func):
        """
        Decorator to enhance existing prediction endpoints with tracking
        
        Usage:
            @learning_system.enhance_existing_prediction_endpoint
            def get_ai_analysis():
                # existing prediction logic
                return prediction_result
        """
        def wrapper(*args, **kwargs):
            # Call original function
            result = original_prediction_func(*args, **kwargs)
            
            # Track the prediction if it's successful
            if isinstance(result, dict) and 'predictions' in result:
                for prediction in result['predictions']:
                    prediction_id = self.track_prediction(prediction)
                    prediction['tracking_id'] = prediction_id
            
            return result
        
        return wrapper
    
    def create_validation_webhook(self, app: Flask):
        """Create webhook endpoint for automatic validation"""
        @app.route('/api/learning/validate-prediction', methods=['POST'])
        def validate_prediction_webhook():
            from flask import request, jsonify
            
            try:
                data = request.get_json()
                prediction_id = data.get('prediction_id')
                actual_price = data.get('actual_price')
                
                if not prediction_id or actual_price is None:
                    return jsonify({'error': 'Missing required parameters'}), 400
                
                result = self.validate_prediction(prediction_id, actual_price)
                
                return jsonify({
                    'success': True,
                    'prediction_id': prediction_id,
                    'validation_result': result
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        logger.info("✅ Validation webhook created")
    
    # Health check methods
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all learning system components"""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'database': {},
            'metrics': {}
        }
        
        try:
            # Check database connection
            with sqlite3.connect(self.learning_db_path, timeout=5.0) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM prediction_records")
                record_count = cursor.fetchone()[0]
                status['database'] = {
                    'status': 'connected',
                    'total_predictions': record_count
                }
        except Exception as e:
            status['database'] = {'status': 'error', 'error': str(e)}
            status['overall_status'] = 'degraded'
        
        # Check components
        components_status = {
            'prediction_tracker': self.prediction_tracker is not None,
            'learning_engine': self.learning_engine is not None,
            'backtest_engine': self.backtest_engine is not None,
            'data_manager': self.data_manager is not None
        }
        
        status['components'] = components_status
        
        if not all(components_status.values()):
            status['overall_status'] = 'degraded'
        
        # Add performance metrics if available
        try:
            recent_performance = self.get_performance_summary(days=7)
            status['metrics'] = {
                'recent_accuracy': recent_performance.get('accuracy_rate', 0.0),
                'total_predictions_7d': recent_performance.get('total_predictions', 0),
                'learning_insights': len(self.get_learning_insights(5))
            }
        except Exception as e:
            status['metrics'] = {'error': str(e)}
        
        return status

# Global instance for easy integration
learning_system = LearningSystemIntegration()

def integrate_learning_system_with_app(app: Flask) -> LearningSystemIntegration:
    """
    Main integration function to add learning system to existing Flask app
    
    Usage in app.py:
        from learning_system_integration import integrate_learning_system_with_app
        learning_system = integrate_learning_system_with_app(app)
    """
    try:
        integration = LearningSystemIntegration(app)
        
        # Create validation webhook
        integration.create_validation_webhook(app)
        
        # Add health check endpoint
        @app.route('/api/learning/health')
        def learning_system_health():
            from flask import jsonify
            health = integration.health_check()
            status_code = 200 if health['overall_status'] == 'healthy' else 503
            return jsonify(health), status_code
        
        logger.info("🎯 Learning system fully integrated with Flask app")
        return integration
        
    except Exception as e:
        logger.error(f"❌ Failed to integrate learning system: {e}")
        raise

if __name__ == "__main__":
    # Test the integration
    from flask import Flask
    
    app = Flask(__name__)
    learning_system = integrate_learning_system_with_app(app)
    
    print("✅ Learning system integration test completed")
    print(f"Health status: {learning_system.health_check()}")
