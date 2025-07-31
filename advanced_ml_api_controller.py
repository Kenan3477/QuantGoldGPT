#!/usr/bin/env python3
"""
Advanced ML Prediction API Controller for GoldGPT
Comprehensive RESTful API with WebSocket integration, background tasks, and security
"""

import asyncio
import json
import sqlite3
import threading
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from functools import wraps
from collections import defaultdict, deque
import secrets
import hashlib

from flask import Blueprint, request, jsonify, g, current_app
from flask_socketio import emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our advanced ML engine
try:
    from advanced_ml_prediction_engine import AdvancedMLPredictionEngine, PredictionResult, get_advanced_ml_predictions
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced ML Engine not available: {e}")
    ML_ENGINE_AVAILABLE = False
    AdvancedMLPredictionEngine = None
    PredictionResult = None
    get_advanced_ml_predictions = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLAPIController:
    """
    Advanced ML Prediction API Controller
    Manages all ML prediction endpoints, WebSocket updates, and background tasks
    """
    
    def __init__(self, app=None, socketio=None):
        self.app = app
        self.socketio = socketio
        self.ml_engine = None
        self.scheduler = None
        self.config = self._load_config()
        self.performance_tracker = PerformanceTracker()
        self.rate_limiter = None
        self.api_keys = {}
        self.active_connections = set()
        
        # Initialize database
        self._init_database()
        
        # Initialize ML engine if available
        if ML_ENGINE_AVAILABLE:
            self._init_ml_engine()
        
        # Setup rate limiting
        if app:
            self._setup_rate_limiting()
            
        # Start background scheduler
        self._start_scheduler()
        
        logger.info("✅ ML API Controller initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with defaults"""
        default_config = {
            'prediction_timeframes': ['5min', '15min', '1h', '4h', '24h'],
            'prediction_schedule_interval': 300,  # 5 minutes
            'validation_schedule_interval': 1800,  # 30 minutes
            'model_retrain_interval': 3600,  # 1 hour
            'confidence_threshold': 0.6,
            'feature_importance_threshold': 0.05,
            'learning_rate': 0.01,
            'max_predictions_per_minute': 10,
            'websocket_update_interval': 30,
            'data_sources': {
                'technical': {'enabled': True, 'weight': 0.3},
                'sentiment': {'enabled': True, 'weight': 0.25},
                'macro': {'enabled': True, 'weight': 0.25},
                'pattern': {'enabled': True, 'weight': 0.1},
                'momentum': {'enabled': True, 'weight': 0.1}
            }
        }
        
        # Try to load from file
        try:
            with open('ml_api_config.json', 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            # Save default config
            with open('ml_api_config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def _init_database(self):
        """Initialize prediction tracking database"""
        conn = sqlite3.connect('goldgpt_ml_api.db')
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                price_change REAL NOT NULL,
                price_change_percent REAL NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                support_level REAL NOT NULL,
                resistance_level REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                features_used TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                target_timestamp DATETIME NOT NULL,
                actual_price REAL,
                validation_timestamp DATETIME,
                accuracy_score REAL,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                accuracy_score REAL NOT NULL,
                total_predictions INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                avg_confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature importance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # API usage tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                client_ip TEXT NOT NULL,
                api_key TEXT,
                response_time REAL NOT NULL,
                status_code INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ ML API database initialized")
    
    def _init_ml_engine(self):
        """Initialize the ML prediction engine"""
        try:
            self.ml_engine = AdvancedMLPredictionEngine()
            logger.info("✅ Advanced ML Engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML engine: {e}")
            self.ml_engine = None
    
    def _setup_rate_limiting(self):
        """Setup rate limiting for API endpoints"""
        self.rate_limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"]
        )
        logger.info("✅ Rate limiting configured")
    
    def _start_scheduler(self):
        """Start background task scheduler"""
        self.scheduler = BackgroundScheduler()
        
        # Schedule prediction generation
        self.scheduler.add_job(
            func=self._generate_scheduled_predictions,
            trigger=IntervalTrigger(seconds=self.config['prediction_schedule_interval']),
            id='prediction_generation',
            name='Generate ML Predictions',
            replace_existing=True
        )
        
        # Schedule prediction validation
        self.scheduler.add_job(
            func=self._validate_predictions,
            trigger=IntervalTrigger(seconds=self.config['validation_schedule_interval']),
            id='prediction_validation',
            name='Validate ML Predictions',
            replace_existing=True
        )
        
        # Schedule model retraining
        self.scheduler.add_job(
            func=self._retrain_models,
            trigger=IntervalTrigger(seconds=self.config['model_retrain_interval']),
            id='model_retraining',
            name='Retrain ML Models',
            replace_existing=True
        )
        
        # Schedule WebSocket updates
        self.scheduler.add_job(
            func=self._broadcast_updates,
            trigger=IntervalTrigger(seconds=self.config['websocket_update_interval']),
            id='websocket_updates',
            name='Broadcast WebSocket Updates',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("✅ Background scheduler started")
    
    def create_blueprint(self) -> Blueprint:
        """Create Flask blueprint with all API endpoints"""
        bp = Blueprint('ml_api', __name__, url_prefix='/api/ml-predictions')
        
        # Authentication decorator
        def require_auth(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    if self._validate_api_key(token):
                        g.api_key = token
                        return f(*args, **kwargs)
                return jsonify({'error': 'Invalid authentication'}), 401
            return decorated_function
        
        # Logging decorator
        def log_request(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                start_time = time.time()
                try:
                    response = f(*args, **kwargs)
                    status_code = response[1] if isinstance(response, tuple) else 200
                    self._log_api_usage(
                        endpoint=request.endpoint,
                        client_ip=request.remote_addr,
                        api_key=getattr(g, 'api_key', None),
                        response_time=time.time() - start_time,
                        status_code=status_code
                    )
                    return response
                except Exception as e:
                    self._log_api_usage(
                        endpoint=request.endpoint,
                        client_ip=request.remote_addr,
                        api_key=getattr(g, 'api_key', None),
                        response_time=time.time() - start_time,
                        status_code=500
                    )
                    raise
            return decorated_function
        
        @bp.route('/<timeframe>', methods=['GET'])
        @self.rate_limiter.limit("60 per minute")
        @log_request
        def get_predictions_by_timeframe(timeframe):
            """Get latest predictions for specific timeframe"""
            try:
                if timeframe not in self.config['prediction_timeframes']:
                    return jsonify({
                        'error': f'Invalid timeframe. Available: {self.config["prediction_timeframes"]}'
                    }), 400
                
                predictions = self._get_latest_predictions(timeframe)
                
                return jsonify({
                    'success': True,
                    'timeframe': timeframe,
                    'predictions': predictions,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_predictions': len(predictions)
                })
                
            except Exception as e:
                logger.error(f"Error getting predictions for {timeframe}: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @bp.route('/all', methods=['GET'])
        @self.rate_limiter.limit("30 per minute")
        @log_request
        def get_all_predictions():
            """Get predictions for all timeframes"""
            try:
                all_predictions = {}
                
                for timeframe in self.config['prediction_timeframes']:
                    all_predictions[timeframe] = self._get_latest_predictions(timeframe)
                
                return jsonify({
                    'success': True,
                    'predictions': all_predictions,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'timeframes': self.config['prediction_timeframes']
                })
                
            except Exception as e:
                logger.error(f"Error getting all predictions: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @bp.route('/accuracy', methods=['GET'])
        @self.rate_limiter.limit("30 per minute")
        @log_request
        def get_accuracy_stats():
            """Get historical accuracy statistics"""
            try:
                days = request.args.get('days', 7, type=int)
                strategy = request.args.get('strategy')
                timeframe = request.args.get('timeframe')
                
                accuracy_stats = self._get_accuracy_statistics(days, strategy, timeframe)
                
                return jsonify({
                    'success': True,
                    'accuracy_stats': accuracy_stats,
                    'period_days': days,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting accuracy stats: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @bp.route('/refresh', methods=['POST'])
        @self.rate_limiter.limit("5 per minute")
        @log_request
        def refresh_predictions():
            """Force regenerate predictions"""
            try:
                timeframes = request.json.get('timeframes', self.config['prediction_timeframes'])
                
                # Generate new predictions
                new_predictions = self._force_generate_predictions(timeframes)
                
                # Broadcast via WebSocket
                self._broadcast_prediction_update(new_predictions)
                
                return jsonify({
                    'success': True,
                    'message': 'Predictions refreshed successfully',
                    'new_predictions': len(new_predictions),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error refreshing predictions: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @bp.route('/features', methods=['GET'])
        @self.rate_limiter.limit("30 per minute")
        @log_request
        def get_feature_importance():
            """Get feature importance data"""
            try:
                strategy = request.args.get('strategy')
                days = request.args.get('days', 7, type=int)
                
                feature_importance = self._get_feature_importance(strategy, days)
                
                return jsonify({
                    'success': True,
                    'feature_importance': feature_importance,
                    'strategy': strategy,
                    'period_days': days,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting feature importance: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @bp.route('/config', methods=['GET', 'POST'])
        @self.rate_limiter.limit("10 per minute")
        @require_auth
        @log_request
        def manage_config():
            """Get or update configuration"""
            if request.method == 'GET':
                return jsonify({
                    'success': True,
                    'config': self.config,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            elif request.method == 'POST':
                try:
                    new_config = request.json
                    self._update_config(new_config)
                    
                    return jsonify({
                        'success': True,
                        'message': 'Configuration updated successfully',
                        'config': self.config,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error updating config: {e}")
                    return jsonify({'error': 'Invalid configuration'}), 400
        
        @bp.route('/status', methods=['GET'])
        @self.rate_limiter.limit("60 per minute")
        @log_request
        def get_system_status():
            """Get system status and health metrics"""
            try:
                status = {
                    'ml_engine_available': self.ml_engine is not None,
                    'scheduler_running': self.scheduler.running if self.scheduler else False,
                    'active_connections': len(self.active_connections),
                    'last_prediction_time': self._get_last_prediction_time(),
                    'total_predictions_today': self._get_daily_prediction_count(),
                    'system_performance': self.performance_tracker.get_metrics(),
                    'config': {
                        'prediction_timeframes': self.config['prediction_timeframes'],
                        'prediction_interval': self.config['prediction_schedule_interval'],
                        'validation_interval': self.config['validation_schedule_interval']
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'status': status
                })
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return bp
    
    def setup_websocket_handlers(self, socketio):
        """Setup WebSocket event handlers"""
        @socketio.on('connect')
        def handle_connect():
            self.active_connections.add(request.sid)
            emit('connection_established', {
                'message': 'Connected to ML prediction updates',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            logger.info(f"WebSocket client connected: {request.sid}")
        
        @socketio.on('disconnect')
        def handle_disconnect():
            self.active_connections.discard(request.sid)
            logger.info(f"WebSocket client disconnected: {request.sid}")
        
        @socketio.on('subscribe_predictions')
        def handle_subscribe(data):
            timeframes = data.get('timeframes', self.config['prediction_timeframes'])
            emit('subscription_confirmed', {
                'timeframes': timeframes,
                'message': 'Subscribed to prediction updates',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        @socketio.on('request_prediction_update')
        def handle_prediction_request(data):
            try:
                timeframe = data.get('timeframe', '1h')
                if timeframe in self.config['prediction_timeframes']:
                    predictions = self._get_latest_predictions(timeframe)
                    emit('prediction_update', {
                        'timeframe': timeframe,
                        'predictions': predictions,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                else:
                    emit('error', {
                        'message': f'Invalid timeframe: {timeframe}',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
            except Exception as e:
                logger.error(f"Error handling prediction request: {e}")
                emit('error', {
                    'message': 'Failed to get predictions',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
    
    def _generate_scheduled_predictions(self):
        """Background task to generate predictions"""
        try:
            if not self.ml_engine:
                return
            
            logger.info("🔄 Generating scheduled predictions...")
            
            new_predictions = []
            for timeframe in self.config['prediction_timeframes']:
                try:
                    predictions = asyncio.run(self.ml_engine.get_comprehensive_predictions(timeframe))
                    self._save_predictions(predictions, timeframe)
                    new_predictions.extend(predictions)
                except Exception as e:
                    logger.error(f"Error generating predictions for {timeframe}: {e}")
            
            if new_predictions:
                self._broadcast_prediction_update(new_predictions)
                logger.info(f"✅ Generated {len(new_predictions)} new predictions")
            
        except Exception as e:
            logger.error(f"Error in scheduled prediction generation: {e}")
    
    def _validate_predictions(self):
        """Background task to validate past predictions"""
        try:
            logger.info("🔄 Validating past predictions...")
            
            # Get predictions ready for validation
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            current_time = datetime.now(timezone.utc)
            cursor.execute('''
                SELECT * FROM ml_predictions 
                WHERE status = 'active' 
                AND target_timestamp <= ?
                AND validation_timestamp IS NULL
            ''', (current_time,))
            
            predictions_to_validate = cursor.fetchall()
            
            if not predictions_to_validate:
                conn.close()
                return
            
            # Get current gold price for validation
            try:
                from price_storage_manager import get_current_gold_price
                current_price = get_current_gold_price()
            except:
                current_price = 3350.0  # Fallback
            
            validated_count = 0
            for pred in predictions_to_validate:
                pred_id, strategy, timeframe, orig_price, pred_price = pred[0], pred[1], pred[2], pred[3], pred[4]
                
                # Calculate accuracy
                accuracy = self._calculate_prediction_accuracy(pred_price, current_price, orig_price)
                
                # Update prediction with validation
                cursor.execute('''
                    UPDATE ml_predictions 
                    SET actual_price = ?, validation_timestamp = ?, accuracy_score = ?, status = 'validated'
                    WHERE id = ?
                ''', (current_price, current_time, accuracy, pred_id))
                
                validated_count += 1
                
                # Update performance tracking
                self._update_performance_tracking(strategy, timeframe, accuracy)
            
            conn.commit()
            conn.close()
            
            if validated_count > 0:
                logger.info(f"✅ Validated {validated_count} predictions")
                self._broadcast_validation_update(validated_count)
            
        except Exception as e:
            logger.error(f"Error in prediction validation: {e}")
    
    def _retrain_models(self):
        """Background task to retrain ML models"""
        try:
            if not self.ml_engine:
                return
            
            logger.info("🔄 Retraining ML models...")
            
            # Get performance data to determine if retraining is needed
            recent_accuracy = self._get_recent_accuracy()
            
            if recent_accuracy < self.config['confidence_threshold']:
                logger.info(f"Model performance below threshold ({recent_accuracy:.2f}), initiating retraining...")
                
                # Retrain models with recent data
                asyncio.run(self.ml_engine.retrain_models())
                
                # Update feature importance
                self._update_feature_importance()
                
                logger.info("✅ Model retraining completed")
                self._broadcast_model_update()
            else:
                logger.info(f"Model performance satisfactory ({recent_accuracy:.2f}), skipping retraining")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
    
    def _broadcast_updates(self):
        """Background task to broadcast periodic updates"""
        try:
            if not self.active_connections:
                return
            
            # Get latest predictions
            latest_predictions = {}
            for timeframe in self.config['prediction_timeframes']:
                latest_predictions[timeframe] = self._get_latest_predictions(timeframe)
            
            # Get system metrics
            metrics = self.performance_tracker.get_metrics()
            
            # Broadcast to all connected clients
            if self.socketio:
                self.socketio.emit('periodic_update', {
                    'predictions': latest_predictions,
                    'metrics': metrics,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error in periodic broadcast: {e}")
    
    def _broadcast_prediction_update(self, predictions):
        """Broadcast new predictions via WebSocket"""
        if self.socketio and self.active_connections:
            try:
                self.socketio.emit('new_predictions', {
                    'predictions': [asdict(p) for p in predictions],
                    'count': len(predictions),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Error broadcasting prediction update: {e}")
    
    def _broadcast_validation_update(self, validated_count):
        """Broadcast validation results via WebSocket"""
        if self.socketio and self.active_connections:
            try:
                self.socketio.emit('validation_update', {
                    'validated_count': validated_count,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Error broadcasting validation update: {e}")
    
    def _broadcast_model_update(self):
        """Broadcast model retraining completion via WebSocket"""
        if self.socketio and self.active_connections:
            try:
                self.socketio.emit('model_update', {
                    'message': 'ML models have been retrained',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Error broadcasting model update: {e}")
    
    def _get_latest_predictions(self, timeframe: str) -> List[Dict]:
        """Get latest predictions for a timeframe"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM ml_predictions 
                WHERE timeframe = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', (timeframe,))
            
            predictions = []
            for row in cursor.fetchall():
                pred = {
                    'id': row[0],
                    'strategy_name': row[1],
                    'timeframe': row[2],
                    'current_price': row[3],
                    'predicted_price': row[4],
                    'price_change': row[5],
                    'price_change_percent': row[6],
                    'direction': row[7],
                    'confidence': row[8],
                    'support_level': row[9],
                    'resistance_level': row[10],
                    'stop_loss': row[11],
                    'take_profit': row[12],
                    'features_used': json.loads(row[13]),
                    'timestamp': row[14],
                    'target_timestamp': row[15],
                    'actual_price': row[16],
                    'validation_timestamp': row[17],
                    'accuracy_score': row[18],
                    'status': row[19]
                }
                predictions.append(pred)
            
            conn.close()
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting latest predictions: {e}")
            return []
    
    def _save_predictions(self, predictions: List[Dict], timeframe: str):
        """Save predictions to database"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            for pred in predictions:
                target_timestamp = datetime.now(timezone.utc) + self._get_timeframe_delta(timeframe)
                
                cursor.execute('''
                    INSERT INTO ml_predictions (
                        strategy_name, timeframe, current_price, predicted_price,
                        price_change, price_change_percent, direction, confidence,
                        support_level, resistance_level, stop_loss, take_profit,
                        features_used, target_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pred.strategy_name, pred.timeframe, pred.current_price, pred.predicted_price,
                    pred.price_change, pred.price_change_percent, pred.direction, pred.confidence,
                    pred.support_level, pred.resistance_level, pred.stop_loss, pred.take_profit,
                    json.dumps(pred.features_used), target_timestamp
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta"""
        timeframe_map = {
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '24h': timedelta(hours=24)
        }
        return timeframe_map.get(timeframe, timedelta(hours=1))
    
    def _calculate_prediction_accuracy(self, predicted_price: float, actual_price: float, original_price: float) -> float:
        """Calculate prediction accuracy score"""
        try:
            predicted_change = predicted_price - original_price
            actual_change = actual_price - original_price
            
            # Direction accuracy (50% weight)
            direction_correct = (predicted_change > 0) == (actual_change > 0)
            direction_score = 1.0 if direction_correct else 0.0
            
            # Magnitude accuracy (50% weight)
            if abs(actual_change) == 0:
                magnitude_score = 1.0 if abs(predicted_change) < original_price * 0.01 else 0.0
            else:
                relative_error = abs(predicted_change - actual_change) / abs(actual_change)
                magnitude_score = max(0.0, 1.0 - relative_error)
            
            return (direction_score * 0.5) + (magnitude_score * 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0
    
    def _get_accuracy_statistics(self, days: int, strategy: Optional[str], timeframe: Optional[str]) -> Dict:
        """Get accuracy statistics"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            # Build query with optional filters
            query = '''
                SELECT strategy_name, timeframe, AVG(accuracy_score) as avg_accuracy,
                       COUNT(*) as total_predictions, SUM(CASE WHEN accuracy_score > 0.5 THEN 1 ELSE 0 END) as accurate_predictions
                FROM ml_predictions 
                WHERE validation_timestamp >= datetime('now', '-' || ? || ' days')
                AND validation_timestamp IS NOT NULL
            '''
            params = [days]
            
            if strategy:
                query += ' AND strategy_name = ?'
                params.append(strategy)
            
            if timeframe:
                query += ' AND timeframe = ?'
                params.append(timeframe)
            
            query += ' GROUP BY strategy_name, timeframe'
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            stats = {
                'overall': {'avg_accuracy': 0.0, 'total_predictions': 0, 'accurate_predictions': 0},
                'by_strategy': {},
                'by_timeframe': {}
            }
            
            total_accuracy_sum = 0
            total_predictions = 0
            total_accurate = 0
            
            for row in results:
                strategy_name, tf, avg_acc, total_pred, acc_pred = row
                
                total_accuracy_sum += avg_acc * total_pred
                total_predictions += total_pred
                total_accurate += acc_pred
                
                if strategy_name not in stats['by_strategy']:
                    stats['by_strategy'][strategy_name] = {'avg_accuracy': avg_acc, 'total_predictions': total_pred, 'accurate_predictions': acc_pred}
                
                if tf not in stats['by_timeframe']:
                    stats['by_timeframe'][tf] = {'avg_accuracy': avg_acc, 'total_predictions': total_pred, 'accurate_predictions': acc_pred}
            
            if total_predictions > 0:
                stats['overall']['avg_accuracy'] = total_accuracy_sum / total_predictions
                stats['overall']['total_predictions'] = total_predictions
                stats['overall']['accurate_predictions'] = total_accurate
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting accuracy statistics: {e}")
            return {'overall': {'avg_accuracy': 0.0, 'total_predictions': 0, 'accurate_predictions': 0}}
    
    def _get_feature_importance(self, strategy: Optional[str], days: int) -> Dict:
        """Get feature importance data"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            query = '''
                SELECT feature_name, AVG(importance_score) as avg_importance
                FROM feature_importance 
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
            '''
            params = [days]
            
            if strategy:
                query += ' AND strategy_name = ?'
                params.append(strategy)
            
            query += ' GROUP BY feature_name ORDER BY avg_importance DESC'
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            features = {}
            for feature_name, avg_importance in results:
                features[feature_name] = avg_importance
            
            conn.close()
            return features
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _update_config(self, new_config: Dict):
        """Update system configuration"""
        # Validate configuration
        required_keys = ['prediction_timeframes', 'prediction_schedule_interval', 'confidence_threshold']
        for key in required_keys:
            if key not in new_config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Update configuration
        self.config.update(new_config)
        
        # Save to file
        with open('ml_api_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Restart scheduler with new intervals
        if self.scheduler:
            self.scheduler.shutdown()
            self._start_scheduler()
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        # Simple API key validation - in production, use proper authentication
        valid_keys = ['goldgpt-api-key', 'demo-key', 'test-key']
        return api_key in valid_keys
    
    def _log_api_usage(self, endpoint: str, client_ip: str, api_key: Optional[str], response_time: float, status_code: int):
        """Log API usage for analytics"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_usage (endpoint, client_ip, api_key, response_time, status_code)
                VALUES (?, ?, ?, ?, ?)
            ''', (endpoint, client_ip, api_key, response_time, status_code))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging API usage: {e}")
    
    def _get_last_prediction_time(self) -> Optional[str]:
        """Get timestamp of last prediction"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT MAX(timestamp) FROM ml_predictions')
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else None
            
        except Exception as e:
            logger.error(f"Error getting last prediction time: {e}")
            return None
    
    def _get_daily_prediction_count(self) -> int:
        """Get count of predictions generated today"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM ml_predictions 
                WHERE DATE(timestamp) = DATE('now')
            ''')
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting daily prediction count: {e}")
            return 0
    
    def _get_recent_accuracy(self) -> float:
        """Get recent model accuracy"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(accuracy_score) FROM ml_predictions 
                WHERE validation_timestamp >= datetime('now', '-24 hours')
                AND validation_timestamp IS NOT NULL
            ''')
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else 0.7
            
        except Exception as e:
            logger.error(f"Error getting recent accuracy: {e}")
            return 0.7
    
    def _update_performance_tracking(self, strategy: str, timeframe: str, accuracy: float):
        """Update performance tracking"""
        try:
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            # Get current performance data
            cursor.execute('''
                SELECT total_predictions, correct_predictions, avg_confidence 
                FROM ml_performance 
                WHERE strategy_name = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (strategy, timeframe))
            
            result = cursor.fetchone()
            if result:
                total_pred, correct_pred, avg_conf = result
                new_total = total_pred + 1
                new_correct = correct_pred + (1 if accuracy > 0.5 else 0)
                new_accuracy = new_correct / new_total
            else:
                new_total = 1
                new_correct = 1 if accuracy > 0.5 else 0
                new_accuracy = accuracy
            
            # Insert new performance record
            cursor.execute('''
                INSERT INTO ml_performance (strategy_name, timeframe, accuracy_score, total_predictions, correct_predictions, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (strategy, timeframe, new_accuracy, new_total, new_correct, 0.7))  # Default confidence
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def _update_feature_importance(self):
        """Update feature importance data"""
        try:
            if not self.ml_engine:
                return
            
            # Get feature importance from ML engine
            feature_importance = asyncio.run(self.ml_engine.get_feature_importance())
            
            conn = sqlite3.connect('goldgpt_ml_api.db')
            cursor = conn.cursor()
            
            for strategy, features in feature_importance.items():
                for feature_name, importance in features.items():
                    cursor.execute('''
                        INSERT INTO feature_importance (strategy_name, feature_name, importance_score)
                        VALUES (?, ?, ?)
                    ''', (strategy, feature_name, importance))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")
    
    def _force_generate_predictions(self, timeframes: List[str]) -> List[Dict]:
        """Force generate predictions for specified timeframes"""
        try:
            if not self.ml_engine:
                return []
            
            all_predictions = []
            for timeframe in timeframes:
                if timeframe in self.config['prediction_timeframes']:
                    predictions = asyncio.run(self.ml_engine.get_comprehensive_predictions(timeframe))
                    self._save_predictions(predictions, timeframe)
                    all_predictions.extend([asdict(p) for p in predictions])
            
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error force generating predictions: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the API controller"""
        if self.scheduler:
            self.scheduler.shutdown()
        logger.info("✅ ML API Controller shutdown complete")


class PerformanceTracker:
    """Track system performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        self.metrics[name].append((time.time(), value))
        # Keep only last 100 metrics
        if len(self.metrics[name]) > 100:
            self.metrics[name].popleft()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_metrics = {}
        
        for name, values in self.metrics.items():
            if values:
                recent_values = [v for t, v in values if time.time() - t < 3600]  # Last hour
                if recent_values:
                    current_metrics[name] = {
                        'average': sum(recent_values) / len(recent_values),
                        'min': min(recent_values),
                        'max': max(recent_values),
                        'count': len(recent_values)
                    }
        
        current_metrics['uptime_seconds'] = time.time() - self.start_time
        
        return current_metrics


# Global API controller instance
ml_api_controller = None

def create_ml_api_controller(app, socketio):
    """Factory function to create ML API controller"""
    global ml_api_controller
    ml_api_controller = MLAPIController(app, socketio)
    return ml_api_controller
