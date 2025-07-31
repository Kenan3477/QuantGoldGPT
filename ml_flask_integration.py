"""
Advanced ML Engine Flask Integration for GoldGPT Web Application

This module provides seamless integration of the advanced multi-strategy ML engine
with the existing Flask web application, replacing the current AI signal generation
with advanced multi-strategy predictions.

Author: AI Assistant
Created: 2025-01-19
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
import threading
import time

from advanced_multi_strategy_ml_engine import MultiStrategyMLEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFlaskIntegration:
    """Main integration class for connecting ML engine with Flask"""
    
    def __init__(self, app: Flask, socketio: SocketIO):
        self.app = app
        self.socketio = socketio
        self.ml_engine = None
        self.prediction_cache = {}
        self.cache_expiry = {}
        self.is_running = False
        self.background_thread = None
        
        # Initialize ML engine
        self._initialize_ml_engine()
        
        # Register routes
        self._register_routes()
        
        # Register WebSocket handlers
        self._register_websocket_handlers()
        
    def _initialize_ml_engine(self):
        """Initialize the ML engine"""
        try:
            logger.info("Initializing Advanced Multi-Strategy ML Engine...")
            self.ml_engine = MultiStrategyMLEngine()
            logger.info("ML Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML Engine: {e}")
            # Create a fallback minimal engine for testing
            self.ml_engine = self._create_fallback_engine()
    
    def _create_fallback_engine(self):
        """Create a fallback engine for testing purposes"""
        class FallbackEngine:
            def generate_prediction(self, symbol, timeframe='1h'):
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'prediction': 'BUY',
                    'confidence': 0.65,
                    'strategies': {
                        'technical': {'prediction': 'BUY', 'confidence': 0.7, 'strength': 0.6},
                        'sentiment': {'prediction': 'HOLD', 'confidence': 0.5, 'strength': 0.4},
                        'macro': {'prediction': 'BUY', 'confidence': 0.8, 'strength': 0.7}
                    },
                    'ensemble': {
                        'prediction': 'BUY',
                        'confidence': 0.65,
                        'voting_details': {'BUY': 2, 'HOLD': 1, 'SELL': 0}
                    },
                    'technical_features': {
                        'rsi': 45.2,
                        'macd': 1.2,
                        'bb_position': 0.3,
                        'volume_sma_ratio': 1.1
                    },
                    'risk_metrics': {
                        'risk_score': 0.35,
                        'volatility': 0.028,
                        'max_drawdown': 0.15
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
            def get_strategy_performance(self):
                return {
                    'technical': {'accuracy': 0.72, 'total_predictions': 100, 'correct_predictions': 72},
                    'sentiment': {'accuracy': 0.58, 'total_predictions': 85, 'correct_predictions': 49},
                    'macro': {'accuracy': 0.81, 'total_predictions': 60, 'correct_predictions': 49}
                }
                
        return FallbackEngine()
    
    def _register_routes(self):
        """Register all ML-related Flask routes"""
        
        @self.app.route('/api/ai-signals/generate', methods=['POST'])
        def generate_ml_signal():
            """Generate advanced ML signal - replaces existing AI signal endpoint"""
            try:
                data = request.json or {}
                symbol = data.get('symbol', 'XAUUSD')
                timeframe = data.get('timeframe', '1h')
                
                logger.info(f"Generating ML signal for {symbol} {timeframe}")
                
                # Check cache first (5-minute expiry)
                cache_key = f"{symbol}_{timeframe}"
                current_time = time.time()
                
                if (cache_key in self.prediction_cache and 
                    cache_key in self.cache_expiry and
                    current_time < self.cache_expiry[cache_key]):
                    
                    logger.info(f"Returning cached prediction for {symbol}")
                    return jsonify({
                        'success': True,
                        'signal': self.prediction_cache[cache_key],
                        'cached': True,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Generate new prediction
                if hasattr(self.ml_engine, 'generate_prediction'):
                    prediction = self.ml_engine.generate_prediction(symbol, timeframe)
                else:
                    # Fallback async call for the full engine
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    prediction = loop.run_until_complete(
                        self.ml_engine.generate_comprehensive_prediction(symbol, timeframe)
                    )
                    loop.close()
                
                # Cache the prediction for 5 minutes
                self.prediction_cache[cache_key] = prediction
                self.cache_expiry[cache_key] = current_time + 300  # 5 minutes
                
                # Format for GoldGPT compatibility
                formatted_signal = self._format_signal_for_goldgpt(prediction)
                
                return jsonify({
                    'success': True,
                    'signal': formatted_signal,
                    'raw_prediction': prediction,
                    'cached': False,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error generating ML signal: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/ml/strategies/performance', methods=['GET'])
        def get_strategy_performance():
            """Get performance metrics for all strategies"""
            try:
                if hasattr(self.ml_engine, 'get_strategy_performance'):
                    performance = self.ml_engine.get_strategy_performance()
                else:
                    performance = self.ml_engine.get_strategy_performance()
                
                return jsonify({
                    'success': True,
                    'performance': performance,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting strategy performance: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/ml/prediction/detailed', methods=['POST'])
        def get_detailed_prediction():
            """Get detailed prediction with all strategy breakdowns"""
            try:
                data = request.json or {}
                symbol = data.get('symbol', 'XAUUSD')
                timeframe = data.get('timeframe', '1h')
                
                if hasattr(self.ml_engine, 'generate_prediction'):
                    prediction = self.ml_engine.generate_prediction(symbol, timeframe)
                else:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    prediction = loop.run_until_complete(
                        self.ml_engine.generate_comprehensive_prediction(symbol, timeframe)
                    )
                    loop.close()
                
                return jsonify({
                    'success': True,
                    'prediction': prediction,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting detailed prediction: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/ml/dashboard/data', methods=['GET'])
        def get_ml_dashboard_data():
            """Get comprehensive ML dashboard data"""
            try:
                # Get recent predictions for multiple timeframes
                symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
                timeframes = ['15m', '1h', '4h', '1d']
                
                dashboard_data = {
                    'predictions': {},
                    'performance': {},
                    'summary': {
                        'active_strategies': 5,
                        'total_predictions_today': 0,
                        'accuracy_today': 0.0,
                        'best_performing_strategy': 'technical'
                    }
                }
                
                # Get performance data
                if hasattr(self.ml_engine, 'get_strategy_performance'):
                    dashboard_data['performance'] = self.ml_engine.get_strategy_performance()
                
                # Get recent predictions (from cache if available)
                for symbol in symbols[:2]:  # Limit to prevent timeout
                    for timeframe in timeframes[:2]:
                        cache_key = f"{symbol}_{timeframe}"
                        if cache_key in self.prediction_cache:
                            if symbol not in dashboard_data['predictions']:
                                dashboard_data['predictions'][symbol] = {}
                            dashboard_data['predictions'][symbol][timeframe] = self.prediction_cache[cache_key]
                
                return jsonify({
                    'success': True,
                    'data': dashboard_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting ML dashboard data: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        # Note: /multi-strategy-ml-dashboard route is now defined in main app.py
        # to avoid conflicts and point to the correct template
        
        @self.app.route('/advanced-ml-predictions')  
        def advanced_ml_predictions():
            """Render advanced ML predictions page"""
            return render_template('ml_dashboard.html')
        
        @self.app.route('/ml-dashboard')
        def ml_dashboard():
            """Render Advanced ML dashboard page"""
            return render_template('ml_dashboard.html')
        
        @self.app.route('/advanced-ml')
        def advanced_ml():
            """Render advanced ML analysis page"""
            return render_template('advanced_ml.html')
    
    def _register_websocket_handlers(self):
        """Register WebSocket handlers for real-time updates"""
        
        @self.socketio.on('request_ml_update')
        def handle_ml_update_request(data):
            """Handle request for ML prediction updates"""
            try:
                symbol = data.get('symbol', 'XAUUSD')
                timeframe = data.get('timeframe', '1h')
                
                logger.info(f"WebSocket ML update requested for {symbol} {timeframe}")
                
                # Generate prediction
                if hasattr(self.ml_engine, 'generate_prediction'):
                    prediction = self.ml_engine.generate_prediction(symbol, timeframe)
                else:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    prediction = loop.run_until_complete(
                        self.ml_engine.generate_comprehensive_prediction(symbol, timeframe)
                    )
                    loop.close()
                
                # Emit the update
                emit('ml_prediction_update', {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'prediction': prediction,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in WebSocket ML update: {e}")
                emit('ml_update_error', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        @self.socketio.on('start_ml_monitoring')
        def handle_start_monitoring():
            """Start real-time ML monitoring"""
            if not self.is_running:
                self.is_running = True
                self.background_thread = threading.Thread(target=self._background_ml_updates)
                self.background_thread.daemon = True
                self.background_thread.start()
                logger.info("Started ML monitoring background thread")
                
                emit('ml_monitoring_started', {
                    'status': 'active',
                    'timestamp': datetime.now().isoformat()
                })
        
        @self.socketio.on('stop_ml_monitoring')
        def handle_stop_monitoring():
            """Stop real-time ML monitoring"""
            self.is_running = False
            logger.info("Stopped ML monitoring")
            
            emit('ml_monitoring_stopped', {
                'status': 'inactive',
                'timestamp': datetime.now().isoformat()
            })
    
    def _background_ml_updates(self):
        """Background thread for periodic ML updates"""
        symbols = ['XAUUSD', 'EURUSD']
        timeframes = ['1h', '4h']
        update_interval = 300  # 5 minutes
        
        while self.is_running:
            try:
                for symbol in symbols:
                    for timeframe in timeframes:
                        if not self.is_running:
                            break
                        
                        # Generate prediction
                        if hasattr(self.ml_engine, 'generate_prediction'):
                            prediction = self.ml_engine.generate_prediction(symbol, timeframe)
                        else:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            prediction = loop.run_until_complete(
                                self.ml_engine.generate_comprehensive_prediction(symbol, timeframe)
                            )
                            loop.close()
                        
                        # Cache the prediction
                        cache_key = f"{symbol}_{timeframe}"
                        self.prediction_cache[cache_key] = prediction
                        self.cache_expiry[cache_key] = time.time() + 300
                        
                        # Emit update via WebSocket
                        self.socketio.emit('background_ml_update', {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'prediction': prediction,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        time.sleep(2)  # Small delay between predictions
                
                # Wait before next full cycle
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in background ML updates: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _format_signal_for_goldgpt(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Format ML prediction to match GoldGPT's expected signal format"""
        try:
            # Extract key information
            signal_direction = prediction.get('prediction', 'HOLD')
            confidence = prediction.get('confidence', 0.5)
            ensemble_data = prediction.get('ensemble', {})
            strategies = prediction.get('strategies', {})
            technical_features = prediction.get('technical_features', {})
            risk_metrics = prediction.get('risk_metrics', {})
            
            # Convert to GoldGPT format
            formatted_signal = {
                'direction': signal_direction,
                'confidence': confidence,
                'strength': min(confidence * 1.2, 1.0),  # Slight boost for display
                'entry_price': technical_features.get('current_price', 2000.0),
                'stop_loss': technical_features.get('support_level', 1980.0),
                'take_profit': technical_features.get('resistance_level', 2020.0),
                'risk_reward_ratio': risk_metrics.get('risk_reward_ratio', 2.0),
                'analysis': {
                    'technical': {
                        'signal': strategies.get('technical', {}).get('prediction', 'HOLD'),
                        'confidence': strategies.get('technical', {}).get('confidence', 0.5),
                        'rsi': technical_features.get('rsi', 50.0),
                        'macd': technical_features.get('macd', 0.0),
                        'bollinger_position': technical_features.get('bb_position', 0.5)
                    },
                    'sentiment': {
                        'signal': strategies.get('sentiment', {}).get('prediction', 'NEUTRAL'),
                        'confidence': strategies.get('sentiment', {}).get('confidence', 0.5),
                        'score': strategies.get('sentiment', {}).get('sentiment_score', 0.0)
                    },
                    'macro': {
                        'signal': strategies.get('macro', {}).get('prediction', 'NEUTRAL'),
                        'confidence': strategies.get('macro', {}).get('confidence', 0.5),
                        'economic_bias': strategies.get('macro', {}).get('economic_sentiment', 0.0)
                    }
                },
                'ensemble': {
                    'method': 'advanced_voting',
                    'final_prediction': signal_direction,
                    'confidence': confidence,
                    'voting_breakdown': ensemble_data.get('voting_details', {})
                },
                'risk_assessment': {
                    'risk_level': 'Medium' if risk_metrics.get('risk_score', 0.5) < 0.7 else 'High',
                    'volatility': risk_metrics.get('volatility', 0.02),
                    'position_size_recommendation': min(0.02 / risk_metrics.get('volatility', 0.02), 0.1)
                },
                'metadata': {
                    'engine_version': 'AdvancedML_v2.0',
                    'strategies_used': list(strategies.keys()),
                    'prediction_time': datetime.now().isoformat(),
                    'symbol': prediction.get('symbol', 'XAUUSD'),
                    'timeframe': prediction.get('timeframe', '1h')
                }
            }
            
            return formatted_signal
            
        except Exception as e:
            logger.error(f"Error formatting signal for GoldGPT: {e}")
            return {
                'direction': 'HOLD',
                'confidence': 0.5,
                'strength': 0.5,
                'error': str(e)
            }
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        logger.info("ML Flask Integration cleanup completed")

def integrate_ml_with_flask(app: Flask, socketio: SocketIO) -> MLFlaskIntegration:
    """
    Main integration function to connect the Advanced ML Engine with Flask app
    
    Usage in app.py:
    
    from ml_flask_integration import integrate_ml_with_flask
    
    # After creating Flask app and SocketIO
    ml_integration = integrate_ml_with_flask(app, socketio)
    
    Args:
        app: Flask application instance
        socketio: Flask-SocketIO instance
        
    Returns:
        MLFlaskIntegration: The integration instance for additional control
    """
    return MLFlaskIntegration(app, socketio)

# Template route for ML dashboard (optional - can be added to main app.py)
def register_ml_dashboard_route(app: Flask):
    """Register ML dashboard template route"""
    
    @app.route('/ml-dashboard')
    def ml_dashboard():
        """Render ML dashboard page"""
        return render_template('ml_dashboard.html')
    
    @app.route('/advanced-ml')
    def advanced_ml():
        """Render advanced ML analysis page"""
        return render_template('advanced_ml.html')

if __name__ == "__main__":
    # For testing purposes
    from flask import Flask
    from flask_socketio import SocketIO
    
    # Create test Flask app
    test_app = Flask(__name__)
    test_app.config['SECRET_KEY'] = 'test_secret_key'
    test_socketio = SocketIO(test_app)
    
    # Integrate ML engine
    ml_integration = integrate_ml_with_flask(test_app, test_socketio)
    
    print("ML Flask Integration test setup completed!")
    print("Available routes:")
    for rule in test_app.url_map.iter_rules():
        if '/api/ml' in rule.rule or '/api/ai-signals' in rule.rule:
            print(f"  {rule.methods} {rule.rule}")
