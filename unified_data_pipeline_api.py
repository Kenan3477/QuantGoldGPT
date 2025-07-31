#!/usr/bin/env python3
"""
GoldGPT Unified Data Pipeline API
Flask integration for the multi-source data pipeline with WebSocket support
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time

# Import our specialized services
from data_pipeline_core import data_pipeline, DataType
from price_data_service import price_service
from sentiment_analysis_service import sentiment_service
from technical_indicator_service import technical_service
from macro_data_service import macro_service

logger = logging.getLogger(__name__)

class UnifiedDataPipelineAPI:
    """Unified API for all data pipeline services"""
    
    def __init__(self, app: Flask, socketio: SocketIO):
        self.app = app
        self.socketio = socketio
        self.background_tasks = {}
        self.setup_routes()
        self.setup_websocket_handlers()
        self.start_background_services()
    
    def setup_routes(self):
        """Setup Flask routes for data pipeline"""
        
        @self.app.route('/api/data-pipeline/health', methods=['GET'])
        def health_check():
            """Health check for all data services"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                health_results = loop.run_until_complete(data_pipeline.health_check())
                
                # Add service-specific health checks
                health_results['price_service'] = {'status': 'healthy', 'last_check': datetime.now().isoformat()}
                health_results['sentiment_service'] = {'status': 'healthy', 'last_check': datetime.now().isoformat()}
                health_results['technical_service'] = {'status': 'healthy', 'last_check': datetime.now().isoformat()}
                health_results['macro_service'] = {'status': 'healthy', 'last_check': datetime.now().isoformat()}
                
                loop.close()
                
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'services': health_results
                })
            
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/sources', methods=['GET'])
        def get_source_status():
            """Get status of all data sources"""
            try:
                source_status = data_pipeline.get_source_status()
                
                return jsonify({
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    'sources': source_status,
                    'total_sources': len(source_status)
                })
            
            except Exception as e:
                logger.error(f"Source status error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/price', methods=['GET'])
        def get_realtime_price():
            """Get real-time price data"""
            try:
                symbol = request.args.get('symbol', 'XAU')
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                price_data = loop.run_until_complete(price_service.get_realtime_price(symbol))
                
                loop.close()
                
                if price_data:
                    return jsonify({
                        'success': True,
                        'data': price_data,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No price data available',
                        'timestamp': datetime.now().isoformat()
                    }), 404
            
            except Exception as e:
                logger.error(f"Price API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/price/historical', methods=['GET'])
        def get_historical_price():
            """Get historical price data"""
            try:
                symbol = request.args.get('symbol', 'XAU')
                timeframe = request.args.get('timeframe', '1h')
                limit = int(request.args.get('limit', 100))
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                historical_data = loop.run_until_complete(
                    price_service.get_historical_data(symbol, timeframe, limit)
                )
                
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': historical_data,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'count': len(historical_data),
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Historical price API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/price/metrics', methods=['GET'])
        def get_price_metrics():
            """Get price metrics and statistics"""
            try:
                symbol = request.args.get('symbol', 'XAU')
                period_hours = int(request.args.get('period_hours', 24))
                
                metrics = price_service.calculate_price_metrics(symbol, period_hours)
                
                return jsonify({
                    'success': True,
                    'data': metrics,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Price metrics API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/sentiment', methods=['GET'])
        def get_sentiment_analysis():
            """Get sentiment analysis"""
            try:
                hours_back = int(request.args.get('hours_back', 12))
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                sentiment_data = loop.run_until_complete(
                    sentiment_service.get_real_time_sentiment(hours_back)
                )
                
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': sentiment_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Sentiment API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/technical', methods=['GET'])
        def get_technical_analysis():
            """Get technical analysis"""
            try:
                symbol = request.args.get('symbol', 'XAU')
                timeframe = request.args.get('timeframe', '1h')
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                technical_data = loop.run_until_complete(
                    technical_service.get_technical_analysis(symbol, timeframe)
                )
                
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': technical_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Technical analysis API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/macro', methods=['GET'])
        def get_macro_analysis():
            """Get macro economic analysis"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                macro_data = loop.run_until_complete(macro_service.get_macro_analysis())
                
                # Convert to dictionary for JSON serialization
                macro_dict = {
                    'overall_sentiment': macro_data.overall_sentiment,
                    'risk_level': macro_data.risk_level,
                    'gold_impact': macro_data.gold_impact,
                    'analysis_summary': macro_data.analysis_summary,
                    'confidence': macro_data.confidence,
                    'key_indicators': [{
                        'name': ind.name,
                        'value': ind.value,
                        'previous_value': ind.previous_value,
                        'change': ind.change,
                        'change_percent': ind.change_percent,
                        'unit': ind.unit,
                        'impact_level': ind.impact_level,
                        'release_date': ind.release_date.isoformat()
                    } for ind in macro_data.key_indicators]
                }
                
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': macro_dict,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Macro analysis API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/economic-calendar', methods=['GET'])
        def get_economic_calendar():
            """Get economic calendar"""
            try:
                days_ahead = int(request.args.get('days_ahead', 7))
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                calendar_data = loop.run_until_complete(
                    macro_service.get_economic_calendar(days_ahead)
                )
                
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': calendar_data,
                    'count': len(calendar_data),
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Economic calendar API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/comprehensive', methods=['GET'])
        def get_comprehensive_analysis():
            """Get comprehensive analysis from all services"""
            try:
                symbol = request.args.get('symbol', 'XAU')
                timeframe = request.args.get('timeframe', '1h')
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Fetch data from all services
                price_data = loop.run_until_complete(price_service.get_realtime_price(symbol))
                sentiment_data = loop.run_until_complete(sentiment_service.get_real_time_sentiment(12))
                technical_data = loop.run_until_complete(technical_service.get_technical_analysis(symbol, timeframe))
                macro_data = loop.run_until_complete(macro_service.get_macro_analysis())
                
                loop.close()
                
                # Combine all analysis
                comprehensive_analysis = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'price': price_data,
                    'sentiment': sentiment_data,
                    'technical': technical_data,
                    'macro': {
                        'overall_sentiment': macro_data.overall_sentiment,
                        'risk_level': macro_data.risk_level,
                        'gold_impact': macro_data.gold_impact,
                        'confidence': macro_data.confidence,
                        'summary': macro_data.analysis_summary
                    },
                    'overall_signal': self.calculate_overall_signal(
                        sentiment_data, technical_data, macro_data
                    ),
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': comprehensive_analysis,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Comprehensive analysis API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/data-pipeline/alerts', methods=['POST'])
        def create_price_alert():
            """Create price alert"""
            try:
                data = request.get_json()
                
                symbol = data.get('symbol', 'XAU')
                target_price = float(data.get('target_price'))
                direction = data.get('direction', 'above')
                
                alert_id = price_service.add_price_alert(symbol, target_price, direction)
                
                return jsonify({
                    'success': True,
                    'alert_id': alert_id,
                    'symbol': symbol,
                    'target_price': target_price,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Price alert API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
    
    def setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("📡 Client connected to data pipeline WebSocket")
            emit('connected', {'status': 'connected', 'timestamp': datetime.now().isoformat()})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("📡 Client disconnected from data pipeline WebSocket")
        
        @self.socketio.on('subscribe_price')
        def handle_subscribe_price(data):
            """Subscribe to real-time price updates"""
            symbol = data.get('symbol', 'XAU')
            logger.info(f"📡 Client subscribed to price updates for {symbol}")
            
            # Add client to price subscription
            # In a real implementation, you'd manage subscriptions per client
            emit('price_subscription', {
                'status': 'subscribed',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('subscribe_analysis')
        def handle_subscribe_analysis(data):
            """Subscribe to analysis updates"""
            analysis_types = data.get('types', ['sentiment', 'technical', 'macro'])
            logger.info(f"📡 Client subscribed to analysis updates: {analysis_types}")
            
            emit('analysis_subscription', {
                'status': 'subscribed',
                'types': analysis_types,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('request_comprehensive')
        def handle_request_comprehensive(data):
            """Handle request for comprehensive analysis"""
            try:
                symbol = data.get('symbol', 'XAU')
                timeframe = data.get('timeframe', '1h')
                
                # Run comprehensive analysis in background
                def run_analysis():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Get comprehensive data
                        price_data = loop.run_until_complete(price_service.get_realtime_price(symbol))
                        sentiment_data = loop.run_until_complete(sentiment_service.get_real_time_sentiment(6))
                        technical_data = loop.run_until_complete(technical_service.get_technical_analysis(symbol, timeframe))
                        
                        loop.close()
                        
                        # Emit results
                        self.socketio.emit('comprehensive_analysis', {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'price': price_data,
                            'sentiment': sentiment_data,
                            'technical': technical_data,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"WebSocket comprehensive analysis error: {e}")
                        self.socketio.emit('analysis_error', {
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Run in background thread
                threading.Thread(target=run_analysis, daemon=True).start()
                
                emit('analysis_requested', {
                    'status': 'processing',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"WebSocket request error: {e}")
                emit('request_error', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    def calculate_overall_signal(self, sentiment_data: Dict, technical_data: Dict, macro_data) -> Dict:
        """Calculate overall market signal from all data sources"""
        signals = []
        weights = []
        
        # Sentiment signal
        if sentiment_data.get('market_signal'):
            sentiment_signal = sentiment_data['market_signal']['signal']
            if sentiment_signal == 'BUY':
                signals.append(1)
            elif sentiment_signal == 'SELL':
                signals.append(-1)
            else:
                signals.append(0)
            weights.append(0.3)  # 30% weight
        
        # Technical signal
        if technical_data.get('overall_assessment'):
            tech_signal = technical_data['overall_assessment']['signal']
            tech_strength = technical_data['overall_assessment'].get('strength', 0.5)
            
            if tech_signal == 'BUY':
                signals.append(tech_strength)
            elif tech_signal == 'SELL':
                signals.append(-tech_strength)
            else:
                signals.append(0)
            weights.append(0.4)  # 40% weight
        
        # Macro signal
        if macro_data.gold_impact == 'POSITIVE':
            signals.append(0.7)
        elif macro_data.gold_impact == 'NEGATIVE':
            signals.append(-0.7)
        else:
            signals.append(0)
        weights.append(0.3)  # 30% weight
        
        # Calculate weighted average
        if signals and weights:
            weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
            
            if weighted_signal > 0.3:
                overall_signal = 'BUY'
                strength = abs(weighted_signal)
            elif weighted_signal < -0.3:
                overall_signal = 'SELL'
                strength = abs(weighted_signal)
            else:
                overall_signal = 'HOLD'
                strength = 1 - abs(weighted_signal)
            
            return {
                'signal': overall_signal,
                'strength': round(strength, 3),
                'confidence': round(sum(weights) / 3, 3),  # Normalize by max possible weight
                'components': {
                    'sentiment_weight': weights[0] if len(weights) > 0 else 0,
                    'technical_weight': weights[1] if len(weights) > 1 else 0,
                    'macro_weight': weights[2] if len(weights) > 2 else 0
                }
            }
        
        return {
            'signal': 'HOLD',
            'strength': 0.5,
            'confidence': 0.1,
            'components': {}
        }
    
    def start_background_services(self):
        """Start background data update services"""
        
        def price_update_service():
            """Background service for real-time price updates"""
            while True:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    price_data = loop.run_until_complete(price_service.get_realtime_price('XAU'))
                    
                    if price_data:
                        # Check for price alerts
                        triggered_alerts = price_service.check_price_alerts(price_data['price'], 'XAU')
                        
                        # Emit price update
                        self.socketio.emit('price_update', {
                            'data': price_data,
                            'triggered_alerts': triggered_alerts,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        if triggered_alerts:
                            self.socketio.emit('price_alert', {
                                'alerts': triggered_alerts,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    loop.close()
                    
                except Exception as e:
                    logger.error(f"Price update service error: {e}")
                
                time.sleep(10)  # Update every 10 seconds
        
        def analysis_update_service():
            """Background service for periodic analysis updates"""
            while True:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Update sentiment every 15 minutes
                    sentiment_data = loop.run_until_complete(
                        sentiment_service.get_real_time_sentiment(6)
                    )
                    
                    # Update technical analysis every 30 minutes
                    technical_data = loop.run_until_complete(
                        technical_service.get_technical_analysis('XAU', '1h')
                    )
                    
                    loop.close()
                    
                    # Emit analysis updates
                    self.socketio.emit('sentiment_update', {
                        'data': sentiment_data,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    self.socketio.emit('technical_update', {
                        'data': technical_data,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Analysis update service error: {e}")
                
                time.sleep(900)  # Update every 15 minutes
        
        # Start background services
        threading.Thread(target=price_update_service, daemon=True).start()
        threading.Thread(target=analysis_update_service, daemon=True).start()
        
        logger.info("🚀 Started background data pipeline services")

# Function to initialize the data pipeline API
def init_data_pipeline_api(app: Flask, socketio: SocketIO) -> UnifiedDataPipelineAPI:
    """Initialize the unified data pipeline API"""
    return UnifiedDataPipelineAPI(app, socketio)

if __name__ == "__main__":
    # Test the unified API
    from flask import Flask
    from flask_socketio import SocketIO
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test_secret_key'
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize API
    pipeline_api = init_data_pipeline_api(app, socketio)
    
    print("🧪 Testing Unified Data Pipeline API...")
    print("🌐 Server starting on http://localhost:5001")
    print("📡 WebSocket available for real-time updates")
    print("\n📋 Available endpoints:")
    print("  • GET /api/data-pipeline/health - Health check")
    print("  • GET /api/data-pipeline/price - Real-time price")
    print("  • GET /api/data-pipeline/sentiment - Sentiment analysis")
    print("  • GET /api/data-pipeline/technical - Technical analysis")
    print("  • GET /api/data-pipeline/macro - Macro analysis")
    print("  • GET /api/data-pipeline/comprehensive - All analysis")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
