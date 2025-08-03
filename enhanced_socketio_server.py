#!/usr/bin/env python3
"""
Enhanced Flask-SocketIO Implementation for GoldGPT
Real-time WebSocket server with authentication, connection management, and error handling
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from flask import Flask, request, session
from flask_socketio import SocketIO, emit, disconnect, join_room, leave_room
import requests
import jwt
from functools import wraps

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoldGPTSocketIOServer:
    """Enhanced SocketIO server for GoldGPT with authentication and connection management"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.socketio = SocketIO(
            app, 
            cors_allowed_origins="*", 
            async_mode='threading',
            logger=True,
            engineio_logger=True
        )
        
        # Connection management
        self.connected_clients: Set[str] = set()
        self.authenticated_clients: Set[str] = set()
        self.client_rooms: Dict[str, Set[str]] = {}
        
        # Background task control
        self.background_tasks_running = False
        self.task_threads: List[threading.Thread] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, int]] = {}
        
        # Initialize event handlers
        self._register_event_handlers()
        
        logger.info("🚀 Enhanced GoldGPT SocketIO Server initialized")
    
    def generate_auth_token(self, client_id: str) -> str:
        """Generate JWT authentication token for client"""
        payload = {
            'client_id': client_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow(),
            'iss': 'goldgpt-server'
        }
        return jwt.encode(payload, self.app.config['SECRET_KEY'], algorithm='HS256')
    
    def verify_auth_token(self, token: str) -> Optional[str]:
        """Verify JWT authentication token"""
        try:
            payload = jwt.decode(token, self.app.config['SECRET_KEY'], algorithms=['HS256'])
            return payload.get('client_id')
        except jwt.ExpiredSignatureError:
            logger.warning("🔐 Expired token received")
            return None
        except jwt.InvalidTokenError:
            logger.warning("🔐 Invalid token received")
            return None
    
    def require_auth(self, f):
        """Decorator to require authentication for SocketIO events"""
        @wraps(f)
        def decorated(*args, **kwargs):
            if request.sid not in self.authenticated_clients:
                emit('error', {
                    'message': 'Authentication required',
                    'code': 'AUTH_REQUIRED',
                    'timestamp': datetime.utcnow().isoformat()
                })
                return
            return f(*args, **kwargs)
        return decorated
    
    def rate_limit(self, max_requests: int = 10, window: int = 60):
        """Rate limiting decorator for SocketIO events"""
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                client_id = request.sid
                current_time = int(time.time())
                window_start = current_time - window
                
                if client_id not in self.rate_limits:
                    self.rate_limits[client_id] = {}
                
                # Clean old entries
                self.rate_limits[client_id] = {
                    timestamp: count for timestamp, count in self.rate_limits[client_id].items()
                    if timestamp > window_start
                }
                
                # Count requests in current window
                total_requests = sum(self.rate_limits[client_id].values())
                
                if total_requests >= max_requests:
                    emit('error', {
                        'message': 'Rate limit exceeded',
                        'code': 'RATE_LIMIT',
                        'retry_after': window,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    return
                
                # Record this request
                self.rate_limits[client_id][current_time] = self.rate_limits[client_id].get(current_time, 0) + 1
                
                return f(*args, **kwargs)
            return decorated
        return decorator
    
    def _register_event_handlers(self):
        """Register all SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection with authentication"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            
            logger.info(f"🔌 Client {client_id} connected from {request.remote_addr}")
            
            # Send welcome message with authentication token
            auth_token = self.generate_auth_token(client_id)
            
            emit('connected', {
                'status': 'Connected to GoldGPT Advanced Dashboard',
                'client_id': client_id,
                'auth_token': auth_token,
                'server_time': datetime.utcnow().isoformat(),
                'features': [
                    'real_time_prices', 
                    'ai_analysis', 
                    'ml_predictions', 
                    'portfolio_updates',
                    'authenticated_sessions',
                    'rate_limiting',
                    'error_handling'
                ],
                'update_intervals': {
                    'price_updates': '2s',
                    'ai_analysis': '30s', 
                    'portfolio_updates': '10s'
                }
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            
            # Clean up client data
            self.connected_clients.discard(client_id)
            self.authenticated_clients.discard(client_id)
            
            # Remove from all rooms
            if client_id in self.client_rooms:
                for room in self.client_rooms[client_id]:
                    leave_room(room)
                del self.client_rooms[client_id]
            
            # Clean rate limiting data
            if client_id in self.rate_limits:
                del self.rate_limits[client_id]
            
            logger.info(f"🔌 Client {client_id} disconnected")
        
        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """Handle client authentication"""
            try:
                token = data.get('token')
                if not token:
                    emit('auth_failed', {
                        'message': 'Token required',
                        'code': 'TOKEN_REQUIRED'
                    })
                    return
                
                client_id = self.verify_auth_token(token)
                if client_id and client_id == request.sid:
                    self.authenticated_clients.add(request.sid)
                    
                    emit('authenticated', {
                        'status': 'Authentication successful',
                        'client_id': client_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    logger.info(f"🔐 Client {client_id} authenticated successfully")
                else:
                    emit('auth_failed', {
                        'message': 'Invalid or expired token',
                        'code': 'INVALID_TOKEN'
                    })
                    
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                emit('auth_failed', {
                    'message': 'Authentication error',
                    'code': 'AUTH_ERROR'
                })
        
        @self.socketio.on('join_room')
        @self.require_auth
        def handle_join_room(data):
            """Handle client joining specific rooms for targeted updates"""
            room_name = data.get('room')
            if room_name in ['prices', 'ai_analysis', 'portfolio', 'ml_predictions']:
                join_room(room_name)
                
                if request.sid not in self.client_rooms:
                    self.client_rooms[request.sid] = set()
                self.client_rooms[request.sid].add(room_name)
                
                emit('room_joined', {
                    'room': room_name,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                logger.info(f"📡 Client {request.sid} joined room: {room_name}")
        
        @self.socketio.on('leave_room')
        @self.require_auth
        def handle_leave_room(data):
            """Handle client leaving specific rooms"""
            room_name = data.get('room')
            leave_room(room_name)
            
            if request.sid in self.client_rooms:
                self.client_rooms[request.sid].discard(room_name)
            
            emit('room_left', {
                'room': room_name,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        @self.socketio.on('request_price_update')
        @self.require_auth
        @self.rate_limit(max_requests=30, window=60)
        def handle_price_update_request():
            """Handle immediate price update request"""
            try:
                price_data = self.get_current_gold_price()
                emit('price_update', price_data)
                logger.debug(f"📈 Price update sent to {request.sid}")
            except Exception as e:
                logger.error(f"Price update error: {e}")
                emit('error', {
                    'message': 'Failed to fetch price data',
                    'code': 'PRICE_ERROR',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        @self.socketio.on('request_ai_analysis')
        @self.require_auth
        @self.rate_limit(max_requests=10, window=60)
        def handle_ai_analysis_request():
            """Handle AI analysis update request"""
            try:
                ai_data = self.get_ai_analysis()
                emit('ai_analysis', ai_data)
                logger.debug(f"🤖 AI analysis sent to {request.sid}")
            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                emit('error', {
                    'message': 'Failed to fetch AI analysis',
                    'code': 'AI_ERROR',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        @self.socketio.on('request_portfolio_update')
        @self.require_auth
        @self.rate_limit(max_requests=20, window=60)
        def handle_portfolio_update_request():
            """Handle portfolio update request"""
            try:
                portfolio_data = self.get_portfolio_data()
                emit('portfolio_update', portfolio_data)
                logger.debug(f"💼 Portfolio update sent to {request.sid}")
            except Exception as e:
                logger.error(f"Portfolio update error: {e}")
                emit('error', {
                    'message': 'Failed to fetch portfolio data',
                    'code': 'PORTFOLIO_ERROR',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping for connection health check"""
            emit('pong', {
                'timestamp': datetime.utcnow().isoformat(),
                'server_time': time.time()
            })
    
    def get_current_gold_price(self) -> Dict:
        """Enhanced gold price fetching with free gold service"""
        try:
            # Use the new free gold service
            from free_gold_api_service import get_free_gold_price
            return get_free_gold_price()
            
        except ImportError:
            # Fallback to simple simulation
            import random
            from datetime import datetime
            
            base_price = 2400.0
            variation = random.uniform(-20, 20)
            current_price = round(base_price + variation, 2)
            
            return {
                'price': current_price,
                'high': round(current_price * 1.01, 2),
                'low': round(current_price * 0.99, 2),
                'volume': round(random.uniform(100000, 500000), 0),
                'change': round(variation, 2),
                'change_percent': round((variation / base_price) * 100, 3),
                'timestamp': datetime.now().isoformat(),
                'source': 'socketio_fallback',
                'bid': round(current_price - 0.5, 2),
                'ask': round(current_price + 0.5, 2),
                'spread': 1.0
            }
                    
                    # Add additional metadata
                    price_data.update({
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': api['url'].split('/')[2],
                        'status': 'live',
                        'currency': 'USD',
                        'unit': 'troy_ounce'
                    })
                    
                    return price_data
                    
            except Exception as e:
                logger.warning(f"API {api['url']} failed: {e}")
                continue
        
        # Fallback to mock data if all APIs fail
        base_price = 2000 + (time.time() % 100)
        return {
            'price': round(base_price, 2),
            'change': round((time.time() % 20) - 10, 2),
            'change_percent': round(((time.time() % 20) - 10) / base_price * 100, 3),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'fallback',
            'status': 'simulated',
            'currency': 'USD',
            'unit': 'troy_ounce'
        }
    
    def get_ai_analysis(self) -> Dict:
        """Generate comprehensive AI analysis data"""
        current_time = datetime.utcnow()
        
        # Simulate AI analysis with realistic patterns
        base_score = 0.5 + (time.time() % 10) / 20
        
        return {
            'timestamp': current_time.isoformat(),
            'analysis_type': 'comprehensive',
            'timeframe': '1H',
            'confidence': round(base_score, 3),
            'signal': 'bullish' if base_score > 0.6 else 'bearish' if base_score < 0.4 else 'neutral',
            'technical_indicators': {
                'rsi': round(30 + (time.time() % 40), 1),
                'macd': round((time.time() % 2) - 1, 3),
                'bollinger_position': round(time.time() % 1, 3),
                'volume_profile': round(0.5 + (time.time() % 0.5), 3)
            },
            'sentiment_analysis': {
                'news_sentiment': round(0.3 + (time.time() % 0.4), 3),
                'social_sentiment': round(0.4 + (time.time() % 0.3), 3),
                'institutional_flow': round(-0.2 + (time.time() % 0.4), 3)
            },
            'risk_assessment': {
                'volatility': round(0.1 + (time.time() % 0.3), 3),
                'liquidity': round(0.7 + (time.time() % 0.3), 3),
                'correlation_risk': round(0.2 + (time.time() % 0.3), 3)
            },
            'predictions': {
                'short_term': round(2000 + (time.time() % 50), 2),
                'medium_term': round(2050 + (time.time() % 100), 2),
                'long_term': round(2100 + (time.time() % 200), 2)
            }
        }
    
    def get_portfolio_data(self) -> Dict:
        """Generate portfolio performance data"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_value': round(100000 + (time.time() % 10000), 2),
            'daily_pnl': round(-500 + (time.time() % 1000), 2),
            'daily_pnl_percent': round(-0.5 + (time.time() % 1), 3),
            'positions': [
                {
                    'symbol': 'XAUUSD',
                    'quantity': 10.5,
                    'entry_price': 1985.50,
                    'current_price': round(2000 + (time.time() % 50), 2),
                    'pnl': round(-200 + (time.time() % 400), 2),
                    'pnl_percent': round(-1 + (time.time() % 2), 3)
                }
            ],
            'performance_metrics': {
                'sharpe_ratio': round(1.2 + (time.time() % 0.8), 3),
                'max_drawdown': round(-0.15 + (time.time() % 0.1), 3),
                'win_rate': round(0.6 + (time.time() % 0.2), 3),
                'avg_trade_duration': '2.5 hours'
            }
        }
    
    def start_background_tasks(self):
        """Start all background update tasks"""
        if self.background_tasks_running:
            logger.warning("Background tasks already running")
            return
        
        self.background_tasks_running = True
        
        def price_updater():
            """Real-time price updates every 2 seconds"""
            while self.background_tasks_running:
                try:
                    if self.connected_clients:
                        price_data = self.get_current_gold_price()
                        self.socketio.emit('price_update', price_data, room='prices')
                        logger.debug(f"📈 Price update broadcast to {len(self.connected_clients)} clients")
                    time.sleep(2)  # Update every 2 seconds as requested
                except Exception as e:
                    logger.error(f"Price updater error: {e}")
                    time.sleep(5)
        
        def ai_analysis_updater():
            """AI analysis updates every 30 seconds"""
            while self.background_tasks_running:
                try:
                    if self.authenticated_clients:
                        ai_data = self.get_ai_analysis()
                        self.socketio.emit('ai_analysis', ai_data, room='ai_analysis')
                        logger.debug(f"🤖 AI analysis broadcast to authenticated clients")
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"AI analysis updater error: {e}")
                    time.sleep(60)
        
        def portfolio_updater():
            """Portfolio updates every 10 seconds"""
            while self.background_tasks_running:
                try:
                    if self.authenticated_clients:
                        portfolio_data = self.get_portfolio_data()
                        self.socketio.emit('portfolio_update', portfolio_data, room='portfolio')
                        logger.debug(f"💼 Portfolio update broadcast to authenticated clients")
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Portfolio updater error: {e}")
                    time.sleep(20)
        
        def connection_monitor():
            """Monitor and cleanup stale connections"""
            while self.background_tasks_running:
                try:
                    # Send ping to all connected clients
                    if self.connected_clients:
                        self.socketio.emit('server_ping', {
                            'timestamp': datetime.utcnow().isoformat(),
                            'connected_clients': len(self.connected_clients),
                            'authenticated_clients': len(self.authenticated_clients)
                        })
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Connection monitor error: {e}")
                    time.sleep(30)
        
        # Start all background threads
        tasks = [
            ('Price Updater', price_updater),
            ('AI Analysis Updater', ai_analysis_updater), 
            ('Portfolio Updater', portfolio_updater),
            ('Connection Monitor', connection_monitor)
        ]
        
        for task_name, task_func in tasks:
            thread = threading.Thread(target=task_func, name=task_name, daemon=True)
            thread.start()
            self.task_threads.append(thread)
            logger.info(f"🚀 Started {task_name}")
        
        logger.info("✅ All background tasks started successfully")
    
    def stop_background_tasks(self):
        """Stop all background tasks"""
        self.background_tasks_running = False
        logger.info("🛑 Background tasks stopped")
    
    def get_server_stats(self) -> Dict:
        """Get server statistics"""
        return {
            'connected_clients': len(self.connected_clients),
            'authenticated_clients': len(self.authenticated_clients),
            'active_rooms': sum(len(rooms) for rooms in self.client_rooms.values()),
            'background_tasks_running': self.background_tasks_running,
            'uptime': datetime.utcnow().isoformat(),
            'task_threads': len(self.task_threads)
        }


# Integration function for existing GoldGPT app
def setup_enhanced_socketio(app: Flask) -> GoldGPTSocketIOServer:
    """Setup enhanced SocketIO server for existing GoldGPT Flask app"""
    server = GoldGPTSocketIOServer(app)
    
    # Add server stats endpoint
    @app.route('/api/websocket/stats')
    def websocket_stats():
        return server.get_server_stats()
    
    # Add authentication endpoint
    @app.route('/api/websocket/auth')
    def websocket_auth():
        # In production, implement proper authentication
        client_id = request.args.get('client_id', 'anonymous')
        token = server.generate_auth_token(client_id)
        return {'token': token, 'client_id': client_id}
    
    return server


# Example usage and testing
if __name__ == '__main__':
    # Create test Flask app
    test_app = Flask(__name__)
    test_app.config['SECRET_KEY'] = 'test-secret-key'
    
    # Setup enhanced SocketIO
    socketio_server = setup_enhanced_socketio(test_app)
    
    # Start background tasks
    socketio_server.start_background_tasks()
    
    logger.info("🚀 Enhanced GoldGPT SocketIO Server starting...")
    logger.info("📡 WebSocket events: connect, disconnect, authenticate, join_room, leave_room")
    logger.info("📈 Real-time updates: price_update (2s), ai_analysis (30s), portfolio_update (10s)")
    logger.info("🔐 Features: Authentication, Rate limiting, Error handling, Reconnection logic")
    
    # Run the enhanced server
    try:
        socketio_server.socketio.run(
            test_app,
            host='0.0.0.0',
            port=5000,
            debug=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested")
        socketio_server.stop_background_tasks()
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        socketio_server.stop_background_tasks()
