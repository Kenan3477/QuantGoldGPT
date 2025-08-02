#!/usr/bin/env python3
"""
Enhanced SocketIO Integration for GoldGPT
Integrates with existing app.py to provide enhanced WebSocket functionality
"""

import os
import sys
import logging
from enhanced_socketio_server import GoldGPTSocketIOServer, setup_enhanced_socketio

# Setup logging
logger = logging.getLogger(__name__)

def integrate_enhanced_socketio(app, existing_socketio):
    """
    Integrate enhanced SocketIO functionality with existing GoldGPT app
    
    Args:
        app: Flask application instance
        existing_socketio: Existing SocketIO instance
    
    Returns:
        Enhanced SocketIO server instance
    """
    
    logger.info("🔄 Integrating enhanced SocketIO functionality...")
    
    try:
        # Create enhanced server instance (reuses existing socketio)
        enhanced_server = GoldGPTSocketIOServer(app)
        enhanced_server.socketio = existing_socketio  # Use existing SocketIO instance
        
        # Re-register enhanced event handlers (will override existing ones)
        enhanced_server._register_event_handlers()
        
        # Add enhanced API endpoints
        @app.route('/api/websocket/stats')
        def websocket_stats():
            """Get WebSocket server statistics"""
            return enhanced_server.get_server_stats()
        
        @app.route('/api/websocket/auth')
        def websocket_auth():
            """Get authentication token for WebSocket connection"""
            try:
                import jwt
                client_id = f"client_{os.urandom(8).hex()}"
                token = enhanced_server.generate_auth_token(client_id)
                return {
                    'success': True,
                    'token': token, 
                    'client_id': client_id,
                    'expires_in': 86400  # 24 hours
                }
            except Exception as e:
                logger.error(f"Auth token generation error: {e}")
                return {
                    'success': False,
                    'error': 'Failed to generate auth token'
                }, 500
        
        @app.route('/api/websocket/test')
        def websocket_test():
            """Test WebSocket broadcasting"""
            try:
                test_data = {
                    'message': 'Test broadcast from server',
                    'timestamp': datetime.utcnow().isoformat(),
                    'server_stats': enhanced_server.get_server_stats()
                }
                
                # Broadcast test message
                existing_socketio.emit('test_broadcast', test_data)
                
                return {
                    'success': True,
                    'message': 'Test broadcast sent',
                    'data': test_data
                }
            except Exception as e:
                logger.error(f"Test broadcast error: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }, 500
        
        # Enhanced background task starter
        def start_enhanced_background_tasks():
            """Start enhanced background tasks with existing system"""
            try:
                enhanced_server.start_background_tasks()
                logger.info("✅ Enhanced background tasks started successfully")
            except Exception as e:
                logger.error(f"❌ Failed to start enhanced background tasks: {e}")
                # Fallback to original background tasks
                logger.info("🔄 Falling back to original background task system...")
        
        # Replace the background task starter
        enhanced_server.start_enhanced_background_tasks = start_enhanced_background_tasks
        
        logger.info("✅ Enhanced SocketIO integration completed successfully")
        return enhanced_server
        
    except ImportError as e:
        logger.warning(f"⚠️ Enhanced SocketIO dependencies missing: {e}")
        logger.info("🔄 Continuing with basic SocketIO functionality...")
        return None
        
    except Exception as e:
        logger.error(f"❌ Enhanced SocketIO integration failed: {e}")
        logger.info("🔄 Continuing with basic SocketIO functionality...")
        return None


def add_enhanced_websocket_routes(app):
    """Add enhanced WebSocket-related routes to existing app"""
    
    @app.route('/websocket-test')
    def websocket_test_page():
        """Test page for WebSocket functionality"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>GoldGPT WebSocket Test</title>
            <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background: #d4edda; color: #155724; }
                .disconnected { background: #f8d7da; color: #721c24; }
                .message { background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 3px solid #007bff; }
                button { padding: 10px 15px; margin: 5px; }
                #messages { height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>🚀 GoldGPT WebSocket Test</h1>
            
            <div id="status" class="status disconnected">Disconnected</div>
            
            <div>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="authenticate()">Authenticate</button>
                <button onclick="requestPriceUpdate()">Request Price Update</button>
                <button onclick="requestAIAnalysis()">Request AI Analysis</button>
                <button onclick="requestPortfolioUpdate()">Request Portfolio Update</button>
            </div>
            
            <h3>Messages:</h3>
            <div id="messages"></div>
            
            <script>
                let socket = null;
                let authToken = null;
                
                function addMessage(message, type = 'info') {
                    const messages = document.getElementById('messages');
                    const div = document.createElement('div');
                    div.className = 'message';
                    div.innerHTML = `<strong>[${new Date().toLocaleTimeString()}]</strong> ${message}`;
                    messages.appendChild(div);
                    messages.scrollTop = messages.scrollHeight;
                }
                
                function updateStatus(status, message) {
                    const statusEl = document.getElementById('status');
                    statusEl.className = `status ${status}`;
                    statusEl.textContent = message;
                }
                
                function connect() {
                    if (socket) {
                        socket.disconnect();
                    }
                    
                    socket = io();
                    
                    socket.on('connected', (data) => {
                        updateStatus('connected', 'Connected');
                        authToken = data.auth_token;
                        addMessage(`Connected! Features: ${data.features.join(', ')}`);
                        addMessage(`Auth token received: ${authToken.substring(0, 20)}...`);
                    });
                    
                    socket.on('disconnect', (reason) => {
                        updateStatus('disconnected', 'Disconnected');
                        addMessage(`Disconnected: ${reason}`);
                    });
                    
                    socket.on('authenticated', (data) => {
                        addMessage('✅ Authentication successful!');
                    });
                    
                    socket.on('auth_failed', (data) => {
                        addMessage(`❌ Authentication failed: ${data.message}`);
                    });
                    
                    socket.on('price_update', (data) => {
                        addMessage(`📈 Price Update: $${data.price} (${data.change >= 0 ? '+' : ''}${data.change})`);
                    });
                    
                    socket.on('ai_analysis', (data) => {
                        addMessage(`🤖 AI Analysis: ${data.signal} (confidence: ${(data.confidence * 100).toFixed(1)}%)`);
                    });
                    
                    socket.on('portfolio_update', (data) => {
                        addMessage(`💼 Portfolio: $${data.total_value.toLocaleString()} (P&L: ${data.daily_pnl >= 0 ? '+' : ''}$${data.daily_pnl})`);
                    });
                    
                    socket.on('error', (data) => {
                        addMessage(`❌ Error [${data.code}]: ${data.message}`);
                    });
                    
                    socket.on('test_broadcast', (data) => {
                        addMessage(`📡 Test Broadcast: ${data.message}`);
                    });
                    
                    socket.on('server_ping', (data) => {
                        addMessage(`🏓 Server ping - ${data.connected_clients} clients connected`);
                    });
                }
                
                function disconnect() {
                    if (socket) {
                        socket.disconnect();
                        socket = null;
                        authToken = null;
                    }
                }
                
                function authenticate() {
                    if (!socket || !authToken) {
                        addMessage('❌ Not connected or no auth token');
                        return;
                    }
                    
                    socket.emit('authenticate', { token: authToken });
                }
                
                function requestPriceUpdate() {
                    if (!socket) {
                        addMessage('❌ Not connected');
                        return;
                    }
                    socket.emit('request_price_update');
                }
                
                function requestAIAnalysis() {
                    if (!socket) {
                        addMessage('❌ Not connected');
                        return;
                    }
                    socket.emit('request_ai_analysis');
                }
                
                function requestPortfolioUpdate() {
                    if (!socket) {
                        addMessage('❌ Not connected');
                        return;
                    }
                    socket.emit('request_portfolio_update');
                }
                
                // Auto-connect on page load
                window.onload = function() {
                    connect();
                };
            </script>
        </body>
        </html>
        '''
    
    @app.route('/websocket-docs')
    def websocket_docs():
        """Documentation for WebSocket API"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>GoldGPT WebSocket API Documentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
                .event { background: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; }
                code { background: #f1f1f1; padding: 2px 5px; border-radius: 3px; }
                pre { background: #f8f9fa; padding: 15px; overflow-x: auto; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🚀 GoldGPT WebSocket API Documentation</h1>
            
            <h2>Connection</h2>
            <div class="endpoint">
                <h3>Connect to WebSocket</h3>
                <p><code>ws://localhost:5000/socket.io/</code></p>
                <p>Establishes WebSocket connection with Socket.IO protocol.</p>
            </div>
            
            <h2>Events</h2>
            
            <div class="event">
                <h3>📡 connected</h3>
                <p><strong>Direction:</strong> Server → Client</p>
                <p><strong>Description:</strong> Sent when client successfully connects</p>
                <pre>{
  "status": "Connected to GoldGPT Advanced Dashboard",
  "client_id": "unique_client_id",
  "auth_token": "jwt_token",
  "features": ["real_time_prices", "ai_analysis", "portfolio_updates"],
  "update_intervals": {
    "price_updates": "2s",
    "ai_analysis": "30s",
    "portfolio_updates": "10s"
  }
}</pre>
            </div>
            
            <div class="event">
                <h3>🔐 authenticate</h3>
                <p><strong>Direction:</strong> Client → Server</p>
                <p><strong>Description:</strong> Authenticate using JWT token</p>
                <pre>{ "token": "jwt_token_here" }</pre>
            </div>
            
            <div class="event">
                <h3>📈 price_update</h3>
                <p><strong>Direction:</strong> Server → Client (every 2 seconds)</p>
                <p><strong>Description:</strong> Real-time gold price updates</p>
                <pre>{
  "price": 2000.50,
  "change": +5.25,
  "change_percent": 0.26,
  "timestamp": "2025-08-01T12:00:00Z",
  "source": "api.metals.live",
  "status": "live"
}</pre>
            </div>
            
            <div class="event">
                <h3>🤖 ai_analysis</h3>
                <p><strong>Direction:</strong> Server → Client (every 30 seconds)</p>
                <p><strong>Description:</strong> AI trading analysis updates</p>
                <pre>{
  "timestamp": "2025-08-01T12:00:00Z",
  "signal": "bullish",
  "confidence": 0.75,
  "technical_indicators": {
    "rsi": 65.5,
    "macd": 0.25,
    "bollinger_position": 0.8
  },
  "predictions": {
    "short_term": 2005.00,
    "medium_term": 2050.00,
    "long_term": 2100.00
  }
}</pre>
            </div>
            
            <div class="event">
                <h3>💼 portfolio_update</h3>
                <p><strong>Direction:</strong> Server → Client (every 10 seconds)</p>
                <p><strong>Description:</strong> Portfolio performance updates</p>
                <pre>{
  "total_value": 105000.00,
  "daily_pnl": 2500.00,
  "daily_pnl_percent": 2.4,
  "positions": [{
    "symbol": "XAUUSD",
    "quantity": 10.5,
    "current_price": 2000.50,
    "pnl": 525.25
  }]
}</pre>
            </div>
            
            <h2>Client Requests</h2>
            
            <div class="event">
                <h3>📈 request_price_update</h3>
                <p><strong>Rate Limit:</strong> 30 requests/minute</p>
                <p><strong>Description:</strong> Request immediate price update</p>
            </div>
            
            <div class="event">
                <h3>🤖 request_ai_analysis</h3>
                <p><strong>Rate Limit:</strong> 10 requests/minute</p>
                <p><strong>Description:</strong> Request immediate AI analysis</p>
            </div>
            
            <div class="event">
                <h3>💼 request_portfolio_update</h3>
                <p><strong>Rate Limit:</strong> 20 requests/minute</p>
                <p><strong>Description:</strong> Request immediate portfolio update</p>
            </div>
            
            <h2>Room Management</h2>
            
            <div class="event">
                <h3>📡 join_room</h3>
                <p><strong>Description:</strong> Join specific update rooms</p>
                <p><strong>Available rooms:</strong> prices, ai_analysis, portfolio, ml_predictions</p>
                <pre>{ "room": "prices" }</pre>
            </div>
            
            <h2>Error Handling</h2>
            
            <div class="event">
                <h3>❌ error</h3>
                <p><strong>Direction:</strong> Server → Client</p>
                <p><strong>Description:</strong> Error notifications</p>
                <pre>{
  "message": "Rate limit exceeded",
  "code": "RATE_LIMIT",
  "timestamp": "2025-08-01T12:00:00Z",
  "retry_after": 60
}</pre>
            </div>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <h3>GET /api/websocket/stats</h3>
                <p>Get WebSocket server statistics</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/websocket/auth</h3>
                <p>Get authentication token for WebSocket connection</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/websocket/test</h3>
                <p>Send test broadcast to all connected clients</p>
            </div>
            
            <h2>Integration Examples</h2>
            
            <h3>JavaScript Client</h3>
            <pre>const socket = io();

socket.on('connected', (data) => {
  console.log('Connected:', data);
  // Authenticate
  socket.emit('authenticate', { token: data.auth_token });
});

socket.on('authenticated', () => {
  // Join rooms for specific updates
  socket.emit('join_room', { room: 'prices' });
  socket.emit('join_room', { room: 'ai_analysis' });
});

socket.on('price_update', (data) => {
  console.log('Price:', data.price);
  // Update UI
});</pre>
            
            <p><a href="/websocket-test">🧪 Try the WebSocket Test Page</a></p>
        </body>
        </html>
        '''


# Fallback functions for compatibility
def enhanced_get_current_gold_price():
    """Enhanced gold price function with multiple API fallbacks"""
    import requests
    import time
    from datetime import datetime
    
    apis = [
        {
            'url': 'https://api.metals.live/v1/spot/gold',
            'parser': lambda r: {
                'price': r.json().get('price', 0),
                'change': r.json().get('change', 0),
                'change_percent': r.json().get('change_percent', 0)
            }
        }
    ]
    
    for api in apis:
        try:
            response = requests.get(api['url'], timeout=5)
            if response.status_code == 200:
                price_data = api['parser'](response)
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
    
    # Fallback
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


if __name__ == '__main__':
    print("🚀 Enhanced SocketIO Integration Module")
    print("📝 This module provides enhanced WebSocket functionality for GoldGPT")
    print("🔗 Import this module in your main app.py to enable enhanced features")
    print("📖 Visit /websocket-docs for API documentation")
    print("🧪 Visit /websocket-test for testing")
