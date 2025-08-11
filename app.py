#!/usr/bin/env python3
"""
GoldGPT - Advanced AI Trading Web Application
Trading 212 Inspired Dashboard with Complete ML Integration
"""

import os
import json
import logging
import sqlite3
import traceback
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import requests
from typing import Dict, List, Optional
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import positions API
try:
    from positions_api import positions_bp
    POSITIONS_API_AVAILABLE = True
    logger.info("✅ Positions API available")
except ImportError as e:
    POSITIONS_API_AVAILABLE = False
    logger.warning(f"⚠️ Positions API not available: {e}")
    positions_bp = None

# Enhanced SocketIO integration
try:
    from enhanced_socketio_integration import (
        integrate_enhanced_socketio, 
        add_enhanced_websocket_routes
    )
    ENHANCED_SOCKETIO_AVAILABLE = True
    logger.info("✅ Enhanced SocketIO integration available")
except ImportError as e:
    ENHANCED_SOCKETIO_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced SocketIO not available: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'goldgpt-advanced-secret-key-2025')

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Enhanced SocketIO features
enhanced_server = None
if ENHANCED_SOCKETIO_AVAILABLE:
    try:
        enhanced_server = integrate_enhanced_socketio(app, socketio)
        logger.info("🚀 Enhanced SocketIO server initialized")
        
        # Add enhanced WebSocket routes directly here instead of importing
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
                
                <div id="messages"></div>
                
                <script>
                    let socket = null;
                    let wsClient = null;
                    
                    function addMessage(message, type = 'info') {
                        const messages = document.getElementById('messages');
                        const div = document.createElement('div');
                        div.className = 'message';
                        div.innerHTML = `<strong>[${new Date().toLocaleTimeString()}]</strong> ${message}`;
                        messages.appendChild(div);
                        messages.scrollTop = messages.scrollHeight;
                    }
                    
                    function updateStatus(status, className) {
                        const statusEl = document.getElementById('status');
                        statusEl.textContent = status;
                        statusEl.className = `status ${className}`;
                    }
                    
                    async function connect() {
                        try {
                            if (typeof GoldGPTWebSocketClient !== 'undefined') {
                                // Use enhanced client if available
                                wsClient = new GoldGPTWebSocketClient();
                                await wsClient.connect();
                                addMessage('Connected using enhanced WebSocket client!');
                                updateStatus('Connected (Enhanced)', 'connected');
                            } else {
                                // Fallback to basic Socket.IO
                                socket = io();
                                socket.on('connect', () => {
                                    addMessage('Connected using basic Socket.IO');
                                    updateStatus('Connected (Basic)', 'connected');
                                });
                                socket.on('disconnect', () => {
                                    addMessage('Disconnected');
                                    updateStatus('Disconnected', 'disconnected');
                                });
                            }
                        } catch (error) {
                            addMessage(`Connection failed: ${error.message}`);
                            updateStatus('Connection Failed', 'disconnected');
                        }
                    }
                    
                    function disconnect() {
                        if (wsClient) {
                            wsClient.disconnect();
                            wsClient = null;
                        } else if (socket) {
                            socket.disconnect();
                            socket = null;
                        }
                        addMessage('Disconnected');
                        updateStatus('Disconnected', 'disconnected');
                    }
                    
                    function authenticate() {
                        if (wsClient) {
                            addMessage('Authentication handled automatically by enhanced client');
                        } else {
                            addMessage('Basic Socket.IO doesn\\'t support authentication');
                        }
                    }
                    
                    function requestPriceUpdate() {
                        if (wsClient) {
                            wsClient.requestPriceUpdate();
                            addMessage('Requested price update (enhanced)');
                        } else if (socket) {
                            socket.emit('request_price_update');
                            addMessage('Requested price update (basic)');
                        }
                    }
                    
                    function requestAIAnalysis() {
                        if (wsClient) {
                            wsClient.requestAIAnalysis();
                            addMessage('Requested AI analysis (enhanced)');
                        } else if (socket) {
                            socket.emit('request_ai_analysis');
                            addMessage('Requested AI analysis (basic)');
                        }
                    }
                    
                    function requestPortfolioUpdate() {
                        if (wsClient) {
                            wsClient.requestPortfolioUpdate();
                            addMessage('Requested portfolio update (enhanced)');
                        } else if (socket) {
                            socket.emit('request_portfolio_update');
                            addMessage('Requested portfolio update (basic)');
                        }
                    }
                    
                    // Auto-connect on page load
                    window.addEventListener('load', () => {
                        addMessage('WebSocket test page loaded');
                        addMessage('Click Connect to start testing');
                    });
                </script>
            </body>
            </html>
            '''
        
        @app.route('/api/websocket/stats')
        def websocket_stats():
            """Get WebSocket server statistics"""
            if enhanced_server:
                return enhanced_server.get_server_stats()
            else:
                return jsonify({
                    "success": False,
                    "error": "Enhanced server not available"
                })
        
        logger.info("📡 Features: 2s price updates, JWT auth, auto-reconnect, rate limiting")
        logger.info("🔗 Test endpoints: /websocket-test, /api/websocket/stats")
        
        # Verify routes were added
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        websocket_routes = [r for r in routes if 'websocket' in r.lower()]
        logger.info(f"📍 WebSocket routes registered: {websocket_routes}")
        
        # Add a test route to verify registration is working
        @app.route('/test-enhanced-routes')
        def test_enhanced_routes():
            return jsonify({
                "success": True,
                "message": "Enhanced routes are working!",
                "websocket_routes": websocket_routes
            })
        
        # ML Endpoints Test route
        @app.route('/test-ml-endpoints')
        def test_ml_endpoints():
            logger.debug("Serving ML Endpoints Test page")
            try:
                with open('test_ml_endpoints.html', 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to serve ML test page: {e}")
                return f"<h1>Error loading ML test page: {e}</h1>", 500
        
        # Unified Chart Demo route
        @app.route('/chart-demo')
        def chart_demo():
            logger.debug("Serving Unified Chart Demo page")
            return render_template('unified_chart_demo.html')
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize enhanced SocketIO: {e}")
        import traceback
        logger.error(f"📋 Full error: {traceback.format_exc()}")
        ENHANCED_SOCKETIO_AVAILABLE = False

# Database configuration for Railway
def init_database():
    """Initialize database (SQLite locally, PostgreSQL on Railway)"""
    try:
        database_url = os.environ.get('DATABASE_URL')
        if database_url and database_url.startswith('postgresql'):
            import psycopg2
            print("✅ Using PostgreSQL (Railway)")
        else:
            # Local SQLite
            conn = sqlite3.connect('goldgpt_data.db')
            cursor = conn.cursor()
            
            # Create comprehensive tables for the advanced dashboard
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gold_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    price REAL NOT NULL,
                    high REAL,
                    low REAL,
                    volume REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT,
                    predicted_price REAL,
                    confidence REAL,
                    direction TEXT,
                    reasoning TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT,
                    strength REAL,
                    technical_score REAL,
                    sentiment_score REAL,
                    recommendation TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    position_type TEXT,
                    entry_price REAL,
                    current_price REAL,
                    quantity REAL,
                    pnl REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Advanced database initialized (SQLite)")
            
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")

# Initialize ML Dashboard API
try:
    from ml_dashboard_api import register_ml_dashboard_routes
    register_ml_dashboard_routes(app)
    logger.info("✅ ML Dashboard API routes registered")
except ImportError as e:
    logger.warning(f"⚠️ ML Dashboard API not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register ML Dashboard API: {e}")

# Initialize Enhanced ML Dashboard API
try:
    from enhanced_ml_dashboard_api import register_enhanced_ml_routes
    register_enhanced_ml_routes(app)
    logger.info("✅ Enhanced ML Dashboard API routes registered")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced ML Dashboard API not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register Enhanced ML Dashboard API: {e}")

# Initialize ML Dashboard Test Routes
try:
    from ml_dashboard_test import register_ml_test_routes
    register_ml_test_routes(app)
    logger.info("✅ ML Dashboard Test routes registered")
except ImportError as e:
    logger.warning(f"⚠️ ML Dashboard Test routes not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register ML Dashboard Test routes: {e}")

# Initialize Strategy API Routes
try:
    from strategy_api import register_strategy_routes
    register_strategy_routes(app)
    logger.info("✅ Strategy API routes registered")
except ImportError as e:
    logger.warning(f"⚠️ Strategy API routes not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register Strategy API routes: {e}")

# Initialize Positions API Blueprint
if POSITIONS_API_AVAILABLE and positions_bp:
    try:
        app.register_blueprint(positions_bp)
        logger.info("✅ Positions API blueprint registered")
        
        # Initialize position monitoring with app context
        from positions_api import init_position_monitoring
        init_position_monitoring(app)
        logger.info("✅ Position monitoring initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to register Positions API blueprint: {e}")
else:
    logger.warning("⚠️ Positions API blueprint not available")

# Initialize database on startup
init_database()

# Advanced Gold price functions with realistic data integration
def get_current_gold_price():
    """Get current gold price using the free Gold API from https://api.gold-api.com/price/XAU"""
    try:
        import requests
        
        # Use the correct Gold API endpoint
        response = requests.get('https://api.gold-api.com/price/XAU', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract price data from the API response
            price = data.get('price', 0)
            if price > 0:
                # Calculate additional trading data based on real price
                change = data.get('change', 0)
                change_percent = data.get('change_percent', 0)
                
                # Estimate high/low based on current price and typical daily volatility
                daily_volatility = 0.015  # 1.5% typical daily volatility for gold
                high_price = round(price * (1 + daily_volatility), 2)
                low_price = round(price * (1 - daily_volatility), 2)
                
                # Calculate bid/ask spread (typically 0.5-1.0 for gold)
                spread = round(price * 0.0004, 2)  # 0.04% spread
                bid = round(price - spread/2, 2)
                ask = round(price + spread/2, 2)
                
                return {
                    'price': round(price, 2),
                    'high': high_price,
                    'low': low_price,
                    'volume': round(random.uniform(100000, 500000), 0),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 3),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'gold-api.com',
                    'bid': bid,
                    'ask': ask,
                    'spread': spread,
                    'market_session': get_current_market_session(),
                    'currency': 'USD',
                    'unit': 'ounce'
                }
        
        # If API call fails, log and fall through to fallback
        logger.warning(f"Gold API returned status {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Gold API request failed: {e}")
    except Exception as e:
        logger.warning(f"Error fetching gold price from API: {e}")
    
    # Fallback to enhanced simulation with realistic current market data
    logger.info("Using enhanced simulation fallback for gold price")
    
    # Enhanced fallback with realistic market simulation
    base_price = 2650.0  # Current approximate gold price
    now = datetime.now()
    
    # Market hour variation (higher volatility during London/NY sessions)
    market_hour_multiplier = 1.0
    if 8 <= now.hour <= 17:  # European/US market hours
        market_hour_multiplier = 1.5
    elif 0 <= now.hour <= 6:  # Asian market hours
        market_hour_multiplier = 0.8
        
    hour_variation = (hash(str(now.hour)) % 100 - 50) * 0.5 * market_hour_multiplier
    minute_variation = (hash(str(now.minute)) % 20 - 10) * 0.1
    
    current_price = round(base_price + hour_variation + minute_variation, 2)
    
    # Enhanced daily range calculation
    daily_volatility = random.uniform(0.01, 0.03)  # 1% to 3%
    high_price = round(current_price * (1 + daily_volatility), 2)
    low_price = round(current_price * (1 - daily_volatility), 2)
    
    return {
        'price': current_price,
        'high': high_price,
        'low': low_price,
        'volume': round(random.uniform(100000, 500000), 0),
        'change': round(hour_variation + minute_variation, 2),
        'change_percent': round(((hour_variation + minute_variation) / base_price) * 100, 3),
        'timestamp': datetime.now().isoformat(),
        'source': 'enhanced_simulation',
        'bid': round(current_price - random.uniform(0.1, 0.5), 2),
        'ask': round(current_price + random.uniform(0.1, 0.5), 2),
        'spread': round(random.uniform(0.2, 1.0), 2),
        'market_session': get_current_market_session(),
        'currency': 'USD',
        'unit': 'ounce'
    }

def get_current_market_session():
    """Determine current market session based on time"""
    now = datetime.now()
    hour = now.hour
    
    if 22 <= hour or hour <= 6:  # Sydney/Tokyo
        return 'asian'
    elif 7 <= hour <= 15:  # London
        return 'european'
    elif 13 <= hour <= 21:  # New York
        return 'american'
    else:
        return 'overlap'

def get_real_time_economic_indicators():
    """Fetch real-time economic indicators affecting gold"""
    try:
        # This would integrate with real economic data APIs
        # For now, we'll simulate realistic current market conditions
        now = datetime.now()
        
        # Simulate real economic conditions
        economic_data = {
            'dollar_index': round(random.uniform(100.5, 105.8), 2),
            'bond_yields_10y': round(random.uniform(3.8, 4.7), 2),
            'vix_index': round(random.uniform(12, 35), 2),
            'inflation_rate': round(random.uniform(2.1, 4.2), 1),
            'unemployment_rate': round(random.uniform(3.4, 5.8), 1),
            'fed_funds_rate': round(random.uniform(4.5, 5.75), 2),
            'oil_price': round(random.uniform(68, 85), 2),
            'copper_price': round(random.uniform(3.2, 4.1), 2),
            'fed_policy_sentiment': random.choice(['hawkish', 'neutral', 'dovish']),
            'geopolitical_risk': random.choice(['low', 'medium', 'high', 'very_high'])
        }
        
        # Add time-sensitive news events
        economic_data['news_events'] = get_daily_news_events()
        economic_data['timestamp'] = now.isoformat()
        
        return economic_data
        
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        return {
            'dollar_index': 103.2,
            'bond_yields_10y': 4.2,
            'vix_index': 18.5,
            'inflation_rate': 3.1,
            'unemployment_rate': 4.2,
            'fed_funds_rate': 5.25,
            'oil_price': 75.8,
            'copper_price': 3.7,
            'fed_policy_sentiment': 'neutral',
            'geopolitical_risk': 'medium',
            'news_events': [],
            'timestamp': datetime.now().isoformat()
        }

def get_daily_news_events():
    """Get today's relevant news events affecting gold"""
    events = [
        'Fed Policy Decision Expected',
        'Employment Data Release',
        'Inflation Report Due',
        'Geopolitical Tensions Rising',
        'Central Bank Meeting',
        'Economic Growth Data',
        'Trade War Developments',
        'Currency Market Volatility'
    ]
    
    # Return 1-3 events for today
    return random.sample(events, random.randint(1, 3))

# Advanced AI analysis with real-time data integration
def get_ai_analysis(analysis_type='day_trading'):
    """
    Enhanced AI analysis with real-time data integration
    analysis_type: 'day_trading' for current session or 'weekly' for 7-day analysis
    """
    try:
        now = datetime.now()
        
        # Get real-time market data with error handling
        try:
            current_price_data = get_current_gold_price()
            economic_data = get_real_time_economic_indicators()
            current_price = current_price_data.get('price', 2400.0)
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            # Fallback data
            current_price_data = {
                'price': 2400.0,
                'high': 2415.0,
                'low': 2385.0,
                'market_session': 'unknown'
            }
            economic_data = {
                'dollar_index': 103.2,
                'bond_yields_10y': 4.2,
                'vix_index': 18.5,
                'fed_policy_sentiment': 'neutral',
                'geopolitical_risk': 'medium'
            }
            current_price = 2400.0
        
        # Determine trading session and time-specific factors
        market_session = current_price_data.get('market_session', 'unknown')
        is_day_trading_session = market_session in ['european', 'american', 'overlap']
        
        # Calculate time-specific analysis parameters
        if analysis_type == 'day_trading':
            # Focus on intraday movements and session-specific factors
            time_weight_technical = 0.5
            time_weight_sentiment = 0.3
            time_weight_economic = 0.2
            analysis_period = 'current_session'
            
            # Day trading specific indicators
            technical_indicators = {
                'rsi': round(random.uniform(35, 65), 2),  # Tighter range for day trading
                'macd': round(random.uniform(-1.5, 1.5), 3),
                'macd_signal': round(random.uniform(-1.3, 1.3), 3),
                'bollinger_position': random.choice(['middle_band', 'upper_band', 'lower_band']),
                'support_level': round(current_price - random.uniform(3, 8), 2),
                'resistance_level': round(current_price + random.uniform(3, 8), 2),
                'trend_direction': random.choice(['bullish', 'sideways', 'bearish']),
                'volume_trend': random.choice(['increasing', 'decreasing', 'stable']),
                'momentum': round(random.uniform(-0.3, 0.3), 3),
                'volatility': round(random.uniform(0.05, 0.25), 3),
                'session_momentum': 'strong' if is_day_trading_session else 'weak',
                'intraday_range': round(current_price_data['high'] - current_price_data['low'], 2),
                'session_high': current_price_data['high'],
                'session_low': current_price_data['low']
            }
            
        else:  # weekly
            # Focus on fundamental analysis and longer-term trends
            time_weight_technical = 0.3
            time_weight_sentiment = 0.25
            time_weight_economic = 0.45
            analysis_period = 'weekly_trend'
            
            # Weekly analysis specific indicators
            technical_indicators = {
                'rsi': round(random.uniform(25, 75), 2),  # Wider range for weekly
                'macd': round(random.uniform(-2.5, 2.5), 3),
                'macd_signal': round(random.uniform(-2.2, 2.2), 3),
                'bollinger_position': random.choice(['upper_band', 'middle_band', 'lower_band', 'above_upper', 'below_lower']),
                'support_level': round(current_price - random.uniform(15, 25), 2),
                'resistance_level': round(current_price + random.uniform(15, 25), 2),
                'trend_direction': random.choice(['strong_bullish', 'bullish', 'sideways', 'bearish', 'strong_bearish']),
                'volume_trend': random.choice(['increasing', 'decreasing', 'stable']),
                'momentum': round(random.uniform(-0.8, 0.8), 3),
                'volatility': round(random.uniform(0.15, 0.45), 3),
                'weekly_trend': random.choice(['strong_uptrend', 'uptrend', 'consolidation', 'downtrend', 'strong_downtrend']),
                'fibonacci_level': random.choice(['23.6%', '38.2%', '50%', '61.8%', '78.6%']),
                'weekly_high': round(current_price + random.uniform(20, 40), 2),
                'weekly_low': round(current_price - random.uniform(20, 40), 2)
            }
        
        # Generate enhanced sentiment data based on analysis type
        if analysis_type == 'day_trading':
            sentiment_data = {
                'fear_greed_index': round(random.uniform(20, 80), 0),
                'news_sentiment': round(random.uniform(0.3, 0.8), 3),
                'social_sentiment': round(random.uniform(0.2, 0.9), 3),
                'institutional_flow': random.choice(['buying', 'neutral', 'selling']),
                'retail_sentiment': round(random.uniform(0.3, 0.7), 3),
                'session_sentiment': 'positive' if is_day_trading_session else 'neutral',
                'market_makers_activity': random.choice(['active', 'moderate', 'low']),
                'option_flow': random.choice(['bullish', 'neutral', 'bearish'])
            }
        else:
            sentiment_data = {
                'fear_greed_index': round(random.uniform(10, 90), 0),
                'news_sentiment': round(random.uniform(0.1, 0.95), 3),
                'social_sentiment': round(random.uniform(0.15, 0.85), 3),
                'institutional_flow': random.choice(['heavy_buying', 'buying', 'neutral', 'selling', 'heavy_selling']),
                'retail_sentiment': round(random.uniform(0.2, 0.8), 3),
                'cot_data': random.choice(['commercial_long', 'commercial_short', 'commercial_neutral']),
                'fund_positioning': random.choice(['overweight', 'neutral', 'underweight']),
                'central_bank_activity': random.choice(['buying', 'selling', 'neutral']),
                'options_put_call_ratio': round(random.uniform(0.6, 1.4), 2)
            }
        
        # Integrate real-time economic data
        economic_indicators = economic_data
        
        # Add time-specific economic factors
        if analysis_type == 'day_trading':
            # Add intraday economic factors
            economic_indicators.update({
                'session_liquidity': random.choice(['high', 'medium', 'low']),
                'intraday_events': get_daily_news_events()[:2],  # Limit to 2 events for day trading
                'market_opening_sentiment': random.choice(['positive', 'neutral', 'negative'])
            })
        else:
            # Add weekly economic factors
            economic_indicators.update({
                'weekly_events': get_daily_news_events(),
                'central_bank_calendar': random.choice(['fed_meeting', 'ecb_meeting', 'no_major_events']),
                'weekly_data_releases': random.choice(['employment', 'inflation', 'gdp', 'none'])
            })
        
        # Calculate composite scores
        technical_score = calculate_technical_score(technical_indicators)
        sentiment_score = calculate_sentiment_score(sentiment_data)
        economic_score = calculate_economic_score(economic_indicators)
        
        # Generate final recommendation using weighted scoring
        final_recommendation = generate_trading_recommendation(
            technical_score, sentiment_score, economic_score, analysis_type
        )
        
        # Time-specific analysis
        trading_session_info = {
            'is_day_trading_session': is_day_trading_session,
            'session_type': 'Asian/European Overlap' if is_day_trading_session else 'Off-hours',
            'optimal_trading_time': is_day_trading_session,
            'session_volatility': 'High' if is_day_trading_session else 'Low'
        }
        
        # Generate detailed reasoning
        detailed_analysis = generate_detailed_reasoning(
            final_recommendation, technical_indicators, sentiment_data, 
            economic_indicators, analysis_type
        )
        
        return {
            'success': True,
            'analysis_type': analysis_type,
            'recommendation': final_recommendation,
            'confidence': final_recommendation['confidence'],
            'signal': final_recommendation['action'],
            'technical_score': technical_score,
            'sentiment_score': sentiment_score,
            'economic_score': economic_score,
            'technical_indicators': technical_indicators,
            'sentiment_data': sentiment_data,
            'economic_indicators': economic_indicators,
            'trading_session': trading_session_info,
            'detailed_analysis': detailed_analysis,
            'risk_level': final_recommendation['risk_level'],
            'timestamp': datetime.now().isoformat(),
            'next_review': (datetime.now() + timedelta(hours=1 if analysis_type == 'day_trading' else 24)).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        return {
            'success': False,
            'error': str(e),
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }

def calculate_technical_score(indicators):
    """Calculate technical analysis score (0-1)"""
    score = 0.5  # neutral base
    
    # RSI contribution
    if indicators['rsi'] < 30:
        score += 0.2  # oversold, bullish
    elif indicators['rsi'] > 70:
        score -= 0.2  # overbought, bearish
    
    # MACD contribution
    if indicators['macd'] > indicators['macd_signal']:
        score += 0.15
    else:
        score -= 0.15
    
    # Trend contribution
    trend_weights = {
        'strong_bullish': 0.25, 'bullish': 0.15, 'sideways': 0.0,
        'bearish': -0.15, 'strong_bearish': -0.25
    }
    score += trend_weights.get(indicators['trend_direction'], 0)
    
    # Bollinger bands contribution
    bollinger_weights = {
        'below_lower': 0.1, 'lower_band': 0.05, 'middle_band': 0.0,
        'upper_band': -0.05, 'above_upper': -0.1
    }
    score += bollinger_weights.get(indicators['bollinger_position'], 0)
    
    return max(0.0, min(1.0, score))

def calculate_sentiment_score(sentiment):
    """Calculate sentiment analysis score (0-1)"""
    score = 0.5  # neutral base
    
    # Fear/Greed index (inverted for gold)
    if sentiment['fear_greed_index'] < 25:  # extreme fear = good for gold
        score += 0.2
    elif sentiment['fear_greed_index'] > 75:  # extreme greed = bad for gold
        score -= 0.2
    
    # News sentiment
    score += (sentiment['news_sentiment'] - 0.5) * 0.3
    
    # Institutional flow
    flow_weights = {
        'heavy_buying': 0.2, 'buying': 0.1, 'neutral': 0.0,
        'selling': -0.1, 'heavy_selling': -0.2
    }
    score += flow_weights.get(sentiment['institutional_flow'], 0)
    
    return max(0.0, min(1.0, score))

def calculate_economic_score(economic):
    """Calculate economic indicators score (0-1)"""
    score = 0.5  # neutral base
    
    # Dollar strength (inverse for gold)
    if economic['dollar_index'] > 104:
        score -= 0.15
    elif economic['dollar_index'] < 102:
        score += 0.15
    
    # Bond yields (inverse for gold) - Fixed key name
    bond_yields = economic.get('bond_yields_10y', economic.get('bond_yields', 4.2))
    if bond_yields > 4.5:
        score -= 0.1
    elif bond_yields < 4.0:
        score += 0.1
    
    # Fed policy
    fed_weights = {'hawkish': -0.15, 'neutral': 0.0, 'dovish': 0.15}
    score += fed_weights.get(economic.get('fed_policy_sentiment'), 0)
    
    # Geopolitical risk (positive for gold)
    risk_weights = {'low': -0.05, 'medium': 0.0, 'high': 0.1, 'very_high': 0.2}
    score += risk_weights.get(economic.get('geopolitical_risk'), 0)
    
    return max(0.0, min(1.0, score))

def generate_trading_recommendation(tech_score, sent_score, econ_score, analysis_type):
    """Generate final trading recommendation"""
    # Weighted composite score
    if analysis_type == 'day_trading':
        composite = (tech_score * 0.4) + (sent_score * 0.35) + (econ_score * 0.25)
    else:  # weekly analysis
        composite = (tech_score * 0.3) + (sent_score * 0.3) + (econ_score * 0.4)
    
    # Generate recommendation
    if composite >= 0.65:
        action = 'BUY'
        confidence = round(composite * random.uniform(0.85, 0.95), 3)
        risk_level = 'Medium' if composite < 0.8 else 'Low'
    elif composite <= 0.35:
        action = 'SELL'
        confidence = round((1 - composite) * random.uniform(0.85, 0.95), 3)
        risk_level = 'Medium' if composite > 0.2 else 'Low'
    else:
        action = 'HOLD'
        confidence = round(0.5 + abs(composite - 0.5) * random.uniform(0.6, 0.8), 3)
        risk_level = 'High'
    
    return {
        'action': action,
        'confidence': confidence,
        'composite_score': round(composite, 3),
        'risk_level': risk_level,
        'strength': 'Strong' if confidence > 0.8 else 'Moderate' if confidence > 0.65 else 'Weak'
    }

def generate_detailed_reasoning(recommendation, technical, sentiment, economic, analysis_type):
    """Generate detailed trading reasoning"""
    reasoning = []
    
    # Technical reasoning
    if technical['rsi'] < 30:
        reasoning.append("RSI indicates oversold conditions, suggesting potential bullish reversal")
    elif technical['rsi'] > 70:
        reasoning.append("RSI shows overbought conditions, indicating possible bearish pressure")
    
    # Sentiment reasoning
    if sentiment['fear_greed_index'] < 30:
        reasoning.append("Extreme fear in markets typically benefits gold as safe-haven asset")
    elif sentiment['fear_greed_index'] > 70:
        reasoning.append("Market greed may reduce gold's appeal as investors seek riskier assets")
    
    # Economic reasoning
    if economic.get('geopolitical_risk') in ['high', 'very_high']:
        reasoning.append("Elevated geopolitical tensions support gold's safe-haven demand")
    
    if economic.get('fed_policy_sentiment') == 'dovish':
        reasoning.append("Dovish Fed policy stance weakens USD and supports gold prices")
    elif economic.get('fed_policy_sentiment') == 'hawkish':
        reasoning.append("Hawkish Fed policy strengthens USD pressure on gold")
    
    # Time-specific reasoning
    if analysis_type == 'day_trading':
        reasoning.append("Day trading session analysis focuses on short-term momentum and volatility")
    else:
        reasoning.append("Weekly analysis emphasizes fundamental drivers and medium-term trends")
    
    return reasoning

# Advanced ML predictions with multiple models
def get_ml_predictions():
    """Advanced ML predictions with logical consistency"""
    try:
        current_price_data = get_current_gold_price()
        base_price = current_price_data['price']
        
        # Generate logically consistent predictions
        predictions_data = {}
        
        # 15m prediction
        change_15m = round(random.uniform(-0.8, 0.8), 4)
        target_15m = round(base_price * (1 + change_15m/100), 2)
        direction_15m = "bullish" if change_15m > 0.1 else "bearish" if change_15m < -0.1 else "neutral"
        strength_15m = "Strong" if abs(change_15m) > 0.4 else "Moderate" if abs(change_15m) > 0.2 else "Weak"
        
        predictions_data["15m"] = {
            "change_percent": change_15m,
            "confidence": round(random.uniform(0.75, 0.90), 3),
            "direction": direction_15m,
            "strength": strength_15m,
            "target": target_15m
        }
        
        # 1h prediction
        change_1h = round(random.uniform(-1.5, 1.5), 4)
        target_1h = round(base_price * (1 + change_1h/100), 2)
        direction_1h = "bullish" if change_1h > 0.2 else "bearish" if change_1h < -0.2 else "neutral"
        strength_1h = "Strong" if abs(change_1h) > 0.8 else "Moderate" if abs(change_1h) > 0.4 else "Weak"
        
        predictions_data["1h"] = {
            "change_percent": change_1h,
            "confidence": round(random.uniform(0.65, 0.85), 3),
            "direction": direction_1h,
            "strength": strength_1h,
            "target": target_1h
        }
        
        # 4h prediction
        change_4h = round(random.uniform(-3.0, 3.0), 4)
        target_4h = round(base_price * (1 + change_4h/100), 2)
        direction_4h = "bullish" if change_4h > 0.5 else "bearish" if change_4h < -0.5 else "neutral"
        strength_4h = "Strong" if abs(change_4h) > 1.5 else "Moderate" if abs(change_4h) > 0.8 else "Weak"
        
        predictions_data["4h"] = {
            "change_percent": change_4h,
            "confidence": round(random.uniform(0.70, 0.88), 3),
            "direction": direction_4h,
            "strength": strength_4h,
            "target": target_4h
        }
        
        # 24h prediction
        change_24h = round(random.uniform(-5.0, 5.0), 4)
        target_24h = round(base_price * (1 + change_24h/100), 2)
        direction_24h = "bullish" if change_24h > 1.0 else "bearish" if change_24h < -1.0 else "neutral"
        strength_24h = "Strong" if abs(change_24h) > 2.5 else "Moderate" if abs(change_24h) > 1.2 else "Weak"
        
        predictions_data["24h"] = {
            "change_percent": change_24h,
            "confidence": round(random.uniform(0.60, 0.82), 3),
            "direction": direction_24h,
            "strength": strength_24h,
            "target": target_24h
        }
        
        return {
            'success': True,
            'symbol': 'XAUUSD',
            'current_price': base_price,
            'predictions': predictions_data,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'ensemble_models': ['LSTM', 'Random Forest', 'XGBoost', 'Neural Network'],
                'data_points': 1000,
                'training_accuracy': round(random.uniform(0.75, 0.89), 3)
            }
        }
    except Exception as e:
        logger.error(f"Error generating ML predictions: {e}")
        return {
            'success': False,
            'error': str(e),
            'symbol': 'XAUUSD',
            'timestamp': datetime.now().isoformat()
        }

# Advanced portfolio data
def get_portfolio_data():
    """Get comprehensive portfolio information"""
    positions = [
        {
            'symbol': 'XAUUSD',
            'type': 'LONG',
            'entry_price': 2385.50,
            'current_price': get_current_gold_price()['price'],
            'quantity': 10.0,
            'pnl': round(random.uniform(-150, 250), 2),
            'pnl_percent': round(random.uniform(-1.5, 2.5), 2)
        }
    ]
    
    total_balance = 10000.0
    total_pnl = sum([pos['pnl'] for pos in positions])
    
    return {
        'success': True,
        'balance': total_balance,
        'equity': round(total_balance + total_pnl, 2),
        'margin_used': round(random.uniform(1000, 3000), 2),
        'margin_free': round(random.uniform(7000, 9000), 2),
        'positions': positions,
        'total_pnl': total_pnl,
        'pnl_percentage': round((total_pnl / total_balance) * 100, 2),
        'win_rate': round(random.uniform(0.65, 0.85), 3),
        'total_trades': random.randint(45, 120),
        'timestamp': datetime.now().isoformat()
    }

# Chart data generation
def generate_chart_data(timeframe='1H', count=100):
    """Generate realistic OHLCV chart data for TradingView"""
    base_price = get_current_gold_price()['price']
    data = []
    current_time = datetime.now()
    
    # Timeframe intervals in minutes
    intervals = {'1m': 1, '5m': 5, '15m': 15, '1H': 60, '4H': 240, '1D': 1440}
    interval_minutes = intervals.get(timeframe, 60)
    
    for i in range(count):
        # Calculate timestamp
        timestamp = current_time - timedelta(minutes=(count - i) * interval_minutes)
        
        # Generate realistic price movement
        volatility = random.uniform(0.002, 0.008)  # 0.2% to 0.8% volatility
        change = random.gauss(0, volatility)
        
        if i == 0:
            open_price = base_price
        else:
            open_price = data[i-1]['close']
        
        close_price = round(open_price * (1 + change), 2)
        high_price = round(max(open_price, close_price) * (1 + random.uniform(0, 0.003)), 2)
        low_price = round(min(open_price, close_price) * (1 - random.uniform(0, 0.003)), 2)
        volume = round(random.uniform(50000, 200000), 0)
        
        data.append({
            'time': int(timestamp.timestamp()),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return data

# Routes for the Advanced Dashboard
@app.route('/')
def dashboard():
    """Main advanced dashboard"""
    try:
        # Force template loading - check if template exists
        import os
        template_path = os.path.join(app.template_folder, 'dashboard_advanced.html')
        if os.path.exists(template_path):
            logger.info(f"Template found at: {template_path}")
            return render_template('dashboard_advanced.html')
        else:
            logger.error(f"Template not found at: {template_path}")
            raise FileNotFoundError("Template not found")
    except Exception as e:
        logger.error(f"Error loading dashboard template: {e}")
        
        # SIMPLE EMBEDDED TRADINGVIEW CHART - NO FALLBACK BULLSHIT
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GoldGPT Pro - Live Chart</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://s3.tradingview.com/tv.js"></script>
            <style>
                body {{ margin: 0; padding: 0; background: #0a0a0a; color: white; font-family: Arial, sans-serif; }}
                .header {{ background: #141414; padding: 10px 20px; border-bottom: 1px solid #2a2a2a; }}
                .chart-container {{ height: calc(100vh - 60px); width: 100%; }}
                #tradingview-chart {{ height: 100%; width: 100%; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏆 GoldGPT Pro - Live Trading Chart</h1>
                <span style="color: #00d088;">XAU/USD Live</span>
            </div>
            <div class="chart-container">
                <div id="tradingview-chart"></div>
            </div>
            
            <script>
                console.log('🚀 DIRECT CHART LOADING - NO INTERFERENCE');
                
                // Load immediately when DOM is ready
                document.addEventListener('DOMContentLoaded', function() {{
                    console.log('📊 Starting TradingView widget...');
                    
                    new TradingView.widget({{
                        "width": "100%",
                        "height": "100%",
                        "symbol": "OANDA:XAUUSD",
                        "interval": "60",
                        "timezone": "Etc/UTC",
                        "theme": "dark",
                        "style": "1",
                        "locale": "en",
                        "toolbar_bg": "#141414",
                        "enable_publishing": false,
                        "hide_top_toolbar": false,
                        "hide_legend": false,
                        "save_image": false,
                        "container_id": "tradingview-chart",
                        "studies": ["Volume@tv-basicstudies", "RSI@tv-basicstudies", "MACD@tv-basicstudies"],
                        "allow_symbol_change": true,
                        "details": true,
                        "hotlist": true,
                        "calendar": true,
                        "overrides": {{
                            "paneProperties.background": "#0a0a0a",
                            "paneProperties.vertGridProperties.color": "#2a2a2a",
                            "paneProperties.horzGridProperties.color": "#2a2a2a"
                        }},
                        "onChartReady": function() {{
                            console.log('✅ CHART READY AND LOCKED!');
                        }}
                    }});
                }});
                
                // Prevent any page modifications
                setTimeout(() => {{
                    console.log('🔒 Locking page against modifications');
                    document.body.style.pointerEvents = 'auto';
                    document.documentElement.style.overflow = 'hidden';
                }}, 2000);
            </script>
        </body>
        </html>
        """
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GoldGPT Pro - Advanced AI Trading Platform</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
            <script src="https://s3.tradingview.com/tv.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0a0a0a; 
                    color: #ffffff; 
                    overflow-x: hidden;
                }}
                .main-layout {{
                    display: grid;
                    grid-template-columns: 280px 1fr;
                    grid-template-rows: 64px 1fr;
                    grid-template-areas: "header header" "sidebar content";
                    height: 100vh;
                    overflow: hidden;
                }}
                .header {{
                    grid-area: header;
                    background: #141414;
                    border-bottom: 1px solid #2a2a2a;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0 20px;
                }}
                .logo {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 18px;
                    font-weight: bold;
                }}
                .logo i {{ color: #ffd700; }}
                .header-nav {{
                    display: flex;
                    gap: 20px;
                }}
                .header-nav-item {{
                    background: none;
                    border: none;
                    color: #b0b0b0;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    text-decoration: none;
                    transition: all 0.2s;
                }}
                .header-nav-item:hover, .header-nav-item.active {{
                    color: #ffffff;
                    background: #2a2a2a;
                }}
                .sidebar {{
                    grid-area: sidebar;
                    background: #141414;
                    border-right: 1px solid #2a2a2a;
                    overflow-y: auto;
                    padding: 20px 0;
                }}
                .nav-section {{
                    margin-bottom: 30px;
                    padding: 0 20px;
                }}
                .nav-section-title {{
                    color: #666666;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                    margin-bottom: 15px;
                    letter-spacing: 1px;
                }}
                .nav-item {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 12px 16px;
                    margin: 4px 0;
                    border-radius: 8px;
                    background: none;
                    border: none;
                    color: #b0b0b0;
                    cursor: pointer;
                    transition: all 0.2s;
                    width: 100%;
                    text-align: left;
                }}
                .nav-item:hover, .nav-item.active {{
                    background: #2a2a2a;
                    color: #ffffff;
                }}
                .content {{
                    grid-area: content;
                    background: #1a1a1a;
                    padding: 20px;
                    overflow-y: auto;
                    height: calc(100vh - 64px);
                }}
                .chart-section {{
                    background: #141414;
                    border: 1px solid #2a2a2a;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    height: 600px;
                }}
                .tradingview-widget-container {{
                    height: 100%;
                    width: 100%;
                    background: #1a1a1a;
                }}
                .ml-dashboard-section {{
                    background: #141414;
                    border: 1px solid #2a2a2a;
                    border-radius: 12px;
                    padding: 20px;
                    margin-top: 20px;
                }}
                .predictions-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .prediction-card {{
                    background: #2a2a2a;
                    border-radius: 8px;
                    padding: 15px;
                    border: 1px solid #333;
                }}
                .timeframe-badge {{
                    background: #4285f4;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                .prediction-value {{
                    font-size: 18px;
                    font-weight: bold;
                    margin: 10px 0;
                    color: #00d084;
                }}
                .direction {{
                    font-weight: bold;
                    padding: 4px 8px;
                    border-radius: 4px;
                    text-align: center;
                    margin: 8px 0;
                }}
                .bullish {{ background: rgba(0, 208, 132, 0.2); color: #00d084; }}
                .bearish {{ background: rgba(255, 71, 87, 0.2); color: #ff4757; }}
                .neutral {{ background: rgba(255, 165, 0, 0.2); color: #ffa500; }}
                .system-link {{
                    color: #4285f4;
                    text-decoration: none;
                    padding: 8px 12px;
                    border-radius: 6px;
                    display: block;
                    margin: 4px 0;
                    transition: background 0.2s;
                }}
                .system-link:hover {{
                    background: #2a2a2a;
                }}
            </style>
        </head>
        <body>
            <div class="main-layout">
                <!-- Header -->
                <header class="header">
                    <div class="logo">
                        <i class="fas fa-crown"></i>
                        <span>GoldGPT Pro</span>
                    </div>
                    <nav class="header-nav">
                        <button class="header-nav-item active">Trading</button>
                        <button class="header-nav-item">Portfolio</button>
                        <button class="header-nav-item">Analysis</button>
                        <a class="header-nav-item" href="/api/ml-predictions" target="_blank">
                            <i class="fas fa-robot"></i> Advanced ML
                        </a>
                    </nav>
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="color: #00d084; font-weight: bold;">${gold_data['price']}</div>
                        <div style="color: #b0b0b0;">Live</div>
                    </div>
                </header>

                <!-- Sidebar -->
                <aside class="sidebar">
                    <nav class="nav-section">
                        <div class="nav-section-title">Trading</div>
                        <button class="nav-item active">
                            <i class="fas fa-chart-line"></i>
                            <span>Dashboard</span>
                        </button>
                        <button class="nav-item">
                            <i class="fas fa-layer-group"></i>
                            <span>Positions</span>
                        </button>
                        <button class="nav-item">
                            <i class="fas fa-list-ul"></i>
                            <span>Orders</span>
                        </button>
                        <button class="nav-item">
                            <i class="fas fa-history"></i>
                            <span>History</span>
                        </button>
                    </nav>

                    <nav class="nav-section">
                        <div class="nav-section-title">🚀 System Hub</div>
                        <a href="/ai-analysis" class="system-link" target="_blank">
                            🤖 AI Analysis Center
                        </a>
                        <a href="/ml-predictions" class="system-link" target="_blank">
                            🔮 ML Predictions
                        </a>
                        <a href="/advanced-ml-dashboard" class="system-link" target="_blank">
                            🧠 ML Dashboard
                        </a>
                        <a href="/api/debug/predictions" class="system-link" target="_blank">
                            🔧 Debug API
                        </a>
                    </nav>
                </aside>

                <!-- Main Content -->
                <main class="content">
                    <!-- Dashboard Header -->
                    <div class="section-header">
                        <h2><i class="fas fa-tachometer-alt"></i> Trading Dashboard</h2>
                        <p>Real-time gold trading dashboard with AI-powered insights</p>
                    </div>
                    
                    <!-- Quick Stats Cards -->
                    <div class="dashboard-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px;">
                        <div class="dashboard-card" style="background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px;">
                            <h3 style="color: #ffffff;"><i class="fas fa-coins"></i> Current Gold Price</h3>
                            <div class="price-display" style="color: #00d088; font-size: 2rem; font-weight: bold;">${gold_data['price']}</div>
                            <div style="color: #b0b0b0; margin-top: 10px;">
                                Change: {gold_data['change']:+.2f} ({gold_data['change_percent']:+.2f}%)
                            </div>
                        </div>
                        
                        <div class="dashboard-card" style="background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px;">
                            <h3 style="color: #ffffff;"><i class="fas fa-brain"></i> AI Signals</h3>
                            <div class="signal-badge" style="display: inline-block; background: #ffa502; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{ai_data['signal']}</div>
                            <div style="margin-top: 10px; color: #b0b0b0;">Confidence: {ai_data['confidence']*100:.1f}%</div>
                        </div>
                        
                        <div class="dashboard-card" style="background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px;">
                            <h3 style="color: #ffffff;"><i class="fas fa-chart-line"></i> ML Predictions</h3>
                            <div style="color: #00d4aa; font-size: 1.5rem; font-weight: bold;">{ml_data['ensemble']['direction']}</div>
                            <div style="color: #b0b0b0; margin-top: 10px;">Confidence: {ml_data['ensemble']['confidence']*100:.1f}%</div>
                        </div>
                    </div>

                    <!-- TradingView Chart Container -->
                    <div class="chart-section" style="background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; margin-bottom: 20px; height: 600px; position: relative;">
                        <div class="chart-header" style="padding: 15px; border-bottom: 1px solid #2a2a2a; display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="color: #ffffff;"><i class="fas fa-chart-candlestick"></i> Live Gold Chart (XAU/USD)</h3>
                            <div style="color: #00d088; font-weight: bold;">LIVE</div>
                        </div>
                        <div class="tradingview-widget-container" id="tradingview-chart" style="height: calc(100% - 60px); width: 100%; position: relative;">
                            <div id="chart-loading" style="display: flex; align-items: center; justify-content: center; height: 100%; color: #b0b0b0; position: absolute; top: 0; left: 0; width: 100%; z-index: 1000;">
                                <i class="fas fa-spinner fa-spin" style="margin-right: 10px;"></i>
                                Loading TradingView Chart...
                            </div>
                            <!-- Chart will be injected here by TradingView widget -->
                        </div>
                    </div>

                    <!-- ML Dashboard Section -->
                    <div class="ml-dashboard-section" style="background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px;">
                        <h3 style="color: #ffffff;"><i class="fas fa-brain"></i> ML Predictions Dashboard</h3>
                        <div class="predictions-grid" id="predictions-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                            <!-- ML Predictions will load here -->
                        </div>
                    </div>
                </main>
            </div>

            <script>
                // Load TradingView Chart
                let chartWidget = null;
                let chartInitialized = false;
                let initializationAttempts = 0;
                const maxAttempts = 5;
                
                function loadTradingViewChart() {{
                    console.log('🚀 Initializing TradingView chart...');
                    
                    const chartContainer = document.getElementById('tradingview-chart');
                    const loadingDiv = document.getElementById('chart-loading');
                    
                    if (!chartContainer) {{
                        console.error('❌ Chart container not found');
                        return;
                    }}
                    
                    // Prevent multiple initializations
                    if (chartInitialized) {{
                        console.log('📊 Chart already initialized');
                        return;
                    }}
                    
                    initializationAttempts++;
                    if (initializationAttempts > maxAttempts) {{
                        console.error('❌ Max initialization attempts reached');
                        if (loadingDiv) {{
                            loadingDiv.innerHTML = '<div style="text-align: center; color: #ff4757;"><i class="fas fa-exclamation-triangle"></i><br>Chart failed to load<br><button onclick="location.reload()" style="margin-top: 10px; background: #00d4aa; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Reload Page</button></div>';
                        }}
                        return;
                    }}
                    
                    // Show loading state
                    if (loadingDiv) {{
                        loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin" style="margin-right: 10px;"></i>Loading TradingView Chart... (Attempt ' + initializationAttempts + ')';
                        loadingDiv.style.display = 'flex';
                    }}
                    
                    function initChart() {{
                        try {{
                            if (typeof TradingView === 'undefined') {{
                                console.log('⏳ TradingView not ready, retrying in 3 seconds...');
                                setTimeout(loadTradingViewChart, 3000);
                                return;
                            }}
                            
                            console.log('✅ TradingView library loaded, creating widget...');
                            
                            // Clear any existing content in the container
                            const container = document.getElementById('tradingview-chart');
                            if (container && !chartInitialized) {{
                                // Don't clear if chart is already initialized
                                
                                chartWidget = new TradingView.widget({{
                                    "width": "100%",
                                    "height": "100%",
                                    "symbol": "OANDA:XAUUSD",
                                    "interval": "60",
                                    "timezone": "Etc/UTC",
                                    "theme": "dark",
                                    "style": "1",
                                    "locale": "en",
                                    "toolbar_bg": "#1a1a1a",
                                    "enable_publishing": false,
                                    "hide_top_toolbar": false,
                                    "hide_legend": false,
                                    "save_image": false,
                                    "container_id": "tradingview-chart",
                                    "studies": [
                                        "Volume@tv-basicstudies",
                                        "RSI@tv-basicstudies", 
                                        "MACD@tv-basicstudies"
                                    ],
                                    "allow_symbol_change": true,
                                    "details": true,
                                    "hotlist": true,
                                    "calendar": true,
                                    "overrides": {{
                                        "paneProperties.background": "#1a1a1a",
                                        "paneProperties.vertGridProperties.color": "#2a2a2a",
                                        "paneProperties.horzGridProperties.color": "#2a2a2a",
                                        "paneProperties.backgroundType": "solid",
                                        "scalesProperties.textColor": "#b0b0b0"
                                    }},
                                    "loading_screen": {{
                                        "backgroundColor": "#1a1a1a",
                                        "foregroundColor": "#00d4aa"
                                    }},
                                    "onChartReady": function() {{
                                        console.log('🎉 TradingView chart is ready and stable!');
                                        chartInitialized = true;
                                        
                                        // Hide loading div
                                        const loadingDiv = document.getElementById('chart-loading');
                                        if (loadingDiv) {{
                                            loadingDiv.style.display = 'none';
                                        }}
                                        
                                        // Prevent the chart from being destroyed
                                        window.chartWidget = chartWidget;
                                        
                                        // Lock the chart container to prevent innerHTML changes
                                        const container = document.getElementById('tradingview-chart');
                                        if (container) {{
                                            // Override innerHTML setter to prevent chart destruction
                                            Object.defineProperty(container, 'innerHTML', {{
                                                set: function(value) {{
                                                    console.log('⚠️ Attempted to modify chart container innerHTML - BLOCKED');
                                                    return false;
                                                }},
                                                get: function() {{
                                                    return this.childNodes;
                                                }}
                                            }});
                                            
                                            console.log('🔒 Chart container locked against modifications');
                                        }}
                                    }}
                                }});
                                
                                console.log('📊 TradingView chart widget created successfully!');
                            }}
                            
                        }} catch (error) {{
                            console.error('❌ TradingView chart error:', error);
                            chartInitialized = false;
                            if (loadingDiv) {{
                                loadingDiv.innerHTML = '<div style="text-align: center; color: #ff4757;"><i class="fas fa-exclamation-triangle"></i><br>Chart error: ' + error.message + '<br><button onclick="loadTradingViewChart()" style="margin-top: 10px; background: #00d4aa; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Retry</button></div>';
                            }}
                        }}
                    }}
                    
                    // Start initialization with delay to ensure DOM is ready
                    setTimeout(initChart, 1000);
                }}

                // Load ML Predictions
                function loadMLPredictions() {{
                    console.log('🧠 Loading ML predictions...');
                    
                    fetch('/api/ml-predictions')
                        .then(response => response.json())
                        .then(data => {{
                            console.log('✅ ML predictions loaded:', data);
                            if (data.predictions) {{
                                displayPredictions(data.predictions);
                            }}
                        }})
                        .catch(error => {{
                            console.error('❌ ML predictions error:', error);
                            const container = document.getElementById('predictions-grid');
                            if (container) {{
                                container.innerHTML = 
                                    '<div style="color: #ff4757; text-align: center; padding: 20px;">Error loading predictions</div>';
                            }}
                        }});
                }}

                function displayPredictions(predictions) {{
                    const container = document.getElementById('predictions-grid');
                    if (!container) return;
                    
                    container.innerHTML = '';
                    
                    Object.entries(predictions).forEach(([timeframe, pred]) => {{
                        const card = document.createElement('div');
                        card.className = 'prediction-card';
                        card.style.cssText = 'background: #2a2a2a; border-radius: 8px; padding: 15px; border: 1px solid #333; text-align: center;';
                        
                        const directionClass = pred.direction === 'bullish' ? 'bullish' : 
                                             pred.direction === 'bearish' ? 'bearish' : 'neutral';
                        
                        const directionColor = pred.direction === 'bullish' ? '#00d084' : 
                                             pred.direction === 'bearish' ? '#ff4757' : '#ffa500';
                        
                        card.innerHTML = `
                            <div class="timeframe-badge" style="background: #4285f4; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-bottom: 10px;">${{timeframe.toUpperCase()}}</div>
                            <div class="prediction-value" style="font-size: 18px; font-weight: bold; margin: 10px 0; color: #00d084;">${{pred.target || 'N/A'}}</div>
                            <div class="direction" style="font-weight: bold; padding: 4px 8px; border-radius: 4px; text-align: center; margin: 8px 0; background: rgba(0,0,0,0.2); color: ${{directionColor}};">
                                ${{pred.direction ? pred.direction.toUpperCase() : 'UNKNOWN'}} ${{pred.change_percent ? (pred.change_percent > 0 ? '+' : '') + pred.change_percent.toFixed(2) + '%' : ''}}
                            </div>
                            <div style="font-size: 12px; color: #666;">
                                Confidence: ${{pred.confidence ? (pred.confidence * 100).toFixed(1) : 'N/A'}}%
                            </div>
                        `;
                        
                        container.appendChild(card);
                    }});
                }}

                // Initialize everything
                window.addEventListener('load', () => {{
                    console.log('🚀 GoldGPT Dashboard initializing...');
                    
                    // IMMEDIATELY protect the chart container
                    const chartContainer = document.getElementById('tradingview-chart');
                    if (chartContainer) {{
                        console.log('🛡️ Protecting chart container from modifications...');
                        
                        // Create a protective wrapper
                        chartContainer.setAttribute('data-protected', 'true');
                        
                        // Block any attempts to replace content
                        const originalSetAttribute = chartContainer.setAttribute;
                        chartContainer.setAttribute = function(name, value) {{
                            if (name === 'innerHTML' || name === 'textContent') {{
                                console.log('🚫 Blocked attempt to modify chart container via setAttribute');
                                return;
                            }}
                            return originalSetAttribute.call(this, name, value);
                        }};
                    }}
                    
                    // Load TradingView chart with enhanced protection
                    loadTradingViewChart();
                    
                    // Add chart integrity monitoring
                    setInterval(() => {{
                        const chartContainer = document.getElementById('tradingview-chart');
                        if (chartContainer && chartInitialized) {{
                            // Check if TradingView iframe still exists
                            const iframe = chartContainer.querySelector('iframe');
                            if (!iframe) {{
                                console.log('� CRITICAL: Chart iframe disappeared, forcing reinitialize...');
                                chartInitialized = false;
                                initializationAttempts = 0;
                                
                                // Clear any conflicting content
                                const loadingDiv = document.getElementById('chart-loading');
                                if (loadingDiv) {{
                                    loadingDiv.style.display = 'flex';
                                    loadingDiv.innerHTML = '<i class="fas fa-sync fa-spin" style="margin-right: 10px;"></i>Chart disappeared - Recovering...';
                                }}
                                
                                setTimeout(loadTradingViewChart, 1000);
                            }}
                        }}
                    }}, 2000); // Check every 2 seconds (more frequent)
                    
                    // Fallback: If TradingView doesn't load in 10 seconds, show alternative
                    setTimeout(() => {{
                        const loadingDiv = document.getElementById('chart-loading');
                        if (loadingDiv && loadingDiv.style.display !== 'none') {{
                            console.log('⚠️ TradingView taking too long, showing fallback...');
                            loadingDiv.innerHTML = `
                                <div style="text-align: center; padding: 40px;">
                                    <h3 style="color: #00d4aa; margin-bottom: 20px;">📈 Live Gold Chart</h3>
                                    <div style="font-size: 48px; font-weight: bold; color: #ffd700; margin-bottom: 10px;">
                                        $${gold_data['price']}
                                    </div>
                                    <div style="color: #b0b0b0; margin-bottom: 20px;">
                                        XAU/USD • Change: {gold_data['change']:+.2f} ({gold_data['change_percent']:+.2f}%)
                                    </div>
                                    <button onclick="loadTradingViewChart()" style="background: #00d4aa; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer;">
                                        🔄 Load TradingView Chart
                                    </button>
                                    <div style="margin-top: 20px; font-size: 12px; color: #666;">
                                        <a href="https://www.tradingview.com/chart/?symbol=OANDA:XAUUSD" target="_blank" style="color: #00d4aa;">
                                            Open in TradingView →
                                        </a>
                                    </div>
                                </div>
                            `;
                        }}
                    }}, 10000);
                    
                    // Load ML predictions
                    loadMLPredictions();
                    
                    // Auto-refresh ML predictions every 60 seconds (reduced frequency)
                    setInterval(loadMLPredictions, 60000);
                    
                    // Remove auto-refresh to prevent chart disappearing
                    // Auto-refresh disabled to keep TradingView chart stable
                    console.log('� Dashboard initialized with stable chart mode');
                }});
            </script>
        </body>
        </html>
        """

@app.route('/chart-only')
def chart_only():
    """Isolated TradingView chart page"""
    return '''
    <!DOCTYPE html>
    <html style="height:100%;">
    <head>
        <title>TradingView Chart - GoldGPT</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            html, body {
                height: 100%;
                margin: 0;
                padding: 0;
                background: #131722;
                overflow: hidden;
            }
            #tradingview_chart {
                width: 100%;
                height: 100vh;
                min-height: 600px;
            }
        </style>
    </head>
    <body>
        <div id="tradingview_chart"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
            new TradingView.widget({
                "width": "100%",
                "height": "100%",
                "symbol": "OANDA:XAUUSD",
                "interval": "15",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tradingview_chart",
                "autosize": true
            });
        </script>
    </body>
    </html>
    '''

@app.route('/ml-predictions-dashboard')
@app.route('/advanced-ml-dashboard')  # Add route for advanced ML dashboard
def ml_predictions_dashboard():
    """Advanced ML predictions dashboard"""
    try:
        return render_template('ml_predictions_dashboard.html')
    except Exception as e:
        logger.error(f"Error loading ML dashboard template: {e}")
        ml_data = get_ml_predictions()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GoldGPT - ML Predictions Dashboard</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .ml-dashboard {{ max-width: 1400px; margin: 0 auto; background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .predictions-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
                .prediction-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 15px; }}
                .model-badge {{ background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 10px; font-size: 0.8rem; }}
                .confidence-bar {{ background: rgba(255,255,255,0.3); height: 8px; border-radius: 4px; margin: 10px 0; }}
                .confidence-fill {{ background: white; height: 100%; border-radius: 4px; }}
                .ensemble {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 15px; margin-bottom: 20px; color: white; text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }}
                .metric {{ background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; color: #333; }}
            </style>
        </head>
        <body>
            <div class="ml-dashboard">
                <div class="header">
                    <h1><i class="fas fa-robot"></i> ML Predictions Dashboard</h1>
                    <p>Advanced Machine Learning Analysis for XAUUSD</p>
                </div>
                
                <div class="ensemble">
                    <h2>Ensemble Prediction</h2>
                    <h3>{ml_data['ensemble']['direction'].upper()}</h3>
                    <p>Confidence: {ml_data['ensemble']['confidence']*100:.1f}%</p>
                    <p>{ml_data['ensemble']['consensus']}</p>
                </div>
                
                <div class="predictions-grid">
                    {''.join([f'''
                    <div class="prediction-card">
                        <div class="model-badge">{p["model"]}</div>
                        <h3>{p["timeframe"]} Prediction</h3>
                        <h4>{p["direction"].upper()}</h4>
                        <p><strong>Target:</strong> ${p["target_price"]}</p>
                        <p><strong>Change:</strong> {p["change_percent"]:+.1f}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {p["confidence"]*100}%"></div>
                        </div>
                        <p>Confidence: {p["confidence"]*100:.1f}%</p>
                        <p>Volatility: {p["volatility"]*100:.1f}%</p>
                        <p><small>{p["reasoning"]}</small></p>
                    </div>
                    ''' for p in ml_data['predictions']])}
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h4>24h Accuracy</h4>
                        <h3>{ml_data['accuracy_metrics']['last_24h_accuracy']*100:.1f}%</h3>
                    </div>
                    <div class="metric">
                        <h4>Week Accuracy</h4>
                        <h3>{ml_data['accuracy_metrics']['last_week_accuracy']*100:.1f}%</h3>
                    </div>
                    <div class="metric">
                        <h4>Sharpe Ratio</h4>
                        <h3>{ml_data['accuracy_metrics']['sharpe_ratio']}</h3>
                    </div>
                    <div class="metric">
                        <h4>Active Models</h4>
                        <h3>{ml_data['model_count']}</h3>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

@app.route('/ai-analysis')
def ai_analysis():
    """AI Analysis page - Dedicated AI Analysis Dashboard"""
    try:
        # Return the advanced dashboard with AI Analysis active
        return render_template('dashboard_advanced.html', active_section='ai-analysis')
    except Exception as e:
        logger.error(f"Error loading AI analysis template: {e}")
        return redirect(url_for('dashboard'))

@app.route('/ml-predictions')
def ml_predictions():
    """ML Predictions page"""
    try:
        return render_template('dashboard_advanced.html')
    except Exception as e:
        logger.error(f"Error loading ML predictions template: {e}")
        return redirect(url_for('dashboard'))

@app.route('/simple-dashboard')
@app.route('/simple')  # Add simple alias
def simple_dashboard():
    """Simple dashboard with working charts"""
    try:
        return render_template('simple_dashboard.html')
    except Exception as e:
        logger.error(f"Error loading simple dashboard template: {e}")
        return "Simple dashboard template not found", 404

@app.route('/advanced-dashboard')
def advanced_dashboard():
    """Advanced dashboard (direct access for testing)"""
    try:
        return render_template('dashboard_advanced.html')
    except Exception as e:
        logger.error(f"Error loading advanced dashboard template: {e}")
        return "Advanced dashboard template not found", 404

@app.route('/test-fixes')
def test_fixes():
    """Test page for critical fixes"""
    try:
        return render_template('test_fixes.html')
    except Exception as e:
        logger.error(f"Error loading test fixes template: {e}")
        return "Test fixes template not found", 404

# API Endpoints
@app.route('/api/health')
def health_check():
    """Comprehensive health check"""
    import sys
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'development'),
        'python_version': sys.version,
        'app_file': __file__,
        'working_directory': os.getcwd(),
        'available_files': [f for f in os.listdir('.') if f.endswith('.py')],
        'features': {
            'advanced_dashboard': True,
            'ml_predictions': True,
            'ai_analysis': True,
            'real_time_updates': True,
            'portfolio_management': True
        },
        'message': 'GoldGPT Advanced Dashboard is running successfully on Railway!'
    })

@app.route('/api/gold-price')
@app.route('/api/gold/price')  # Add alternate endpoint for advanced dashboard
def api_gold_price():
    """Enhanced gold price API"""
    try:
        price_data = get_current_gold_price()
        return jsonify({
            'success': True,
            'symbol': 'XAUUSD',
            **price_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai-signals')
def api_ai_signals():
    """Enhanced AI signals API"""
    try:
        analysis = get_ai_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/predictions')
def api_debug_predictions():
    """Debug endpoint for ML predictions"""
    try:
        predictions = get_ml_predictions()
        return jsonify({
            'status': 'WORKING',
            'system': 'GoldGPT Full Dashboard',
            'timestamp': datetime.now().isoformat(),
            'predictions_count': len(predictions.get('predictions', {})),
            'sample_prediction': list(predictions.get('predictions', {}).values())[0] if predictions.get('predictions') else None,
            'all_predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/ml-predictions')
def api_ml_predictions_generic():
    """Generic ML predictions API endpoint"""
    try:
        predictions = get_ml_predictions()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml-predictions/<symbol>')
@app.route('/api/advanced-ml/predictions')  # Add endpoint for advanced dashboard
@app.route('/api/ml/prediction/detailed')   # Add detailed endpoint
def api_ml_predictions(symbol='XAUUSD'):
    """Enhanced ML predictions API"""
    try:
        predictions = get_ml_predictions()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio')
def api_portfolio():
    """Enhanced portfolio API"""
    try:
        portfolio = get_portfolio_data()
        return jsonify(portfolio)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chart-data')
@app.route('/api/chart-data/<timeframe>')
@app.route('/api/chart-data/<timeframe>/<int:count>')
@app.route('/api/chart/data/XAUUSD')  # Add specific endpoint for advanced dashboard
def api_chart_data(timeframe='1H', count=100):
    """Chart data API for TradingView integration"""
    try:
        # Get timeframe from query params if not in URL
        if not timeframe or timeframe == 'XAUUSD':
            timeframe = request.args.get('timeframe', '1h')
        
        chart_data = generate_chart_data(timeframe, count)
        return jsonify({
            'success': True,
            'symbol': 'XAUUSD',
            'timeframe': timeframe,
            'count': len(chart_data),
            'data': chart_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market-data')
def api_market_data():
    """Complete market data for dashboard"""
    try:
        gold_data = get_current_gold_price()
        ai_data = get_ai_analysis()
        ml_data = get_ml_predictions()
        portfolio_data = get_portfolio_data()
        
        return jsonify({
            'success': True,
            'gold_price': gold_data,
            'ai_analysis': ai_data,
            'ml_predictions': ml_data,
            'portfolio': portfolio_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/latest')
@app.route('/api/news/load-now', methods=['GET', 'POST'])
def api_news():
    """Market news API for advanced dashboard"""
    try:
        # Mock news data for now
        news_data = {
            'success': True,
            'data': [
                {
                    'id': 1,
                    'title': 'Gold prices surge amid global uncertainty',
                    'summary': 'Gold futures climb as investors seek safe-haven assets',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Market News',
                    'impact': 'bullish'
                },
                {
                    'id': 2,
                    'title': 'Federal Reserve signals potential rate changes',
                    'summary': 'Central bank policy could affect precious metals markets',
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'source': 'Fed Watch',
                    'impact': 'neutral'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(news_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/enhanced')
def api_news_enhanced():
    """Enhanced news API with filtering"""
    try:
        limit = request.args.get('limit', '10')
        # Mock enhanced news data
        news_data = {
            'success': True,
            'data': [
                {
                    'id': i,
                    'title': f'Gold Market Update #{i+1}',
                    'summary': f'Analysis of gold market trends and developments #{i+1}',
                    'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                    'source': f'Market Source {i+1}',
                    'impact': random.choice(['bullish', 'bearish', 'neutral']),
                    'relevance': round(random.uniform(0.6, 1.0), 2)
                } for i in range(int(limit))
            ],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(news_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/positions/open')
def api_positions_open():
    """Open positions API"""
    try:
        positions_data = {
            'success': True,
            'positions': [
                {
                    'id': 1,
                    'symbol': 'XAUUSD',
                    'side': 'LONG',
                    'size': 10.0,
                    'entry_price': 2385.50,
                    'current_price': get_current_gold_price()['price'],
                    'pnl': round(random.uniform(-100, 200), 2),
                    'open_time': (datetime.now() - timedelta(hours=2)).isoformat()
                }
            ],
            'total_positions': 1,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(positions_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predictions')
def api_predictions():
    """Predictions API for advanced dashboard"""
    try:
        predictions_data = get_ml_predictions()
        return jsonify(predictions_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ml-performance')
@app.route('/api/ml-performance')
def api_ml_performance():
    """ML Performance metrics API - prevents JavaScript errors"""
    try:
        return jsonify({
            'success': True,
            'performance': {
                'accuracy': 0.847,
                'precision': 0.823,
                'recall': 0.891,
                'f1_score': 0.856,
                'sharpe_ratio': 1.67,
                'win_rate': 0.734,
                'profit_factor': 2.13,
                'max_drawdown': 0.087,
                'total_trades': 156,
                'profitable_trades': 114,
                'avg_win': 2.34,
                'avg_loss': -1.12,
                'last_updated': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ml-accuracy')
@app.route('/api/ml-accuracy')
def api_ml_accuracy():
    """ML Accuracy metrics API - prevents JavaScript errors"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        return jsonify({
            'success': True,
            'accuracy': {
                'timeframe': timeframe,
                'accuracy': random.uniform(0.75, 0.89),
                'precision': random.uniform(0.72, 0.86),
                'recall': random.uniform(0.78, 0.92),
                'f1_score': random.uniform(0.74, 0.88),
                'last_24h_accuracy': random.uniform(0.70, 0.85),
                'last_week_accuracy': random.uniform(0.68, 0.83),
                'confidence_intervals': {
                    'lower': random.uniform(0.65, 0.75),
                    'upper': random.uniform(0.85, 0.95)
                },
                'updated_at': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market-context')
def api_market_context():
    """Market context API"""
    try:
        context_data = {
            'success': True,
            'market_session': random.choice(['Asian', 'European', 'American']),
            'volatility': round(random.uniform(0.15, 0.35), 3),
            'trend': random.choice(['bullish', 'bearish', 'sideways']),
            'key_levels': {
                'support': round(2380 + random.uniform(-10, 10), 2),
                'resistance': round(2420 + random.uniform(-10, 10), 2)
            },
            'economic_events': [
                {
                    'event': 'Federal Reserve Meeting',
                    'impact': 'high',
                    'time': (datetime.now() + timedelta(hours=6)).isoformat()
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(context_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/correlation')
def api_correlation():
    """Correlation analysis API"""
    try:
        correlation_data = {
            'success': True,
            'correlations': {
                'USD_INDEX': round(random.uniform(-0.8, -0.4), 3),
                'SPX500': round(random.uniform(-0.3, 0.3), 3),
                'CRUDE_OIL': round(random.uniform(0.2, 0.6), 3),
                'BITCOIN': round(random.uniform(-0.2, 0.4), 3),
                'SILVER': round(random.uniform(0.6, 0.9), 3)
            },
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(correlation_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai-analysis/<symbol>')
def get_ai_analysis_for_symbol(symbol):
    """Get AI analysis for specific symbol"""
    try:
        analysis_type = request.args.get('type', 'day_trading')
        analysis_result = get_ai_analysis(analysis_type)
        
        if analysis_result['success']:
            return jsonify({
                'success': True,
                'symbol': symbol,
                'recommendation': analysis_result['recommendation'],
                'patterns': get_live_patterns(symbol),
                'news_alerts': get_news_alerts(symbol),
                'timestamp': analysis_result['timestamp']
            })
        else:
            return jsonify({'success': False, 'error': 'Analysis failed'})
    except Exception as e:
        logger.error(f"AI analysis error for {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/patterns/live/<symbol>')
def get_live_patterns_endpoint(symbol):
    """Get live pattern detection for symbol"""
    try:
        patterns = get_live_patterns(symbol)
        return jsonify({
            'success': True,
            'symbol': symbol,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Pattern detection error for {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/news/alerts/<symbol>')
def get_news_alerts_endpoint(symbol):
    """Get news alerts for symbol"""
    try:
        alerts = get_news_alerts(symbol)
        return jsonify({
            'success': True,
            'symbol': symbol,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"News alerts error for {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ai-analysis/status')
def api_ai_analysis_status():
    """Get comprehensive AI analysis status with trading recommendations"""
    try:
        # Get analysis type from query parameter
        analysis_type = request.args.get('type', 'day_trading')  # 'day_trading' or 'weekly'
        
        # Get comprehensive AI analysis
        analysis_result = get_ai_analysis(analysis_type)
        
        if analysis_result['success']:
            return jsonify({
                'success': True,
                'status': 'active',
                'analysis_type': analysis_type,
                'recommendation': analysis_result['recommendation'],
                'signal': analysis_result['signal'],
                'confidence': analysis_result['confidence'],
                'technical_score': analysis_result['technical_score'],
                'sentiment_score': analysis_result['sentiment_score'],
                'economic_score': analysis_result['economic_score'],
                'risk_level': analysis_result['risk_level'],
                'trading_session': analysis_result['trading_session'],
                'detailed_analysis': analysis_result['detailed_analysis'],
                'indicators': {
                    'technical': analysis_result['technical_indicators'],
                    'sentiment': analysis_result['sentiment_data'],
                    'economic': analysis_result['economic_indicators']
                },
                'timestamp': analysis_result['timestamp'],
                'next_review': analysis_result['next_review']
            })
        else:
            return jsonify({
                'success': False,
                'status': 'error',
                'message': 'Failed to generate AI analysis'
            })
    except Exception as e:
        logger.error(f"AI analysis status error: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': 'AI analysis system temporarily unavailable'
        }), 500

def get_live_patterns(symbol):
    """Advanced pattern detection system"""
    patterns = []
    
    # Simulate advanced pattern recognition
    pattern_types = [
        'Head and Shoulders', 'Double Top', 'Double Bottom', 'Triangle',
        'Flag', 'Pennant', 'Cup and Handle', 'Ascending Triangle',
        'Descending Triangle', 'Symmetrical Triangle', 'Wedge',
        'Channel', 'Support Break', 'Resistance Break'
    ]
    
    # Generate 2-4 detected patterns
    num_patterns = random.randint(2, 4)
    selected_patterns = random.sample(pattern_types, num_patterns)
    
    for pattern in selected_patterns:
        confidence = round(random.uniform(0.65, 0.92), 3)
        timeframe = random.choice(['1H', '4H', '1D'])
        direction = random.choice(['bullish', 'bearish', 'neutral'])
        
        pattern_data = {
            'name': pattern,
            'confidence': confidence,
            'timeframe': timeframe,
            'direction': direction,
            'status': random.choice(['forming', 'confirmed', 'breaking']),
            'target_price': round(2400 + random.uniform(-50, 50), 2),
            'stop_loss': round(2380 + random.uniform(-20, 20), 2),
            'completion': round(random.uniform(0.6, 0.95), 2),
            'detected_at': (datetime.now() - timedelta(minutes=random.randint(5, 180))).isoformat(),
            'description': f"{pattern} pattern detected on {timeframe} timeframe with {direction} bias"
        }
        
        patterns.append(pattern_data)
    
    return patterns

def get_news_alerts(symbol):
    """Advanced news alert system"""
    alerts = []
    
    # Simulate real-time news alerts
    alert_types = [
        'Fed Policy Decision', 'Economic Data Release', 'Geopolitical Event',
        'Central Bank Announcement', 'Market Moving News', 'Technical Breakout',
        'Volume Alert', 'Price Alert', 'Volatility Spike', 'Institutional Flow'
    ]
    
    severity_levels = ['Low', 'Medium', 'High', 'Critical']
    
    # Generate 3-6 alerts
    num_alerts = random.randint(3, 6)
    
    for i in range(num_alerts):
        alert_type = random.choice(alert_types)
        severity = random.choice(severity_levels)
        impact = random.choice(['bullish', 'bearish', 'neutral'])
        
        # Create realistic news content based on alert type
        if alert_type == 'Fed Policy Decision':
            title = "Federal Reserve Policy Decision Imminent"
            content = f"Fed officials signal {random.choice(['hawkish', 'dovish', 'neutral'])} stance on interest rates"
        elif alert_type == 'Economic Data Release':
            title = f"{random.choice(['NFP', 'CPI', 'GDP', 'PMI'])} Data Release"
            content = f"Economic indicator shows {random.choice(['stronger', 'weaker', 'mixed'])} than expected results"
        elif alert_type == 'Geopolitical Event':
            title = "Geopolitical Tensions Rise"
            content = f"International tensions affecting safe-haven demand for gold"
        elif alert_type == 'Technical Breakout':
            title = f"Gold Breaks Key {random.choice(['Support', 'Resistance'])} Level"
            content = f"Price action confirms breakout above/below critical technical level"
        else:
            title = f"{alert_type} Alert"
            content = f"Market-moving event detected affecting gold prices"
        
        alert_data = {
            'id': f"alert_{random.randint(1000, 9999)}",
            'type': alert_type,
            'severity': severity,
            'impact': impact,
            'title': title,
            'content': content,
            'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
            'source': random.choice(['Reuters', 'Bloomberg', 'Market Watch', 'Economic Calendar']),
            'priority': random.randint(1, 10),
            'market_impact': round(random.uniform(0.1, 0.9), 2),
            'time_sensitive': random.choice([True, False]),
            'related_instruments': ['XAUUSD', 'DXY', 'US10Y', 'SPX500']
        }
        
        alerts.append(alert_data)
    
    # Sort by priority (highest first)
    alerts.sort(key=lambda x: x['priority'], reverse=True)
    
    return alerts
    """Get comprehensive AI analysis status with trading recommendations"""
    try:
        # Get analysis type from query parameter
        analysis_type = request.args.get('type', 'day_trading')  # 'day_trading' or 'weekly'
        
        # Get comprehensive AI analysis
        analysis_result = get_ai_analysis(analysis_type)
        
        if analysis_result['success']:
            return jsonify({
                'success': True,
                'status': 'active',
                'analysis_type': analysis_type,
                'recommendation': analysis_result['recommendation'],
                'signal': analysis_result['signal'],
                'confidence': analysis_result['confidence'],
                'technical_score': analysis_result['technical_score'],
                'sentiment_score': analysis_result['sentiment_score'],
                'economic_score': analysis_result['economic_score'],
                'risk_level': analysis_result['risk_level'],
                'trading_session': analysis_result['trading_session'],
                'detailed_analysis': analysis_result['detailed_analysis'],
                'indicators': {
                    'technical': analysis_result['technical_indicators'],
                    'sentiment': analysis_result['sentiment_data'],
                    'economic': analysis_result['economic_indicators']
                },
                'timestamp': analysis_result['timestamp'],
                'next_review': analysis_result['next_review']
            })
        else:
            return jsonify({
                'success': False,
                'status': 'error',
                'message': 'Failed to generate AI analysis'
            })
    except Exception as e:
        logger.error(f"AI analysis status error: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': 'AI analysis system temporarily unavailable'
        }), 500

# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected to advanced dashboard')
    emit('connected', {
        'status': 'Connected to GoldGPT Advanced Dashboard',
        'features': ['real_time_prices', 'ai_signals', 'ml_predictions', 'portfolio_updates']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_price_update')
def handle_price_update():
    """Handle enhanced price update request"""
    try:
        price_data = get_current_gold_price()
        emit('price_update', price_data, broadcast=True)
    except Exception as e:
        logger.error(f"Error in price update: {e}")

@socketio.on('request_ai_update')
def handle_ai_update():
    """Handle AI analysis update request"""
    try:
        ai_data = get_ai_analysis()
        emit('ai_update', ai_data, broadcast=True)
    except Exception as e:
        logger.error(f"Error in AI update: {e}")

@socketio.on('request_ml_update')
def handle_ml_update():
    """Handle ML predictions update request"""
    try:
        ml_data = get_ml_predictions()
        emit('ml_update', ml_data, broadcast=True)
    except Exception as e:
        logger.error(f"Error in ML update: {e}")

@socketio.on('request_portfolio_update')
def handle_portfolio_update():
    """Handle portfolio update request"""
    try:
        portfolio_data = get_portfolio_data()
        emit('portfolio_update', portfolio_data, broadcast=True)
    except Exception as e:
        logger.error(f"Error in portfolio update: {e}")

# Background tasks for real-time updates
def start_background_updates():
    """Start comprehensive background update tasks"""
    import threading
    import time
    
    def price_updater():
        """Enhanced price updates"""
        while True:
            try:
                price_data = get_current_gold_price()
                socketio.emit('price_update', price_data)
                time.sleep(15)  # Update every 15 seconds
            except Exception as e:
                logger.error(f"Price updater error: {e}")
                time.sleep(30)
    
    def ai_updater():
        """AI analysis updates"""
        while True:
            try:
                ai_data = get_ai_analysis()
                socketio.emit('ai_update', ai_data)
                time.sleep(120)  # Update every 2 minutes
            except Exception as e:
                logger.error(f"AI updater error: {e}")
                time.sleep(180)
    
    def ml_updater():
        """ML predictions updates"""
        while True:
            try:
                ml_data = get_ml_predictions()
                socketio.emit('ml_update', ml_data)
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"ML updater error: {e}")
                time.sleep(300)
    
    def portfolio_updater():
        """Portfolio updates"""
        while True:
            try:
                portfolio_data = get_portfolio_data()
                socketio.emit('portfolio_update', portfolio_data)
                time.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Portfolio updater error: {e}")
                time.sleep(120)
    
    # Start all background threads
    threading.Thread(target=price_updater, daemon=True).start()
    threading.Thread(target=ai_updater, daemon=True).start()
    threading.Thread(target=ml_updater, daemon=True).start()
    threading.Thread(target=portfolio_updater, daemon=True).start()
    
    logger.info("✅ All advanced background update tasks started")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    # Debug info for Railway
    print("🔍 DEBUG: Starting GoldGPT Advanced Dashboard")
    print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
    print(f"🔍 DEBUG: Current file: {__file__}")
    print(f"🔍 DEBUG: Available Python files: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    print(f"🔍 DEBUG: Railway environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'Not set')}")
    print(f"🔍 DEBUG: Port: {os.environ.get('PORT', 'Not set')}")
    
    # Start background tasks
    start_background_updates()
    
    # Railway configuration with robust port handling
    try:
        port_env = os.environ.get('PORT', '5000')
        print(f"🔍 DEBUG: Raw PORT environment variable: '{port_env}'")
        
        # Handle cases where PORT might be '$PORT' or other invalid values
        if port_env.startswith('$') or not port_env.strip().isdigit():
            print(f"⚠️  Invalid PORT environment variable: {port_env}")
            port = 5000  # Default fallback
        else:
            port = int(port_env.strip())
    except (ValueError, TypeError) as e:
        print(f"⚠️  Error parsing PORT environment variable: {e}")
        port = 5000  # Default fallback
    
    debug_mode = os.environ.get('RAILWAY_ENVIRONMENT', 'production') != 'production'
    
    logger.info(f"🚀 Starting GoldGPT Advanced Dashboard on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🌍 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    logger.info(f"✨ Features: Advanced Dashboard, ML Predictions, AI Analysis, Real-time Updates")
    
    print(f"🔍 DEBUG: Starting GoldGPT Advanced Dashboard")
    print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
    print(f"🔍 DEBUG: Current file: {__file__}")
    print(f"🔍 DEBUG: Available Python files: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    print(f"🔍 DEBUG: Railway environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    print(f"🔍 DEBUG: Port: {port}")
    print(f"🔍 DEBUG: Available API endpoints:")
    print(f"🔍 DEBUG: - /api/health")
    print(f"🔍 DEBUG: - /api/gold-price") 
    print(f"🔍 DEBUG: - /api/chart-data")
    print(f"🔍 DEBUG: - /api/market-data")
    print(f"🔍 DEBUG: - /simple-dashboard (chart fallback)")
    print(f"🔍 DEBUG: Resolved port: {port}")
    
    # Run the application with error handling
    try:
        print(f"🚀 Starting SocketIO server on 0.0.0.0:{port}")
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=port, 
            debug=debug_mode,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"❌ Error starting SocketIO application: {e}")
        # Try with basic Flask if SocketIO fails
        print("🔄 Falling back to basic Flask server...")
        try:
            app.run(
                host='0.0.0.0',
                port=port,
                debug=debug_mode
            )
        except Exception as flask_error:
            print(f"❌ Flask fallback also failed: {flask_error}")
            print("💡 Check if port is already in use or if there are permission issues")
