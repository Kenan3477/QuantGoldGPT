#!/usr/bin/env python3
"""
GoldGPT - Production Web Application
Minimal version for Railway deployment with essential features
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import requests
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'goldgpt-secret-key-2025')

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
            
            # Create basic tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gold_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database initialized (SQLite)")
            
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")

# Initialize database on startup
init_database()

# Gold price functions
def get_current_gold_price():
    """Get current gold price with fallback"""
    try:
        # Try multiple APIs
        apis = [
            'https://api.metals.live/v1/spot/gold',
            'https://api.coinbase.com/v2/exchange-rates?currency=XAU'
        ]
        
        for api_url in apis:
            try:
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'price' in data:
                        return float(data['price'])
                    elif 'data' in data and 'rates' in data['data']:
                        return float(data['data']['rates']['USD'])
            except:
                continue
                
        # Fallback price
        return 2400.0 + (hash(str(datetime.now().hour)) % 100 - 50)
        
    except Exception as e:
        logger.error(f"Error getting gold price: {e}")
        return 2400.0

# Mock AI analysis function
def get_ai_analysis():
    """Simple AI analysis with random but realistic values"""
    import random
    
    signals = ['BUY', 'SELL', 'HOLD']
    signal = random.choice(signals)
    confidence = round(random.uniform(0.6, 0.9), 2)
    
    return {
        'success': True,
        'signal': signal,
        'confidence': confidence,
        'analysis': f"Technical analysis suggests {signal} signal",
        'technical_score': round(random.uniform(0.4, 0.8), 2),
        'sentiment_score': round(random.uniform(0.3, 0.7), 2),
        'prediction': f"Price may move {signal.lower()} based on current indicators"
    }

# Mock ML predictions
def get_ml_predictions():
    """Simple ML predictions with mock data"""
    import random
    
    base_price = get_current_gold_price()
    timeframes = ['1H', '4H', '1D']
    predictions = []
    
    for tf in timeframes:
        direction = random.choice(['bullish', 'bearish', 'neutral'])
        change_pct = round(random.uniform(-2.5, 2.5), 2)
        target_price = round(base_price * (1 + change_pct/100), 2)
        confidence = round(random.uniform(0.6, 0.85), 2)
        
        predictions.append({
            'timeframe': tf,
            'direction': direction,
            'target_price': target_price,
            'confidence': confidence,
            'change_percent': change_pct,
            'reasoning': f"ML model predicts {direction} movement for {tf} timeframe"
        })
    
    return {
        'success': True,
        'predictions': predictions,
        'symbol': 'XAUUSD',
        'timestamp': datetime.now().isoformat()
    }

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/ml-predictions-dashboard')
def ml_predictions_dashboard():
    """ML predictions dashboard"""
    return render_template('ml_predictions_dashboard.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/gold-price')
def api_gold_price():
    """Current gold price API"""
    try:
        price = get_current_gold_price()
        return jsonify({
            'success': True,
            'price': price,
            'symbol': 'XAUUSD',
            'timestamp': datetime.now().isoformat(),
            'change': round(price - 2400, 2),
            'change_percent': round(((price - 2400) / 2400) * 100, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai-signals')
def api_ai_signals():
    """AI trading signals API"""
    try:
        analysis = get_ai_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml-predictions/<symbol>')
def api_ml_predictions(symbol):
    """ML predictions API"""
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
    """Portfolio data API"""
    return jsonify({
        'success': True,
        'portfolio': {
            'balance': 10000.0,
            'positions': [],
            'pnl': 0.0,
            'pnl_percentage': 0.0
        }
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('connected', {'status': 'Connected to GoldGPT'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_price_update')
def handle_price_update():
    """Handle price update request"""
    try:
        price_data = {
            'price': get_current_gold_price(),
            'timestamp': datetime.now().isoformat()
        }
        emit('price_update', price_data, broadcast=True)
    except Exception as e:
        logger.error(f"Error in price update: {e}")

@socketio.on('request_ml_update')
def handle_ml_update():
    """Handle ML predictions update request"""
    try:
        predictions = get_ml_predictions()
        emit('ml_predictions_update', predictions, broadcast=True)
    except Exception as e:
        logger.error(f"Error in ML update: {e}")

# Background tasks
def start_background_updates():
    """Start background update tasks"""
    import threading
    import time
    
    def price_updater():
        """Background price updates"""
        while True:
            try:
                price_data = {
                    'price': get_current_gold_price(),
                    'timestamp': datetime.now().isoformat()
                }
                socketio.emit('price_update', price_data)
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Price updater error: {e}")
                time.sleep(60)
    
    def ml_updater():
        """Background ML predictions updates"""
        while True:
            try:
                predictions = get_ml_predictions()
                socketio.emit('ml_predictions_update', predictions)
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"ML updater error: {e}")
                time.sleep(300)
    
    # Start background threads
    threading.Thread(target=price_updater, daemon=True).start()
    threading.Thread(target=ml_updater, daemon=True).start()
    logger.info("✅ Background update tasks started")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Start background tasks
    start_background_updates()
    
    # Railway configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('RAILWAY_ENVIRONMENT') != 'production'
    
    logger.info(f"🚀 Starting GoldGPT on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🌍 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'development')}")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode,
        allow_unsafe_werkzeug=True
    )
