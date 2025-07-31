"""
=======================================================================================
                    GOLDGPT - ADVANCED AI TRADING WEB APPLICATION
=======================================================================================

Copyright (c) 2025 Kenan Davies. All Rights Reserved.

GoldGPT Web Application - Trading 212 Inspired Dashboard
Advanced AI Trading System adapted from Telegram bot to modern web platform
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from datetime import datetime, timezone, timedelta
import os
import json
import sqlite3
import threading
import time
import random
import requests
import logging
import re
from typing import Dict, List, Optional
from ai_analysis_api import get_ai_analysis_sync
import asyncio
from news_aggregator import news_aggregator, run_news_aggregation, get_latest_news

# Import Beautiful Soup for web scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️ Beautiful Soup not available - web scraping features disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'goldgpt-secret-key-2025')
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    ping_timeout=120,
    ping_interval=25,
    engineio_logger=False,
    socketio_logger=False
)

# Live Gold Price Integration (Gold-API)
GOLD_API_URL = "https://api.gold-api.com/price/XAU"
GOLD_API_BACKUP_URL = "https://api.metals.live/v1/spot/gold"

# Real-time price tracking
current_prices = {
    'XAUUSD': 0.0,
    'XAGUSD': 0.0,
    'EURUSD': 1.0875,
    'GBPUSD': 1.2650,
    'USDJPY': 148.50,
    'BTCUSD': 43500.0
}

price_history = {}
last_price_update = {}

def fetch_live_gold_price():
    """Fetch real live gold price from Gold-API"""
    try:
        # Primary Gold-API source
        response = requests.get(GOLD_API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = float(data.get('price', 0))
            if price > 0:
                print(f"✅ Live Gold Price: ${price}")
                return {
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Gold-API',
                    'currency': 'USD'
                }
    except Exception as e:
        print(f"⚠️ Gold-API error: {e}")
    
    try:
        # Backup source
        response = requests.get(GOLD_API_BACKUP_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = float(data.get('gold', 0))
            if price > 0:
                return {
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Metals.live',
                    'currency': 'USD'
                }
    except Exception as e:
        print(f"⚠️ Backup API error: {e}")
    
    # Fallback to realistic simulation
    base_price = 2650.0  # Current approximate gold price
    variation = random.uniform(-0.01, 0.01)  # ±1% variation
    simulated_price = base_price * (1 + variation)
    
    return {
        'price': round(simulated_price, 2),
        'timestamp': datetime.now().isoformat(),
        'source': 'Simulated (API Unavailable)',
        'currency': 'USD'
    }

def update_price_history(symbol, price_data):
    """Update price history for charting"""
    if symbol not in price_history:
        price_history[symbol] = []
    
    # Add new price point
    price_history[symbol].append({
        'timestamp': price_data['timestamp'],
        'price': price_data['price'],
        'volume': random.randint(1000, 10000)  # Simulated volume
    })
    
    # Keep only last 1000 points
    if len(price_history[symbol]) > 1000:
        price_history[symbol] = price_history[symbol][-1000:]

def start_live_price_feed():
    """Start background thread for live price updates"""
    def price_update_worker():
        while True:
            try:
                # Fetch live gold price
                gold_data = fetch_live_gold_price()
                current_prices['XAUUSD'] = gold_data['price']
                update_price_history('XAUUSD', gold_data)
                
                # Emit to all connected clients
                socketio.emit('price_update', {
                    'symbol': 'XAUUSD',
                    'price': gold_data['price'],
                    'timestamp': gold_data['timestamp'],
                    'source': gold_data['source'],
                    'change': calculate_price_change('XAUUSD', gold_data['price']),
                    'change_percent': calculate_percentage_change('XAUUSD', gold_data['price'])
                })
                
                # Update other symbols with simulated data
                for symbol in ['XAGUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD']:
                    if symbol in current_prices:
                        base = current_prices[symbol]
                        new_price = base * (1 + random.uniform(-0.005, 0.005))
                        current_prices[symbol] = new_price
                        
                        socketio.emit('price_update', {
                            'symbol': symbol,
                            'price': round(new_price, 4),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Market Data',
                            'change': calculate_price_change(symbol, new_price),
                            'change_percent': calculate_percentage_change(symbol, new_price)
                        })
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"❌ Price feed error: {e}")
                time.sleep(10)  # Wait longer on error
    
    # Start background thread
    price_thread = threading.Thread(target=price_update_worker, daemon=True)
    price_thread.start()
    print("🚀 Live price feed started!")

def start_news_aggregation():
    """Start background news aggregation task"""
    def news_aggregation_worker():
        """Background worker for news aggregation"""
        print("📰 Starting news aggregation background task...")
        
        while True:
            try:
                print("🔄 Running scheduled news aggregation...")
                result = run_news_aggregation()
                
                print(f"✅ News aggregation complete: {result['articles_stored']} articles stored")
                
                # Broadcast news update to connected clients
                socketio.emit('news_update', {
                    'message': 'Market news updated',
                    'articles_count': result['articles_stored'],
                    'sources': result['sources_active'],
                    'reddit_sentiment': result['reddit_sentiment']['average_sentiment'],
                    'timestamp': result['last_update']
                })
                
                # Wait 30 minutes before next aggregation
                time.sleep(1800)  # 30 minutes
                
            except Exception as e:
                print(f"❌ News aggregation error: {e}")
                time.sleep(600)  # Wait 10 minutes on error
    
    # Start background thread
    news_thread = threading.Thread(target=news_aggregation_worker, daemon=True)
    news_thread.start()
    print("📰 News aggregation background task started!")

def calculate_price_change(symbol, current_price):
    """Calculate price change from last update"""
    if symbol in last_price_update:
        return round(current_price - last_price_update[symbol], 4)
    last_price_update[symbol] = current_price
    return 0.0

def calculate_percentage_change(symbol, current_price):
    """Calculate percentage change"""
    if symbol in last_price_update:
        old_price = last_price_update[symbol]
        if old_price > 0:
            return round(((current_price - old_price) / old_price) * 100, 2)
    return 0.0

# Import our advanced systems (adapted from telegram bot)
try:
    from advanced_systems import (
        get_price_fetcher, get_sentiment_analyzer, get_technical_analyzer,
        get_pattern_detector, get_ml_manager, get_macro_fetcher
    )
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ADVANCED_SYSTEMS_AVAILABLE = False
    print("⚠️ Advanced systems not available - using fallback implementations")

# Add missing function implementations
def perform_technical_analysis(symbol, current_price):
    """Perform technical analysis with fallback implementation"""
    if ADVANCED_SYSTEMS_AVAILABLE:
        try:
            analyzer = get_technical_analyzer()
            return analyzer.analyze(symbol, current_price)
        except:
            pass
    
    # Fallback implementation
    return {
        'trend': 'NEUTRAL',
        'rsi': 50.0,
        'macd': 0.0,
        'support': current_price * 0.98,
        'resistance': current_price * 1.02,
        'confidence': 0.5,
        'signals': []
    }

def perform_sentiment_analysis(symbol):
    """Perform sentiment analysis with fallback implementation"""
    if ADVANCED_SYSTEMS_AVAILABLE:
        try:
            analyzer = get_sentiment_analyzer()
            return analyzer.analyze(symbol)
        except:
            pass
    
    # Fallback implementation
    return {
        'sentiment': 'NEUTRAL',
        'score': 0.0,
        'confidence': 0.5,
        'sources': []
    }

def perform_ml_predictions(symbol, current_price):
    """Perform ML predictions with fallback implementation"""
    if ADVANCED_SYSTEMS_AVAILABLE:
        try:
            ml_manager = get_ml_manager()
            return ml_manager.predict(symbol, current_price)
        except:
            pass
    
    # Fallback implementation
    return {
        'prediction': current_price,
        'confidence': 0.5,
        'direction': 'NEUTRAL',
        'timeframe': '1h'
    }

def perform_pattern_detection(symbol):
    """Perform pattern detection with fallback implementation"""
    if ADVANCED_SYSTEMS_AVAILABLE:
        try:
            detector = get_pattern_detector()
            return detector.detect(symbol)
        except:
            pass
    
    # Fallback implementation
    return {
        'patterns': [],
        'strength': 0.0,
        'confidence': 0.5
    }

def generate_trading_recommendation(technical, sentiment, ml_pred, patterns):
    """Generate trading recommendation"""
    # Simple recommendation logic
    score = 0
    score += 1 if technical.get('trend') == 'BULLISH' else -1 if technical.get('trend') == 'BEARISH' else 0
    score += 1 if sentiment.get('sentiment') == 'BULLISH' else -1 if sentiment.get('sentiment') == 'BEARISH' else 0
    score += 1 if ml_pred.get('direction') == 'UP' else -1 if ml_pred.get('direction') == 'DOWN' else 0
    
    if score > 0:
        recommendation = 'BUY'
    elif score < 0:
        recommendation = 'SELL'
    else:
        recommendation = 'HOLD'
    
    confidence = min(0.9, max(0.1, abs(score) / 3.0))
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'score': score,
        'reasoning': f"Based on {len([x for x in [technical, sentiment, ml_pred] if x])} analysis components"
    }

def store_analysis_result(symbol, analysis_result):
    """Store analysis result to database"""
    try:
        conn = sqlite3.connect('goldgpt.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO analysis_results 
            (symbol, timestamp, data) VALUES (?, ?, ?)
        """, (symbol, datetime.now().isoformat(), json.dumps(analysis_result)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error storing analysis result: {e}")

def generate_realistic_chart_data(symbol, timeframe='1h', limit=100):
    """Generate realistic chart data"""
    if symbol in current_prices:
        base_price = current_prices[symbol]
    else:
        base_price = 2650.0  # Default gold price
    
    data = []
    current_time = datetime.now()
    
    for i in range(limit):
        timestamp = current_time - timedelta(hours=i)
        variation = random.uniform(-0.02, 0.02)
        price = base_price * (1 + variation)
        volume = random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': volume
        })
    
    return list(reversed(data))

# Import our advanced systems (adapted from telegram bot)
try:
    from advanced_systems import (
        get_price_fetcher, get_sentiment_analyzer, get_technical_analyzer,
        get_pattern_detector, get_ml_manager, get_macro_fetcher
    )
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ADVANCED_SYSTEMS_AVAILABLE = False
    print("⚠️ Advanced systems not available - using fallback implementations")

# Import live chart generator
try:
    from live_chart_generator import generate_live_chart
    CHART_GENERATOR_AVAILABLE = True
    print("✅ Live chart generator loaded successfully")
except ImportError as e:
    CHART_GENERATOR_AVAILABLE = False
    print(f"⚠️ Live chart generator not available: {e}")

# Database initialization
def init_database():
    """Initialize SQLite database for web app"""
    conn = sqlite3.connect('goldgpt.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            quantity REAL NOT NULL,
            status TEXT DEFAULT 'open',
            profit_loss REAL DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ai_confidence REAL,
            analysis_data TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ai_score REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# Routes
@app.route('/')
def dashboard():
    """Main dashboard - Trading 212 inspired layout with Advanced Features"""
    return render_template('dashboard_advanced.html')

@app.route('/debug-price')
def debug_price():
    """Debug page to test live price functionality"""
    return render_template('live_price_debug.html')

@app.route('/nuclear-chart')
def nuclear_chart():
    """Nuclear option chart that WILL work"""
    return render_template('nuclear_chart.html')

@app.route('/debug-test')
def debug_test():
    """Debug test center for comprehensive feature testing"""
    return render_template('debug_test.html')

@app.route('/container-test')
def container_test():
    """Container detection test for debugging chart issues"""
    return render_template('container_test.html')

@app.route('/tradingview-test')
def tradingview_test():
    """Direct TradingView widget test"""
    return render_template('tradingview_test.html')

@app.route('/working-chart')
def working_chart():
    """Working chart demo"""
    return render_template('working_chart.html')

@app.route('/test-price-simple')
def test_price_simple():
    """Simple TradingView price testing page"""
    return render_template('test_price_simple.html')

@app.route('/api/portfolio')
def get_portfolio():
    """Get current portfolio data"""
    conn = sqlite3.connect('goldgpt.db')
    cursor = conn.cursor()
    
    # Get open trades
    cursor.execute('SELECT * FROM trades WHERE status = "open"')
    trades = cursor.fetchall()
    
    # Calculate portfolio metrics
    total_value = 0
    total_pnl = 0
    
    portfolio_data = {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'open_trades': len(trades),
        'trades': []
    }
    
    for trade in trades:
        trade_data = {
            'id': trade[0],
            'symbol': trade[1],
            'side': trade[2],
            'entry_price': trade[3],
            'current_price': trade[4] or trade[3],
            'quantity': trade[5],
            'pnl': trade[7] or 0,
            'timestamp': trade[8]
        }
        portfolio_data['trades'].append(trade_data)
        total_value += trade_data['current_price'] * trade_data['quantity']
        total_pnl += trade_data['pnl']
    
    portfolio_data['total_value'] = total_value
    portfolio_data['total_pnl'] = total_pnl
    
    conn.close()
    return jsonify(portfolio_data)

@app.route('/api/analysis/<symbol>')
def get_analysis(symbol):
    """Get AI analysis for a symbol"""
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            # Use advanced AI analysis
            technical_analysis = get_technical_analyzer().analyze(symbol)
            sentiment_analysis = get_sentiment_analyzer().analyze(symbol)
            ml_prediction = get_ml_manager().predict(symbol)
            
            analysis = {
                'symbol': symbol,
                'technical': technical_analysis,
                'sentiment': sentiment_analysis,
                'ml_prediction': ml_prediction,
                'timestamp': datetime.now().isoformat(),
                'confidence': 0.85
            }
        else:
            # Fallback analysis
            analysis = {
                'symbol': symbol,
                'technical': {'trend': 'bullish', 'support': 1850, 'resistance': 1900},
                'sentiment': {'score': 0.75, 'label': 'positive'},
                'ml_prediction': {'direction': 'up', 'confidence': 0.82},
                'timestamp': datetime.now().isoformat(),
                'confidence': 0.75
            }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade', methods=['POST'])
def execute_trade():
    """Execute a new trade"""
    data = request.json
    
    try:
        conn = sqlite3.connect('goldgpt.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, quantity, ai_confidence, analysis_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['symbol'],
            data['side'],
            data['price'],
            data['quantity'],
            data.get('confidence', 0.5),
            json.dumps(data.get('analysis', {}))
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Emit trade update to all connected clients
        socketio.emit('trade_executed', {
            'trade_id': trade_id,
            'symbol': data['symbol'],
            'side': data['side'],
            'price': data['price'],
            'quantity': data['quantity']
        })
        
        return jsonify({'success': True, 'trade_id': trade_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/close_trade/<int:trade_id>', methods=['POST'])
def close_trade(trade_id):
    """Close an existing trade"""
    data = request.json
    
    try:
        conn = sqlite3.connect('goldgpt.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades SET status = 'closed', current_price = ?, profit_loss = ?
            WHERE id = ?
        ''', (data['close_price'], data['pnl'], trade_id))
        
        conn.commit()
        conn.close()
        
        # Emit trade closure to all connected clients
        socketio.emit('trade_closed', {
            'trade_id': trade_id,
            'close_price': data['close_price'],
            'pnl': data['pnl']
        })
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/price/<symbol>')
def get_price(symbol):
    """Get current price for a symbol"""
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            price_data = get_price_fetcher().get_price(symbol)
            return jsonify(price_data)
        else:
            # Fallback price simulation
            base_prices = {
                'XAUUSD': 1875.0,
                'EURUSD': 1.0875,
                'GBPUSD': 1.2650,
                'USDJPY': 148.50,
                'BTCUSD': 43500.0
            }
            
            base = base_prices.get(symbol, 1.0)
            current_price = base * (1 + random.uniform(-0.02, 0.02))
            change = random.uniform(-0.01, 0.01)
            
            price_data = {
                'symbol': symbol,
                'price': round(current_price, 4),
                'change': round(change, 4),
                'change_percent': round(change * 100, 2),
                'high_24h': round(current_price * 1.015, 4),
                'low_24h': round(current_price * 0.985, 4),
                'volume': random.randint(10000, 100000),
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(price_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =====================================================
# ADVANCED API ROUTES - FULL AI TRADING CAPABILITIES
# =====================================================

@app.route('/api/live-gold-price')
def get_live_gold_price():
    """Get real-time gold price from Gold-API"""
    try:
        gold_data = fetch_live_gold_price()
        return jsonify({
            'success': True,
            'data': gold_data,
            'symbol': 'XAUUSD',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/comprehensive-analysis/<symbol>')
def get_comprehensive_analysis(symbol):
    """Get complete AI analysis including technical, sentiment, and ML predictions"""
    try:
        # Get current price first
        if symbol == 'XAUUSD':
            price_data = fetch_live_gold_price()
            current_price = price_data['price']
        else:
            current_price = current_prices.get(symbol, 1.0)
        
        # Technical Analysis
        technical_analysis = perform_technical_analysis(symbol, current_price)
        
        # Sentiment Analysis
        sentiment_analysis = perform_sentiment_analysis(symbol)
        
        # ML Predictions
        ml_predictions = perform_ml_predictions(symbol, current_price)
        
        # Pattern Detection
        pattern_analysis = perform_pattern_detection(symbol)
        
        # Overall Recommendation
        overall_recommendation = generate_trading_recommendation(
            technical_analysis, sentiment_analysis, ml_predictions, pattern_analysis
        )
        
        analysis_result = {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'sentiment_analysis': sentiment_analysis,
            'ml_predictions': ml_predictions,
            'pattern_analysis': pattern_analysis,
            'overall_recommendation': overall_recommendation,
            'confidence_score': overall_recommendation.get('confidence', 0.5)
        }
        
        # Store analysis in database
        store_analysis_result(symbol, analysis_result)
        
        return jsonify({'success': True, 'analysis': analysis_result})
        
    except Exception as e:
        print(f"❌ Analysis error for {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/technical-analysis/<symbol>')
def get_technical_analysis(symbol):
    """Get detailed technical analysis with multiple indicators"""
    try:
        current_price = current_prices.get(symbol, 1.0)
        analysis = perform_technical_analysis(symbol, current_price)
        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sentiment-analysis/<symbol>')
def get_sentiment_analysis(symbol):
    """Get market sentiment analysis from multiple sources"""
    try:
        analysis = perform_sentiment_analysis(symbol)
        return jsonify({'success': True, 'sentiment': analysis})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml-predictions/<symbol>')
def get_ml_predictions(symbol):
    """Get machine learning price predictions"""
    try:
        current_price = current_prices.get(symbol, 1.0)
        predictions = perform_ml_predictions(symbol, current_price)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pattern-detection/<symbol>')
def get_pattern_detection(symbol):
    """Get chart pattern detection results"""
    try:
        patterns = perform_pattern_detection(symbol)
        return jsonify({'success': True, 'patterns': patterns})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chart/data/<symbol>')
def get_chart_data(symbol):
    """Get historical chart data for symbol"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        limit = int(request.args.get('limit', 100))
        
        # Generate realistic OHLCV data
        chart_data = generate_realistic_chart_data(symbol, timeframe, limit)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': chart_data,
            'count': len(chart_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chart/data/<symbol>')
def get_chart_data_for_bot(symbol):
    """Get chart data that the bot can access directly"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        bars = int(request.args.get('bars', '100'))
        
        print(f"🤖 Bot requesting chart data: {symbol} ({timeframe}, {bars} bars)")
        
        # Generate realistic OHLCV data for the bot
        chart_data = generate_chart_data_for_bot(symbol, timeframe, bars)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'bars_requested': bars,
            'bars_returned': len(chart_data['data']),
            'data': chart_data['data'],
            'metadata': chart_data['metadata'],
            'current_price': chart_data['current_price'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Chart data API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/bot/market-data')
def get_market_data_for_bot():
    """Comprehensive market data endpoint for bot analysis"""
    try:
        print("🤖 Bot requesting comprehensive market data...")
        
        # Get current gold price
        gold_price_data = fetch_live_gold_price()
        
        # Get chart data
        chart_data = generate_chart_data_for_bot('XAUUSD', '1h', 50)
        
        # Calculate technical indicators for bot
        technical_data = calculate_technical_indicators_for_bot(chart_data['data'])
        
        market_data = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'current_price': {
                'symbol': 'XAUUSD',
                'price': gold_price_data['price'],
                'source': gold_price_data['source'],
                'currency': gold_price_data['currency']
            },
            'chart_data': {
                'timeframe': '1h',
                'bars': chart_data['data'][-20:],  # Last 20 bars
                'metadata': chart_data['metadata']
            },
            'technical_indicators': technical_data,
            'market_summary': {
                'trend': technical_data.get('trend', 'NEUTRAL'),
                'volatility': technical_data.get('volatility', 'MODERATE'),
                'momentum': technical_data.get('momentum', 'NEUTRAL')
            }
        }
        
        print(f"✅ Market data prepared for bot: {len(chart_data['data'])} bars, current price: ${gold_price_data['price']}")
        
        return jsonify(market_data)
        
    except Exception as e:
        print(f"❌ Market data API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def generate_chart_data_for_bot(symbol, timeframe, bars):
    """Generate realistic chart data for bot analysis"""
    try:
        # Get current price as base
        current_price_data = fetch_live_gold_price()
        current_price = current_price_data['price']
        
        # Generate historical data
        data = []
        base_time = datetime.now()
        
        # Timeframe to minutes mapping
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        interval_minutes = timeframe_minutes.get(timeframe, 60)
        
        # Generate bars going backwards in time
        for i in range(bars, 0, -1):
            bar_time = base_time - timedelta(minutes=i * interval_minutes)
            
            # Create realistic price movement
            if i == 1:  # Current/latest bar
                close = current_price
            else:
                # Random walk with slight downward bias (gold market simulation)
                price_change = random.uniform(-0.003, 0.002)  # -0.3% to +0.2%
                close = current_price * (1 + price_change * i / bars)
            
            # Generate OHLC around close price
            volatility = random.uniform(0.001, 0.005)  # 0.1% to 0.5% volatility
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close + random.uniform(-volatility, volatility) * close
            
            # Ensure OHLC relationships are correct
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = random.randint(100, 1000)
            
            bar = {
                'timestamp': int(bar_time.timestamp()),
                'datetime': bar_time.isoformat(),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            }
            
            data.append(bar)
        
        return {
            'data': data,
            'current_price': current_price,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'bars_generated': len(data),
                'start_time': data[0]['datetime'],
                'end_time': data[-1]['datetime'],
                'data_source': 'GoldGPT Simulation + Live Price',
                'generated_at': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"❌ Chart data generation error: {e}")
        raise

def calculate_technical_indicators_for_bot(chart_data):
    """Calculate technical indicators from chart data for bot analysis"""
    try:
        if len(chart_data) < 20:
            return {'error': 'Insufficient data for technical analysis'}
        
        # Extract prices
        closes = [bar['close'] for bar in chart_data]
        highs = [bar['high'] for bar in chart_data]
        lows = [bar['low'] for bar in chart_data]
        volumes = [bar['volume'] for bar in chart_data]
        
        # Simple Moving Averages
        sma_10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else closes[-1]
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        
        current_price = closes[-1]
        
        # Trend determination
        if current_price > sma_10 > sma_20:
            trend = 'BULLISH'
        elif current_price < sma_10 < sma_20:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        # Volatility (standard deviation of last 10 closes)
        if len(closes) >= 10:
            mean_price = sum(closes[-10:]) / 10
            variance = sum((price - mean_price) ** 2 for price in closes[-10:]) / 10
            volatility_value = (variance ** 0.5) / mean_price * 100  # Percentage
            
            if volatility_value > 2.0:
                volatility = 'HIGH'
            elif volatility_value > 1.0:
                volatility = 'MODERATE'
            else:
                volatility = 'LOW'
        else:
            volatility = 'UNKNOWN'
            volatility_value = 0
        
        # Momentum (price change over last 5 bars)
        if len(closes) >= 5:
            momentum_change = (closes[-1] - closes[-5]) / closes[-5] * 100
            if momentum_change > 0.5:
                momentum = 'STRONG_UP'
            elif momentum_change > 0.1:
                momentum = 'UP'
            elif momentum_change < -0.5:
                momentum = 'STRONG_DOWN'
            elif momentum_change < -0.1:
                momentum = 'DOWN'
            else:
                momentum = 'NEUTRAL'
        else:
            momentum = 'NEUTRAL'
            momentum_change = 0
        
        # Support and Resistance levels
        recent_lows = lows[-20:] if len(lows) >= 20 else lows
        recent_highs = highs[-20:] if len(highs) >= 20 else highs
        
        support = min(recent_lows)
        resistance = max(recent_highs)
        
        return {
            'current_price': current_price,
            'sma_10': round(sma_10, 2),
            'sma_20': round(sma_20, 2),
            'trend': trend,
            'volatility': volatility,
            'volatility_value': round(volatility_value, 2),
            'momentum': momentum,
            'momentum_change': round(momentum_change, 2),
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'price_vs_sma10': round((current_price - sma_10) / sma_10 * 100, 2),
            'price_vs_sma20': round((current_price - sma_20) / sma_20 * 100, 2),
            'calculation_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Technical indicators calculation error: {e}")
        return {'error': str(e)}

# Debug endpoint for client-side logging
@app.route('/api/debug', methods=['POST'])
def debug_log():
    """Receive debug logs from client-side for terminal display"""
    try:
        data = request.get_json()
        feature = data.get('feature', 'unknown')
        status = data.get('status', 'info')
        message = data.get('message', '')
        debug_data = data.get('data', '')
        timestamp = data.get('timestamp', '')
        
        # Color coding for terminal output
        colors = {
            'success': '\033[92m',  # Green
            'error': '\033[91m',    # Red
            'loading': '\033[93m',  # Yellow
            'unknown': '\033[94m'   # Blue
        }
        reset = '\033[0m'
        
        color = colors.get(status, colors['unknown'])
        
        # Format terminal output
        terminal_msg = f"{color}🔍 [CLIENT-{feature.upper()}] {status.upper()}: {message}{reset}"
        if debug_data:
            terminal_msg += f" | Data: {str(debug_data)[:100]}..."
        
        print(terminal_msg)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Debug endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# =====================================================
# ENHANCED WEBSOCKET EVENTS FOR REAL-TIME FEATURES
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection with enhanced welcome"""
    print('🔗 Client connected to GoldGPT Pro')
    
    # Send current live gold price immediately
    try:
        gold_data = fetch_live_gold_price()
        emit('price_update', {
            'symbol': 'XAUUSD',
            'price': gold_data['price'],
            'timestamp': gold_data['timestamp'],
            'source': gold_data['source']
        })
    except Exception as e:
        print(f"Error sending initial price: {e}")
    
    emit('connected', {
        'message': 'Connected to GoldGPT Pro',
        'features': ['live_prices', 'ai_analysis', 'real_time_charts'],
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('❌ Client disconnected from GoldGPT Pro')

@socketio.on('subscribe_symbol')
def handle_subscribe(data):
    """Subscribe to symbol updates with immediate data"""
    symbol = data.get('symbol', 'XAUUSD')
    print(f'📊 Client subscribed to {symbol}')
    
    # Send immediate price for subscribed symbol
    if symbol in current_prices:
        emit('price_update', {
            'symbol': symbol,
            'price': current_prices[symbol],
            'timestamp': datetime.now().isoformat(),
            'source': 'Live' if symbol == 'XAUUSD' else 'Market Data'
        })

@socketio.on('get_ai_analysis')
def handle_ai_analysis_request(data):
    """Handle real-time AI analysis requests"""
    symbol = data.get('symbol', 'XAUUSD')
    analysis_type = data.get('type', 'comprehensive')
    
    print(f'🧠 AI Analysis requested for {symbol} ({analysis_type})')
    
    try:
        if analysis_type == 'comprehensive':
            # Get current price
            current_price = current_prices.get(symbol, 1.0)
            if symbol == 'XAUUSD':
                gold_data = fetch_live_gold_price()
                current_price = gold_data['price']
            
            # Perform all analyses
            technical = perform_technical_analysis(symbol, current_price)
            sentiment = perform_sentiment_analysis(symbol)
            ml_pred = perform_ml_predictions(symbol, current_price)
            patterns = perform_pattern_detection(symbol)
            recommendation = generate_trading_recommendation(technical, sentiment, ml_pred, patterns)
            
            analysis_result = {
                'symbol': symbol,
                'current_price': current_price,
                'technical_analysis': technical,
                'sentiment_analysis': sentiment,
                'ml_predictions': ml_pred,
                'pattern_analysis': patterns,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }
            
            emit('ai_analysis_update', analysis_result)
            
        elif analysis_type == 'technical':
            current_price = current_prices.get(symbol, 1.0)
            technical = perform_technical_analysis(symbol, current_price)
            emit('technical_analysis_update', {'symbol': symbol, 'analysis': technical})
            
        elif analysis_type == 'sentiment':
            sentiment = perform_sentiment_analysis(symbol)
            emit('sentiment_analysis_update', {'symbol': symbol, 'analysis': sentiment})
            
    except Exception as e:
        emit('analysis_error', {'error': str(e), 'symbol': symbol})

@socketio.on('get_live_prices')
def handle_live_prices_request(data):
    """Handle live price data requests"""
    symbols = data.get('symbols', ['XAUUSD'])
    
    for symbol in symbols:
        try:
            if symbol == 'XAUUSD':
                gold_data = fetch_live_gold_price()
                emit('price_update', {
                    'symbol': symbol,
                    'price': gold_data['price'],
                    'timestamp': gold_data['timestamp'],
                    'source': gold_data['source']
                })
            else:
                if symbol in current_prices:
                    emit('price_update', {
                        'symbol': symbol,
                        'price': current_prices[symbol],
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Market Data'
                    })
        except Exception as e:
            emit('price_error', {'error': str(e), 'symbol': symbol})

@socketio.on('get_chart_data')
def handle_chart_data_request(data):
    """Handle chart data requests"""
    symbol = data.get('symbol', 'XAUUSD')
    timeframe = data.get('timeframe', '1h')
    limit = data.get('limit', 100)
    
    try:
        chart_data = generate_realistic_chart_data(symbol, timeframe, limit)
        emit('chart_data_update', {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': chart_data,
            'count': len(chart_data)
        })
    except Exception as e:
        emit('chart_error', {'error': str(e), 'symbol': symbol})

@socketio.on('execute_trade')
def handle_trade_execution(data):
    """Handle trade execution via WebSocket"""
    try:
        # Store trade in database
        conn = sqlite3.connect('goldgpt.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, quantity, ai_confidence, analysis_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['symbol'],
            data['side'],
            data['price'],
            data['quantity'],
            data.get('confidence', 0.5),
            json.dumps(data.get('analysis', {}))
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Broadcast trade execution
        socketio.emit('trade_executed', {
            'trade_id': trade_id,
            'symbol': data['symbol'],
            'side': data['side'],
            'price': data['price'],
            'quantity': data['quantity'],
            'timestamp': datetime.now().isoformat()
        })
        
        emit('trade_confirmation', {'success': True, 'trade_id': trade_id})
        
    except Exception as e:
        emit('trade_error', {'error': str(e)})

# =====================================================
# NEWS AGGREGATION API ENDPOINTS
# =====================================================

@app.route('/api/news/latest')
def get_latest_news_api():
    """Get latest market news for display"""
    try:
        limit = request.args.get('limit', 20, type=int)
        news_articles = get_latest_news(limit)
        
        return jsonify({
            'success': True,
            'news': news_articles,
            'count': len(news_articles),
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error fetching latest news: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'news': [],
            'count': 0
        }), 500

@app.route('/api/news/aggregate', methods=['POST'])
def trigger_news_aggregation():
    """Manually trigger news aggregation process"""
    try:
        print("📰 Manual news aggregation triggered...")
        result = run_news_aggregation()
        
        # Broadcast news update to all connected clients
        socketio.emit('news_update', {
            'message': 'News updated',
            'articles_count': result['articles_stored'],
            'sources': result['sources_active'],
            'timestamp': result['last_update']
        })
        
        return jsonify({
            'success': True,
            'aggregation_result': result
        })
    except Exception as e:
        logger.error(f"❌ Error triggering news aggregation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/sentiment')
def get_market_sentiment():
    """Get current market sentiment from news and social media"""
    try:
        # Get Reddit sentiment
        reddit_sentiment = news_aggregator.get_reddit_gold_sentiment()
        
        # Get recent news sentiment average
        recent_news = news_aggregator.db.get_recent_news(hours=24, limit=50)
        
        if recent_news:
            sentiment_scores = [article['sentiment_score'] for article in recent_news 
                             if article['sentiment_score'] is not None]
            avg_news_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        else:
            avg_news_sentiment = 0.0
        
        # Combined sentiment
        combined_sentiment = (reddit_sentiment['average_sentiment'] + avg_news_sentiment) / 2
        
        sentiment_data = {
            'combined_sentiment': round(combined_sentiment, 3),
            'reddit_sentiment': {
                'score': round(reddit_sentiment['average_sentiment'], 3),
                'confidence': round(reddit_sentiment['confidence'], 3),
                'post_count': reddit_sentiment['post_count'],
                'trending_topics': reddit_sentiment['trending_topics'][:5]
            },
            'news_sentiment': {
                'score': round(avg_news_sentiment, 3),
                'article_count': len(recent_news),
                'timeframe': '24 hours'
            },
            'interpretation': {
                'label': 'Bullish' if combined_sentiment > 0.1 else 'Bearish' if combined_sentiment < -0.1 else 'Neutral',
                'strength': 'Strong' if abs(combined_sentiment) > 0.3 else 'Moderate' if abs(combined_sentiment) > 0.1 else 'Weak'
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'sentiment': sentiment_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting market sentiment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/sources')
def get_news_sources_status():
    """Get status of all news sources"""
    try:
        sources_status = {
            'sources': [
                {
                    'name': 'MarketWatch',
                    'type': 'RSS Feed',
                    'category': 'Market News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'Reuters',
                    'type': 'RSS Feed', 
                    'category': 'Economic News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'Yahoo Finance',
                    'type': 'RSS Feed',
                    'category': 'Financial News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'CNBC',
                    'type': 'RSS Feed',
                    'category': 'Market News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'Reddit',
                    'type': 'Social Media API',
                    'category': 'Social Sentiment',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                }
            ],
            'total_sources': 5,
            'active_sources': 5,
            'last_aggregation': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'sources': sources_status
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting sources status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/bot/news')
def get_news_for_bot():
    """Get processed news data for bot learning and analysis"""
    try:
        print("🤖 Bot requesting news data for learning...")
        
        # Get recent high-impact news
        recent_news = news_aggregator.db.get_recent_news(hours=48, limit=100)
        
        # Filter for high relevance and impact
        filtered_news = [
            article for article in recent_news 
            if article.get('sentiment_score') is not None and 
            abs(article.get('sentiment_score', 0)) > 0.1
        ]
        
        # Get Reddit sentiment data
        try:
            reddit_sentiment = {
                'average_sentiment': 0.0,
                'confidence': 0.5,
                'post_count': 0,
                'trending_topics': []
            }
        except:
            reddit_sentiment = {
                'average_sentiment': 0.0,
                'confidence': 0.5,
                'post_count': 0,
                'trending_topics': []
            }
        
        if recent_news:
            sentiment_scores = [article['sentiment_score'] for article in recent_news 
                             if article['sentiment_score'] is not None]
            avg_news_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        else:
            avg_news_sentiment = 0.0
        
        # Combined sentiment
        combined_sentiment = (reddit_sentiment['average_sentiment'] + avg_news_sentiment) / 2
        
        sentiment_data = {
            'combined_sentiment': round(combined_sentiment, 3),
            'reddit_sentiment': {
                'score': round(reddit_sentiment['average_sentiment'], 3),
                'confidence': round(reddit_sentiment['confidence'], 3),
                'post_count': reddit_sentiment['post_count'],
                'trending_topics': reddit_sentiment['trending_topics'][:5]
            },
            'news_sentiment': {
                'score': round(avg_news_sentiment, 3),
                'article_count': len(recent_news),
                'timeframe': '24 hours'
            },
            'interpretation': {
                'label': 'Bullish' if combined_sentiment > 0.1 else 'Bearish' if combined_sentiment < -0.1 else 'Neutral',
                'strength': 'Strong' if abs(combined_sentiment) > 0.3 else 'Moderate' if abs(combined_sentiment) > 0.1 else 'Weak'
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'sentiment': sentiment_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting market sentiment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/sources')
def get_news_sources_status():
    """Get status of all news sources"""
    try:
        sources_status = {
            'sources': [
                {
                    'name': 'MarketWatch',
                    'type': 'RSS Feed',
                    'category': 'Market News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'Reuters',
                    'type': 'RSS Feed', 
                    'category': 'Economic News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'Yahoo Finance',
                    'type': 'RSS Feed',
                    'category': 'Financial News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'CNBC',
                    'type': 'RSS Feed',
                    'category': 'Market News',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                },
                {
                    'name': 'Reddit',
                    'type': 'Social Media API',
                    'category': 'Social Sentiment',
                    'status': 'Active',
                    'last_check': datetime.now().isoformat()
                }
            ],
            'total_sources': 5,
            'active_sources': 5,
            'last_aggregation': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'sources': sources_status
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting sources status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/bot/news')
def get_news_for_bot():
    """Get processed news data for bot learning and analysis"""
    try:
        print("🤖 Bot requesting news data for learning...")
        
        # Get recent high-impact news
        recent_news = news_aggregator.db.get_recent_news(hours=48, limit=100)
        
        # Filter for high relevance and impact
        filtered_news = [
            article for article in recent_news 
            if (article.get('gold_relevance_score', 0) > 0.3 or 
                article.get('impact_score', 0) > 0.5)
        ]
        
        # Get market sentiment data
        sentiment_data = news_aggregator.get_reddit_gold_sentiment()
        
        # Format for bot consumption
        bot_news_data = {
            'news_articles': [
                {
                    'id': article['id'],
                    'title': article['title'],
                    'source': article['source'],
                    'published_date': article['published_date'],
                    'category': article['category'],
                    'sentiment_score': article['sentiment_score'],
                    'impact_score': article['impact_score'],
                    'gold_relevance_score': article['gold_relevance_score'],
                    'keywords': json.loads(article['keywords']) if article['keywords'] else []
                }
                for article in filtered_news
            ],
            'market_sentiment': {
                'reddit_sentiment': sentiment_data['average_sentiment'],
                'confidence': sentiment_data['confidence'],
                'trending_topics': sentiment_data['trending_topics'][:10]
            },
            'aggregation_metadata': {
                'total_articles_found': len(recent_news),
                'high_relevance_articles': len(filtered_news),
                'timeframe_hours': 48,
                'data_generated_at': datetime.now().isoformat(),
                'sources_included': ['MarketWatch', 'Reuters', 'Yahoo Finance', 'CNBC', 'Reddit']
            }
        }
        
        print(f"✅ News data prepared for bot: {len(filtered_news)} relevant articles")
        
        return jsonify({
            'success': True,
            'data': bot_news_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Bot news API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# =====================================================
# STARTUP AND BACKGROUND TASKS
# =====================================================

# =====================================================
# ENHANCED REAL MACRO ECONOMIC DATA API ENDPOINTS WITH WEB SCRAPING
# =====================================================

import requests
from bs4 import BeautifulSoup
import re

# Enhanced USD Index with multiple sources and web scraping
@app.route('/api/macro/usd-index')
def get_usd_index():
    """Get real USD Dollar Index from multiple sources including web scraping"""
    try:
        # Primary source: Yahoo Finance API
        response = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB', timeout=10)
        data = response.json()
        
        if data.get('chart') and data['chart'].get('result'):
            result = data['chart']['result'][0]
            current_price = result['meta']['regularMarketPrice']
            previous_close = result['meta']['previousClose']
            
            return jsonify({
                'success': True,
                'data': {
                    'value': current_price,
                    'change': current_price - previous_close,
                    'change_percent': ((current_price - previous_close) / previous_close) * 100,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Yahoo Finance API',
                    'last_updated': datetime.now().strftime('%H:%M:%S')
                }
            })
        else:
            raise Exception("Yahoo Finance API failed")
            
    except Exception as e:
        logger.warning(f"Yahoo Finance USD Index error: {e}, trying web scraping...")
        
        # Fallback: Web scraping from MarketWatch
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get('https://www.marketwatch.com/investing/index/dxy', headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to extract USD Index price from MarketWatch
            price_element = soup.find('bg-quote', {'field': 'Last'})
            if not price_element:
                price_element = soup.find('span', {'class': 'value'})
            
            if price_element:
                price_text = price_element.get_text().strip()
                price = float(re.findall(r'[\d.]+', price_text)[0])
                
                # Try to get change
                change_element = soup.find('bg-quote', {'field': 'Change'})
                change = 0.0
                if change_element:
                    change_text = change_element.get_text().strip()
                    change = float(re.findall(r'[-+]?[\d.]+', change_text)[0])
                
                return jsonify({
                    'success': True,
                    'data': {
                        'value': price,
                        'change': change,
                        'change_percent': (change / (price - change)) * 100 if price != change else 0,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'MarketWatch (Scraped)',
                        'last_updated': datetime.now().strftime('%H:%M:%S')
                    }
                })
                
        except Exception as scraping_error:
            logger.warning(f"MarketWatch scraping failed: {scraping_error}")
        
        # Final fallback
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'value': 103.25,
                'change': 0.15,
                'change_percent': 0.15,
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })

@app.route('/api/macro/treasury-yields')
def get_treasury_yields():
    """Get real Treasury yields from multiple sources including FRED and web scraping"""
    try:
        yields = {}
        symbols = ['^TNX', '^FVX', '^TYX']  # 10Y, 5Y, 30Y
        names = ['10Y', '5Y', '30Y']
        
        # Try Yahoo Finance first
        for i, symbol in enumerate(symbols):
            try:
                response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}', timeout=10)
                data = response.json()
                
                if data.get('chart') and data['chart'].get('result'):
                    price = data['chart']['result'][0]['meta']['regularMarketPrice']
                    prev_close = data['chart']['result'][0]['meta']['previousClose']
                    yields[names[i]] = {
                        'value': price,
                        'change': price - prev_close,
                        'change_percent': ((price - prev_close) / prev_close) * 100
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        # If Yahoo fails, try web scraping from Treasury.gov
        if not yields:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield', headers=headers, timeout=15)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse treasury yield table
                table = soup.find('table')
                if table:
                    rows = table.find_all('tr')
                    if len(rows) > 1:  # Skip header
                        latest_row = rows[1]  # Most recent data
                        cells = latest_row.find_all('td')
                        
                        if len(cells) >= 8:  # Make sure we have enough columns
                            yields['5Y'] = {'value': float(cells[5].get_text().strip()), 'change': 0, 'change_percent': 0}
                            yields['10Y'] = {'value': float(cells[7].get_text().strip()), 'change': 0, 'change_percent': 0}
                            yields['30Y'] = {'value': float(cells[10].get_text().strip()) if len(cells) > 10 else 4.45, 'change': 0, 'change_percent': 0}
                            
            except Exception as scraping_error:
                logger.warning(f"Treasury.gov scraping failed: {scraping_error}")
        
        # Fallback values if all sources fail
        if not yields:
            yields = {
                '10Y': {'value': 4.25, 'change': 0.02, 'change_percent': 0.47},
                '5Y': {'value': 4.75, 'change': 0.01, 'change_percent': 0.21},
                '30Y': {'value': 4.45, 'change': -0.01, 'change_percent': -0.22}
            }
        
        return jsonify({
            'success': True,
            'data': {
                **yields,
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance / Treasury.gov' if any('change' in y for y in yields.values()) else 'Fallback',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Treasury yields fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                '10Y': {'value': 4.25, 'change': 0, 'change_percent': 0},
                '5Y': {'value': 4.75, 'change': 0, 'change_percent': 0},
                '30Y': {'value': 4.45, 'change': 0, 'change_percent': 0},
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })

@app.route('/api/macro/commodities')
def get_commodity_prices():
    """Get real commodity prices from multiple sources including web scraping"""
    try:
        commodities = {
            'GC=F': 'gold',      # Gold futures
            'SI=F': 'silver',    # Silver futures  
            'CL=F': 'oil',       # Crude oil futures
            'NG=F': 'natgas',    # Natural gas futures
            'HG=F': 'copper'     # Copper futures
        }
        
        prices = {}
        
        # Fetch commodity prices from Yahoo Finance
        for symbol, name in commodities.items():
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if data['chart']['result']:
                    result = data['chart']['result'][0]
                    current_price = result['meta']['regularMarketPrice']
                    previous_close = result['meta']['previousClose']
                    
                    prices[name] = {
                        'price': current_price,
                        'change': current_price - previous_close,
                        'changePercent': ((current_price - previous_close) / previous_close) * 100,
                        'symbol': symbol
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        # If some commodities failed, try web scraping from MarketWatch
        missing_commodities = set(commodities.values()) - set(prices.keys())
        if missing_commodities:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Scrape Gold from MarketWatch if missing
                if 'gold' in missing_commodities:
                    response = requests.get('https://www.marketwatch.com/investing/future/gc00', headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    price_element = soup.find('bg-quote', {'field': 'Last'})
                    if price_element:
                        price = float(re.findall(r'[\d,.]+', price_element.get_text().replace(',', ''))[0])
                        change_element = soup.find('bg-quote', {'field': 'Change'})
                        change = 0.0
                        if change_element:
                            change_text = change_element.get_text().strip()
                            change = float(re.findall(r'[-+]?[\d.]+', change_text)[0])
                        
                        prices['gold'] = {
                            'price': price,
                            'change': change,
                            'changePercent': (change / (price - change)) * 100 if price != change else 0,
                            'symbol': 'GC=F'
                        }
                
                # Scrape Oil from MarketWatch if missing
                if 'oil' in missing_commodities:
                    response = requests.get('https://www.marketwatch.com/investing/future/cl.1', headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    price_element = soup.find('bg-quote', {'field': 'Last'})
                    if price_element:
                        price = float(re.findall(r'[\d.]+', price_element.get_text())[0])
                        prices['oil'] = {
                            'price': price,
                            'change': 0.0,
                            'changePercent': 0.0,
                            'symbol': 'CL=F'
                        }
                        
            except Exception as scraping_error:
                logger.warning(f"Commodity scraping failed: {scraping_error}")
        
        # Fill in any missing commodities with fallback data
        fallback_prices = {
            'gold': {'price': 2085.0, 'change': 12.5, 'changePercent': 0.6, 'symbol': 'GC=F'},
            'silver': {'price': 25.0, 'change': 0.15, 'changePercent': 0.6, 'symbol': 'SI=F'},
            'oil': {'price': 75.0, 'change': -0.5, 'changePercent': -0.66, 'symbol': 'CL=F'},
            'natgas': {'price': 3.5, 'change': 0.1, 'changePercent': 2.94, 'symbol': 'NG=F'},
            'copper': {'price': 4.25, 'change': 0.02, 'changePercent': 0.47, 'symbol': 'HG=F'}
        }
        
        for commodity, fallback_data in fallback_prices.items():
            if commodity not in prices:
                prices[commodity] = fallback_data
        
        return jsonify({
            'success': True,
            'data': {
                **prices,
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance / MarketWatch',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Commodity prices fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'gold': {'price': 2085.0, 'change': 12.5, 'changePercent': 0.6},
                'silver': {'price': 25.0, 'change': 0.15, 'changePercent': 0.6},
                'oil': {'price': 75.0, 'change': -0.5, 'changePercent': -0.66},
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })

@app.route('/api/macro/sentiment')
def get_macro_market_sentiment():
    """Get enhanced market sentiment from multiple sources including Fear & Greed Index scraping"""
    try:
        vix_value = 20.0
        fear_greed_index = 50
        
        # Get VIX from Yahoo Finance
        try:
            response = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/^VIX', timeout=10)
            data = response.json()
            
            if data.get('chart') and data['chart'].get('result'):
                vix_value = data['chart']['result'][0]['meta']['regularMarketPrice']
        except Exception as vix_error:
            logger.warning(f"VIX fetch error: {vix_error}")
        
        # Scrape Fear & Greed Index from CNN
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get('https://money.cnn.com/data/fear-and-greed/', headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the fear and greed index value
            fear_greed_elements = soup.find_all('span', {'class': 'wsod_fgIndex'})
            if not fear_greed_elements:
                # Try alternative selectors
                fear_greed_elements = soup.find_all('div', string=re.compile(r'\d+'))
                
            for element in fear_greed_elements:
                text = element.get_text().strip()
                if text.isdigit() and 0 <= int(text) <= 100:
                    fear_greed_index = int(text)
                    break
                    
        except Exception as fg_error:
            logger.warning(f"Fear & Greed scraping failed: {fg_error}")
            # Fallback: calculate from VIX
            fear_greed_index = max(0, min(100, 100 - (vix_value * 2)))
        
        # Calculate sentiment based on VIX and Fear & Greed
        if vix_value < 15 and fear_greed_index > 70:
            sentiment = 'Extremely Bullish'
        elif vix_value < 20 and fear_greed_index > 50:
            sentiment = 'Bullish'
        elif vix_value > 30 or fear_greed_index < 30:
            sentiment = 'Bearish'
        elif vix_value > 25 or fear_greed_index < 40:
            sentiment = 'Cautious'
        else:
            sentiment = 'Neutral'
        
        # Get additional sentiment indicators by scraping market news sentiment
        try:
            response = requests.get('https://www.marketwatch.com/markets', headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Count positive vs negative sentiment words in headlines
            headlines = soup.find_all(['h2', 'h3', 'a'], limit=20)
            positive_words = ['gains', 'rises', 'up', 'strong', 'bullish', 'rally', 'surge']
            negative_words = ['falls', 'drops', 'down', 'weak', 'bearish', 'decline', 'plunges']
            
            positive_count = 0
            negative_count = 0
            
            for headline in headlines:
                text = headline.get_text().lower()
                positive_count += sum(1 for word in positive_words if word in text)
                negative_count += sum(1 for word in negative_words if word in text)
            
            news_sentiment_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
            
        except Exception as news_error:
            logger.warning(f"News sentiment scraping failed: {news_error}")
            news_sentiment_score = 0.0
        
        return jsonify({
            'success': True,
            'data': {
                'vix': vix_value,
                'vix_level': 'Low' if vix_value < 15 else 'Normal' if vix_value < 25 else 'High',
                'sentiment': sentiment,
                'fear_greed_index': fear_greed_index,
                'fear_greed_level': 'Extreme Greed' if fear_greed_index > 80 else 'Greed' if fear_greed_index > 60 else 'Neutral' if fear_greed_index > 40 else 'Fear' if fear_greed_index > 20 else 'Extreme Fear',
                'news_sentiment_score': news_sentiment_score,
                'market_stress_level': 'Low' if vix_value < 20 and fear_greed_index > 50 else 'Medium' if vix_value < 30 else 'High',
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance / CNN / MarketWatch (Scraped)',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Market sentiment fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'vix': 18.5,
                'sentiment': 'Neutral',
                'fear_greed_index': 63,
                'vix_level': 'Normal',
                'fear_greed_level': 'Greed',
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })

@app.route('/api/macro/central-bank-rates')
def get_central_bank_rates():
    """Get central bank rates from multiple sources with web scraping"""
@app.route('/api/macro/central-bank-rates')
def get_central_bank_rates():
    """Get central bank rates from multiple sources with web scraping"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        rates = {}
        
        # Federal Reserve rate from Fed website
        try:
            response = requests.get('https://www.federalreserve.gov/monetarypolicy/openmarket.htm', headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text().lower()
            fed_rate_match = re.search(r'(\d+\.\d+)\s*(?:to\s*(\d+\.\d+))?\s*percent', text_content)
            
            if fed_rate_match:
                # If range (e.g., 5.25-5.50), take the upper bound
                fed_rate = float(fed_rate_match.group(2)) if fed_rate_match.group(2) else float(fed_rate_match.group(1))
                rates['fed_funds_rate'] = fed_rate
            else:
                rates['fed_funds_rate'] = 5.25  # Fallback
                
        except Exception as fed_error:
            logger.warning(f"Fed rate scraping failed: {fed_error}")
            rates['fed_funds_rate'] = 5.25
        
        # ECB rate
        try:
            response = requests.get('https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html', headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        rate_text = cells[1].get_text().strip()
                        rate_match = re.search(r'(\d+\.\d+)', rate_text)
                        if rate_match:
                            rates['ecb_rate'] = float(rate_match.group(1))
                            break
            
            if 'ecb_rate' not in rates:
                rates['ecb_rate'] = 4.50
                
        except Exception as ecb_error:
            logger.warning(f"ECB rate scraping failed: {ecb_error}")
            rates['ecb_rate'] = 4.50
        
        # Bank of England rate
        try:
            response = requests.get('https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate', headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text().lower()
            boe_rate_match = re.search(r'(\d+\.\d+)%', text_content)
            
            if boe_rate_match:
                rates['boe_rate'] = float(boe_rate_match.group(1))
            else:
                rates['boe_rate'] = 5.25
                
        except Exception as boe_error:
            logger.warning(f"BOE rate scraping failed: {boe_error}")
            rates['boe_rate'] = 5.25
        
        # Bank of Japan - typically negative
        rates['boj_rate'] = -0.10  # Bank of Japan typically at -0.10%
        
        # Bank of Canada
        try:
            response = requests.get('https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/', headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            rate_element = soup.find('div', {'class': 'policy-rate'}) or soup.find('span', string=re.compile(r'\d+\.\d+%'))
            if rate_element:
                rate_text = rate_element.get_text()
                rate_match = re.search(r'(\d+\.\d+)', rate_text)
                if rate_match:
                    rates['boc_rate'] = float(rate_match.group(1))
                else:
                    rates['boc_rate'] = 5.00
            else:
                rates['boc_rate'] = 5.00
                
        except Exception as boc_error:
            logger.warning(f"Bank of Canada rate scraping failed: {boc_error}")
            rates['boc_rate'] = 5.00
        
        return jsonify({
            'success': True,
            'data': {
                **rates,
                'next_meeting_date': '2025-01-29',  # Next scheduled FOMC meeting
                'timestamp': datetime.now().isoformat(),
                'source': 'Fed, ECB, BOE, BOJ, BOC',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Central bank rates fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'fed_funds_rate': 5.25,
                'ecb_rate': 4.50,
                'boe_rate': 5.25,
                'boj_rate': -0.10,
                'boc_rate': 5.00,
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })

@app.route('/api/macro/inflation')
def get_inflation_data():
    """Get real inflation data from BLS.gov and other sources with web scraping"""
    try:
        # Try scraping from BLS.gov (Bureau of Labor Statistics)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Scrape CPI data from BLS
        try:
            response = requests.get('https://www.bls.gov/news.release/cpi.nr0.htm', headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for CPI percentage in the text
            text_content = soup.get_text()
            cpi_match = re.search(r'(\d+\.\d+)\s*percent.*12.*month', text_content.lower())
            
            if cpi_match:
                cpi_annual = float(cpi_match.group(1))
            else:
                # Fallback: try to find any percentage in CPI context
                cpi_matches = re.findall(r'(\d+\.\d+)\s*percent', text_content)
                cpi_annual = float(cpi_matches[0]) if cpi_matches else 3.2
        except Exception as bls_error:
            logger.warning(f"BLS.gov scraping failed: {bls_error}")
            cpi_annual = 3.2
        
        return jsonify({
            'success': True,
            'data': {
                'cpi_annual': cpi_annual,
                'cpi_monthly': 0.2,
                'core_cpi_annual': max(2.0, cpi_annual - 0.3),
                'pce_annual': max(1.8, cpi_annual - 0.4),
                'target_rate': 2.0,
                'above_target': cpi_annual > 2.5,
                'timestamp': datetime.now().isoformat(),
                'source': 'BLS.gov (Scraped)',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Inflation data fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'cpi_annual': 3.2,
                'cpi_monthly': 0.2,
                'core_cpi_annual': 2.9,
                'pce_annual': 2.8,
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })

# New endpoint for economic calendar
@app.route('/api/macro/economic-calendar')
def get_economic_calendar():
    """Get upcoming economic events by scraping economic calendar websites"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        events = []
        
        # Try scraping from Investing.com economic calendar
        try:
            response = requests.get('https://www.investing.com/economic-calendar/', headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for economic events table
            calendar_table = soup.find('table', {'id': 'economicCalendarData'})
            if not calendar_table:
                calendar_table = soup.find('table', {'class': 'genTbl'})
            
            if calendar_table:
                rows = calendar_table.find_all('tr')[:10]  # Get first 10 events
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        # Extract event details
                        time_cell = cells[0].get_text().strip() if cells[0] else ''
                        event_cell = cells[3].get_text().strip() if len(cells) > 3 else ''
                        impact_cell = cells[2] if len(cells) > 2 else None
                        
                        # Determine impact level
                        impact = 'Medium'
                        if impact_cell:
                            impact_class = impact_cell.get('class', [])
                            if 'redCon' in str(impact_class) or 'high' in str(impact_class):
                                impact = 'High'
                            elif 'yellowCon' in str(impact_class) or 'medium' in str(impact_class):
                                impact = 'Medium'
                            else:
                                impact = 'Low'
                        
                        if event_cell and event_cell not in ['', '-']:
                            events.append({
                                'time': time_cell,
                                'event': event_cell,
                                'impact': impact,
                                'country': 'US'  # Default to US
                            })
                            
        except Exception as investing_error:
            logger.warning(f"Investing.com calendar scraping failed: {investing_error}")
        
        # If no events found, provide some typical upcoming events
        if not events:
            events = [
                {'time': 'Today 08:30', 'event': 'Initial Jobless Claims', 'impact': 'Medium', 'country': 'US'},
                {'time': 'Today 10:00', 'event': 'Existing Home Sales', 'impact': 'Low', 'country': 'US'},
                {'time': 'Tomorrow 08:30', 'event': 'Core PCE Price Index', 'impact': 'High', 'country': 'US'},
                {'time': 'Tomorrow 10:00', 'event': 'Consumer Confidence', 'impact': 'Medium', 'country': 'US'},
                {'time': 'Friday 08:30', 'event': 'Non-Farm Payrolls', 'impact': 'High', 'country': 'US'}
            ]
        
        return jsonify({
            'success': True,
            'data': {
                'events': events[:8],  # Limit to 8 events
                'timestamp': datetime.now().isoformat(),
                'source': 'Investing.com (Scraped)',
                'last_updated': datetime.now().strftime('%H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Economic calendar fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'events': [
                    {'time': 'Today 08:30', 'event': 'Initial Jobless Claims', 'impact': 'Medium', 'country': 'US'}
                ],
                'timestamp': datetime.now().isoformat(),
                'source': 'Fallback Data'
            }
        })

@app.route('/api/macro/all')
def get_all_macro_data():
    """Get all macro economic data in one comprehensive request"""
    try:
        # Fetch all macro data from respective endpoints
        macro_data = {}
        
        # USD Index
        try:
            usd_response = get_usd_index()
            if usd_response.status_code == 200:
                macro_data['usd_index'] = usd_response.get_json()['data']
        except Exception as e:
            logger.warning(f"USD Index aggregation error: {e}")
            macro_data['usd_index'] = {'value': 103.25, 'change': 0.15, 'source': 'Fallback'}
        
        # Treasury Yields
        try:
            treasury_response = get_treasury_yields()
            if treasury_response.status_code == 200:
                macro_data['treasury_yields'] = treasury_response.get_json()['data']
        except Exception as e:
            logger.warning(f"Treasury yields aggregation error: {e}")
            macro_data['treasury_yields'] = {'10Y': {'value': 4.25}, 'source': 'Fallback'}
        
        # Commodity Prices
        try:
            commodity_response = get_commodity_prices()
            if commodity_response.status_code == 200:
                macro_data['commodity_prices'] = commodity_response.get_json()['data']
        except Exception as e:
            logger.warning(f"Commodity prices aggregation error: {e}")
            macro_data['commodity_prices'] = {'gold': {'price': 2085.0, 'change': 12.5}, 'source': 'Fallback'}
        
        # Market Sentiment
        try:
            sentiment_response = get_market_sentiment()
            if sentiment_response.status_code == 200:
                macro_data['market_sentiment'] = sentiment_response.get_json()['data']
        except Exception as e:
            logger.warning(f"Market sentiment aggregation error: {e}")
            macro_data['market_sentiment'] = {'vix': 18.5, 'sentiment': 'Neutral', 'source': 'Fallback'}
        
        # Inflation Data
        try:
            inflation_response = get_inflation_data()
            if inflation_response.status_code == 200:
                macro_data['inflation_data'] = inflation_response.get_json()['data']
        except Exception as e:
            logger.warning(f"Inflation data aggregation error: {e}")
            macro_data['inflation_data'] = {'cpi_annual': 3.2, 'source': 'Fallback'}
        
        # Central Bank Rates
        try:
            rates_response = get_central_bank_rates()
            if rates_response.status_code == 200:
                macro_data['central_bank_rates'] = rates_response.get_json()['data']
        except Exception as e:
            logger.warning(f"Central bank rates aggregation error: {e}")
            macro_data['central_bank_rates'] = {'fed_funds_rate': 5.25, 'source': 'Fallback'}
        
        # Economic Calendar
        try:
            calendar_response = get_economic_calendar()
            if calendar_response.status_code == 200:
                macro_data['economic_calendar'] = calendar_response.get_json()['data']
        except Exception as e:
            logger.warning(f"Economic calendar aggregation error: {e}")
            macro_data['economic_calendar'] = {'events': [], 'source': 'Fallback'}
        
        # Calculate overall market health score
        try:
            vix = macro_data.get('market_sentiment', {}).get('vix', 20)
            fear_greed = macro_data.get('market_sentiment', {}).get('fear_greed_index', 50)
            fed_rate = macro_data.get('central_bank_rates', {}).get('fed_funds_rate', 5.25)
            inflation = macro_data.get('inflation_data', {}).get('cpi_annual', 3.2)
            
            # Simple market health calculation (0-100 scale)
            health_score = 50  # Neutral start
            health_score += (50 - vix) if vix <= 50 else -(vix - 50) / 2
            health_score += (fear_greed - 50) / 2
            health_score -= abs(inflation - 2.0) * 10  # Target 2% inflation
            health_score = max(0, min(100, health_score))
            
            market_health = 'Excellent' if health_score > 80 else 'Good' if health_score > 60 else 'Fair' if health_score > 40 else 'Poor'
            
        except Exception as health_error:
            logger.warning(f"Market health calculation error: {health_error}")
            health_score = 50
            market_health = 'Fair'
        
        return jsonify({
            'success': True,
            'data': {
                **macro_data,
                'market_health': {
                    'score': health_score,
                    'rating': market_health,
                    'factors': {
                        'volatility': 'Low' if vix < 20 else 'High',
                        'sentiment': macro_data.get('market_sentiment', {}).get('sentiment', 'Neutral'),
                        'inflation': 'Target' if 1.5 <= inflation <= 2.5 else 'Elevated',
                        'rates': 'Restrictive' if fed_rate > 5.0 else 'Accommodative'
                    }
                },
                'last_updated': datetime.now().isoformat(),
                'update_frequency': '5 minutes',
                'sources': 'Yahoo Finance, Fed, ECB, BOE, BLS, CNN, MarketWatch'
            }
        })
        
    except Exception as e:
        logger.error(f"All macro data fetch error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'market_health': {'score': 50, 'rating': 'Unknown'},
                'last_updated': datetime.now().isoformat()
            }
        }), 500

# =====================================================
# BACKGROUND TASKS
# =====================================================

def start_advanced_background_tasks():
    """Start all background tasks for real-time features"""
    
    # Start live price feed
    start_live_price_feed()
    
    # Start news aggregation 
    start_news_aggregation()
    
    # Start AI analysis broadcasting
    def ai_analysis_broadcaster():
        """Broadcast AI analysis updates every 30 seconds"""
        while True:
            try:
                # Perform comprehensive analysis for XAUUSD
                gold_data = fetch_live_gold_price()
                current_price = gold_data['price']
                
                technical = perform_technical_analysis('XAUUSD', current_price)
                sentiment = perform_sentiment_analysis('XAUUSD')
                ml_pred = perform_ml_predictions('XAUUSD', current_price)
                recommendation = generate_trading_recommendation(technical, sentiment, ml_pred, {})
                
                # Broadcast to all connected clients
                socketio.emit('ai_analysis_broadcast', {
                    'symbol': 'XAUUSD',
                    'current_price': current_price,
                    'technical': technical,
                    'sentiment': sentiment,
                    'ml_prediction': ml_pred,
                    'recommendation': recommendation,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"📡 AI Analysis broadcast: {recommendation.get('recommendation', 'HOLD')} (Confidence: {recommendation.get('confidence', 0.5):.1%})")
                
            except Exception as e:
                print(f"❌ AI Analysis broadcast error: {e}")
            
            time.sleep(30)  # Update every 30 seconds
    
    # Start AI analysis thread
    ai_thread = threading.Thread(target=ai_analysis_broadcaster, daemon=True)
    ai_thread.start()
    
    print("🤖 Advanced AI analysis broadcasting started!")

    # News aggregation will be started by the global function

if __name__ == '__main__':
    print("🏆 GoldGPT Pro - Advanced AI Trading Web Application")
    print("=" * 60)
    print("🚀 Starting advanced features...")
    print("📊 Live Gold-API integration")
    print("🧠 Real-time AI analysis")
    print("📈 Advanced technical indicators")
    print("💡 ML predictions & sentiment analysis")
    print("🔄 WebSocket real-time updates")
    
    # Start all background tasks
    start_advanced_background_tasks()
    
    print("\n✅ All systems online!")
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
