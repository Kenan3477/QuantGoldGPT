#!/usr/bin/env python3
"""
Simple Flask test for the problematic endpoints
"""
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/api/live-gold-price')
def api_live_gold_price():
    """Test live gold price endpoint"""
    print("📞 /api/live-gold-price called")
    try:
        return jsonify({
            'success': True,
            'symbol': 'XAUUSD',
            'price': 2065.50,
            'change': 15.30,
            'change_percent': 0.75,
            'timestamp': time.time(),
            'test': 'This endpoint is working!'
        })
    except Exception as e:
        print(f"❌ Error in live-gold-price: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/signals/active')
def get_active_signals():
    """Test active signals endpoint"""
    print("📞 /api/signals/active called")
    try:
        return jsonify({
            'success': True,
            'signals': [
                {'id': 1, 'type': 'BUY', 'confidence': 0.85, 'price': 2065.50},
                {'id': 2, 'type': 'SELL', 'confidence': 0.70, 'price': 2070.00}
            ],
            'count': 2,
            'timestamp': time.time(),
            'test': 'This endpoint is working!'
        })
    except Exception as e:
        print(f"❌ Error in signals/active: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/')
def index():
    return '''
    <h1>Flask Test Server</h1>
    <ul>
        <li><a href="/api/health">Health Check</a></li>
        <li><a href="/api/live-gold-price">Live Gold Price</a></li>
        <li><a href="/api/signals/active">Active Signals</a></li>
    </ul>
    '''

if __name__ == '__main__':
    print("🚀 Starting simple Flask test server on port 5000")
    print("🔍 This will help debug the endpoint issues")
    app.run(debug=True, port=5000, host='127.0.0.1')
