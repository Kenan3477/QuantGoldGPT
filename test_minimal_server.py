#!/usr/bin/env python3
"""
Minimal Flask route test
"""
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/api/live-gold-price')
def api_live_gold_price():
    """Test endpoint"""
    return jsonify({
        'success': True,
        'test': 'This is a test response',
        'timestamp': time.time()
    })

@app.route('/api/signals/active')
def get_active_signals():
    """Test signals endpoint"""
    return jsonify({
        'success': True,
        'signals': ['test signal 1', 'test signal 2'],
        'count': 2
    })

if __name__ == '__main__':
    print("🔍 Starting minimal test server on port 5001")
    app.run(debug=True, port=5001, host='127.0.0.1')
