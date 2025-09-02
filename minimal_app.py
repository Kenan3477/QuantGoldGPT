#!/usr/bin/env python3
"""
Ultra-Minimal Railway Test - Basic Flask Only
"""
import os
from flask import Flask, jsonify

# Create the most basic Flask app possible
app = Flask(__name__)

@app.route('/')
def health():
    """Basic health check for Railway"""
    return jsonify({
        "status": "healthy",
        "service": "goldgpt",
        "message": "Basic server running"
    })

@app.route('/api/signals/generate')
def basic_signal():
    """Ultra-basic signal generation"""
    import random
    signal_type = random.choice(["BUY", "SELL"])
    price = round(3500 + random.uniform(-50, 50), 2)
    
    return jsonify({
        "success": True,
        "signal": {
            "signal_type": signal_type,
            "entry_price": price,
            "confidence": 0.7,
            "take_profit": price + (20 if signal_type == "BUY" else -20),
            "stop_loss": price - (10 if signal_type == "BUY" else -10)
        },
        "method": "ultra_basic",
        "timestamp": "2025-09-02T14:30:00Z"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting ultra-minimal server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
