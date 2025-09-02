#!/usr/bin/env python3
"""
Ultra-Minimal Railway Test - Basic Flask Only with Debug
"""
import os
import sys
from flask import Flask, jsonify

print("🚀 STARTING MINIMAL APP")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Files available: {os.listdir('.')}")
print(f"Environment variables:")
for key, value in os.environ.items():
    if 'PORT' in key or 'RAILWAY' in key or 'HOST' in key:
        print(f"  {key} = {value}")

# Create the most basic Flask app possible
app = Flask(__name__)

@app.route('/')
def health():
    """Basic health check for Railway"""
    print("🎯 Health check endpoint accessed!")
    return jsonify({
        "status": "healthy",
        "service": "goldgpt",
        "message": "Basic server running",
        "port": os.environ.get('PORT', 'not-set')
    })

@app.route('/api/signals/generate')
def basic_signal():
    """Ultra-basic signal generation"""
    print("📡 Signal generation endpoint accessed!")
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
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 Starting server on 0.0.0.0:{port}")
        print("🔄 Starting Flask app...")
        
        # Force threading to True for Railway
        app.run(
            host='0.0.0.0', 
            port=port, 
            debug=False, 
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
