"""
Ultra-minimal QuantGold for Railway
No dependencies, no complexity, just working
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>QuantGold Dashboard</h1>
    <p>Status: Running on Railway!</p>
    <p><a href="/health">Health Check</a></p>
    <p><a href="/api/signals/generate">Generate Signal</a></p>
    '''

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "app": "QuantGold"})

@app.route('/api/signals/generate')
def generate_signal():
    import random
    return jsonify({
        "success": True,
        "signal": {
            "type": "BUY" if random.random() > 0.5 else "SELL",
            "price": round(2650 + random.uniform(-50, 50), 2),
            "confidence": round(random.uniform(0.6, 0.9), 2)
        }
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting QuantGold on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
