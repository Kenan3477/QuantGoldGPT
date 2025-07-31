#!/usr/bin/env python3
"""
Simple test to check if Flask is working
"""

from flask import Flask, jsonify
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-key'

@app.route('/')
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GoldGPT Test</title>
    </head>
    <body>
        <h1>GoldGPT Test Page</h1>
        <p>If you can see this, Flask is working!</p>
        <script>
            console.log('JavaScript is working too!');
            document.body.style.background = '#0a0a0a';
            document.body.style.color = '#ffffff';
        </script>
    </body>
    </html>
    """

@app.route('/api/test')
def api_test():
    return jsonify({'status': 'working', 'message': 'API is functional'})

if __name__ == '__main__':
    print("🚀 Starting simple Flask test...")
    app.run(debug=True, host='0.0.0.0', port=5001)
