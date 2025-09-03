#!/usr/bin/env python3
"""
Ultra-Basic HTTP Server for Railway - No Flask Dependencies
"""
import os
import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        print(f"📡 Request received: {path}")
        
        if path == '/':
            # Health check endpoint
            response = {
                "status": "healthy",
                "service": "goldgpt-basic",
                "message": "Basic HTTP server running",
                "port": os.environ.get('PORT', 'unknown')
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/api/signals/generate':
            # Signal generation endpoint
            signal_type = random.choice(["BUY", "SELL"])
            price = round(3500 + random.uniform(-50, 50), 2)
            
            response = {
                "success": True,
                "signal": {
                    "signal_type": signal_type,
                    "entry_price": price,
                    "confidence": 0.7,
                    "take_profit": price + (20 if signal_type == "BUY" else -20),
                    "stop_loss": price - (10 if signal_type == "BUY" else -10)
                },
                "method": "basic_http",
                "timestamp": "2025-09-03T10:00:00Z"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        else:
            # 404 for other paths
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

def main():
    """Start the basic HTTP server"""
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 STARTING BASIC HTTP SERVER")
    print(f"🌐 Port: {port}")
    print(f"🐍 Python version: {os.sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📂 Files: {os.listdir('.')}")
    
    try:
        server = HTTPServer(('0.0.0.0', port), SimpleHandler)
        print(f"✅ Server started successfully on 0.0.0.0:{port}")
        print("🔄 Server is ready to handle requests...")
        server.serve_forever()
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
