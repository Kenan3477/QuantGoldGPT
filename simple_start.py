#!/usr/bin/env python3
"""
Simple Railway startup script - Emergency Signal Generation Fix
"""
import os
import sys

def main():
    """Minimal startup for Railway deployment"""
    print("🚀 Starting GoldGPT with Signal Generation Fix...")
    
    try:
        from app import app
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 Starting on port {port}")
        
        # Simple Flask startup without SocketIO complications
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
