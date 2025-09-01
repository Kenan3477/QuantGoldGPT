#!/usr/bin/env python3
"""
Railway-optimized startup script for GoldGPT QuantGold Dashboard
Enhanced with comprehensive error handling and debugging
"""

import os
import sys
import traceback
from datetime import datetime

def log_startup_info():
    """Log comprehensive startup information for Railway debugging"""
    print("="*60)
    print("🚀 GOLDGPT QUANTGOLD DASHBOARD - RAILWAY STARTUP")
    print("="*60)
    print(f"⏰ Startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🌍 Railway environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'unknown')}")
    print(f"🔌 PORT environment: {os.environ.get('PORT', 'not-set')}")
    print(f"📦 Available files: {[f for f in os.listdir('.') if f.endswith('.py')][:5]}")
    print("-"*60)

def main():
    """Main startup function with comprehensive error handling"""
    log_startup_info()
    
    try:
        # Import the main app components
        print("📥 Importing Flask application...")
        from app import app, socketio
        print("✅ Flask app imported successfully")
        
        # Configure port with robust error handling
        port_env = os.environ.get('PORT', '5000')
        print(f"🔍 Raw PORT value: '{port_env}'")
        
        if port_env.startswith('$') or not str(port_env).strip().isdigit():
            print(f"⚠️ Invalid PORT format, using default 5000")
            port = 5000
        else:
            port = int(port_env)
        
        print(f"🌐 Starting server on 0.0.0.0:{port}")
        
        # Railway-optimized startup sequence
        print("🔄 Attempting SocketIO startup (preferred)...")
        try:
            socketio.run(
                app, 
                host='0.0.0.0', 
                port=port, 
                debug=False,
                use_reloader=False,
                log_output=True,
                allow_unsafe_werkzeug=True
            )
        except Exception as socketio_error:
            print(f"❌ SocketIO startup failed: {socketio_error}")
            print("🔄 Falling back to standard Flask...")
            
            # Fallback to standard Flask
            app.run(
                host='0.0.0.0', 
                port=port, 
                debug=False,
                use_reloader=False
            )
            
    except ImportError as import_error:
        print(f"💥 Import error: {import_error}")
        print("🔍 Available modules:")
        for module in ['flask', 'flask_socketio']:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError:
                print(f"  ❌ {module}")
        sys.exit(1)
        
    except ValueError as value_error:
        print(f"💥 Configuration error: {value_error}")
        print(f"🔍 PORT environment variable: '{os.environ.get('PORT')}'")
        sys.exit(1)
        
    except Exception as general_error:
        print(f"💥 Startup failed: {general_error}")
        print("📋 Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
