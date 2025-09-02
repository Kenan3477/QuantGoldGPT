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
        # Import app and socketio
        print("📥 Importing Flask application...")
        from app import app, socketio
        
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 Starting on port {port}")
        
        # Use SocketIO run for production compatibility
        print("🔄 Starting with SocketIO...")
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False,
            log_output=True
        )
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic Flask if SocketIO fails
        try:
            print("🔄 Trying fallback Flask startup...")
            from app import app
            app.run(host='0.0.0.0', port=port, debug=False)
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            sys.exit(1)

if __name__ == "__main__":
    main()
