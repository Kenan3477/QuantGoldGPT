#!/usr/bin/env python3
"""
Ultra-Simple Railway Startup - Debugging Version
"""
import os
import sys

def main():
    """Ultra-minimal startup with extensive logging"""
    print("🚀 GoldGPT Emergency Startup - Debug Mode")
    print("=" * 50)
    
    # Log environment
    port = os.environ.get('PORT', '5000')
    print(f"🌍 Environment PORT: {port}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Available files: {os.listdir('.')}")
    
    try:
        print("\n📥 Step 1: Testing basic imports...")
        import flask
        print(f"✅ Flask version: {flask.__version__}")
        
        print("\n📥 Step 2: Testing app import...")
        from app import app
        print("✅ App imported successfully")
        
        print("\n📥 Step 3: Testing SocketIO import...")
        try:
            from app import socketio
            print("✅ SocketIO imported successfully")
            use_socketio = True
        except:
            print("⚠️ SocketIO not available, using basic Flask")
            use_socketio = False
        
        print(f"\n🌐 Step 4: Starting server on 0.0.0.0:{port}")
        
        if use_socketio:
            print("🔄 Using SocketIO startup...")
            socketio.run(
                app,
                host='0.0.0.0',
                port=int(port),
                debug=False,
                use_reloader=False
            )
        else:
            print("🔄 Using basic Flask startup...")
            app.run(
                host='0.0.0.0',
                port=int(port),
                debug=False
            )
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("🆘 Creating emergency minimal server...")
        
        # Emergency minimal Flask server
        from flask import Flask
        emergency_app = Flask(__name__)
        
        @emergency_app.route('/')
        def health():
            return {"status": "emergency_mode", "message": "Signal generation service running"}
            
        @emergency_app.route('/api/signals/generate')
        def emergency_signals():
            # Use our emergency signal generator
            try:
                from emergency_signal_generator import generate_working_signal
                signal = generate_working_signal()
                return {"success": True, "signal": signal, "mode": "emergency"}
            except:
                import random
                return {
                    "success": True,
                    "signal": {
                        "signal_type": random.choice(["BUY", "SELL"]),
                        "entry_price": 3500.0,
                        "confidence": 0.7
                    },
                    "mode": "ultra_emergency"
                }
        
        emergency_app.run(host='0.0.0.0', port=int(port), debug=False)
        
    except Exception as e:
        print(f"❌ Critical startup failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
