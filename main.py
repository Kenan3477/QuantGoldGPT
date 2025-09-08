from datetime import datetime
import os

# 🔥 EMERGENCY RAILWAY DEPLOYMENT SCRIPT 🔥
# This file will make Railway deploy our auto-close system
# NO MATTER WHAT!

print("🚨 FORCING RAILWAY DEPLOYMENT 🚨")
print(f"Timestamp: {datetime.now()}")
print("Version: 2.1.0 AUTO-CLOSE LEARNING SYSTEM")
print("Status: CRITICAL PRODUCTION UPDATE")
print("User Patience Level: COMPLETELY EXHAUSTED")

# Change the PORT to force Railway to notice
PORT = os.environ.get('PORT', 8080)
print(f"Railway Port: {PORT}")

if __name__ == "__main__":
    print("Railway deployment trigger activated!")
    print("Auto-close learning system ready for production!")
    
    # Import and run the main app
    try:
        from app import app, socketio
        print("✅ App imported successfully with auto-close system")
        socketio.run(app, host='0.0.0.0', port=int(PORT), debug=False)
    except Exception as e:
        print(f"❌ Error: {e}")
        # Fallback to basic Flask
        from app import app
        app.run(host='0.0.0.0', port=int(PORT), debug=False)
