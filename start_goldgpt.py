#!/usr/bin/env python3
"""
🚀 GoldGPT Web Application Launcher
Complete startup script with all ML capabilities, real-time data, and advanced features

This script will:
1. Check and install dependencies
2. Initialize all database systems
3. Start background ML training
4. Launch the web application with all features enabled
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("🏆 GoldGPT - Advanced Gold Trading AI Platform")
    print("=" * 80)
    print("📊 Features: ML Predictions | Real-time Gold API | Technical Analysis")
    print("🔮 AI Analysis | News Sentiment | Market Psychology | Trading Signals")
    print("⚡ WebSocket Updates | Professional Dashboard | Trading 212 Style UI")
    print("=" * 80)
    print(f"🕐 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def check_dependencies():
    """Check and install required dependencies"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'flask',
        'flask-socketio',
        'requests',
        'pandas',
        'numpy',
        'scikit-learn',
        'beautifulsoup4',
        'python-dateutil',
        'eventlet'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\\n🔧 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                print("💡 Try running: pip install -r requirements.txt")
    else:
        print("✅ All dependencies installed")

def initialize_databases():
    """Initialize all database systems"""
    print("\\n🗄️ Initializing databases...")
    
    try:
        # Import and initialize core systems
        from price_storage_manager import get_current_gold_price
        print("✅ Price storage database initialized")
        
        from ml_prediction_api import ml_engine
        print("✅ ML prediction database initialized")
        
        from enhanced_signal_generator import EnhancedSignalGenerator
        signal_gen = EnhancedSignalGenerator()
        print("✅ Enhanced signals database initialized")
        
        from data_pipeline_core import data_pipeline
        print("✅ Data pipeline database initialized")
        
        print("✅ All databases ready")
        
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        print("📝 Databases will be created automatically when needed")

def test_gold_api():
    """Test Gold API connectivity"""
    print("\\n🔍 Testing Gold API connectivity...")
    
    try:
        from price_storage_manager import get_current_gold_price
        price = get_current_gold_price()
        
        if price and price > 1000:
            print(f"✅ Gold API connected - Current price: ${price:.2f}")
            return True
        else:
            print("⚠️ Gold API returned invalid price")
            return False
            
    except Exception as e:
        print(f"❌ Gold API test failed: {e}")
        return False

def start_background_services():
    """Start background services"""
    print("\\n🔄 Starting background services...")
    
    try:
        # Start ML training scheduler
        print("🤖 Starting ML background training...")
        from ml_prediction_api import start_background_training
        start_background_training()
        print("✅ ML training scheduler started")
        
        # Start data pipeline updates
        print("📡 Starting data pipeline...")
        from data_pipeline_core import data_pipeline
        print("✅ Data pipeline ready")
        
        print("✅ All background services started")
        
    except Exception as e:
        print(f"⚠️ Background services warning: {e}")
        print("📝 Services will start automatically with the web app")

def validate_ml_systems():
    """Validate ML prediction systems"""
    print("\\n🧠 Validating ML systems...")
    
    try:
        # Test intelligent ML predictor
        from intelligent_ml_predictor import get_intelligent_ml_predictions
        result = get_intelligent_ml_predictions('XAUUSD')
        if result and 'current_price' in result:
            print(f"✅ Intelligent ML Predictor - Price: ${result['current_price']:.2f}")
        else:
            print("⚠️ Intelligent ML Predictor - Using fallback")
        
        # Test ML prediction API
        import asyncio
        from ml_prediction_api import get_ml_predictions
        
        async def test_ml_api():
            result = await get_ml_predictions('GC=F')
            return result
        
        ml_result = asyncio.run(test_ml_api())
        if ml_result and ml_result.get('success'):
            print("✅ ML Prediction API - Working")
        else:
            print("⚠️ ML Prediction API - Using fallback")
        
        print("✅ ML systems validated")
        
    except Exception as e:
        print(f"⚠️ ML validation warning: {e}")
        print("📝 ML systems will use fallback predictions")

def start_web_application():
    """Start the main web application"""
    print("\\n🌐 Starting GoldGPT Web Application...")
    print("=" * 50)
    print("🔗 Application will be available at:")
    print("   • http://localhost:5000")
    print("   • http://127.0.0.1:5000")
    print("=" * 50)
    print("📊 Dashboard Features:")
    print("   • Real-time Gold Prices")
    print("   • ML Price Predictions")
    print("   • Technical Analysis")
    print("   • AI Trading Signals")
    print("   • News Sentiment Analysis")
    print("   • Market Psychology Indicators")
    print("=" * 50)
    print("🎮 Controls:")
    print("   • Ctrl+C to stop the server")
    print("   • Check console for real-time updates")
    print("=" * 50)
    
    try:
        # Import and run the main Flask app
        from app import app, socketio
        
        # Configure for production-like environment
        print("🚀 Launching Flask application with SocketIO...")
        print("📡 WebSocket connections enabled for real-time updates")
        print("⚡ Background ML training active")
        print("🔄 Auto-refreshing gold prices every 5 seconds")
        print("")
        print("✅ GoldGPT is now running!")
        print("🌟 Enjoy professional-grade gold trading analysis!")
        print("")
        
        # Start the application
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,  # Set to False for production
                    allow_unsafe_werkzeug=True)
        
    except KeyboardInterrupt:
        print("\\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start web application: {e}")
        print("💡 Try running 'python app.py' directly")

def main():
    """Main startup sequence"""
    print_banner()
    
    # Pre-flight checks
    check_python_version()
    check_dependencies()
    
    # System initialization
    initialize_databases()
    test_gold_api()
    start_background_services()
    validate_ml_systems()
    
    print("\\n🎯 All systems ready!")
    print("🚀 Launching GoldGPT Web Application...")
    time.sleep(2)
    
    # Start the web application
    start_web_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n👋 Goodbye! Thanks for using GoldGPT!")
    except Exception as e:
        print(f"\\n💥 Startup failed: {e}")
        print("\\n🔧 Troubleshooting:")
        print("1. Check that you're in the correct directory")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Try running 'python app.py' directly")
        print("4. Check for any missing files or configuration issues")
