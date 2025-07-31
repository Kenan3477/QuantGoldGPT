#!/usr/bin/env python3
"""
GoldGPT Web Application Launcher
Python 3.12+ compatible launcher with SSL compatibility fixes
"""

# Apply Python 3.12 SSL compatibility patch FIRST
try:
    from ssl_compatibility_patch import patch_ssl_for_python312, patch_eventlet_ssl
except ImportError:
    # Create minimal patch if file doesn't exist
    import ssl
    if not hasattr(ssl, 'wrap_socket'):
        def wrap_socket(sock, **kwargs):
            context = ssl.SSLContext()
            return context.wrap_socket(sock, **kwargs)
        ssl.wrap_socket = wrap_socket
        print("✅ Minimal SSL patch applied")

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
import ssl

# Fix SSL compatibility for Python 3.12+
if not hasattr(ssl, 'wrap_socket'):
    ssl.wrap_socket = lambda sock, **kwargs: ssl.SSLContext().wrap_socket(sock, **kwargs)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'flask_socketio',
        'numpy',
        'pandas',
        'requests',
        'sklearn'
        # Note: yfinance removed due to Python 3.12 SSL compatibility issues
        # We use gold-api.com for real-time data instead
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'flask_socketio':
                import flask_socketio
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        # Install missing packages with Python 3.12+ compatible versions
        for package in missing_packages:
            if package == 'flask_socketio':
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Flask-SocketIO==5.3.6'])
            elif package == 'flask':
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Flask==3.0.0'])
            elif package == 'yfinance':
                # Skip yfinance installation as we use gold-api.com instead
                print(f"   ⚠️ Skipping {package} (using alternative Gold API)")
                continue
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        
        # Install Python 3.12+ compatible versions
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'eventlet==0.35.2'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-socketio==5.10.0'])
        
        print("✅ All packages installed successfully!")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("Please use Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_missing_files():
    """Create any missing critical files"""
    
    # Create empty AI analysis file if missing
    ai_analysis_file = Path('ai_analysis_api.py')
    if not ai_analysis_file.exists():
        print("📝 Creating missing ai_analysis_api.py...")
        ai_analysis_file.write_text('''
"""
AI Analysis API Stub
Minimal implementation for development
"""

def get_ai_analysis_sync(symbol):
    """Stub function for AI analysis"""
    return {
        'symbol': symbol,
        'signal': 'NEUTRAL',
        'confidence': 0.5,
        'analysis': 'Analysis not available'
    }
''')
    
    # Create empty news aggregator if missing
    news_aggregator_file = Path('news_aggregator.py')
    if not news_aggregator_file.exists():
        print("📝 Creating missing news_aggregator.py...")
        news_aggregator_file.write_text('''
"""
News Aggregator Stub
Minimal implementation for development
"""

class NewsAggregator:
    def get_latest(self):
        return []

news_aggregator = NewsAggregator()

def run_news_aggregation():
    pass

def get_latest_news():
    return []
''')
    
    # Create empty ML prediction API if missing
    ml_prediction_file = Path('ml_prediction_api.py')
    if not ml_prediction_file.exists():
        print("📝 Creating missing ml_prediction_api.py...")
        ml_prediction_file.write_text('''
"""
ML Prediction API Stub
Minimal implementation for development
"""

def get_ml_predictions(symbol):
    """Stub function for ML predictions"""
    return {
        'symbol': symbol,
        'predictions': {},
        'confidence': 0.5
    }

def train_all_models():
    pass

class MLEngine:
    pass

ml_engine = MLEngine()
''')

def main():
    """Main launcher function"""
    print("🏆 GoldGPT - Advanced AI Trading Web Application")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Create missing files
    create_missing_files()
    
    # Check dependencies
    check_dependencies()
    
    # Set environment variables
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('SECRET_KEY', 'goldgpt-secret-key-2025')
    
    print("\n🚀 Starting GoldGPT Web Application...")
    print("📊 Trading 212 Inspired Dashboard")
    print("🤖 Advanced AI Trading Features")
    print("🔗 WebSocket Real-time Updates")
    
    # Start the application
    try:
        # SSL and Python 3.12+ compatibility fix
        try:
            import ssl
            import warnings
            if not hasattr(ssl, 'wrap_socket'):
                ssl.wrap_socket = ssl.SSLSocket
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        except:
            pass
        
        # Import and start the app
        print("\n✅ Loading application modules...")
        from app import app, socketio
        
        print("✅ Application modules loaded successfully")
        print("🌐 Starting server on http://localhost:5000")
        print("📱 Opening browser in 3 seconds...")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open('http://localhost:5000')
                print("🌐 Browser opened automatically")
            except Exception as e:
                print(f"⚠️ Could not open browser automatically: {e}")
                print("Please manually open http://localhost:5000")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the server with Python 3.12+ compatibility
        print("🚀 Server starting...")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 GoldGPT Application stopped by user")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\n🔧 This usually means missing dependencies.")
        print("Try running: pip install -r requirements.txt")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if port 5000 is available")
        print("   2. Verify all dependencies are installed")
        print("   3. Check Python version (3.8+ required)")
        print("   4. Try running: pip install -r requirements.txt")
        input("Press Enter to exit...")

if __name__ == '__main__':
    main()
