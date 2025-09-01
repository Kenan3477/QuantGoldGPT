#!/usr/bin/env python3
"""
WSGI Entry Point for GoldGPT Production Deployment
Optimized for Railway with Gunicorn
"""

import os
import sys
import logging
from app import app, socketio

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configure for production
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting GoldGPT in production mode on port {port}")
    logger.info(f"🔍 Debug mode: {debug_mode}")
    logger.info(f"🌍 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    
    try:
        # Use SocketIO with gunicorn-compatible settings
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=debug_mode,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

# Export the application for gunicorn
application = app

if __name__ != "__main__":
    # When running with gunicorn, ensure proper initialization
    logger.info("🔧 Initializing GoldGPT for gunicorn...")
    logger.info("✅ GoldGPT application ready for production")
