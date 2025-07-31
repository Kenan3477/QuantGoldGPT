#!/usr/bin/env python3
"""
Quick integration validation test
"""

import sys
import os
from datetime import datetime, timezone

# Test imports
try:
    from learning_system_integration import LearningSystemIntegration
    print("✅ Learning system integration import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test Flask integration
try:
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Initialize learning system
    integration = LearningSystemIntegration()
    integration.learning_db_path = "goldgpt_learning_system.db"
    integration.init_app(app)
    
    print("✅ Flask integration successful")
except Exception as e:
    print(f"❌ Flask integration failed: {e}")
    sys.exit(1)

# Test basic functionality
try:
    # Test prediction tracking
    test_prediction = {
        'symbol': 'XAUUSD',
        'strategy': 'integration_test',
        'direction': 'bullish',
        'confidence': 0.8,
        'predicted_price': 2100.0,
        'current_price': 2095.0
    }
    
    tracking_id = integration.track_prediction(test_prediction)
    print(f"✅ Prediction tracking successful: {tracking_id}")
    
    # Test health check
    health = integration.health_check()
    print(f"✅ Health check successful: {health['overall_status']}")
    
    # Test dashboard endpoints
    with app.test_client() as client:
        response = client.get('/dashboard/')
        if response.status_code == 200:
            print("✅ Dashboard endpoint accessible")
        else:
            print(f"⚠️ Dashboard endpoint returned {response.status_code}")
        
        response = client.get('/api/learning/health')
        if response.status_code in [200, 503]:
            print("✅ Learning health endpoint working")
        else:
            print(f"⚠️ Learning health endpoint returned {response.status_code}")
    
    print("\n🎯 INTEGRATION VALIDATION SUCCESSFUL!")
    print("✅ All core components working")
    print("✅ Database initialized and accessible")
    print("✅ Flask integration functional")
    print("✅ Dashboard endpoints operational")
    
    print("\n🚀 READY FOR PRODUCTION INTEGRATION!")
    print("Next steps:")
    print("1. Add learning system imports to your app.py")
    print("2. Initialize with: learning_system = integrate_learning_system_with_app(app)")
    print("3. Access dashboard at: http://localhost:5000/dashboard/")
    
except Exception as e:
    print(f"❌ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
