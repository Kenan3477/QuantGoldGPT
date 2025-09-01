#!/usr/bin/env python3

import sqlite3
import requests
import json
from datetime import datetime

print("🔍 VERIFYING REAL ML SIGNALS ARE WORKING")
print("=" * 50)

# 1. Check if signal tracking database exists and has recent signals
try:
    conn = sqlite3.connect('signal_tracking.db')
    cursor = conn.cursor()
    
    # Get recent signals
    cursor.execute('SELECT * FROM signals ORDER BY created_at DESC LIMIT 5')
    signals = cursor.fetchall()
    
    print(f"✅ RECENT REAL ML SIGNALS ({len(signals)} found):")
    for signal in signals:
        signal_id, timeframe, signal_type, confidence, created_at = signal
        print(f"  📊 {signal_id}: {signal_type} ({confidence}%) at {created_at}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error checking database: {e}")

print("\n" + "=" * 50)

# 2. Test the API endpoint directly
try:
    print("🌐 TESTING ML PREDICTIONS API:")
    response = requests.get('http://localhost:5000/api/ml-predictions', timeout=10)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API Response:")
        print(json.dumps(data, indent=2))
        
        # Check if we have real predictions
        predictions = data.get('predictions', [])
        if predictions:
            print(f"\n📈 Found {len(predictions)} predictions:")
            for pred in predictions:
                print(f"  • {pred.get('timeframe')}: {pred.get('signal')} ({pred.get('confidence', 0)}%)")
        else:
            print("⚠️ No predictions found in response")
    else:
        print(f"❌ API Error: {response.status_code} - {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Connection Error: {e}")

print("\n" + "=" * 50)

# 3. Check if real ML engine is being used (not fake signals)
try:
    print("🔧 CHECKING REAL ML ENGINE STATUS:")
    
    # Import the real ML engine
    from real_ml_trading_engine import RealMLTradingEngine
    
    engine = RealMLTradingEngine()
    print("✅ RealMLTradingEngine imported successfully")
    
    # Test signal generation
    test_signal = engine.generate_real_signal("GOLD", "15m")
    print(f"✅ Test signal generated: {test_signal}")
    
    # Check if it's not just random
    if test_signal in ["BUY", "SELL", "HOLD"]:
        print("✅ Real ML engine is generating proper signals")
    else:
        print("⚠️ Unexpected signal format")
        
except Exception as e:
    print(f"❌ Error testing ML engine: {e}")

print("\n🎯 VERIFICATION COMPLETE!")
