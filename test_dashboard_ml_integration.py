#!/usr/bin/env python3
"""
Test the Fixed ML Predictions in the Web Dashboard
"""

import requests
import json
import time
from datetime import datetime

def test_dashboard_ml_predictions():
    """Test the ML predictions through the web dashboard"""
    print("🧪 Testing Fixed ML Predictions in Web Dashboard")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Test endpoints
    endpoints = [
        ("/api/advanced-ml/health", "Health Check"),
        ("/api/advanced-ml/quick-prediction", "Quick Prediction"),
        ("/api/advanced-ml/predict", "Full Predictions"),
        ("/api/advanced-ml/strategies", "Strategy Info")
    ]
    
    for endpoint, description in endpoints:
        print(f"\n🔍 Testing {description}: {endpoint}")
        
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {description} successful ({duration:.3f}s)")
                
                # Display key information based on endpoint
                if 'health' in endpoint:
                    print(f"   └─ Status: {data.get('status', 'unknown')}")
                    print(f"   └─ Engine Type: {data.get('engine_type', 'unknown')}")
                    
                elif 'quick-prediction' in endpoint:
                    pred = data.get('prediction', {})
                    print(f"   └─ Price: ${pred.get('current_price', 0):.2f} → ${pred.get('predicted_price', 0):.2f}")
                    print(f"   └─ Change: {pred.get('price_change_percent', 0):.2f}%")
                    print(f"   └─ Direction: {pred.get('direction', 'unknown').upper()}")
                    print(f"   └─ Confidence: {pred.get('confidence', 0):.1%}")
                    
                elif 'predict' in endpoint:
                    predictions = data.get('predictions', {})
                    print(f"   └─ Timeframes: {list(predictions.keys())}")
                    print(f"   └─ Engine: {data.get('engine_version', 'unknown')}")
                    
                    for tf, pred in predictions.items():
                        print(f"     • {tf}: {pred.get('direction', 'unknown')} ({pred.get('confidence', 0):.1%})")
                        
                elif 'strategies' in endpoint:
                    strategies = data.get('strategies', {})
                    print(f"   └─ Engine Type: {data.get('engine_type', 'unknown')}")
                    print(f"   └─ Strategies: {len(strategies)}")
                    
            else:
                print(f"❌ {description} failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   └─ Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   └─ Raw response: {response.text[:100]}...")
                    
        except requests.exceptions.ConnectionError:
            print(f"❌ {description} failed: Connection refused")
            print("   └─ Make sure the Flask app is running on localhost:5000")
            
        except requests.exceptions.Timeout:
            print(f"❌ {description} failed: Request timeout")
            
        except Exception as e:
            print(f"❌ {description} failed: {e}")
    
    print(f"\n✅ Dashboard ML test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_websocket_predictions():
    """Test WebSocket predictions (basic simulation)"""
    print("\n🔌 WebSocket Prediction Test")
    print("=" * 30)
    
    # This would require running WebSocket client
    print("🔄 WebSocket testing requires the Flask app to be running")
    print("📡 Available WebSocket events:")
    print("   • request_dashboard_data - Get dashboard data")
    print("   • request_live_predictions - Get live predictions")
    print("   • request_advanced_ml_prediction - Get advanced ML prediction")

if __name__ == "__main__":
    print("🚀 Testing GoldGPT Fixed ML Dashboard Integration")
    print("=" * 70)
    
    test_dashboard_ml_predictions()
    test_websocket_predictions()
    
    print("\n🎯 How to test manually:")
    print("1. Start your Flask app: python app.py")
    print("2. Open browser: http://localhost:5000")
    print("3. Check the ML Predictions section on the dashboard")
    print("4. Verify that predictions show real analysis instead of false figures")
    print("\n💡 The Fixed ML Engine provides:")
    print("• Real candlestick pattern analysis")
    print("• Actual news sentiment scoring")
    print("• Live economic factor integration")
    print("• Proper technical indicator calculations")
    print("• Multi-factor ensemble predictions")
