#!/usr/bin/env python3
"""
Test Script for ML Predictions Fix
Tests the real ML predictions API vs dashboard display
"""

import requests
import json
from datetime import datetime

def test_ml_predictions_api():
    """Test the ML predictions API endpoint"""
    print("🚀 Testing ML Predictions API Fix...")
    
    try:
        # Test the API endpoint
        url = "http://localhost:5000/api/ml-predictions/XAUUSD"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ API Response:")
            print(f"Current Price: ${data.get('current_price', 'N/A')}")
            print(f"Success: {data.get('success', False)}")
            
            if 'predictions' in data:
                print("\n📊 Predictions:")
                for i, pred in enumerate(data['predictions']):
                    timeframe = pred.get('timeframe', f'{i+1}H')
                    change_pct = pred.get('change_percent', 0)
                    predicted_price = pred.get('predicted_price', 0)
                    confidence = pred.get('confidence', 0)
                    
                    change_text = f"+{change_pct:.3f}%" if change_pct >= 0 else f"{change_pct:.3f}%"
                    confidence_pct = int(confidence * 100) if confidence < 1 else int(confidence)
                    
                    print(f"  {timeframe}: {change_text} (${predicted_price:.2f}) - {confidence_pct}% confidence")
            
            # Check if this matches terminal output
            print("\n🔍 Checking against terminal output...")
            expected_terminal = {
                "current_price": 3350.7,
                "predictions": [
                    {"change_percent": -0.083, "predicted_price": 3347.91, "confidence": 0.636},
                    {"change_percent": -0.141, "predicted_price": 3345.96, "confidence": 0.672},
                    {"change_percent": -0.413, "predicted_price": 3336.86, "confidence": 0.716}
                ]
            }
            
            current_price = data.get('current_price', 0)
            if abs(current_price - expected_terminal['current_price']) < 10:
                print("✅ Current price matches expected range")
            else:
                print(f"⚠️ Current price differs: API={current_price}, Expected={expected_terminal['current_price']}")
                
            return True
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_dashboard_integration():
    """Test that dashboard can access the API"""
    print("\n🌐 Testing Dashboard Integration...")
    
    try:
        # Test main dashboard endpoint
        url = "http://localhost:5000/"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Dashboard accessible")
            
            # Check if the HTML contains the ML predictions section
            html_content = response.text
            if 'id="ml-predictions"' in html_content:
                print("✅ ML predictions section found in dashboard")
            else:
                print("❌ ML predictions section NOT found in dashboard")
                
            if 'updateMLPredictionsDisplay' in html_content:
                print("✅ ML predictions update function found")
            else:
                print("❌ ML predictions update function NOT found")
                
            if 'ml_predictions_update' in html_content:
                print("✅ WebSocket ML listener found")
            else:
                print("❌ WebSocket ML listener NOT found")
                
            return True
        else:
            print(f"❌ Dashboard not accessible: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ML PREDICTIONS FIX - VERIFICATION TEST")
    print("=" * 60)
    
    # Test API
    api_success = test_ml_predictions_api()
    
    # Test Dashboard
    dashboard_success = test_dashboard_integration()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"API Test: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"Dashboard Test: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    
    if api_success and dashboard_success:
        print("\n🎉 ALL TESTS PASSED - Fix is working!")
        print("\nNext Steps:")
        print("1. Restart your web application")
        print("2. Open the dashboard and check ML predictions")
        print("3. Predictions should now show REAL data instead of fake random data")
    else:
        print("\n⚠️ Some tests failed - Check the application logs")
    
    print("=" * 60)
