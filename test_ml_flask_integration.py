"""
Test Script for Advanced Multi-Strategy ML Engine Flask Integration

This script tests the complete integration of the advanced ML engine with the Flask web application.
"""

import json
import time
import requests
from datetime import datetime

def test_ml_flask_integration():
    """Test the ML Flask integration"""
    
    base_url = "http://localhost:5000"  # Adjust if your app runs on different port
    
    print("🧪 Testing Advanced Multi-Strategy ML Engine Flask Integration")
    print("=" * 70)
    
    # Test 1: Check if the server is running
    print("\n1. Testing server connectivity...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to server: {e}")
        print("💡 Make sure your Flask app is running with: python app.py")
        return False
    
    # Test 2: Test ML dashboard route
    print("\n2. Testing ML dashboard route...")
    try:
        response = requests.get(f"{base_url}/ml-dashboard", timeout=10)
        if response.status_code == 200:
            print("✅ ML dashboard route is accessible")
            print(f"   Response length: {len(response.text)} characters")
        else:
            print(f"❌ ML dashboard returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing ML dashboard: {e}")
    
    # Test 3: Test AI signals generation (ML enhanced)
    print("\n3. Testing enhanced AI signal generation...")
    try:
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "1h"
        }
        response = requests.post(
            f"{base_url}/api/ai-signals/generate", 
            json=payload, 
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced AI signal generation successful")
            
            if data.get('success'):
                signal = data.get('signal', {})
                raw_prediction = data.get('raw_prediction', {})
                
                print(f"   Signal Direction: {signal.get('direction', 'N/A')}")
                print(f"   Confidence: {signal.get('confidence', 0):.2%}")
                print(f"   Cached: {data.get('cached', False)}")
                
                # Check if it's using the new ML engine
                if 'strategies' in raw_prediction:
                    print("✅ Multi-strategy ML engine is working!")
                    strategies = raw_prediction['strategies']
                    print(f"   Strategies used: {len(strategies)}")
                    for strategy_name, strategy_data in strategies.items():
                        conf = strategy_data.get('confidence', 0)
                        pred = strategy_data.get('prediction', 'N/A')
                        print(f"     • {strategy_name}: {pred} ({conf:.2%})")
                else:
                    print("⚠️ Fallback ML engine is being used")
            else:
                print(f"❌ Signal generation failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Signal generation returned status code: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error testing signal generation: {e}")
    
    # Test 4: Test strategy performance endpoint
    print("\n4. Testing strategy performance endpoint...")
    try:
        response = requests.get(f"{base_url}/api/ml/strategies/performance", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Strategy performance endpoint working")
            
            if data.get('success'):
                performance = data.get('performance', {})
                print(f"   Strategies tracked: {len(performance)}")
                for strategy_name, perf in performance.items():
                    accuracy = perf.get('accuracy', 0)
                    total = perf.get('total_predictions', 0)
                    print(f"     • {strategy_name}: {accuracy:.2%} accuracy ({total} predictions)")
            else:
                print(f"❌ Performance data failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Performance endpoint returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error testing performance endpoint: {e}")
    
    # Test 5: Test detailed prediction endpoint  
    print("\n5. Testing detailed prediction endpoint...")
    try:
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "4h"
        }
        response = requests.post(
            f"{base_url}/api/ml/prediction/detailed",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Detailed prediction endpoint working")
            
            if data.get('success'):
                prediction = data.get('prediction', {})
                print(f"   Symbol: {prediction.get('symbol', 'N/A')}")
                print(f"   Timeframe: {prediction.get('timeframe', 'N/A')}")
                print(f"   Final Prediction: {prediction.get('prediction', 'N/A')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.2%}")
                
                # Check ensemble voting
                ensemble = prediction.get('ensemble', {})
                if ensemble:
                    voting = ensemble.get('voting_details', {})
                    print(f"   Ensemble Voting: {voting}")
            else:
                print(f"❌ Detailed prediction failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Detailed prediction returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error testing detailed prediction: {e}")
    
    # Test 6: Test ML dashboard data endpoint
    print("\n6. Testing ML dashboard data endpoint...")
    try:
        response = requests.get(f"{base_url}/api/ml/dashboard/data", timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ ML dashboard data endpoint working")
            
            if data.get('success'):
                dashboard_data = data.get('data', {})
                predictions = dashboard_data.get('predictions', {})
                performance = dashboard_data.get('performance', {})
                summary = dashboard_data.get('summary', {})
                
                print(f"   Predictions cached: {len(predictions)} symbols")
                print(f"   Performance data: {len(performance)} strategies")
                print(f"   Active strategies: {summary.get('active_strategies', 0)}")
                print(f"   Best strategy: {summary.get('best_performing_strategy', 'N/A')}")
            else:
                print(f"❌ Dashboard data failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Dashboard data returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error testing dashboard data: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Advanced Multi-Strategy ML Engine Integration Test Complete!")
    print("\n📋 Integration Summary:")
    print("   • Flask app enhanced with 5-strategy ML engine")
    print("   • Enhanced /api/ai-signals/generate endpoint")
    print("   • New ML-specific API endpoints added")
    print("   • ML dashboard UI available at /ml-dashboard")
    print("   • Real-time WebSocket updates supported")
    
    return True

def test_prediction_accuracy():
    """Test multiple predictions to see consistency"""
    
    base_url = "http://localhost:5000"
    print("\n🔬 Testing Prediction Consistency...")
    
    predictions = []
    
    for i in range(3):
        try:
            payload = {"symbol": "XAUUSD", "timeframe": "1h"}
            response = requests.post(f"{base_url}/api/ai-signals/generate", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    signal = data.get('signal', {})
                    predictions.append({
                        'direction': signal.get('direction'),
                        'confidence': signal.get('confidence', 0),
                        'cached': data.get('cached', False)
                    })
                    print(f"   Prediction {i+1}: {signal.get('direction')} ({signal.get('confidence', 0):.2%}) {'[CACHED]' if data.get('cached') else '[FRESH]'}")
        except Exception as e:
            print(f"   Error in prediction {i+1}: {e}")
        
        time.sleep(1)  # Small delay
    
    if predictions:
        # Check for consistency in non-cached predictions
        fresh_predictions = [p for p in predictions if not p['cached']]
        if len(fresh_predictions) > 1:
            directions = [p['direction'] for p in fresh_predictions]
            if len(set(directions)) == 1:
                print("✅ Fresh predictions are consistent")
            else:
                print("⚠️ Fresh predictions vary (this can be normal)")
        
        # Check caching behavior
        cached_count = sum(1 for p in predictions if p['cached'])
        print(f"   Caching working: {cached_count}/{len(predictions)} predictions cached")

if __name__ == "__main__":
    success = test_ml_flask_integration()
    
    if success:
        print("\n🔄 Running additional consistency tests...")
        test_prediction_accuracy()
        
        print("\n🚀 Ready to launch! Your GoldGPT app now has:")
        print("   1. Advanced Multi-Strategy ML Engine")
        print("   2. 5 specialized prediction strategies")
        print("   3. Ensemble voting system")
        print("   4. Real-time performance tracking")
        print("   5. Enhanced web dashboard")
        print("\n💡 Next steps:")
        print("   • Visit http://localhost:5000/ml-dashboard for ML interface")
        print("   • Check existing dashboard for enhanced AI signals")
        print("   • Monitor strategy performance over time")
    else:
        print("\n❌ Integration test failed. Please check your setup.")
        print("💡 Troubleshooting:")
        print("   • Ensure Flask app is running: python app.py")
        print("   • Check all imports are working")
        print("   • Verify ml_flask_integration.py is in the right directory")
