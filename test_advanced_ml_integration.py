#!/usr/bin/env python3
"""
Advanced ML API Integration Test Script
Tests the comprehensive ML prediction system backend and Flask integration
"""

import asyncio
import json
import requests
import time
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoints():
    """Test all ML API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Advanced ML API Endpoints")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-health",
            'name': 'Health Check',
            'expected_keys': ['status', 'timestamp']
        },
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-demo",
            'name': 'Demo Endpoint',
            'expected_keys': ['available_endpoints', 'supported_timeframes']
        },
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-predictions/status",
            'name': 'ML System Status',
            'expected_keys': ['success', 'status']
        },
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-predictions/1h",
            'name': 'Predictions for 1h timeframe',
            'expected_keys': ['success', 'predictions', 'timestamp']
        },
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-predictions/all",
            'name': 'All Predictions',
            'expected_keys': ['success', 'predictions', 'timeframes']
        },
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-predictions/accuracy",
            'name': 'Accuracy Statistics',
            'expected_keys': ['success', 'accuracy_stats']
        },
        {
            'method': 'GET',
            'url': f"{base_url}/api/ml-predictions/features",
            'name': 'Feature Importance',
            'expected_keys': ['success', 'feature_importance']
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        try:
            print(f"\n� Testing {endpoint['name']}...")
            print(f"   {endpoint['method']} {endpoint['url']}")
            
            response = requests.get(endpoint['url'], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check expected keys
                missing_keys = []
                for key in endpoint['expected_keys']:
                    if key not in data:
                        missing_keys.append(key)
                
                if not missing_keys:
                    print(f"   ✅ SUCCESS - All expected keys present")
                    results.append((endpoint['name'], True, "All expected keys present"))
                else:
                    print(f"   ⚠️  PARTIAL - Missing keys: {missing_keys}")
                    results.append((endpoint['name'], False, f"Missing keys: {missing_keys}"))
                
                # Show sample data
                if isinstance(data, dict) and 'success' in data:
                    print(f"   📊 Success: {data['success']}")
                
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                results.append((endpoint['name'], False, f"HTTP {response.status_code}"))
            
        except requests.exceptions.ConnectionError:
            print(f"   ❌ FAILED - Connection refused (server not running?)")
            results.append((endpoint['name'], False, "Connection refused"))
        except Exception as e:
            print(f"   ❌ FAILED - {str(e)}")
            results.append((endpoint['name'], False, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, details in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {name:30} {details}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Advanced ML API is working correctly.")
    elif passed > total // 2:
        print("⚠️  Most tests passed. Some features may not be fully available.")
    else:
        print("❌ Many tests failed. Check server status and configuration.")
    
    return passed, total

def test_websocket_connection():
    """Test WebSocket connection"""
    print("\n� Testing WebSocket Connection")
    print("=" * 30)
    
    try:
        import socketio
        
        sio = socketio.Client()
        
        @sio.event
        def connect():
            print("✅ WebSocket connected successfully")
            sio.emit('subscribe_predictions', {'timeframes': ['1h']})
        
        @sio.event
        def disconnect():
            print("🔌 WebSocket disconnected")
        
        @sio.event
        def new_predictions(data):
            print(f"📊 Received prediction update: {len(data.get('predictions', []))} predictions")
        
        @sio.event
        def connection_established(data):
            print(f"✅ Connection established: {data['message']}")
        
        print("Connecting to WebSocket...")
        sio.connect('http://localhost:5000')
        
        # Wait a bit for events
        time.sleep(2)
        
        sio.disconnect()
        
        return True
        
    except ImportError:
        print("⚠️ python-socketio not installed, skipping WebSocket test")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def test_advanced_ml_integration():
    """Test advanced ML components directly"""
    print("\n🤖 Testing Advanced ML Integration")
    print("=" * 35)
    
    try:
        # Test ML API controller import
        from advanced_ml_api_controller import MLAPIController
        print("✅ MLAPIController import successful")
        
        # Test Flask integration import
        from advanced_ml_flask_integration import MLFlaskIntegration
        print("✅ MLFlaskIntegration import successful")
        
        # Test client-side components
        client_js_path = "static/js/advanced-ml-api-client.js"
        if os.path.exists(client_js_path):
            print("✅ Client-side JavaScript available")
        else:
            print("⚠️ Client-side JavaScript not found")
        
        # Test CSS styles
        css_path = "static/css/advanced-ml-dashboard.css"
        if os.path.exists(css_path):
            print("✅ Dashboard CSS styles available")
        else:
            print("⚠️ Dashboard CSS not found")
        
        # Test demo template
        demo_path = "templates/advanced_ml_demo.html"
        if os.path.exists(demo_path):
            print("✅ Demo template available")
        else:
            print("⚠️ Demo template not found")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_post_endpoints():
    """Test POST endpoints"""
    print("\n� Testing POST Endpoints")
    print("=" * 25)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test refresh predictions
        print("Testing prediction refresh...")
        response = requests.post(
            f"{base_url}/api/ml-predictions/refresh",
            json={'timeframes': ['1h', '4h']},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Refresh successful: {data.get('message', 'No message')}")
            return True
        else:
            print(f"❌ Refresh failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ POST endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Advanced ML API Integration Test Suite")
    print("=" * 45)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Test 1: Component Integration
    print("Test 1: Component Integration")
    integration_success = test_advanced_ml_integration()
    test_results.append(("Component Integration", integration_success))
    
    # Test 2: API Endpoints
    print("\nTest 2: API Endpoints")
    passed, total = test_api_endpoints()
    api_success = passed == total
    test_results.append(("API Endpoints", api_success))
    
    # Test 3: WebSocket
    print("\nTest 3: WebSocket Connection")
    websocket_success = test_websocket_connection()
    test_results.append(("WebSocket", websocket_success))
    
    # Test 4: POST Endpoints
    print("\nTest 4: POST Endpoints")
    post_success = test_post_endpoints()
    test_results.append(("POST Endpoints", post_success))
    
    # Final Summary
    print("\n" + "=" * 45)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 45)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    total_passed = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"\n📊 Overall Results: {total_passed}/{total_tests} test categories passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Advanced ML API system is fully functional")
        print()
        print("🔗 Available endpoints:")
        print("   • http://localhost:5000/api/ml-predictions/all")
        print("   • http://localhost:5000/api/ml-predictions/status")
        print("   • http://localhost:5000/ml-dashboard")
        print("   • http://localhost:5000/templates/advanced_ml_demo.html")
        
        return True
    else:
        print("⚠️ Some tests failed. Please check the server configuration.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)
    results['system_status'] = test_advanced_ml_system_status(base_url)
    
    # Test 3: Advanced ML Endpoints
    results['advanced_endpoints'] = test_advanced_ml_endpoints(base_url)
    
    # Test 4: Main Prediction Endpoint
    results['prediction'] = test_prediction_endpoint(base_url)
    
    # Test 5: Strategy Performance
    results['performance'] = test_strategy_performance_endpoint(base_url)
    
    # Test 6: Demo Page
    results['demo_page'] = test_demo_page(base_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if isinstance(result, bool) and result)
    total = len([r for r in results.values() if isinstance(r, bool)])
    
    print(f"Tests Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        elif isinstance(result, dict):
            endpoint_results = [r.get('success', False) for r in result.values()]
            endpoint_passed = sum(endpoint_results)
            endpoint_total = len(endpoint_results)
            print(f"   {test_name}: {endpoint_passed}/{endpoint_total} endpoints working")
    
    print("\n🎯 RECOMMENDATIONS:")
    
    if not results['system_status']:
        print("   • Check if ML models are properly loaded")
        print("   • Verify advanced ML engine initialization")
    
    if isinstance(results['advanced_endpoints'], dict):
        failed_endpoints = [k for k, v in results['advanced_endpoints'].items() if not v.get('success', False)]
        if failed_endpoints:
            print(f"   • Fix failed endpoints: {', '.join(failed_endpoints)}")
    
    if not results['prediction']:
        print("   • Check ML prediction system configuration")
        print("   • Verify database connections and model files")
    
    if not results['demo_page']:
        print("   • Check template files and static assets")

