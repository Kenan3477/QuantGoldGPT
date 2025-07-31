#!/usr/bin/env python3
"""
Test script to verify ML Predictions Dashboard is working correctly
"""

import requests
import json
import time

def test_ml_predictions_dashboard():
    """Test the new ML Predictions Dashboard functionality"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing ML Predictions Dashboard...")
    print("=" * 60)
    
    tests = [
        {
            "name": "ML Predictions Page",
            "url": f"{base_url}/ml-predictions",
            "expected_status": 200,
            "expected_content": ["ML Predictions Dashboard", "Advanced Machine Learning", "🤖"]
        },
        {
            "name": "Advanced ML API - Predictions",
            "url": f"{base_url}/api/advanced-ml/predictions",
            "expected_status": 200,
            "expected_content": ["predictions", "success"]
        },
        {
            "name": "Advanced ML API - Status",
            "url": f"{base_url}/api/advanced-ml/status",
            "expected_status": 200,
            "expected_content": ["ml_engine", "status"]
        },
        {
            "name": "Advanced ML API - Health",
            "url": f"{base_url}/api/advanced-ml/health",
            "expected_status": 200,
            "expected_content": ["healthy", "services"]
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   URL: {test['url']}")
        
        try:
            response = requests.get(test['url'], timeout=10)
            status_ok = response.status_code == test['expected_status']
            
            # Check content
            content_ok = True
            if test['expected_content']:
                response_text = response.text.lower()
                missing_content = []
                for content in test['expected_content']:
                    if content.lower() not in response_text:
                        missing_content.append(content)
                        content_ok = False
                
                if not content_ok:
                    print(f"   ❌ Missing content: {missing_content}")
            
            if status_ok and content_ok:
                print(f"   ✅ PASSED (Status: {response.status_code})")
                results.append(True)
            else:
                print(f"   ❌ FAILED (Status: {response.status_code}, Content OK: {content_ok})")
                results.append(False)
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ FAILED (Connection Error: {str(e)})")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! ML Predictions Dashboard is fully operational!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    test_ml_predictions_dashboard()
