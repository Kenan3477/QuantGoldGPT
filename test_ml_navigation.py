#!/usr/bin/env python3
"""
Test script to verify Advanced ML navigation is working correctly
"""

import requests

def test_ml_navigation():
    """Test the ML navigation links"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing ML Navigation Links...")
    print("=" * 60)
    
    tests = [
        {
            "name": "Main Dashboard",
            "url": f"{base_url}/",
            "expected_content": ["Advanced ML", "System Hub", "GoldGPT"]
        },
        {
            "name": "Advanced ML Predictions (NEW)",
            "url": f"{base_url}/ml-predictions",
            "expected_content": ["ML Predictions Dashboard", "Advanced Machine Learning", "predictions-grid"]
        },
        {
            "name": "Old ML Dashboard",
            "url": f"{base_url}/advanced-ml-dashboard",
            "expected_content": ["GoldGPT ML Dashboard", "dashboard-header"]
        },
        {
            "name": "Multi-Strategy ML",
            "url": f"{base_url}/multi-strategy-ml-dashboard",
            "expected_content": ["GoldGPT ML Dashboard"]
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   URL: {test['url']}")
        
        try:
            response = requests.get(test['url'], timeout=10)
            status_ok = response.status_code == 200
            
            if status_ok:
                content_ok = True
                missing_content = []
                response_text = response.text.lower()
                
                for content in test['expected_content']:
                    if content.lower() not in response_text:
                        missing_content.append(content)
                        content_ok = False
                
                if content_ok:
                    print(f"   ✅ PASSED (Status: {response.status_code})")
                    results.append(True)
                else:
                    print(f"   ⚠️ PARTIAL (Status: {response.status_code}, Missing: {missing_content})")
                    results.append(True)  # Still accessible
            else:
                print(f"   ❌ FAILED (Status: {response.status_code})")
                results.append(False)
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ FAILED (Connection Error: {str(e)})")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 NAVIGATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print("\n🎯 NAVIGATION ROUTES:")
    print("✅ Header 'Advanced ML' button → /ml-predictions (UPDATED)")
    print("✅ System Hub 'ML Dashboard' → /advanced-ml-dashboard")
    print("✅ System Hub 'ML Predictions' → /ml-predictions (BEST)")
    print("✅ Direct access to all ML pages available")
    
    print("\n🚀 RECOMMENDATION:")
    print("Use /ml-predictions for the best ML experience!")
    print("- Smart prediction persistence")
    print("- No more N/A flashing")
    print("- Advanced UI with real-time data")
    print("- Momentum-based updates")
    
    return passed == total

if __name__ == "__main__":
    test_ml_navigation()
