#!/usr/bin/env python3
"""
Simple Advanced ML Dashboard Test
================================

Basic validation of the advanced ML dashboard system.
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """Test the API endpoints"""
    base_url = "http://localhost:5000/api/advanced-ml"
    
    endpoints = [
        '/predictions',
        '/performance', 
        '/feature-importance',
        '/market-analysis',
        '/learning-data',
        '/performance-metrics'
    ]
    
    results = {}
    print("🔍 Testing API Endpoints")
    print("=" * 40)
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                success = data.get('success', False)
                
                if success:
                    print(f"✅ {endpoint} - {duration:.3f}s - {len(data)} keys")
                    results[endpoint] = {'status': 'success', 'duration': duration, 'keys': len(data)}
                else:
                    print(f"❌ {endpoint} - Invalid response")
                    results[endpoint] = {'status': 'invalid_response'}
            else:
                print(f"❌ {endpoint} - HTTP {response.status_code}")
                results[endpoint] = {'status': 'http_error', 'code': response.status_code}
                
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)}")
            results[endpoint] = {'status': 'error', 'error': str(e)}
    
    return results

def test_dashboard_page():
    """Test the main dashboard page"""
    try:
        print("\n🎨 Testing Dashboard Page")
        print("=" * 40)
        
        start_time = time.time()
        response = requests.get("http://localhost:5000/advanced-ml-dashboard", timeout=10)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            content = response.text
            
            # Check for key elements
            required_elements = [
                'advanced-ml-dashboard',
                'Chart.js',
                'Socket.IO',
                'predictions-grid',
                'performance-overview'
            ]
            
            found = sum(1 for element in required_elements if element in content)
            print(f"✅ Dashboard page loaded - {duration:.3f}s")
            print(f"   Found {found}/{len(required_elements)} required elements")
            
            return {'status': 'success', 'duration': duration, 'elements_found': found}
        else:
            print(f"❌ Dashboard page - HTTP {response.status_code}")
            return {'status': 'http_error', 'code': response.status_code}
            
    except Exception as e:
        print(f"❌ Dashboard page - Error: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def test_static_files():
    """Test static files"""
    print("\n📁 Testing Static Files")
    print("=" * 40)
    
    static_files = [
        '/static/css/advanced-ml-dashboard-new.css',
        '/static/js/advanced-ml-dashboard.js'
    ]
    
    results = {}
    
    for file_path in static_files:
        try:
            start_time = time.time()
            response = requests.get(f"http://localhost:5000{file_path}", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                size_kb = len(response.content) / 1024
                print(f"✅ {file_path.split('/')[-1]} - {size_kb:.1f}KB")
                results[file_path] = {'status': 'success', 'size_kb': size_kb}
            else:
                print(f"❌ {file_path.split('/')[-1]} - HTTP {response.status_code}")
                results[file_path] = {'status': 'http_error', 'code': response.status_code}
                
        except Exception as e:
            print(f"❌ {file_path.split('/')[-1]} - Error: {str(e)}")
            results[file_path] = {'status': 'error', 'error': str(e)}
    
    return results

def test_refresh_endpoint():
    """Test the refresh predictions endpoint"""
    try:
        print("\n🔄 Testing Refresh Endpoint")
        print("=" * 40)
        
        start_time = time.time()
        response = requests.post("http://localhost:5000/api/advanced-ml/refresh-predictions", timeout=5)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Refresh predictions - {duration:.3f}s")
                return {'status': 'success', 'duration': duration}
            else:
                print(f"❌ Refresh predictions - Invalid response")
                return {'status': 'invalid_response'}
        else:
            print(f"❌ Refresh predictions - HTTP {response.status_code}")
            return {'status': 'http_error', 'code': response.status_code}
            
    except Exception as e:
        print(f"❌ Refresh predictions - Error: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def main():
    """Run all tests"""
    print("🚀 Advanced ML Dashboard - Simple Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    api_results = test_api_endpoints()
    dashboard_result = test_dashboard_page()
    static_results = test_static_files()
    refresh_result = test_refresh_endpoint()
    
    total_duration = time.time() - start_time
    
    # Calculate summary
    all_results = list(api_results.values()) + [dashboard_result, refresh_result] + list(static_results.values())
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.get('status') == 'success')
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🏆 Test Summary")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Duration: {total_duration:.3f}s")
    
    # Save results
    results = {
        'summary': {
            'total_tests': total_tests,
            'passed': passed_tests,
            'success_rate': f"{success_rate:.1f}%",
            'total_duration': f"{total_duration:.3f}s"
        },
        'api_results': api_results,
        'dashboard_result': dashboard_result,
        'static_results': static_results,
        'refresh_result': refresh_result,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('simple_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: simple_test_results.json")
    
    return 0 if success_rate >= 70 else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
