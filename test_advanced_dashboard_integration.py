#!/usr/bin/env python3
"""
Advanced ML Dashboard Integration Test Suite
============================================

Comprehensive testing for the GoldGPT Advanced ML Dashboard system including:
- API endpoint validation
- WebSocket functionality testing
- Frontend integration validation
- Performance benchmarking
- Real-time features testing
"""

import asyncio
import aiohttp
import socketio
import json
import time
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import requests
import threading

class AdvancedDashboardTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/advanced-ml"
        self.websocket_url = base_url
        self.sio = socketio.SimpleClient()
        self.test_results = []
        self.websocket_responses = {}
        
    def log_test(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration': f"{duration:.3f}s",
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name} ({duration:.3f}s)")
        if details:
            print(f"     └─ {details}")
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test all API endpoints"""
        print("\n🔍 Testing API Endpoints")
        print("=" * 50)
        
        endpoints = [
            ('/predictions', 'GET', 'Multi-timeframe Predictions'),
            ('/performance', 'GET', 'Performance Metrics'),
            ('/feature-importance', 'GET', 'Feature Importance'),
            ('/refresh-predictions', 'POST', 'Refresh Predictions'),
            ('/market-analysis', 'GET', 'Market Analysis'),
            ('/learning-data', 'GET', 'Learning Data'),
            ('/performance-metrics', 'GET', 'System Performance')
        ]
        
        results = {}
        
        for endpoint, method, description in endpoints:
            start_time = time.time()
            try:
                url = f"{self.api_url}{endpoint}"
                
                if method == 'GET':
                    response = requests.get(url, timeout=10)
                else:
                    response = requests.post(url, timeout=10)
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate response structure
                    if 'success' in data and data['success']:
                        results[endpoint] = {
                            'status': 'success',
                            'response_time': duration,
                            'data_keys': list(data.keys())
                        }
                        self.log_test(f"{description} API", True, 
                                    f"Response time: {duration:.3f}s, Keys: {len(data.keys())}", duration)
                    else:
                        results[endpoint] = {'status': 'invalid_response', 'response_time': duration}
                        self.log_test(f"{description} API", False, 
                                    "Invalid response structure", duration)
                else:
                    results[endpoint] = {'status': 'http_error', 'code': response.status_code}
                    self.log_test(f"{description} API", False, 
                                f"HTTP {response.status_code}", duration)
                    
            except Exception as e:
                duration = time.time() - start_time
                results[endpoint] = {'status': 'error', 'error': str(e)}
                self.log_test(f"{description} API", False, str(e), duration)
        
        return results
    
    def test_websocket_events(self) -> Dict[str, Any]:
        """Test WebSocket event handling"""
        print("\n🔌 Testing WebSocket Events")
        print("=" * 50)
        
        events_to_test = [
            ('request_dashboard_data', {'types': ['all']}, 'Dashboard Data'),
            ('request_live_predictions', {'timeframe': '1h'}, 'Live Predictions'),
            ('request_learning_update', {}, 'Learning Updates'),
            ('request_advanced_ml_prediction', {'timeframe': '1H'}, 'Advanced ML Prediction')
        ]
        
        results = {}
        
        try:
            # Connect to WebSocket
            start_time = time.time()
            self.sio.connect(self.websocket_url)
            connection_time = time.time() - start_time
            self.log_test("WebSocket Connection", True, 
                        f"Connected in {connection_time:.3f}s", connection_time)
            
            for event_name, event_data, description in events_to_test:
                start_time = time.time()
                
                try:
                    # Set up response handler
                    response_received = threading.Event()
                    response_data = {}
                    
                    def response_handler(data):
                        nonlocal response_data
                        response_data.update(data)
                        response_received.set()
                    
                    # Map event to expected response
                    response_event_map = {
                        'request_dashboard_data': 'dashboard_data_update',
                        'request_live_predictions': 'live_prediction_update',
                        'request_learning_update': 'learning_update',
                        'request_advanced_ml_prediction': 'advanced_ml_prediction'
                    }
                    
                    response_event = response_event_map.get(event_name)
                    if response_event:
                        self.sio.on(response_event, response_handler)
                    
                    # Emit event
                    self.sio.emit(event_name, event_data)
                    
                    # Wait for response
                    if response_received.wait(timeout=5):
                        duration = time.time() - start_time
                        
                        if response_data.get('success'):
                            results[event_name] = {
                                'status': 'success',
                                'response_time': duration,
                                'response_keys': list(response_data.keys())
                            }
                            self.log_test(f"{description} WebSocket", True,
                                        f"Response time: {duration:.3f}s", duration)
                        else:
                            results[event_name] = {
                                'status': 'error_response',
                                'error': response_data.get('error', 'Unknown error')
                            }
                            self.log_test(f"{description} WebSocket", False,
                                        response_data.get('error', 'Error response'), duration)
                    else:
                        duration = time.time() - start_time
                        results[event_name] = {'status': 'timeout'}
                        self.log_test(f"{description} WebSocket", False,
                                    "Response timeout", duration)
                        
                except Exception as e:
                    duration = time.time() - start_time
                    results[event_name] = {'status': 'exception', 'error': str(e)}
                    self.log_test(f"{description} WebSocket", False, str(e), duration)
            
            # Disconnect
            self.sio.disconnect()
            
        except Exception as e:
            self.log_test("WebSocket Connection", False, str(e))
            results['connection_error'] = str(e)
        
        return results
    
    def test_frontend_integration(self) -> Dict[str, Any]:
        """Test frontend template and static files"""
        print("\n🎨 Testing Frontend Integration")
        print("=" * 50)
        
        results = {}
        
        # Test dashboard template
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/advanced-ml-dashboard", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                content = response.text
                
                # Check for key elements
                key_elements = [
                    'advanced-ml-dashboard',
                    'Chart.js',
                    'Socket.IO',
                    'dashboard-grid',
                    'predictions-grid',
                    'performance-overview'
                ]
                
                found_elements = sum(1 for element in key_elements if element in content)
                
                results['dashboard_template'] = {
                    'status': 'success',
                    'response_time': duration,
                    'elements_found': f"{found_elements}/{len(key_elements)}"
                }
                
                self.log_test("Dashboard Template", True,
                            f"Found {found_elements}/{len(key_elements)} key elements", duration)
            else:
                results['dashboard_template'] = {
                    'status': 'http_error',
                    'code': response.status_code
                }
                self.log_test("Dashboard Template", False,
                            f"HTTP {response.status_code}", duration)
                
        except Exception as e:
            duration = time.time() - start_time
            results['dashboard_template'] = {'status': 'error', 'error': str(e)}
            self.log_test("Dashboard Template", False, str(e), duration)
        
        # Test static files
        static_files = [
            ('/static/css/advanced-ml-dashboard-new.css', 'Dashboard CSS'),
            ('/static/js/advanced-ml-dashboard.js', 'Dashboard JavaScript')
        ]
        
        for file_path, description in static_files:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{file_path}", timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    size_kb = len(response.content) / 1024
                    results[file_path] = {
                        'status': 'success',
                        'response_time': duration,
                        'size_kb': f"{size_kb:.1f}KB"
                    }
                    self.log_test(f"{description} File", True,
                                f"Size: {size_kb:.1f}KB", duration)
                else:
                    results[file_path] = {
                        'status': 'http_error',
                        'code': response.status_code
                    }
                    self.log_test(f"{description} File", False,
                                f"HTTP {response.status_code}", duration)
                    
            except Exception as e:
                duration = time.time() - start_time
                results[file_path] = {'status': 'error', 'error': str(e)}
                self.log_test(f"{description} File", False, str(e), duration)
        
        return results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("\n⚡ Running Performance Benchmarks")
        print("=" * 50)
        
        results = {}
        
        # API Response Time Benchmark
        try:
            endpoint = f"{self.api_url}/predictions"
            response_times = []
            
            for i in range(10):
                start_time = time.time()
                response = requests.get(endpoint, timeout=5)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    response_times.append(duration)
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                min_time = min(response_times)
                
                results['api_performance'] = {
                    'average_ms': f"{avg_time * 1000:.1f}",
                    'max_ms': f"{max_time * 1000:.1f}",
                    'min_ms': f"{min_time * 1000:.1f}",
                    'requests_tested': len(response_times)
                }
                
                self.log_test("API Performance Benchmark", True,
                            f"Avg: {avg_time * 1000:.1f}ms, Max: {max_time * 1000:.1f}ms", avg_time)
            else:
                results['api_performance'] = {'status': 'no_valid_responses'}
                self.log_test("API Performance Benchmark", False, "No valid responses")
                
        except Exception as e:
            results['api_performance'] = {'status': 'error', 'error': str(e)}
            self.log_test("API Performance Benchmark", False, str(e))
        
        return results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""
🏆 ADVANCED ML DASHBOARD TEST REPORT
{'=' * 60}

📊 Overall Results:
   • Total Tests: {total_tests}
   • Passed: {passed_tests}
   • Failed: {total_tests - passed_tests}
   • Success Rate: {success_rate:.1f}%

📋 Detailed Results:
"""
        
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            report += f"   {status} {result['test']} ({result['duration']})\n"
            if result['details']:
                report += f"      └─ {result['details']}\n"
        
        report += f"\n🕒 Test completed at: {datetime.now().isoformat()}\n"
        
        return report
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🚀 Starting Advanced ML Dashboard Integration Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        api_results = self.test_api_endpoints()
        websocket_results = self.test_websocket_events()
        frontend_results = self.test_frontend_integration()
        performance_results = self.test_performance_benchmarks()
        
        total_duration = time.time() - start_time
        
        # Generate final report
        report = self.generate_test_report()
        print(report)
        
        # Save detailed results
        detailed_results = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed': sum(1 for r in self.test_results if r['success']),
                'success_rate': f"{sum(1 for r in self.test_results if r['success']) / len(self.test_results) * 100:.1f}%",
                'total_duration': f"{total_duration:.3f}s"
            },
            'api_results': api_results,
            'websocket_results': websocket_results,
            'frontend_results': frontend_results,
            'performance_results': performance_results,
            'test_details': self.test_results
        }
        
        return detailed_results

def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:5000"
    
    print(f"🎯 Testing GoldGPT Advanced ML Dashboard at: {base_url}")
    
    tester = AdvancedDashboardTester(base_url)
    results = tester.run_all_tests()
    
    # Save results to file
    with open('advanced_dashboard_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: advanced_dashboard_test_results.json")
    
    # Return appropriate exit code
    success_rate = float(results['summary']['success_rate'].rstrip('%'))
    return 0 if success_rate >= 70 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
