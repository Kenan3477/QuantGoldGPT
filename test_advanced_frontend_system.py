#!/usr/bin/env python3
"""
GoldGPT Advanced Frontend System Integration Test
Comprehensive testing of the Trading212-inspired dashboard system
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

class AdvancedFrontendSystemTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results with timestamp"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'status': status,
            'details': details
        }
        self.test_results.append(result)
        
        status_icon = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }.get(status, '❓')
        
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   └─ {details}")
    
    def test_dashboard_accessibility(self):
        """Test if the advanced ML dashboard is accessible"""
        try:
            response = requests.get(f"{self.base_url}/advanced-ml-dashboard", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                # Check for key dashboard components
                required_elements = [
                    'dashboard-container',
                    'predictions-grid',
                    'analysis-section',
                    'learning-dashboard',
                    'predictionChart',
                    'advanced-ml-dashboard.css',
                    'advanced-ml-dashboard.js'
                ]
                
                missing_elements = []
                for element in required_elements:
                    if element not in content:
                        missing_elements.append(element)
                
                if not missing_elements:
                    self.log_test("Dashboard Accessibility", "PASS", 
                                f"All required elements present (Response: {response.status_code})")
                else:
                    self.log_test("Dashboard Accessibility", "WARNING",
                                f"Missing elements: {', '.join(missing_elements)}")
                    
            else:
                self.log_test("Dashboard Accessibility", "FAIL",
                            f"HTTP {response.status_code}: {response.reason}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Dashboard Accessibility", "FAIL", f"Request error: {str(e)}")
    
    def test_static_assets(self):
        """Test if CSS and JavaScript assets are served correctly"""
        assets = [
            '/static/css/advanced-ml-dashboard.css',
            '/static/js/advanced-ml-dashboard.js',
            '/static/css/advanced-ml-dashboard-new.css'
        ]
        
        for asset in assets:
            try:
                response = requests.get(f"{self.base_url}{asset}", timeout=5)
                
                if response.status_code == 200:
                    content_length = len(response.content)
                    if content_length > 1000:  # Reasonable file size check
                        self.log_test(f"Static Asset {asset}", "PASS",
                                    f"Loaded successfully ({content_length} bytes)")
                    else:
                        self.log_test(f"Static Asset {asset}", "WARNING",
                                    f"File seems small ({content_length} bytes)")
                else:
                    self.log_test(f"Static Asset {asset}", "FAIL",
                                f"HTTP {response.status_code}: {response.reason}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Static Asset {asset}", "FAIL", f"Request error: {str(e)}")
    
    def test_advanced_ml_api_endpoints(self):
        """Test if the Advanced ML API endpoints are available"""
        endpoints = [
            '/api/advanced-ml/status',
            '/api/advanced-ml/predictions',
            '/api/advanced-ml/accuracy-stats',
            '/api/advanced-ml/feature-importance',
            '/api/advanced-ml/health'
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if 'success' in data and data.get('success'):
                            self.log_test(f"API Endpoint {endpoint}", "PASS",
                                        "Valid JSON response with success=True")
                        else:
                            self.log_test(f"API Endpoint {endpoint}", "WARNING",
                                        f"Response: {data}")
                    except json.JSONDecodeError:
                        self.log_test(f"API Endpoint {endpoint}", "WARNING",
                                    "Response is not valid JSON")
                        
                elif response.status_code == 404:
                    self.log_test(f"API Endpoint {endpoint}", "FAIL",
                                "Endpoint not found - Advanced ML API may not be integrated")
                else:
                    self.log_test(f"API Endpoint {endpoint}", "FAIL",
                                f"HTTP {response.status_code}: {response.reason}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"API Endpoint {endpoint}", "FAIL", f"Request error: {str(e)}")
    
    def test_websocket_availability(self):
        """Test if WebSocket endpoint is available"""
        try:
            import socketio
            sio = socketio.Client()
            
            @sio.event
            def connect():
                self.log_test("WebSocket Connection", "PASS", "Successfully connected to Socket.IO")
                sio.disconnect()
            
            @sio.event
            def connect_error(data):
                self.log_test("WebSocket Connection", "FAIL", f"Connection error: {data}")
            
            sio.connect(self.base_url, timeout=5)
            time.sleep(1)  # Give time for connection
            
        except ImportError:
            self.log_test("WebSocket Connection", "WARNING", "socketio library not available for testing")
        except Exception as e:
            self.log_test("WebSocket Connection", "FAIL", f"Connection failed: {str(e)}")
    
    def test_responsive_design_elements(self):
        """Test if the dashboard includes responsive design elements"""
        try:
            response = requests.get(f"{self.base_url}/advanced-ml-dashboard", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                responsive_indicators = [
                    '@media',  # CSS media queries
                    'viewport',  # Viewport meta tag
                    'responsive',  # Responsive classes
                    'grid-template-columns',  # CSS Grid
                    'flex-wrap',  # Flexbox responsive
                ]
                
                found_indicators = []
                for indicator in responsive_indicators:
                    if indicator in content:
                        found_indicators.append(indicator)
                
                if len(found_indicators) >= 3:
                    self.log_test("Responsive Design", "PASS",
                                f"Found indicators: {', '.join(found_indicators)}")
                else:
                    self.log_test("Responsive Design", "WARNING",
                                f"Limited responsive indicators: {', '.join(found_indicators)}")
                    
        except requests.exceptions.RequestException as e:
            self.log_test("Responsive Design", "FAIL", f"Request error: {str(e)}")
    
    def test_javascript_functionality(self):
        """Test if JavaScript components are properly integrated"""
        try:
            response = requests.get(f"{self.base_url}/advanced-ml-dashboard", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                js_components = [
                    'AdvancedMLDashboard',  # Main class
                    'Chart.js',  # Chart library
                    'socket.io',  # WebSocket library
                    'addEventListener',  # Event handling
                    'WebSocket',  # WebSocket support
                    'fetch(',  # API calls
                ]
                
                found_components = []
                for component in js_components:
                    if component in content:
                        found_components.append(component)
                
                if len(found_components) >= 4:
                    self.log_test("JavaScript Integration", "PASS",
                                f"Found components: {', '.join(found_components)}")
                else:
                    self.log_test("JavaScript Integration", "WARNING",
                                f"Limited JS components: {', '.join(found_components)}")
                    
        except requests.exceptions.RequestException as e:
            self.log_test("JavaScript Integration", "FAIL", f"Request error: {str(e)}")
    
    def test_trading212_design_elements(self):
        """Test if Trading212-inspired design elements are present"""
        try:
            response = requests.get(f"{self.base_url}/static/css/advanced-ml-dashboard-new.css", timeout=5)
            
            if response.status_code == 200:
                css_content = response.text
                
                design_elements = [
                    '#0066cc',  # Primary blue color
                    '#00b386',  # Success green color
                    'box-shadow',  # Card shadows
                    'border-radius',  # Rounded corners
                    'transition',  # Smooth animations
                    'hover:',  # Hover effects
                    '@keyframes',  # Custom animations
                ]
                
                found_elements = []
                for element in design_elements:
                    if element in css_content:
                        found_elements.append(element)
                
                if len(found_elements) >= 5:
                    self.log_test("Trading212 Design System", "PASS",
                                f"Found design elements: {len(found_elements)}/7")
                else:
                    self.log_test("Trading212 Design System", "WARNING",
                                f"Limited design elements: {len(found_elements)}/7")
                    
        except requests.exceptions.RequestException as e:
            self.log_test("Trading212 Design System", "FAIL", f"CSS request error: {str(e)}")
    
    def test_performance_metrics(self):
        """Test dashboard loading performance"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/advanced-ml-dashboard", timeout=10)
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                content_size = len(response.content)
                
                if load_time < 2.0:  # Target: < 2 seconds
                    self.log_test("Dashboard Performance", "PASS",
                                f"Load time: {load_time:.2f}s, Size: {content_size:,} bytes")
                elif load_time < 5.0:
                    self.log_test("Dashboard Performance", "WARNING",
                                f"Slow load time: {load_time:.2f}s, Size: {content_size:,} bytes")
                else:
                    self.log_test("Dashboard Performance", "FAIL",
                                f"Very slow load time: {load_time:.2f}s")
                    
        except requests.exceptions.RequestException as e:
            self.log_test("Dashboard Performance", "FAIL", f"Request error: {str(e)}")
    
    def run_comprehensive_tests(self):
        """Run all frontend system tests"""
        print("🧪 Starting GoldGPT Advanced Frontend System Tests...")
        print("=" * 60)
        
        # Core functionality tests
        self.test_dashboard_accessibility()
        self.test_static_assets()
        self.test_advanced_ml_api_endpoints()
        
        # Integration tests
        self.test_websocket_availability()
        self.test_javascript_functionality()
        self.test_responsive_design_elements()
        
        # Design and performance tests
        self.test_trading212_design_elements()
        self.test_performance_metrics()
        
        print("\n" + "=" * 60)
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warning_tests = len([r for r in self.test_results if r['status'] == 'WARNING'])
        
        print("📊 FRONTEND SYSTEM TEST REPORT")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Warnings: {warning_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🚀 FRONTEND SYSTEM STATUS: EXCELLENT")
            print("   The advanced dashboard is fully functional with Trading212-inspired design!")
        elif success_rate >= 60:
            print("\n✅ FRONTEND SYSTEM STATUS: GOOD")
            print("   Core functionality working, minor issues to address.")
        else:
            print("\n⚠️ FRONTEND SYSTEM STATUS: NEEDS ATTENTION")
            print("   Several components need fixes before production use.")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status_icon = {'PASS': '✅', 'FAIL': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️'}.get(result['status'], '❓')
            print(f"   {status_icon} {result['test_name']}: {result['status']}")
            if result['details']:
                print(f"      └─ {result['details']}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"frontend_test_report_{timestamp}.json"
        
        try:
            with open(report_filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total_tests': total_tests,
                        'passed': passed_tests,
                        'failed': failed_tests,
                        'warnings': warning_tests,
                        'success_rate': success_rate
                    },
                    'test_results': self.test_results
                }, f, indent=2)
            
            print(f"\n💾 Detailed report saved: {report_filename}")
            
        except Exception as e:
            print(f"\n⚠️ Could not save report: {str(e)}")

def main():
    """Run comprehensive frontend system tests"""
    tester = AdvancedFrontendSystemTester()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()
