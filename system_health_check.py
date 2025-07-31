#!/usr/bin/env python3
"""
Lightweight System Health Check for GoldGPT
==========================================

Quick system status check that won't overwhelm CPU/Memory
"""

import requests
import time
import json
from datetime import datetime

class LightweightSystemCheck:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        
    def check_system_health(self):
        """Quick system health check"""
        print("🔍 Checking GoldGPT System Health...")
        print("=" * 50)
        
        try:
            # Check basic connectivity
            start_time = time.time()
            response = requests.get(f"{self.base_url}/", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ Web Server: Running ({response_time:.2f}s)")
            else:
                print(f"⚠️ Web Server: HTTP {response.status_code}")
                return False
            
        except Exception as e:
            print(f"❌ Web Server: Not responding - {e}")
            return False
        
        # Check system status endpoint
        try:
            response = requests.get(f"{self.base_url}/api/system-status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    cpu_percent = data.get('system_status', {}).get('current_resources', {}).get('cpu_percent', 0)
                    memory_percent = data.get('system_status', {}).get('current_resources', {}).get('memory_percent', 0)
                    processing_paused = data.get('processing_paused', False)
                    
                    print(f"📊 CPU Usage: {cpu_percent:.1f}%")
                    print(f"💾 Memory Usage: {memory_percent:.1f}%")
                    print(f"⏸️ Processing Paused: {'Yes' if processing_paused else 'No'}")
                    
                    # Cache stats
                    cache_stats = data.get('cache_stats', {})
                    if cache_stats:
                        print(f"📦 Cache Hit Rate: {cache_stats.get('hit_rate', 'N/A')}")
                        print(f"🗄️ Cache Entries: {cache_stats.get('entry_count', 'N/A')}")
                    
                    # System recommendations
                    recommendations = data.get('system_status', {}).get('recommendations', [])
                    if recommendations:
                        print("\n💡 Recommendations:")
                        for rec in recommendations:
                            print(f"   • {rec}")
                    
                    return True
                else:
                    print(f"⚠️ System Status: {data.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"⚠️ System Status API: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ System Status: Error - {e}")
            return False
    
    def check_api_endpoints(self):
        """Quick check of key API endpoints"""
        print("\n🔍 Checking Key API Endpoints...")
        print("=" * 50)
        
        endpoints = [
            ('/api/advanced-ml/status', 'ML System Status'),
            ('/api/system-status', 'System Status'),
            ('/api/cache/stats', 'Cache Stats')
        ]
        
        results = {}
        
        for endpoint, description in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    results[endpoint] = True
                    print(f"✅ {description}: OK ({response_time:.2f}s)")
                else:
                    results[endpoint] = False
                    print(f"❌ {description}: HTTP {response.status_code}")
                    
            except Exception as e:
                results[endpoint] = False
                print(f"❌ {description}: Error - {str(e)[:50]}")
        
        success_rate = sum(results.values()) / len(results) * 100
        print(f"\n📊 API Health: {success_rate:.0f}% ({sum(results.values())}/{len(results)} endpoints)")
        
        return success_rate >= 66  # At least 2/3 endpoints working
    
    def performance_test(self):
        """Quick performance test"""
        print("\n⚡ Quick Performance Test...")
        print("=" * 50)
        
        try:
            # Test ML predictions endpoint (cached should be fast)
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/advanced-ml/predictions", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ ML Predictions: {response_time:.2f}s")
                    
                    # Check if we got actual predictions
                    predictions = data.get('multi_timeframe', [])
                    print(f"📊 Predictions Generated: {len(predictions)}")
                    
                    if response_time < 5.0:
                        print("🚀 Performance: Excellent")
                    elif response_time < 10.0:
                        print("✅ Performance: Good")
                    else:
                        print("⚠️ Performance: Slow")
                    
                    return True
                else:
                    print(f"❌ ML Predictions: API Error - {data.get('error', 'Unknown')}")
                    return False
            else:
                print(f"❌ ML Predictions: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Performance Test: {e}")
            return False
    
    def get_optimization_suggestions(self):
        """Get optimization suggestions based on current state"""
        print("\n💡 Optimization Suggestions...")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/api/system-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('high_cpu'):
                    print("🔥 HIGH CPU DETECTED:")
                    print("   • Stop unnecessary background processes")
                    print("   • Clear cache: POST /api/cache/clear")
                    print("   • Force cleanup: POST /api/system/force-cleanup")
                
                if data.get('high_memory'):
                    print("💾 HIGH MEMORY DETECTED:")
                    print("   • Clear browser tabs")
                    print("   • Restart VS Code Python extension")
                    print("   • Clear cache and restart Flask app")
                
                if data.get('processing_paused'):
                    print("⏸️ PROCESSING PAUSED:")
                    print("   • System is automatically managing resources")
                    print("   • Wait for CPU/Memory to normalize")
                    print("   • Consider reducing refresh rates")
                
                cache_stats = data.get('cache_stats', {})
                hit_rate = cache_stats.get('hit_rate', '0%')
                if hit_rate and float(hit_rate.rstrip('%')) < 50:
                    print("📦 LOW CACHE EFFICIENCY:")
                    print("   • Cache hit rate is low")
                    print("   • System may be generating too many unique requests")
                    print("   • Consider increasing cache TTL")
                
                if not any([data.get('high_cpu'), data.get('high_memory'), data.get('processing_paused')]):
                    print("✅ System running optimally!")
                    print("   • No immediate optimizations needed")
                    print("   • Cache hit rate: " + hit_rate)
                
        except Exception as e:
            print(f"❌ Could not get optimization data: {e}")
    
    def force_cleanup_if_needed(self):
        """Force cleanup if system is under stress"""
        try:
            response = requests.get(f"{self.base_url}/api/system-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('high_cpu') or data.get('high_memory'):
                    print("\n🚑 Forcing system cleanup...")
                    
                    cleanup_response = requests.post(f"{self.base_url}/api/system/force-cleanup", timeout=10)
                    if cleanup_response.status_code == 200:
                        cleanup_data = cleanup_response.json()
                        if cleanup_data.get('success'):
                            print("✅ System cleanup completed")
                            return True
                        else:
                            print(f"❌ Cleanup failed: {cleanup_data.get('message')}")
                    else:
                        print(f"❌ Cleanup request failed: HTTP {cleanup_response.status_code}")
                else:
                    print("✅ No cleanup needed - system resources are normal")
                    return True
                    
        except Exception as e:
            print(f"❌ Cleanup check failed: {e}")
            return False
    
    def run_health_check(self):
        """Run complete health check"""
        print("🏥 GoldGPT System Health Check")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Health checks
        system_ok = self.check_system_health()
        api_ok = self.check_api_endpoints()
        performance_ok = self.performance_test()
        
        # Overall status
        print("\n🏆 Overall System Status")
        print("=" * 50)
        
        if system_ok and api_ok and performance_ok:
            print("✅ SYSTEM HEALTHY")
            print("   All systems operational")
        elif system_ok and api_ok:
            print("⚠️ SYSTEM FUNCTIONAL")
            print("   Basic functionality working, performance may be slow")
        elif system_ok:
            print("⚠️ SYSTEM PARTIAL")
            print("   Web server running, but some APIs may be down")
        else:
            print("❌ SYSTEM CRITICAL")
            print("   Major issues detected, immediate attention required")
        
        # Optimization suggestions
        self.get_optimization_suggestions()
        
        # Auto-cleanup if needed
        print()
        self.force_cleanup_if_needed()
        
        print(f"\n✅ Health check completed at {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Main execution"""
    checker = LightweightSystemCheck()
    checker.run_health_check()

if __name__ == "__main__":
    main()
