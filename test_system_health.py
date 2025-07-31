#!/usr/bin/env python3
"""
System Health Quick Test
Tests the system status API and validates all components
"""

import requests
import json
from datetime import datetime

def test_system_status():
    """Test the system status endpoint"""
    print("🔧 Testing System Status API...")
    print("=" * 50)
    
    try:
        # Test system status endpoint
        response = requests.get('http://127.0.0.1:5000/api/system-status', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Status API: {response.status_code}")
            print(f"🌟 Overall Status: {data.get('overall_status', 'unknown')}")
            print(f"📅 Timestamp: {data.get('timestamp', 'unknown')}")
            print()
            
            # Check individual systems
            systems = data.get('systems', {})
            print("📊 Individual System Status:")
            print("-" * 30)
            
            for system_name, system_data in systems.items():
                status = system_data.get('status', 'unknown')
                health = system_data.get('health', 'unknown')
                details = system_data.get('details', 'No details')
                
                # Choose emoji based on health
                if health == 'excellent':
                    emoji = '🟢'
                elif health == 'good':
                    emoji = '🟡'
                elif health == 'warning':
                    emoji = '🟠'
                else:
                    emoji = '🔴'
                
                print(f"{emoji} {system_name.replace('_', ' ').title()}")
                print(f"   Status: {status}")
                print(f"   Health: {health}")
                print(f"   Details: {details}")
                
                if 'health_score' in system_data:
                    print(f"   Score: {system_data['health_score']}%")
                print()
                
        else:
            print(f"❌ System Status API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_individual_endpoints():
    """Test individual system endpoints"""
    print("🧪 Testing Individual System Endpoints...")
    print("=" * 50)
    
    endpoints = [
        ('/api/validation-status', 'Validation System'),
        ('/api/ai-analysis/status', 'AI Analysis'),
        ('/api/portfolio', 'Portfolio System'),
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f'http://127.0.0.1:5000{endpoint}', timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: OK")
            else:
                print(f"⚠️ {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)[:50]}...")

def main():
    """Run all tests"""
    print("🚀 GoldGPT System Health Check")
    print("=" * 50)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test main system status
    test_system_status()
    
    # Test individual endpoints
    test_individual_endpoints()
    
    print("🏁 System Health Check Complete!")

if __name__ == "__main__":
    main()
