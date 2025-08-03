#!/usr/bin/env python3
"""
Test ML Dashboard API functionality
"""
import requests
import json
import sys

def test_ml_dashboard():
    """Test ML Dashboard API endpoints"""
    base_url = 'http://localhost:5000/api'
    endpoints = ['/ml-predictions', '/ml-health', '/ml-performance', '/ml-accuracy']
    
    print('🧠 Testing ML Dashboard API Endpoints...')
    print('=' * 50)
    
    results = {}
    
    for endpoint in endpoints:
        try:
            print(f'🔍 Testing {endpoint}...')
            response = requests.get(f'{base_url}{endpoint}', timeout=10)
            print(f'   Status: {response.status_code}')
            
            if response.status_code == 200:
                data = response.json()
                results[endpoint] = data
                
                if endpoint == '/ml-predictions':
                    predictions = data.get('predictions', [])
                    print(f'   📈 Found {len(predictions)} predictions')
                    
                    for pred in predictions[:3]:  # Show first 3
                        timeframe = pred.get('timeframe', 'Unknown')
                        direction = pred.get('direction', 'N/A')
                        confidence = pred.get('confidence', 0)
                        target = pred.get('target_price', 'N/A')
                        print(f'      {timeframe}: {direction} ({confidence}%) → ${target}')
                        
                elif endpoint == '/ml-health':
                    health = data.get('health', {})
                    status = health.get('status', 'Unknown')
                    systems = health.get('systems', {})
                    active = sum(systems.values()) if systems else 0
                    total = len(systems) if systems else 0
                    
                    print(f'   🏥 System status: {status}')
                    print(f'   🔧 Active systems: {active}/{total}')
                    
                    for system, active in systems.items():
                        status_icon = '✅' if active else '❌'
                        print(f'      {status_icon} {system}')
                        
            else:
                print(f'   ❌ Error {response.status_code}: {response.text[:100]}')
                
        except requests.exceptions.ConnectionError:
            print(f'   ❌ Connection failed - Server not running on localhost:5000')
            return False
        except Exception as e:
            print(f'   ❌ Error: {str(e)[:100]}')
            
        print()
    
    return True

if __name__ == '__main__':
    if test_ml_dashboard():
        print('✅ ML Dashboard API test completed')
    else:
        print('❌ ML Dashboard API test failed')
        sys.exit(1)
