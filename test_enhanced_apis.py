#!/usr/bin/env python3
"""
Test enhanced signals APIs
"""
import requests

def test_enhanced_apis():
    try:
        print("Testing enhanced signals APIs...")
        
        # Test active signals
        r1 = requests.get('http://localhost:5000/api/enhanced-signals/active')
        print(f"Active signals status: {r1.status_code}")
        if r1.status_code == 200:
            data = r1.json()
            print(f"Active signals count: {data.get('count', 0)}")
        
        # Test performance
        r2 = requests.get('http://localhost:5000/api/enhanced-signals/performance') 
        print(f"Performance status: {r2.status_code}")
        if r2.status_code == 200:
            data = r2.json()
            perf = data.get('performance', {})
            print(f"Total signals: {perf.get('total_signals', 0)}")
            print(f"Successful signals: {perf.get('successful_signals', 0)}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_enhanced_apis()
