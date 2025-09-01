#!/usr/bin/env python3
"""
Test the minimal server
"""
import requests
import json

def test_minimal():
    print("🔍 Testing minimal server on port 5001")
    
    try:
        # Test live-gold-price
        resp1 = requests.get("http://127.0.0.1:5001/api/live-gold-price", timeout=5)
        print(f"Live Gold Price: {resp1.status_code} - {resp1.json()}")
        
        # Test signals/active  
        resp2 = requests.get("http://127.0.0.1:5001/api/signals/active", timeout=5)
        print(f"Active Signals: {resp2.status_code} - {resp2.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_minimal()
