#!/usr/bin/env python3
"""
Test Positions API Endpoint
"""
import requests
import json

def test_positions_api():
    try:
        url = "http://localhost:5000/api/positions/open"
        print(f"🔍 Testing: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response:")
            print(f"   Type: {type(data)}")
            print(f"   Length: {len(data) if isinstance(data, list) else 'N/A'}")
            
            if isinstance(data, list) and len(data) == 0:
                print("✅ Correct: Empty array returned (no positions)")
            elif isinstance(data, list) and len(data) > 0:
                print("⚠️  Warning: API returned positions when none should exist")
                for pos in data[:3]:  # Show first 3
                    print(f"   - Position: {pos}")
            else:
                print(f"❌ Unexpected response format: {data}")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask app may not be running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_positions_api()
