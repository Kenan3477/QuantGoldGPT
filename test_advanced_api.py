import requests
import json

# Test the new advanced signal endpoints
base_url = "http://localhost:5000"

def test_endpoint(endpoint, method='GET'):
    try:
        url = f"{base_url}{endpoint}"
        print(f"\n🔍 Testing {method} {endpoint}")
        
        if method == 'GET':
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(json.dumps(data, indent=2))
            return data
        else:
            print(f"❌ Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

# Test all new advanced signal endpoints
print("🎯 Testing Advanced Signal System APIs")
print("=" * 50)

# 1. Test signal generation
test_endpoint("/api/generate-signal")

# 2. Test active signals
test_endpoint("/api/active-signals")

# 3. Test signal statistics
test_endpoint("/api/signal-stats")

# 4. Test force signal generation
test_endpoint("/api/force-signal", "POST")

print("\n✅ All tests completed!")
