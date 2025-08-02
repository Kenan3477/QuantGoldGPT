import requests
import json

def test_api_endpoint(url, method='GET', data=None):
    try:
        if method == 'GET':
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=data, timeout=5)
        
        print(f"\n=== {url} ===")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None

# Test all endpoints
base_url = "http://127.0.0.1:5000/api"

print("Testing ML Dashboard APIs...")
health = test_api_endpoint(f"{base_url}/ml-health")
performance = test_api_endpoint(f"{base_url}/ml-performance")
accuracy = test_api_endpoint(f"{base_url}/ml-accuracy")
predictions = test_api_endpoint(f"{base_url}/ml-predictions", 'POST', {"timeframes": ["15m", "1h"]})
