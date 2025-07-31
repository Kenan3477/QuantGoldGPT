import requests
import time

print("🧪 Testing Flask app connection...")

try:
    response = requests.get('http://localhost:5000/ping', timeout=5)
    print(f"✅ Connection successful!")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:100]}")
except requests.exceptions.ConnectionError:
    print("❌ Connection refused - Flask app might not be running on localhost:5000")
except requests.exceptions.Timeout:
    print("❌ Connection timeout - Flask app is not responding")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n🔍 Testing different URLs...")

urls = [
    'http://localhost:5000/',
    'http://127.0.0.1:5000/',
    'http://localhost:5000/ping',
    'http://127.0.0.1:5000/ping'
]

for url in urls:
    try:
        print(f"Testing: {url}")
        response = requests.get(url, timeout=3)
        print(f"  ✅ Status: {response.status_code}, Length: {len(response.text)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(0.5)
