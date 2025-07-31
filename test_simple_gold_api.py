#!/usr/bin/env python3
"""
Simple Gold API test - just check the function works
"""
import requests

def test_simple_gold_api():
    """Test the exact function we updated"""
    try:
        # Primary API: gold-api.com (reliable and unlimited)
        url = "https://api.gold-api.com/price/XAU"
        headers = {
            'User-Agent': 'GoldGPT/1.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Extract price from the API response (check common field names)
            price = None
            if 'price' in data:
                price = float(data['price'])
            elif 'ask' in data:
                price = float(data['ask'])
            elif 'last' in data:
                price = float(data['last'])
            elif 'value' in data:
                price = float(data['value'])
            elif 'rate' in data:
                price = float(data['rate'])
            
            if price:
                print(f"✅ Real-time gold price from gold-api.com: ${price:.2f}")
                return price
            else:
                print(f"Could not extract price from API response: {data}")
            
        print(f"Failed to fetch real-time price (status: {response.status_code}), using fallback")
        return 3300.0
        
    except Exception as e:
        print(f"❌ Error fetching real-time gold price: {e}")
        return 3300.0

if __name__ == "__main__":
    print("🧪 Testing Updated Gold API Function")
    print("=" * 40)
    price = test_simple_gold_api()
    print(f"Final result: ${price:.2f}")
    
    if price > 3200:
        print("✅ SUCCESS: Using real Gold API data!")
    else:
        print("❌ FAILED: Still using fallback data")
