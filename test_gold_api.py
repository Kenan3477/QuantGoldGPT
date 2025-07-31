#!/usr/bin/env python3
"""
Test the new Gold API endpoint
"""
import requests
import json

def test_gold_api():
    print('🧪 Testing Gold API: https://api.gold-api.com/price/XAU')
    print('=' * 50)
    
    try:
        response = requests.get('https://api.gold-api.com/price/XAU', timeout=10)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'Response Data: {json.dumps(data, indent=2)}')
            
            # Check for common price fields
            price_fields = ['price', 'ask', 'last', 'value', 'rate']
            for field in price_fields:
                if field in data:
                    print(f'✅ Found price field "{field}": {data[field]}')
        else:
            print(f'❌ Failed with status: {response.status_code}')
            print(f'Response: {response.text}')
            
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    test_gold_api()
