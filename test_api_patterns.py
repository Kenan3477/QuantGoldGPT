import requests
import json

print("Testing pattern detection API on Railway...")

try:
    # Test the Railway deployment
    response = requests.get('https://web-production-41882.up.railway.app/api/live/patterns', timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        patterns = data.get('current_patterns', [])
        print(f"✅ API returned {len(patterns)} patterns")
        print(f"📊 Data source: {data.get('data_source', 'Unknown')}")
        print(f"🔢 Live pattern count: {data.get('live_pattern_count', 0)}")
        print(f"💰 Current price: ${data.get('current_price', 0)}")
        
        if patterns:
            print("\n📈 Top 3 patterns:")
            for i, pattern in enumerate(patterns[:3]):
                pattern_name = pattern.get('pattern', 'Unknown')
                confidence = pattern.get('confidence', 0)
                signal = pattern.get('signal', 'UNKNOWN')
                timestamp = pattern.get('timestamp', 'Unknown')
                print(f"{i+1}. {pattern_name}: {confidence:.1f}% ({signal}) - {timestamp}")
        else:
            print("❌ No patterns returned from API")
            print("Full response keys:", list(data.keys()))
    else:
        print(f"❌ API returned status code: {response.status_code}")
        print("Response:", response.text[:500])
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
