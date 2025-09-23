import requests
import json

def expose_railway_truth():
    """
    Expose what Railway is ACTUALLY returning vs what the console logs claim
    """
    print("🕵️ EXPOSING THE TRUTH ABOUT RAILWAY DEPLOYMENT")
    print("=" * 60)
    
    url = "https://web-production-41882.up.railway.app/api/signals/generate"
    
    try:
        # Test signal generation multiple times
        for i in range(3):
            print(f"\n🎯 TEST #{i+1}: Raw API Response")
            print("-" * 40)
            
            response = requests.get(url, timeout=15)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Raw Response Text:")
            print(response.text)
            print()
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print("Parsed JSON:")
                    print(json.dumps(data, indent=2))
                    
                    # Analyze the response structure
                    print("\n🔍 ANALYSIS:")
                    if 'signal' in data:
                        signal = data['signal']
                        
                        # Check for technical analysis indicators
                        technical_fields = ['rsi', 'macd', 'technical_analysis', 'volatility', 'bias', 'reasoning']
                        found_technical = [field for field in technical_fields if field in str(data).lower()]
                        
                        if found_technical:
                            print(f"✅ REAL TECHNICAL ANALYSIS: Found {found_technical}")
                        else:
                            print("❌ FAKE SIGNALS: No technical indicators found")
                            print("   This looks like random generation!")
                            
                        # Check signal structure
                        print(f"Signal Type: {signal.get('type', 'N/A')}")
                        print(f"Price: ${signal.get('price', 'N/A')}")
                        print(f"Confidence: {signal.get('confidence', 'N/A')}")
                        
                        # Check if price is realistic for gold
                        price = signal.get('price', 0)
                        if isinstance(price, (int, float)):
                            if 2500 <= price <= 4000:
                                print(f"✅ Price in gold range: ${price}")
                            else:
                                print(f"❌ Suspicious price: ${price} (not typical for gold)")
                        
                    print("\n" + "="*50)
                except json.JSONDecodeError:
                    print("❌ Invalid JSON response!")
                    
    except Exception as e:
        print(f"💥 Request failed: {e}")
        
    print("\n🎯 VERDICT:")
    print("If you see simple responses with just 'type', 'price', 'confidence'")  
    print("and NO 'rsi', 'macd', 'technical_analysis' fields...")
    print("Then your Railway deployment is using FAKE random signals!")

if __name__ == "__main__":
    expose_railway_truth()
