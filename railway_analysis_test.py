"""
Railway Technical Analysis Verification Report
Testing: https://web-production-41882.up.railway.app/
"""

import requests
import json
from datetime import datetime

def test_technical_analysis():
    base_url = "https://web-production-41882.up.railway.app"
    
    print("🔍 RAILWAY TECHNICAL ANALYSIS VERIFICATION")
    print("=" * 50)
    print(f"🌐 Testing: {base_url}")
    print(f"⏰ Test Time: {datetime.now()}")
    print()
    
    # Test signal generation multiple times to check for patterns
    signals = []
    for i in range(3):
        try:
            print(f"🎯 Signal Test #{i+1}")
            response = requests.get(f"{base_url}/api/signals/generate", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                signals.append(data)
                print(f"   ✅ Success: {response.status_code}")
                
                if 'signal' in data:
                    signal = data['signal']
                    signal_type = signal.get('type', 'N/A')
                    price = signal.get('price', 0)
                    confidence = signal.get('confidence', 0)
                    
                    print(f"   📊 Signal: {signal_type}")
                    print(f"   💰 Price: ${price}")
                    print(f"   🎯 Confidence: {confidence}")
                    
                    # Check for technical analysis indicators in response
                    response_text = json.dumps(data, indent=2)
                    technical_keywords = ['rsi', 'macd', 'sma', 'technical', 'analysis', 'volatility', 'bias']
                    found_keywords = [kw for kw in technical_keywords if kw.lower() in response_text.lower()]
                    
                    if found_keywords:
                        print(f"   ✅ Technical indicators found: {found_keywords}")
                    else:
                        print("   ⚠️ No technical indicators in response")
                        
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
            
        print()
    
    # Analysis of results
    print("📊 ANALYSIS SUMMARY")
    print("-" * 30)
    
    if len(signals) > 1:
        prices = [s.get('signal', {}).get('price', 0) for s in signals if 'signal' in s]
        types = [s.get('signal', {}).get('type', '') for s in signals if 'signal' in s]
        
        if prices:
            price_range = max(prices) - min(prices)
            print(f"💰 Price range: ${min(prices):.2f} - ${max(prices):.2f} (range: ${price_range:.2f})")
            
            # Check if prices are realistic for gold
            avg_price = sum(prices) / len(prices)
            if 2500 <= avg_price <= 4000:
                print("✅ Prices are in realistic gold range")
            else:
                print("⚠️ Prices may be random (outside typical gold range)")
                
        if types:
            buy_count = types.count('BUY')
            sell_count = types.count('SELL')
            print(f"📈 Signal distribution: {buy_count} BUY, {sell_count} SELL")
            
            if buy_count == len(types) or sell_count == len(types):
                print("⚠️ All signals are the same type - possible issue")
            else:
                print("✅ Mixed signal types - good sign")
    
    # Final verdict
    print("\n🎯 VERDICT")
    print("-" * 20)
    
    if signals:
        # Look for signs of real technical analysis
        has_technical_indicators = any('rsi' in str(s).lower() or 'macd' in str(s).lower() for s in signals)
        has_realistic_prices = any(2500 <= s.get('signal', {}).get('price', 0) <= 4000 for s in signals)
        
        if has_technical_indicators:
            print("✅ REAL TECHNICAL ANALYSIS DETECTED!")
            print("   Your Railway deployment is using genuine technical analysis")
        elif has_realistic_prices:
            print("⚠️ POSSIBLY REAL ANALYSIS")
            print("   Prices look realistic but no technical indicators visible")
        else:
            print("❌ LIKELY RANDOM SIGNALS")
            print("   No technical indicators and unrealistic prices")
    else:
        print("❌ NO SIGNALS GENERATED")
        print("   Railway deployment may be down or misconfigured")

if __name__ == "__main__":
    test_technical_analysis()
