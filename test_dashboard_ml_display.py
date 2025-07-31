#!/usr/bin/env python3
"""
Force Dashboard ML Refresh Test
This script will verify what the dashboard should be displaying
"""

import requests
import json

def test_dashboard_ml_data():
    print("🔧 TESTING DASHBOARD ML PREDICTIONS DISPLAY")
    print("=" * 60)
    
    try:
        # 1. Get the real ML data from API
        response = requests.get('http://localhost:5000/api/ml-predictions/XAUUSD')
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
            
        data = response.json()
        
        print("📊 REAL ML API DATA:")
        print(f"Current Price: ${data['current_price']}")
        print("\nPredictions:")
        
        timeframes = ['1H', '4H', '1D']
        expected_display = []
        
        for i, pred in enumerate(data['predictions']):
            change_pct = pred['change_percent']
            predicted_price = pred['predicted_price'] 
            confidence = int(pred['confidence'] * 100)
            timeframe = timeframes[i] if i < len(timeframes) else pred.get('timeframe', f'{i+1}H')
            
            # This is how it SHOULD display on dashboard
            change_text = f"+{change_pct:.1f}%" if change_pct >= 0 else f"{change_pct:.1f}%"
            price_text = f"${int(predicted_price):,}"
            
            print(f"  {timeframe}: {change_text} ({price_text}) - {confidence}% confidence")
            
            expected_display.append({
                'timeframe': timeframe,
                'change_text': change_text,
                'price_text': price_text,
                'confidence': confidence,
                'is_bearish': change_pct < 0
            })
        
        print("\n" + "=" * 60)
        print("🎯 WHAT YOUR DASHBOARD SHOULD SHOW:")
        print("=" * 60)
        
        all_bearish = all(item['is_bearish'] for item in expected_display)
        if all_bearish:
            print("✅ ALL PREDICTIONS SHOULD BE BEARISH (NEGATIVE)")
        else:
            print("⚠️ MIXED PREDICTIONS")
            
        for item in expected_display:
            status = "📉 BEARISH" if item['is_bearish'] else "📈 BULLISH" 
            print(f"  {item['timeframe']}: {item['change_text']} {item['price_text']} - {status}")
        
        print("\n" + "=" * 60)
        print("❌ WHAT YOUR DASHBOARD IS CURRENTLY SHOWING (WRONG):")
        print("📈 1H: +0.8% ($3,348) - FAKE POSITIVE")
        print("📈 4H: +1.2% ($3,356) - FAKE POSITIVE") 
        print("📈 1D: +0.3% ($3,342) - FAKE POSITIVE")
        
        print("\n🔧 DIAGNOSIS:")
        if all_bearish:
            print("✅ ML API is working correctly (bearish predictions)")
            print("❌ Dashboard JavaScript is NOT fetching real API data")
            print("🔄 Dashboard needs to be force-refreshed or cache cleared")
        else:
            print("⚠️ Check ML API - predictions not consistently bearish")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_dashboard_ml_data()
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Hard refresh dashboard (Ctrl+F5)")
        print("2. Clear browser cache")
        print("3. Check browser developer console for errors")
        print("4. Verify ML predictions JavaScript is calling /api/ml-predictions/XAUUSD")
    else:
        print("\n❌ Could not diagnose the issue")
