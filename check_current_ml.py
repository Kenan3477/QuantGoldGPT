#!/usr/bin/env python3
"""
Check Current ML Predictions API Response
"""
import requests
import json

def check_current_predictions():
    print("🔍 Checking Current ML Predictions...")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:5000/api/ml-predictions')
        data = response.json()
        
        print("📊 API Response Summary:")
        print(f"Status: {response.status_code}")
        print(f"Current Price: ${data.get('current_price', 'Unknown')}")
        print(f"Source: {data.get('source', 'Unknown')}")
        print(f"Data Quality: {data.get('data_quality', 'Unknown')}")
        print(f"Generated At: {data.get('generated_at', 'Unknown')}")
        
        predictions = data.get('predictions', [])
        print(f"\n📈 Predictions ({len(predictions)} found):")
        
        for i, pred in enumerate(predictions):
            print(f"{i+1}. {pred.get('timeframe', 'Unknown')}: ${pred.get('predicted_price', 0):.2f} ({pred.get('change_percent', 0):+.3f}%) - {pred.get('direction', 'Unknown')} [{pred.get('confidence', 0):.0%}]")
        
        # Check if it's using real or fake data
        if 'fallback' in data and data['fallback']:
            print("\n❌ ISSUE: Using fallback/fake predictions!")
        elif data.get('source') == 'intelligent_ml_api':
            print("\n✅ Good: Using intelligent ML engine")
        elif data.get('source') == 'enhanced_ml_engine':
            print("\n✅ Good: Using enhanced ML engine")
        else:
            print(f"\n⚠️ Warning: Unknown source: {data.get('source')}")
        
        # Check for economic factors
        if 'economic_factors' in data:
            print("✅ Enhanced features detected")
        else:
            print("❌ Missing enhanced features")
        
        return data
        
    except Exception as e:
        print(f"❌ Error checking predictions: {e}")
        return None

if __name__ == "__main__":
    check_current_predictions()
