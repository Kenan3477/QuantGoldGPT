#!/usr/bin/env python3

import requests
import json
from datetime import datetime

def test_timeframe_predictions():
    """Test timeframe predictions API to see if synthetic analysis is working"""
    try:
        print("🔍 Testing Timeframe Predictions API...")
        
        url = "http://localhost:5000/api/timeframe-predictions"
        response = requests.get(url, timeout=10)
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("📊 Response Data:")
            print(json.dumps(data, indent=2))
            
            # Check if we have varied signals
            signals = []
            for timeframe, prediction in data.get('predictions', {}).items():
                signal = prediction.get('signal', 'UNKNOWN')
                signals.append(signal)
                print(f"⏰ {timeframe}: {signal}")
            
            # Count unique signals
            unique_signals = set(signals)
            print(f"\n🎯 Unique signals found: {len(unique_signals)}")
            print(f"📋 Signal types: {list(unique_signals)}")
            
            if len(unique_signals) > 1:
                print("✅ SUCCESS: Dynamic signals working!")
            else:
                print("❌ ISSUE: All signals are the same")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_ml_predictions():
    """Test ML predictions API"""
    try:
        print("\n🔍 Testing ML Predictions API...")
        
        url = "http://localhost:5000/api/ml-predictions"
        response = requests.get(url, timeout=10)
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("📊 ML Predictions:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_gold_price():
    """Test gold price API"""
    try:
        print("\n🔍 Testing Gold Price API...")
        
        url = "http://localhost:5000/api/live-gold-price"
        response = requests.get(url, timeout=10)
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"💰 Gold Price: ${data.get('price', 'N/A')}")
            print(f"📅 Updated: {data.get('updated_at', 'N/A')}")
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing GoldGPT APIs")
    print("=" * 50)
    
    test_gold_price()
    test_timeframe_predictions()
    test_ml_predictions()
    
    print("\n" + "=" * 50)
    print("✅ API Testing Complete")
