#!/usr/bin/env python3
"""
Test script to verify all hardcoded fake prices are replaced
"""
import requests
import json

def test_ml_predictions():
    """Test that ML predictions API returns real prices"""
    print("🧪 Testing ML Predictions API for Hardcoded Prices")
    print("=" * 60)
    
    try:
        url = "http://localhost:5000/api/ml-predictions/XAUUSD"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            current_price = data.get('current_price', 0)
            
            print(f"✅ API Response Status: {response.status_code}")
            print(f"📊 Current Price: ${current_price:.2f}")
            
            # Check for fake prices
            if current_price == 2600.0 or current_price == 2650.0:
                print("❌ STILL USING FAKE HARDCODED PRICE!")
                return False
            elif 3300 <= current_price <= 3400:
                print("✅ Using real-time price (in expected range)")
                
                # Check predictions
                predictions = data.get('predictions', [])
                print(f"\n📈 Predictions ({len(predictions)} timeframes):")
                
                for pred in predictions:
                    timeframe = pred.get('timeframe', 'N/A')
                    pred_price = pred.get('predicted_price', 0)
                    change_pct = pred.get('change_percent', 0)
                    
                    print(f"  {timeframe}: ${pred_price:.2f} ({change_pct:+.2f}%)")
                    
                    # Check if predictions are anchored to real price
                    price_diff = abs(pred_price - current_price)
                    max_expected_diff = current_price * 0.05  # 5% max difference
                    
                    if price_diff > max_expected_diff:
                        print(f"    ❌ Prediction too far from current price!")
                        return False
                    else:
                        print(f"    ✅ Anchored to real price")
                
                return True
            else:
                print(f"⚠️  Unexpected price range: ${current_price:.2f}")
                return False
                
        else:
            print(f"❌ API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_price_storage():
    """Test price storage manager directly"""
    print("\n🧪 Testing Price Storage Manager")
    print("=" * 40)
    
    try:
        from price_storage_manager import get_current_gold_price, get_comprehensive_price_data
        
        # Test simple price
        simple_price = get_current_gold_price()
        print(f"Simple Price: ${simple_price:.2f}")
        
        # Test comprehensive data
        price_data = get_comprehensive_price_data()
        print(f"Comprehensive Data: ${price_data.get('price', 0):.2f}")
        print(f"Source: {price_data.get('source', 'unknown')}")
        
        if simple_price == 2600.0 or simple_price == 2650.0:
            print("❌ Price storage returning fake prices!")
            return False
        elif simple_price > 3300:
            print("✅ Price storage working correctly")
            return True
        else:
            print(f"⚠️  Unexpected price: ${simple_price:.2f}")
            return False
            
    except Exception as e:
        print(f"❌ Price Storage Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE FAKE PRICE DETECTION TEST")
    print("=" * 80)
    
    # Test 1: Price Storage Manager
    storage_ok = test_price_storage()
    
    # Test 2: ML Predictions API
    api_ok = test_ml_predictions()
    
    print("\n" + "=" * 80)
    print("📋 FINAL RESULTS:")
    print(f"  Price Storage Manager: {'✅ PASS' if storage_ok else '❌ FAIL'}")
    print(f"  ML Predictions API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if storage_ok and api_ok:
        print("\n🎉 SUCCESS! All hardcoded fake prices have been replaced!")
        print("✅ System now uses real-time gold prices")
    else:
        print("\n❌ FAILED! Some hardcoded prices still exist")
        print("🔧 Check the system for remaining fake values")
