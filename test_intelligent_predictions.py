#!/usr/bin/env python3
"""
Test the intelligent predictions API to make sure fake prices are gone
"""
import requests
import json

def test_intelligent_predictions():
    print("🚀 Testing Intelligent ML Predictions API")
    print("=" * 50)
    
    try:
        # Test the Flask endpoint if it's running
        url = "http://localhost:5000/api/ml-predictions/XAUUSD"
        print(f"Testing: {url}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            current_price = data.get('current_price', 0)
            
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Current Price: ${current_price:.2f}")
            
            # Check if we're still getting fake prices
            if 2600 <= current_price <= 2700:
                print("❌ STILL USING FAKE PRICES!")
                return False
            elif 3300 <= current_price <= 3400:
                print("✅ Using REAL-TIME prices!")
                
                # Check predictions
                predictions = data.get('predictions', [])
                print(f"\n📈 Found {len(predictions)} predictions:")
                
                all_realistic = True
                for pred in predictions:
                    tf = pred.get('timeframe', 'N/A')
                    price = pred.get('predicted_price', 0)
                    change = pred.get('change_percent', 0)
                    
                    print(f"  {tf}: ${price:.2f} ({change:+.2f}%)")
                    
                    # Check if prediction is realistic (within 2% of current)
                    if abs(price - current_price) > (current_price * 0.02):
                        print(f"    ⚠️ Large deviation from current price")
                        all_realistic = False
                    else:
                        print(f"    ✅ Realistic prediction")
                
                return all_realistic
            else:
                print(f"⚠️ Unexpected price: ${current_price:.2f}")
                return False
                
        else:
            print(f"❌ API Error: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Flask app not running")
        
        # Test the direct function instead
        print("\n🧪 Testing direct intelligent predictor...")
        try:
            from intelligent_ml_predictor import get_intelligent_ml_predictions
            
            result = get_intelligent_ml_predictions("XAUUSD")
            current_price = result['current_price']
            
            print(f"📊 Direct test - Current Price: ${current_price:.2f}")
            
            if 3300 <= current_price <= 3400:
                print("✅ Direct test - Using REAL prices!")
                
                predictions = result['predictions']
                for pred in predictions:
                    tf = pred['timeframe']
                    price = pred['predicted_price']
                    change = pred['change_percent']
                    print(f"  {tf}: ${price:.2f} ({change:+.2f}%)")
                
                return True
            else:
                print(f"❌ Direct test - Wrong price range: ${current_price:.2f}")
                return False
                
        except Exception as e:
            print(f"❌ Direct test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_intelligent_predictions()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS! Intelligent predictions working with REAL prices!")
        print("✅ No more fake price targets (2634, 2674, 2629)")
        print("✅ All predictions anchored to real-time gold prices")
    else:
        print("❌ FAILED! Still issues with price predictions")
        print("🔧 Check the system for remaining problems")
