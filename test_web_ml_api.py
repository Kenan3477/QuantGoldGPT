#!/usr/bin/env python3
"""
Test Fixed ML Predictions via Web API
"""
import requests
import json

def test_web_ml_predictions():
    print("🌐 Testing Fixed ML Predictions via Web API")
    print("=" * 60)
    
    # Test the web API endpoint
    try:
        url = "http://localhost:5000/api/ml-predictions/XAUUSD"
        print(f"Testing endpoint: {url}")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Web API Response Successful!")
            
            # Check current price
            current_price = data.get('current_price', 0)
            print(f"Current Price: ${current_price:.2f}")
            
            # Validate it's using real-time price (not fake $2600)
            if current_price > 3000:
                print("✅ Using real-time prices (not fake $2600)")
            else:
                print(f"❌ Still using fake/old prices: ${current_price:.2f}")
            
            # Check predictions
            predictions = data.get('predictions', {})
            if predictions:
                print(f"\n📈 ML Predictions ({len(predictions)} timeframes):")
                for timeframe, pred in predictions.items():
                    if isinstance(pred, dict):
                        pred_price = pred.get('predicted_price', 0)
                        direction = pred.get('direction', 'unknown')
                        change_pct = pred.get('price_change_percent', 0)
                        confidence = pred.get('confidence', 0)
                        
                        print(f"  {timeframe}: ${pred_price:.2f} ({direction})")
                        print(f"    Change: {change_pct:+.2f}% | Confidence: {confidence:.1%}")
                        
                        # Check if prediction is realistic
                        if abs(change_pct) > 2.0:
                            print(f"    ⚠️ Large change detected")
                        else:
                            print(f"    ✅ Realistic prediction")
            
            # Check API source
            api_source = data.get('api_source', 'unknown')
            print(f"\nAPI Source: {api_source}")
            
            if api_source == 'enhanced_ml':
                print("✅ Using enhanced ML system!")
            elif api_source == 'fallback':
                print("⚠️ Using fallback system")
            else:
                print("❓ Unknown API source")
                
        else:
            print(f"❌ Web API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Web application is not running")
        print("Please start the app with: python app.py")
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_web_ml_predictions()
