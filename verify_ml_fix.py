#!/usr/bin/env python3
"""
Final Verification of ML Prediction Accuracy
"""
import requests
import json
import time

def verify_ml_predictions():
    print("🎯 Final Verification of ML Prediction Accuracy...")
    print("=" * 60)
    
    try:
        # Get current gold price first
        price_response = requests.get('http://localhost:5000/api/gold-price', timeout=10)
        if price_response.status_code == 200:
            price_data = price_response.json()
            current_market_price = price_data.get('price', 0)
            print(f"📈 Current Market Price: ${current_market_price:.2f}")
        else:
            current_market_price = 3394.80  # Fallback
            print(f"⚠️  Using fallback price: ${current_market_price:.2f}")
        
        # Get ML predictions
        ml_response = requests.get('http://localhost:5000/api/ml-predictions', timeout=15)
        
        if ml_response.status_code == 200:
            data = ml_response.json()
            
            print(f"\n✅ ML API Status: {data.get('success', False)}")
            api_current_price = data.get('current_price', 0)
            print(f"💰 API Current Price: ${api_current_price:.2f}")
            
            # Check if current price is accurate
            if abs(api_current_price - current_market_price) < 1.0:
                print("✅ Current price is accurate")
            else:
                print(f"⚠️  Price mismatch: API=${api_current_price:.2f} vs Market=${current_market_price:.2f}")
            
            predictions = data.get('predictions', [])
            print(f"\n🔮 ML Predictions ({len(predictions)} total):")
            print("-" * 60)
            
            all_predictions_valid = True
            
            for i, pred in enumerate(predictions, 1):
                timeframe = pred.get('timeframe', 'N/A')
                current_price = api_current_price
                predicted_price = pred.get('predicted_price', 0)
                change_percent = pred.get('change_percent', 0)
                direction = pred.get('direction', 'unknown')
                confidence = pred.get('confidence', 0) * 100
                
                print(f"\n  📊 Prediction {i} - {timeframe}")
                print(f"     Current Price: ${current_price:.2f}")
                print(f"     Predicted Price: ${predicted_price:.2f}")
                print(f"     Expected Change: {change_percent:+.3f}%")
                print(f"     Direction: {direction.upper()}")
                print(f"     Confidence: {confidence:.1f}%")
                
                # Mathematical verification
                if current_price > 0:
                    calculated_change = ((predicted_price - current_price) / current_price) * 100
                    price_from_percentage = current_price * (1 + change_percent / 100)
                    
                    print(f"     Calculated Change: {calculated_change:+.3f}%")
                    print(f"     Price from %: ${price_from_percentage:.2f}")
                    
                    # Check math consistency
                    if abs(calculated_change - change_percent) < 0.1:
                        print("     ✅ Math is consistent")
                    else:
                        print("     ❌ Math inconsistency!")
                        all_predictions_valid = False
                    
                    # Check directional logic
                    if change_percent > 0.1 and predicted_price > current_price:
                        print("     ✅ Bullish prediction: higher price")
                    elif change_percent < -0.1 and predicted_price < current_price:
                        print("     ✅ Bearish prediction: lower price")
                    elif abs(change_percent) <= 0.1:
                        print("     ✅ Neutral prediction: stable price")
                    else:
                        print("     ❌ Directional logic error!")
                        all_predictions_valid = False
                        
                    # Check if prediction makes sense
                    if abs(change_percent) > 10:
                        print("     ⚠️  Large price movement predicted")
                    
            print("\n" + "=" * 60)
            if all_predictions_valid:
                print("🎉 ALL ML PREDICTIONS ARE MATHEMATICALLY ACCURATE!")
                print("✅ Issue Fixed: Predictions now show correct percentages")
            else:
                print("❌ Some predictions still have issues")
                
        else:
            print(f"❌ ML API Error: {ml_response.status_code}")
            print(ml_response.text[:500])
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("⏳ Waiting 2 seconds for ML predictions to update...")
    time.sleep(2)
    verify_ml_predictions()
