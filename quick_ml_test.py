#!/usr/bin/env python3
"""
Quick ML Prediction Test
"""
import asyncio
from ml_prediction_api import get_ml_predictions, get_real_time_gold_price

async def quick_test():
    print("🔧 Quick ML Prediction Test")
    
    # Get current price
    current_price = get_real_time_gold_price()
    print(f"Current price: ${current_price:.2f}")
    
    # Get predictions
    try:
        result = await get_ml_predictions("GC=F")
        
        if result.get('success'):
            print("✅ Predictions successful!")
            print(f"Current price in predictions: ${result.get('current_price', 'N/A')}")
            
            predictions = result.get('predictions', {})
            for timeframe, pred in predictions.items():
                if isinstance(pred, dict):
                    predicted_price = pred.get('predicted_price', 0)
                    direction = pred.get('direction', 'unknown')
                    change_pct = pred.get('price_change_percent', 0)
                    print(f"{timeframe}: ${predicted_price:.2f} ({direction}, {change_pct:+.2f}%)")
        else:
            print(f"❌ Predictions failed: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
