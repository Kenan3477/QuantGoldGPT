#!/usr/bin/env python3
"""
Generate Today's Daily ML Predictions
Only one prediction set per day per engine
"""

import asyncio
from datetime import datetime
from dual_ml_prediction_system import DualMLPredictionSystem
from ml_engine_tracker import MLEngineTracker

async def generate_todays_predictions():
    """Generate today's predictions from both ML engines"""
    print("🚀 Generating Today's Daily ML Predictions")
    print("=" * 50)
    
    # Initialize systems
    dual_system = DualMLPredictionSystem()
    tracker = MLEngineTracker('goldgpt_ml_tracking.db')
    
    try:
        # Generate predictions from both engines
        print("📈 Generating predictions from both ML engines...")
        result = await dual_system.get_dual_predictions()
        
        if result.get('success'):
            enhanced_predictions = result.get('enhanced_ml', {}).get('predictions', [])
            intelligent_predictions = result.get('intelligent_ml', {}).get('predictions', [])
            current_price = result.get('enhanced_ml', {}).get('current_price', 0)
            
            print(f"💰 Current Gold Price: ${current_price}")
            
            # Store Enhanced ML predictions
            if enhanced_predictions:
                print("\n📊 Enhanced ML Engine:")
                for pred in enhanced_predictions:
                    tracker.store_prediction(
                        engine_name="enhanced_ml",
                        timeframe=pred['timeframe'],
                        predicted_price=pred['predicted_price'],
                        current_price=current_price,
                        change_percent=pred['change_percent'],
                        direction=pred['direction'],
                        confidence=pred['confidence']
                    )
                    print(f"   {pred['timeframe']}: ${pred['predicted_price']} ({pred['change_percent']:+.2f}%) - {pred['direction']}")
            
            # Store Intelligent ML predictions
            if intelligent_predictions:
                print("\n🤖 Intelligent ML Engine:")
                for pred in intelligent_predictions:
                    tracker.store_prediction(
                        engine_name="intelligent_ml",
                        timeframe=pred['timeframe'],
                        predicted_price=pred['predicted_price'],
                        current_price=current_price,
                        change_percent=pred['change_percent'],
                        direction=pred['direction'],
                        confidence=pred['confidence']
                    )
                    print(f"   {pred['timeframe']}: ${pred['predicted_price']} ({pred['change_percent']:+.2f}%) - {pred['direction']}")
            
            # Show summary
            total_predictions = len(enhanced_predictions) + len(intelligent_predictions)
            print(f"\n✅ Stored {total_predictions} predictions for today ({datetime.now().strftime('%Y-%m-%d')})")
            print("📝 Daily tracking: One prediction set per day per engine")
            
        else:
            print("❌ Failed to generate predictions")
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")

if __name__ == "__main__":
    asyncio.run(generate_todays_predictions())
