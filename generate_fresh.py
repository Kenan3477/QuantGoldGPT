#!/usr/bin/env python3
import asyncio
from dual_ml_prediction_system import DualMLPredictionSystem

async def generate_fresh_predictions():
    print("🚀 Generating fresh dual ML predictions...")

    try:
        system = DualMLPredictionSystem()
        result = await system.get_dual_predictions()
        
        print("✅ Fresh predictions generated!")
        
        # Show Enhanced ML predictions
        enhanced = result.get('enhanced', {})
        if enhanced.get('predictions'):
            print("\n📈 Enhanced ML Engine:")
            for pred in enhanced['predictions']:
                print(f"  {pred['timeframe']}: {pred['direction']} ${pred['predicted_price']} ({pred['change_percent']:+.2f}%)")
        
        # Show Intelligent ML predictions  
        intelligent = result.get('intelligent', {})
        if intelligent.get('predictions'):
            print("\n🤖 Intelligent ML Engine:")
            for pred in intelligent['predictions']:
                print(f"  {pred['timeframe']}: {pred['direction']} ${pred['predicted_price']} ({pred['change_percent']:+.2f}%)")
                
        print(f"\n💰 Current Gold Price: ${enhanced.get('current_price', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(generate_fresh_predictions())
