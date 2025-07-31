import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_prediction_api import get_ml_predictions
    
    async def test():
        try:
            print("🔮 Testing ML Predictions...")
            result = await get_ml_predictions('GC=F')
            print(f'✅ Success: {result.get("success", False)}')
            
            if result.get('predictions'):
                print(f'📊 Available timeframes: {list(result["predictions"].keys())}')
                for tf, pred in result['predictions'].items():
                    direction = pred.get('direction', 'unknown')
                    confidence = pred.get('confidence', 0) * 100
                    predicted_price = pred.get('predicted_price', 0)
                    print(f'  {tf}: {direction.upper()} - {confidence:.1f}% confidence - ${predicted_price:.2f}')
            else:
                print(f'❌ Error: {result.get("error", "Unknown error")}')
                
            print(f"\n📈 Model Version: {result.get('model_version', 'Unknown')}")
            print(f"💰 Current Price: ${result.get('current_price', 0):.2f}")
            
        except Exception as e:
            print(f'❌ Test failed: {e}')
            import traceback
            traceback.print_exc()

    # Run the test
    asyncio.run(test())
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('Make sure ml_prediction_api.py is in the same directory')
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    import traceback
    traceback.print_exc()
