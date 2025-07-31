#!/usr/bin/env python3
"""
Real Prediction Validator
Validates predictions that have reached their target times with real market data
"""

from ml_engine_tracker import MLEngineTracker
from price_storage_manager import get_current_gold_price
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_ready_predictions():
    """Validate predictions that are ready based on real time passage"""
    print("⏰ Running REAL Prediction Validation...")
    print("=" * 50)
    
    # Get current real market price
    current_price = get_current_gold_price()
    print(f"📊 Current XAUUSD price: ${current_price}")
    
    # Initialize tracker
    tracker = MLEngineTracker('goldgpt_ml_tracking.db')
    
    # Validate predictions that have reached their target times
    results = tracker.validate_predictions(current_price, 'XAUUSD')
    
    if results:
        print(f"\n✅ Validated {len(results)} predictions!")
        
        for result in results[:10]:  # Show first 10 results
            status = "✅ CORRECT" if result.direction_correct else "❌ WRONG"
            print(f"  Prediction {result.prediction_id}: {result.accuracy_score:.1f}% accuracy {status}")
        
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")
        
        # Get updated engine stats
        print(f"\n📈 Updated Engine Accuracy:")
        stats = tracker.get_dashboard_stats()
        
        for engine in stats.get('engines', []):
            accuracy = engine['overall_accuracy']
            badge = engine['badge']['label']
            total = engine['total_predictions']
            validated = engine['validated_predictions']
            
            print(f"  {engine['display_name']}: {accuracy:.1f}% accuracy ({badge})")
            print(f"    {validated}/{total} predictions validated")
        
        print(f"\n🎯 Real accuracy data is now available!")
        print(f"💡 Refresh your dashboard to see the updated accuracy badges!")
        
    else:
        print("⚠️ No predictions were ready for validation")

if __name__ == "__main__":
    validate_ready_predictions()
