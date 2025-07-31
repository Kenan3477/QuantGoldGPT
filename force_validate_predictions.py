#!/usr/bin/env python3
"""
Force ML Prediction Validation for Testing
Simulates time passage to validate older predictions
"""

import sqlite3
import json
from datetime import datetime, timedelta
from ml_engine_tracker import MLEngineTracker
from price_storage_manager import get_current_gold_price

def force_validate_predictions():
    """Force validation of predictions for testing accuracy"""
    print("🧪 Force Validating ML Predictions for Testing...")
    
    db_path = 'goldgpt_ml_tracking.db'
    tracker = MLEngineTracker(db_path)
    
    # Get current price for validation
    current_price = get_current_gold_price()
    print(f"📊 Current XAUUSD price: ${current_price}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find predictions that are older than their timeframes
        current_time = datetime.now()
        
        # Get some predictions to validate manually
        cursor.execute("""
            SELECT id, engine_name, timeframe, predicted_price, current_price, 
                   change_percent, direction, prediction_time, target_validation_time
            FROM ml_engine_predictions 
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 10
        """)
        
        predictions = cursor.fetchall()
        validated_count = 0
        
        for pred in predictions:
            pred_id, engine_name, timeframe, predicted_price, orig_price, change_percent, direction, pred_time, target_time = pred
            
            # Simulate that this prediction is now ready for validation
            # Calculate actual metrics
            actual_change_percent = ((current_price - orig_price) / orig_price) * 100
            
            # Determine actual direction
            if actual_change_percent > 0.1:
                actual_direction = 'bullish'
            elif actual_change_percent < -0.1:
                actual_direction = 'bearish'
            else:
                actual_direction = 'neutral'
            
            # Calculate accuracy metrics
            price_accuracy = 100 - abs(((predicted_price - current_price) / current_price) * 100)
            direction_correct = (direction.lower() == actual_direction.lower())
            
            # Overall accuracy score (combination of price and direction accuracy)
            accuracy_score = (price_accuracy * 0.7) + (80 if direction_correct else 20)
            accuracy_score = max(0, min(100, accuracy_score))  # Clamp between 0-100
            
            # Update the prediction with validation data
            cursor.execute("""
                UPDATE ml_engine_predictions
                SET actual_price = ?, actual_change_percent = ?, actual_direction = ?,
                    price_accuracy = ?, direction_correct = ?, accuracy_score = ?,
                    status = 'validated', validation_date = ?
                WHERE id = ?
            """, (
                current_price, actual_change_percent, actual_direction,
                price_accuracy, direction_correct, accuracy_score,
                current_time, pred_id
            ))
            
            validated_count += 1
            print(f"✅ Validated prediction {pred_id} ({engine_name} {timeframe}): {accuracy_score:.1f}% accuracy")
        
        conn.commit()
        print(f"🎯 Force validated {validated_count} predictions")
        
        # Update engine performance stats
        tracker._recalculate_all_engine_stats()
        
        # Show updated stats
        stats = tracker.get_dashboard_stats()
        print("\n📈 Updated Engine Performance:")
        for engine in stats.get('engines', []):
            print(f"  {engine['display_name']}: {engine['overall_accuracy']:.1f}% accuracy ({engine['badge']['label']})")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error during force validation: {e}")

if __name__ == "__main__":
    force_validate_predictions()
