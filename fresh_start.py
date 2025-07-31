#!/usr/bin/env python3
"""
Fresh Start Script - Clear DB and Generate New Predictions
"""

import sqlite3
from dual_ml_prediction_system import DualMLPredictionSystem
from ml_engine_tracker import MLEngineTracker

def fresh_start():
    """Clear database and generate fresh predictions"""
    print("🧹 Starting Fresh - Clearing All Predictions...")
    
    # Clear database
    db_path = 'goldgpt_ml_tracking.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Delete all existing predictions
    cursor.execute('DELETE FROM ml_engine_predictions')
    cursor.execute('DELETE FROM ml_engine_performance')
    cursor.execute('DELETE FROM daily_accuracy_summary')
    conn.commit()
    
    print("✅ Database cleared successfully")
    
    # Check count
    cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
    remaining = cursor.fetchone()[0]
    print(f"📊 Remaining predictions: {remaining}")
    conn.close()
    
    # Generate fresh predictions
    print("\n🚀 Generating Fresh Dual Predictions...")
    system = DualMLPredictionSystem()
    result = system.get_dual_predictions()
    
    if result.get('enhanced') and result.get('intelligent'):
        print("✅ Fresh predictions generated successfully!")
        
        # Show first prediction from each engine
        enhanced_1h = result['enhanced']['predictions'][0]
        intelligent_1h = result['intelligent']['predictions'][0]
        
        print(f"\n📈 Enhanced ML (1H): {enhanced_1h['direction']} - ${enhanced_1h['predicted_price']} ({enhanced_1h['change_percent']:+.2f}%)")
        print(f"🤖 Intelligent ML (1H): {intelligent_1h['direction']} - ${intelligent_1h['predicted_price']} ({intelligent_1h['change_percent']:+.2f}%)")
        
        # Check database count after predictions
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
        new_count = cursor.fetchone()[0]
        print(f"\n📊 New predictions stored: {new_count}")
        conn.close()
        
    else:
        print("❌ Failed to generate fresh predictions")

if __name__ == "__main__":
    fresh_start()
