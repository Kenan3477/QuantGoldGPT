#!/usr/bin/env python3
"""
Quick Fix for ML Prediction Count Issue
Simple check and correction without async complications
"""
import sqlite3
from datetime import datetime

def fix_prediction_count():
    print("🔧 Quick Fix for ML Prediction Count")
    print("=" * 40)
    
    # Check the main tracking database
    try:
        conn = sqlite3.connect('goldgpt_ml_tracking.db')
        cursor = conn.cursor()
        
        # Get current count
        cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
        current_count = cursor.fetchone()[0]
        print(f"📊 Current predictions in tracking DB: {current_count}")
        
        # Show breakdown by engine
        cursor.execute("SELECT engine_name, COUNT(*) FROM ml_engine_predictions GROUP BY engine_name")
        for engine, count in cursor.fetchall():
            print(f"   {engine}: {count} predictions")
        
        # Check if we have more than 6 (should only be 3 per engine max)
        if current_count > 6:
            print(f"⚠️  Too many predictions! Should be max 6, found {current_count}")
            
            # Keep only the latest 3 predictions per engine
            print("🧹 Cleaning up to keep only latest 3 per engine...")
            
            for engine in ['enhanced_ml_prediction_engine', 'intelligent_ml_predictor']:
                # Get all predictions for this engine, ordered by newest first
                cursor.execute("""
                    SELECT id FROM ml_engine_predictions 
                    WHERE engine_name = ? 
                    ORDER BY created_at DESC
                """, (engine,))
                
                prediction_ids = [row[0] for row in cursor.fetchall()]
                
                # Keep only the first 3 (newest), delete the rest
                if len(prediction_ids) > 3:
                    ids_to_delete = prediction_ids[3:]  # Everything after the first 3
                    for pred_id in ids_to_delete:
                        cursor.execute("DELETE FROM ml_engine_predictions WHERE id = ?", (pred_id,))
                    print(f"   Deleted {len(ids_to_delete)} old predictions for {engine}")
            
            conn.commit()
            
            # Check new count
            cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
            new_count = cursor.fetchone()[0]
            print(f"✅ Cleaned up! New count: {new_count}")
            
        else:
            print("✅ Prediction count is acceptable")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

    # Also check other databases that might be causing confusion
    print("\n🔍 Checking other databases:")
    
    other_dbs = ['goldgpt_ml_predictions.db', 'ml_predictions.db']
    for db_name in other_dbs:
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            
            # Check for prediction tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%prediction%'")
            tables = cursor.fetchall()
            
            total_other = 0
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                total_other += count
                if count > 0:
                    print(f"   {db_name}.{table_name}: {count} records")
            
            if total_other > 0:
                print(f"⚠️  Found {total_other} predictions in {db_name} - these should NOT be counted")
            
            conn.close()
            
        except Exception as e:
            print(f"   {db_name}: {e}")

if __name__ == "__main__":
    fix_prediction_count()
