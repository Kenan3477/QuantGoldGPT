#!/usr/bin/env python3
"""
Clean Duplicate ML Predictions
Keep only the latest valid prediction per engine per timeframe for today
"""

import sqlite3
from datetime import datetime

def clean_duplicate_predictions():
    """Remove duplicate predictions and keep only latest per engine/timeframe/day"""
    print("🧹 Cleaning duplicate ML predictions...")
    
    db_path = 'goldgpt_ml_tracking.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # First, show current state
        cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
        total_before = cursor.fetchone()[0]
        print(f"📊 Total predictions before cleanup: {total_before}")
        
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Find duplicates - keep only the latest prediction per engine/timeframe for today
        cursor.execute('''
        DELETE FROM ml_engine_predictions 
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM ml_engine_predictions 
            WHERE DATE(created_at) = ? 
            AND predicted_price IS NOT NULL 
            AND predicted_price > 0
            GROUP BY engine_name, timeframe
        )
        AND DATE(created_at) = ?
        ''', (today, today))
        
        deleted_today = cursor.rowcount
        print(f"🗑️  Deleted {deleted_today} duplicate predictions from today")
        
        # Also clean up any predictions with missing or invalid data
        cursor.execute('''
        DELETE FROM ml_engine_predictions 
        WHERE predicted_price IS NULL 
        OR predicted_price <= 0 
        OR engine_name IS NULL 
        OR timeframe IS NULL
        ''')
        
        deleted_invalid = cursor.rowcount
        print(f"🗑️  Deleted {deleted_invalid} invalid predictions")
        
        # Show final state
        cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
        total_after = cursor.fetchone()[0]
        print(f"📊 Total predictions after cleanup: {total_after}")
        
        # Show today's cleaned predictions
        cursor.execute('''
        SELECT engine_name, timeframe, predicted_price, change_percent, direction, created_at
        FROM ml_engine_predictions 
        WHERE DATE(created_at) = ?
        ORDER BY engine_name, 
                 CASE timeframe 
                     WHEN '1H' THEN 1 
                     WHEN '4H' THEN 2 
                     WHEN '1D' THEN 3 
                 END
        ''', (today,))
        
        predictions = cursor.fetchall()
        print(f"\n📅 Today's Clean Predictions ({today}):")
        
        current_engine = None
        for engine, tf, price, change, direction, created in predictions:
            if engine != current_engine:
                current_engine = engine
                display_name = "Enhanced ML" if "enhanced" in engine else "Intelligent ML"
                print(f"   {display_name}:")
            
            print(f"     {tf}: ${price} ({change:+.2f}%) - {direction}")
        
        conn.commit()
        print("\n✅ Database cleaned successfully!")
        
    except Exception as e:
        print(f"❌ Error cleaning database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    clean_duplicate_predictions()
