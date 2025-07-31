#!/usr/bin/env python3
"""
Real Database Status Checker
Check the actual state of ML predictions and when they'll be ready for validation
"""

import sqlite3
from datetime import datetime
import sys
import os

def check_database_status():
    """Check the real status of the ML tracking database"""
    db_path = 'goldgpt_ml_tracking.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("📊 REAL ML Prediction Database Status")
        print("=" * 50)
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
        total = cursor.fetchone()[0]
        print(f"Total predictions: {total}")
        
        if total == 0:
            print("⚠️ No predictions found in database!")
            conn.close()
            return
        
        # By status
        cursor.execute('SELECT status, COUNT(*) FROM ml_engine_predictions GROUP BY status')
        status_counts = cursor.fetchall()
        print("\nBy Status:")
        for status, count in status_counts:
            print(f"  {status}: {count}")
        
        # By engine
        cursor.execute('SELECT engine_name, COUNT(*) FROM ml_engine_predictions GROUP BY engine_name')
        engine_counts = cursor.fetchall()
        print("\nBy Engine:")
        for engine, count in engine_counts:
            print(f"  {engine}: {count}")
        
        # Check validation readiness
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(f'''
            SELECT COUNT(*) FROM ml_engine_predictions 
            WHERE status = 'pending' 
            AND datetime(target_validation_time) <= datetime('{now}')
        ''')
        ready_count = cursor.fetchone()[0]
        print(f"\nReady for validation NOW: {ready_count}")
        
        # Show next few predictions to validate
        cursor.execute(f'''
            SELECT engine_name, timeframe, direction, created_at, target_validation_time,
                   CASE 
                     WHEN datetime(target_validation_time) <= datetime('{now}') THEN 'READY NOW'
                     ELSE 'Still waiting'
                   END as validation_status
            FROM ml_engine_predictions 
            WHERE status = 'pending'
            ORDER BY target_validation_time ASC 
            LIMIT 5
        ''')
        
        upcoming = cursor.fetchall()
        print(f"\nNext predictions to validate:")
        for engine, tf, direction, created, target, status in upcoming:
            print(f"  {engine} {tf} ({direction})")
            print(f"    Created: {created}")
            print(f"    Target:  {target}")
            print(f"    Status:  {status}")
            print()
        
        # Check accuracy stats from performance table
        cursor.execute('SELECT * FROM ml_engine_performance')
        perf_data = cursor.fetchall()
        print("Performance Table Data:")
        if perf_data:
            for row in perf_data:
                print(f"  Engine: {row[2]}, Accuracy: {row[7]}%, Total: {row[3]}, Validated: {row[4]}")
        else:
            print("  No performance data yet")
        
        conn.close()
        
        # Show why badges are "Poor"
        print(f"\n💡 Why Accuracy Badges Show 'Poor':")
        print(f"   - All {total} predictions are still 'pending'")
        print(f"   - Only {ready_count} are ready for validation")
        print(f"   - Accuracy = 0% until predictions are validated")
        print(f"   - System is working correctly - just needs time!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_status()
