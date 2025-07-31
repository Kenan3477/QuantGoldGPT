#!/usr/bin/env python3
"""
Debug Validation Query
"""

import sqlite3
from datetime import datetime

def debug_validation_query():
    conn = sqlite3.connect('goldgpt_ml_tracking.db')
    cursor = conn.cursor()
    
    print("🔍 Debugging Validation Query...")
    
    # Check what the validation query returns
    cursor.execute("""
        SELECT id, engine_name, timeframe, target_validation_time,
               datetime(target_validation_time) as parsed_target,
               datetime('now', 'utc') as current_utc_time,
               CASE 
                 WHEN datetime(target_validation_time) <= datetime('now', 'utc') THEN 'READY'
                 ELSE 'NOT READY'
               END as status
        FROM ml_engine_predictions 
        WHERE status = 'pending' 
        AND symbol = 'XAUUSD'
        ORDER BY target_validation_time ASC
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    print(f"Found {len(results)} pending predictions:")
    
    for row in results:
        print(f"  ID {row[0]}: {row[1]} {row[2]} - Target: {row[3]} - Status: {row[6]}")
    
    # Check how many are actually ready
    cursor.execute("""
        SELECT COUNT(*) FROM ml_engine_predictions 
        WHERE status = 'pending' 
        AND datetime(target_validation_time) <= datetime('now', 'utc')
        AND symbol = 'XAUUSD'
    """)
    
    ready_count = cursor.fetchone()[0]
    print(f"\nReady for validation: {ready_count}")
    
    conn.close()

if __name__ == "__main__":
    debug_validation_query()
