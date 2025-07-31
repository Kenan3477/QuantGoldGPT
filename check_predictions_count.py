#!/usr/bin/env python3
"""
Check Predictions Count - See what's actually stored
"""
import sqlite3
from datetime import datetime

def check_predictions_count():
    print("🔍 Checking ML Predictions Database")
    print("=" * 40)
    
    conn = sqlite3.connect('goldgpt_ml_tracking.db')
    cursor = conn.cursor()
    
    # Total count
    cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
    total = cursor.fetchone()[0]
    print(f"📊 Total predictions: {total}")
    
    # Count by engine
    cursor.execute('SELECT engine_name, COUNT(*) as count FROM ml_engine_predictions GROUP BY engine_name')
    print("\n🤖 Predictions per engine:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} predictions")
    
    # Count by timeframe
    cursor.execute('SELECT timeframe, COUNT(*) as count FROM ml_engine_predictions GROUP BY timeframe')
    print("\n⏰ Predictions per timeframe:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} predictions")
    
    # Recent predictions
    cursor.execute('''
        SELECT engine_name, timeframe, created_at, predicted_price 
        FROM ml_engine_predictions 
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    print("\n📅 Most recent predictions:")
    for row in cursor.fetchall():
        print(f"   {row[0]} {row[1]}: ${row[3]} at {row[2]}")
    
    conn.close()

if __name__ == "__main__":
    check_predictions_count()
