#!/usr/bin/env python3
import sqlite3
from datetime import datetime, timedelta

# Check fresh predictions
conn = sqlite3.connect('goldgpt_ml_tracking.db')
cursor = conn.cursor()

# Get today's predictions
today = datetime.now().date()
cursor.execute("""
    SELECT engine_name, timeframe, predicted_price, change_percent, direction, confidence, created_at
    FROM ml_engine_predictions 
    WHERE DATE(created_at) = DATE('now')
    ORDER BY created_at DESC
    LIMIT 20
""")

predictions = cursor.fetchall()

print(f"📅 Fresh predictions for {today}:")
print(f"📊 Total fresh predictions: {len(predictions)}")

if predictions:
    print("\n🔥 Latest Predictions:")
    for pred in predictions:
        engine, timeframe, predicted_price, change_percent, direction, confidence, created_at = pred
        print(f"  {engine} ({timeframe}): {direction} ${predicted_price} ({change_percent:+.2f}%) [{confidence:.1%}] - {created_at}")

# Get total count
cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
total = cursor.fetchone()[0]
print(f"\n📈 Total predictions in database: {total}")

conn.close()
