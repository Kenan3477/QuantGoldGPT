#!/usr/bin/env python3
import sqlite3

# Clear ML tracking database
conn = sqlite3.connect('goldgpt_ml_tracking.db')
cursor = conn.cursor()

print("🧹 Clearing ML tracking database...")

# Delete all predictions
cursor.execute('DELETE FROM ml_engine_predictions')
predictions_deleted = cursor.rowcount

cursor.execute('DELETE FROM ml_engine_performance') 
performance_deleted = cursor.rowcount

cursor.execute('DELETE FROM daily_accuracy_summary')
summary_deleted = cursor.rowcount

conn.commit()

print(f"✅ Deleted {predictions_deleted} predictions")
print(f"✅ Deleted {performance_deleted} performance records")
print(f"✅ Deleted {summary_deleted} daily summaries")

# Verify cleanup
cursor.execute('SELECT COUNT(*) FROM ml_engine_predictions')
remaining = cursor.fetchone()[0]
print(f"📊 Remaining predictions: {remaining}")

conn.close()
print("🎯 Database cleared successfully - ready for fresh predictions!")
