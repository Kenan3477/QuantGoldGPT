#!/usr/bin/env python3
"""
Check all ML databases for prediction counts
"""
import sqlite3
import os

def check_all_ml_databases():
    print("🔍 Checking All ML Prediction Databases")
    print("=" * 50)
    
    # List of databases to check
    databases = [
        'ml_predictions.db',
        'goldgpt_ml_predictions.db', 
        'goldgpt_ml_tracking.db',
        'goldgpt_ml_learning.db'
    ]
    
    for db_name in databases:
        if os.path.exists(db_name):
            print(f"\n📊 {db_name}:")
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                print(f"   Tables: {[t[0] for t in tables]}")
                
                # Check common prediction table names
                for table_name in ['ml_predictions', 'ml_engine_predictions', 'predictions']:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"   {table_name}: {count} records")
                    except:
                        pass
                
                conn.close()
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print(f"\n❌ {db_name}: Not found")

if __name__ == "__main__":
    check_all_ml_databases()
