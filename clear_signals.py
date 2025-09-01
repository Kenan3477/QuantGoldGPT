#!/usr/bin/env python3
"""
Clear existing signals from database while keeping generation capability
"""
import sqlite3
import os
import sys

def clear_signals():
    """Clear all existing signals from the database"""
    db_path = 'goldgpt.db'
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Found tables: {[table[0] for table in tables]}")
        
        cleared_count = 0
        
        # Clear ML predictions table if it exists
        try:
            cursor.execute("SELECT COUNT(*) FROM ml_predictions")
            count_before = cursor.fetchone()[0]
            cursor.execute("DELETE FROM ml_predictions")
            print(f"Cleared {count_before} records from ml_predictions table")
            cleared_count += count_before
        except sqlite3.OperationalError:
            print("ml_predictions table doesn't exist")
        
        # Clear AI signals table if it exists
        try:
            cursor.execute("SELECT COUNT(*) FROM ai_signals")
            count_before = cursor.fetchone()[0]
            cursor.execute("DELETE FROM ai_signals")
            print(f"Cleared {count_before} records from ai_signals table")
            cleared_count += count_before
        except sqlite3.OperationalError:
            print("ai_signals table doesn't exist")
        
        # Clear any signal-related tables
        signal_tables = ['signals', 'timeframe_predictions', 'predictions']
        for table in signal_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count_before = cursor.fetchone()[0]
                cursor.execute(f"DELETE FROM {table}")
                print(f"Cleared {count_before} records from {table} table")
                cleared_count += count_before
            except sqlite3.OperationalError:
                pass  # Table doesn't exist
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Successfully cleared {cleared_count} total signal records")
        print("🔧 Signal generation capability preserved - only stored data cleared")
        
    except Exception as e:
        print(f"❌ Error clearing signals: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_signals()
