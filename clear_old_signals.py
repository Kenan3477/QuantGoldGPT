#!/usr/bin/env python3
"""
Clear Old AI Signals Database
"""
import sqlite3
import os

def clear_old_signals():
    db_path = 'goldgpt_signals.db'
    
    if not os.path.exists(db_path):
        print(f"✅ No old signals database to clear")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count existing signals
        cursor.execute("SELECT COUNT(*) FROM trade_signals")
        total_count = cursor.fetchone()[0]
        print(f"📊 Found {total_count} old signals to clear")
        
        # Clear all signals
        cursor.execute("DELETE FROM trade_signals")
        rows_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"✅ Cleared {rows_deleted} old signals from database")
        print("🎯 Dashboard should now show 0 open positions")
        
    except Exception as e:
        print(f"❌ Error clearing old signals: {e}")

if __name__ == "__main__":
    clear_old_signals()
