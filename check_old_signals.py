#!/usr/bin/env python3
"""
Check Old Trade Signals Database
"""
import sqlite3
import os

def check_old_signals():
    db_path = 'goldgpt.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database {db_path} does not exist")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if trade_signals table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_signals'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("❌ trade_signals table does not exist")
            conn.close()
            return
            
        print("✅ trade_signals table found")
        
        # Count open signals
        cursor.execute("SELECT COUNT(*) FROM trade_signals WHERE status='open'")
        open_count = cursor.fetchone()[0]
        print(f"🟢 Open signals: {open_count}")
        
        # Count closed signals  
        cursor.execute("SELECT COUNT(*) FROM trade_signals WHERE status='closed'")
        closed_count = cursor.fetchone()[0]
        print(f"🔴 Closed signals: {closed_count}")
        
        # Show recent signals
        cursor.execute("""
            SELECT id, signal_type, entry_price, target_price, stop_loss, 
                   timestamp, status 
            FROM trade_signals 
            ORDER BY id DESC 
            LIMIT 5
        """)
        
        signals = cursor.fetchall()
        if signals:
            print("\n📈 Recent signals from old system:")
            for signal in signals:
                signal_id, signal_type, entry_price, target_price, stop_loss, timestamp, status = signal
                print(f"  🎯 ID: {signal_id}")
                print(f"     Type: {signal_type}")
                print(f"     Entry: ${entry_price:.2f}")
                print(f"     TP: ${target_price:.2f}")
                print(f"     SL: ${stop_loss:.2f}")
                print(f"     Status: {status}")
                print(f"     Time: {timestamp}")
                print()
        else:
            print("\n✅ No signals found in old system")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking old signals: {e}")

if __name__ == "__main__":
    check_old_signals()
