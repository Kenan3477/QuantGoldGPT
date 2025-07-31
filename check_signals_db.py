#!/usr/bin/env python3
"""
Check Enhanced Signals Database Content
"""
import sqlite3
import os

def check_signals_database():
    db_path = 'goldgpt_enhanced_signals.db'
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database {db_path} does not exist")
        return
        
    print(f"✅ Database {db_path} exists")
    print(f"📊 File size: {os.path.getsize(db_path)} bytes")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n📋 Tables: {[table[0] for table in tables]}")
        
        # Check enhanced_signals table
        if ('enhanced_signals',) in tables:
            cursor.execute("SELECT COUNT(*) FROM enhanced_signals")
            total_count = cursor.fetchone()[0]
            print(f"\n📊 Total signals in database: {total_count}")
            
            cursor.execute("SELECT COUNT(*) FROM enhanced_signals WHERE status='active'")
            active_count = cursor.fetchone()[0]
            print(f"🟢 Active signals: {active_count}")
            
            cursor.execute("SELECT COUNT(*) FROM enhanced_signals WHERE status='closed'")
            closed_count = cursor.fetchone()[0]
            print(f"🔴 Closed signals: {closed_count}")
            
            # Show recent signals
            cursor.execute("""
                SELECT id, signal_type, entry_price, target_price, stop_loss, 
                       timestamp, status, exit_reason
                FROM enhanced_signals 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            
            signals = cursor.fetchall()
            if signals:
                print("\n📈 Recent signals:")
                for signal in signals:
                    signal_id, signal_type, entry_price, target_price, stop_loss, timestamp, status, exit_reason = signal
                    print(f"  🎯 ID: {signal_id}")
                    print(f"     Type: {signal_type.upper()}")
                    print(f"     Entry: ${entry_price:.2f}")
                    print(f"     TP: ${target_price:.2f}")
                    print(f"     SL: ${stop_loss:.2f}")
                    print(f"     Status: {status}")
                    print(f"     Exit Reason: {exit_reason or 'N/A'}")
                    print(f"     Time: {timestamp}")
                    print()
            else:
                print("\n✅ No signals found in database")
        else:
            print("\n❌ enhanced_signals table not found")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_signals_database()
