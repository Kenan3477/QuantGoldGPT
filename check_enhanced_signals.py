#!/usr/bin/env python3
"""
Check enhanced signals database
"""
import sqlite3
import os

def check_enhanced_signals():
    db_file = 'goldgpt_enhanced_signals.db'
    
    if os.path.exists(db_file):
        print(f"📊 Checking {db_file}:")
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Tables: {[t[0] for t in tables]}")
            
            # Check enhanced_signals table
            if any('enhanced_signals' in table[0] for table in tables):
                cursor.execute("SELECT COUNT(*) FROM enhanced_signals")
                total_count = cursor.fetchone()[0]
                print(f"Total enhanced signals: {total_count}")
                
                cursor.execute("SELECT COUNT(*) FROM enhanced_signals WHERE status='closed'")
                closed_count = cursor.fetchone()[0]
                print(f"Closed signals: {closed_count}")
                
                # Show status breakdown
                cursor.execute("SELECT status, COUNT(*) FROM enhanced_signals GROUP BY status")
                status_counts = cursor.fetchall()
                print(f"Status breakdown: {status_counts}")
                
                # Show some sample records
                cursor.execute("SELECT id, signal_type, entry_price, exit_price, profit_loss, timestamp, status FROM enhanced_signals LIMIT 5")
                samples = cursor.fetchall()
                print(f"Sample records:")
                for sample in samples:
                    print(f"  {sample}")
                    
                # If there are closed signals, ask to clean
                if closed_count > 0:
                    print(f"\n❌ Found {closed_count} closed signals that might be causing the issue!")
                    return True
            
            conn.close()
            
        except Exception as e:
            print(f"Error checking {db_file}: {e}")
    else:
        print(f"❌ {db_file} not found")
        
    return False

def clean_enhanced_signals():
    """Clean enhanced signals database"""
    db_file = 'goldgpt_enhanced_signals.db'
    
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Delete all signals
            cursor.execute("DELETE FROM enhanced_signals")
            deleted_count = cursor.rowcount
            
            # Also clean learning metrics
            cursor.execute("DELETE FROM signal_learning_metrics")
            cursor.execute("DELETE FROM performance_history")
            
            conn.commit()
            conn.close()
            
            print(f"🧹 Cleaned {deleted_count} signals from enhanced_signals database")
            return True
            
        except Exception as e:
            print(f"Error cleaning {db_file}: {e}")
            return False
    else:
        print(f"❌ {db_file} not found")
        return False

if __name__ == "__main__":
    has_closed_signals = check_enhanced_signals()
    
    if has_closed_signals:
        print("\n" + "="*50)
        response = input("Do you want to clean the enhanced signals database? (y/N): ")
        
        if response.lower() == 'y':
            if clean_enhanced_signals():
                print("\n🔍 Checking after cleanup...")
                check_enhanced_signals()
        else:
            print("No changes made.")
    else:
        print("✅ No closed signals found in enhanced_signals database")
