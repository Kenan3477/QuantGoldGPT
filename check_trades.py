#!/usr/bin/env python3
"""
Check and clean trades database
"""
import sqlite3
import os

def check_trades_database():
    db_files = ['goldgpt.db', 'goldgpt_news.db', 'goldgpt_news_analysis.db']
    
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"\n📊 Checking {db_file}:")
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                print(f"Tables: {[t[0] for t in tables]}")
                
                # Check for trades table
                if any('trade' in table[0].lower() for table in tables):
                    for table in tables:
                        if 'trade' in table[0].lower():
                            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                            count = cursor.fetchone()[0]
                            print(f"{table[0]} total records: {count}")
                            
                            # Check for status column
                            cursor.execute(f"PRAGMA table_info({table[0]})")
                            columns = cursor.fetchall()
                            has_status = any('status' in col[1].lower() for col in columns)
                            
                            if has_status:
                                cursor.execute(f"SELECT status, COUNT(*) FROM {table[0]} GROUP BY status")
                                status_counts = cursor.fetchall()
                                print(f"Status breakdown: {status_counts}")
                                
                                # Show some sample records
                                cursor.execute(f"SELECT * FROM {table[0]} LIMIT 3")
                                samples = cursor.fetchall()
                                print(f"Sample records: {samples}")
                
                conn.close()
                
            except Exception as e:
                print(f"Error checking {db_file}: {e}")
        else:
            print(f"❌ {db_file} not found")

def clean_trades_database():
    """Clean all trade records"""
    db_files = ['goldgpt.db', 'goldgpt_news.db', 'goldgpt_news_analysis.db']
    
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                # Clean trade-related tables
                for table in tables:
                    table_name = table[0]
                    if 'trade' in table_name.lower():
                        cursor.execute(f"DELETE FROM {table_name}")
                        print(f"🧹 Cleaned {table_name} in {db_file}")
                
                conn.commit()
                conn.close()
                print(f"✅ {db_file} cleaned")
                
            except Exception as e:
                print(f"Error cleaning {db_file}: {e}")

if __name__ == "__main__":
    print("🔍 Checking trades databases...")
    check_trades_database()
    
    print("\n" + "="*50)
    response = input("Do you want to clean all trade records? (y/N): ")
    
    if response.lower() == 'y':
        clean_trades_database()
        print("\n🔍 Checking after cleanup...")
        check_trades_database()
    else:
        print("No changes made.")
