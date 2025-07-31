#!/usr/bin/env python3
"""
Database Status Checker and Repair Tool
Helps diagnose and fix database issues for the GoldGPT learning system
"""

import os
import sqlite3
import time
from datetime import datetime

# Optional psutil import for process checking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - process checking disabled")

def check_database_status():
    """Check the status of the learning system database"""
    print("🔍 GoldGPT Learning System Database Status Check")
    print("=" * 50)
    
    db_path = "goldgpt_learning_system.db"
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        print(f"   Expected location: {os.path.abspath(db_path)}")
        return False
    
    print(f"✅ Database file found: {db_path}")
    
    # Check file size and modification time
    try:
        stat = os.stat(db_path)
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"📊 File size: {size_mb:.2f} MB")
        print(f"🕒 Last modified: {mod_time}")
        
    except Exception as e:
        print(f"⚠️ Could not get file stats: {e}")
    
    # Test database connectivity
    print("\n🔌 Testing database connectivity...")
    
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            # Test basic query
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master")
            table_count = cursor.fetchone()[0]
            
            print(f"✅ Database connection successful")
            print(f"📋 Found {table_count} database objects")
            
            # Check tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'prediction_records', 
                'strategy_performance', 
                'validation_results', 
                'market_conditions', 
                'learning_insights'
            ]
            
            print(f"\n📊 Database Tables:")
            for table in expected_tables:
                if table in tables:
                    # Count records in table
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        print(f"✅ {table}: {count} records")
                    except Exception as e:
                        print(f"⚠️ {table}: Error reading ({e})")
                else:
                    print(f"❌ {table}: Missing")
            
            return True
            
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print("🔒 Database is locked - checking for processes...")
            check_locking_processes()
        else:
            print(f"❌ Database error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def check_locking_processes():
    """Check for processes that might be locking the database"""
    print("\n🔍 Checking for processes using the database...")
    
    if not PSUTIL_AVAILABLE:
        print("⚠️ Process checking not available (psutil not installed)")
        print("   You can manually check Task Manager for python.exe processes")
        return
    
    try:
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline'] or []
                    cmdline_str = ' '.join(cmdline)
                    
                    if any(keyword in cmdline_str.lower() for keyword in ['app.py', 'goldgpt', 'flask']):
                        create_time = datetime.fromtimestamp(proc.info['create_time'])
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline_str,
                            'started': create_time
                        })
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if python_processes:
            print(f"🐍 Found {len(python_processes)} Python process(es) that may be using the database:")
            for proc in python_processes:
                print(f"  PID {proc['pid']}: {proc['name']}")
                print(f"    Started: {proc['started']}")
                print(f"    Command: {proc['cmdline'][:80]}...")
                print()
        else:
            print("✅ No obvious database-locking processes found")
            
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")

def repair_database():
    """Attempt to repair database issues"""
    print("\n🔧 Database Repair Options")
    print("=" * 30)
    
    print("1. Wait for lock to clear (recommended)")
    print("2. Create new database (will lose existing data)")
    print("3. Cancel")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\n⏳ Waiting for database lock to clear...")
        for i in range(10):
            try:
                with sqlite3.connect("goldgpt_learning_system.db", timeout=1.0) as conn:
                    conn.execute("SELECT 1")
                    print("✅ Database is now accessible!")
                    return True
            except sqlite3.OperationalError:
                print(f"   Still locked... waiting ({i+1}/10)")
                time.sleep(2)
        
        print("❌ Database is still locked after waiting")
        return False
        
    elif choice == "2":
        backup_path = f"goldgpt_learning_system_backup_{int(time.time())}.db"
        
        try:
            # Backup existing database
            if os.path.exists("goldgpt_learning_system.db"):
                os.rename("goldgpt_learning_system.db", backup_path)
                print(f"📋 Existing database backed up to: {backup_path}")
            
            # Create new database
            from init_learning_database import initialize_learning_database
            success = initialize_learning_database()
            
            if success:
                print("✅ New database created successfully")
                return True
            else:
                print("❌ Failed to create new database")
                # Restore backup
                if os.path.exists(backup_path):
                    os.rename(backup_path, "goldgpt_learning_system.db")
                    print("📋 Backup restored")
                return False
                
        except Exception as e:
            print(f"❌ Repair failed: {e}")
            return False
    
    else:
        print("Cancelled")
        return False

def main():
    """Main function"""
    print("🩺 GoldGPT Database Diagnostic Tool")
    print("=" * 40)
    
    # Check database status
    status_ok = check_database_status()
    
    if not status_ok:
        print("\n🔧 Database issues detected!")
        repair_choice = input("\nAttempt repair? (y/N): ").lower()
        
        if repair_choice == 'y':
            repair_success = repair_database()
            if repair_success:
                print("\n✅ Database repair completed")
                # Re-check status
                print("\nRe-checking database status...")
                check_database_status()
            else:
                print("\n❌ Database repair failed")
        
    else:
        print("\n✅ Database appears to be healthy")
    
    print("\n💡 Tips:")
    print("- Close the main GoldGPT application before running diagnostics")
    print("- Use Task Manager to end python.exe processes if needed")
    print("- Restart your terminal/command prompt if issues persist")

if __name__ == "__main__":
    main()
