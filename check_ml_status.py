#!/usr/bin/env python3
"""
Check ML Tracking Database Status
"""

import sqlite3
from datetime import datetime

def check_database_status():
    """Check the current status of the ML tracking database"""
    print("📊 ML Tracking Database Status")
    print("=" * 40)
    
    db_path = 'goldgpt_ml_tracking.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
        total = cursor.fetchone()[0]
        print(f"🎯 Total Predictions: {total}")
        
        if total > 0:
            # By engine
            cursor.execute("""
                SELECT engine_name, COUNT(*) 
                FROM ml_engine_predictions 
                GROUP BY engine_name
            """)
            
            engine_counts = cursor.fetchall()
            print("\n📈 By Engine:")
            for engine, count in engine_counts:
                display_name = "Enhanced ML" if engine == "enhanced_ml" else "Intelligent ML"
                print(f"   {display_name}: {count} predictions")
            
            # Today's predictions
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT engine_name, timeframe, predicted_price, change_percent, direction
                FROM ml_engine_predictions 
                WHERE DATE(created_at) = ?
                ORDER BY engine_name, timeframe
            """, (today,))
            
            today_predictions = cursor.fetchall()
            
            if today_predictions:
                print(f"\n📅 Today's Predictions ({today}):")
                current_engine = None
                for engine, timeframe, price, change, direction in today_predictions:
                    display_name = "Enhanced ML" if engine == "enhanced_ml" else "Intelligent ML"
                    if engine != current_engine:
                        print(f"   {display_name}:")
                        current_engine = engine
                    print(f"     {timeframe}: ${price} ({change:+.2f}%) - {direction}")
            else:
                print(f"\n❌ No predictions found for today ({today})")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database_status()
