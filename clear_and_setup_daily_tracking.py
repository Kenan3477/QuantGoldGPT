#!/usr/bin/env python3
"""
Clear ML Tracking Database and Setup Daily Prediction System
Only tracks one prediction set per day per engine
"""

import sqlite3
import json
from datetime import datetime, timedelta
from ml_engine_tracker import MLEngineTracker
from dual_ml_prediction_system import DualMLPredictionSystem

def clear_all_predictions():
    """Clear all predictions from the database"""
    print("🧹 Clearing all ML predictions from database...")
    
    db_path = 'goldgpt_ml_tracking.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count existing predictions
        cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
        count_before = cursor.fetchone()[0]
        print(f"📊 Found {count_before} existing predictions")
        
        # Clear all predictions
        cursor.execute("DELETE FROM ml_engine_predictions")
        cursor.execute("DELETE FROM ml_engine_performance")
        cursor.execute("DELETE FROM daily_accuracy_summary")
        
        # Reset auto-increment
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='ml_engine_predictions'")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='ml_engine_performance'")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='daily_accuracy_summary'")
        
        conn.commit()
        
        # Verify cleanup
        cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
        count_after = cursor.fetchone()[0]
        
        print(f"✅ Cleared {count_before} predictions, {count_after} remaining")
        conn.close()
        
    except Exception as e:
        print(f"❌ Error clearing predictions: {e}")

def setup_daily_prediction_tracking():
    """Setup daily prediction tracking - one prediction set per day per engine"""
    print("🎯 Setting up daily prediction tracking system...")
    
    db_path = 'goldgpt_ml_tracking.db'
    tracker = MLEngineTracker(db_path)
    
    try:
        # Generate today's predictions from both engines
        print("📈 Generating today's predictions...")
        dual_system = DualMLPredictionSystem()
        
        # Get predictions from both engines
        result = dual_system.get_dual_predictions()
        
        if result.get('success'):
            enhanced_predictions = result.get('enhanced_ml', {}).get('predictions', [])
            intelligent_predictions = result.get('intelligent_ml', {}).get('predictions', [])
            
            current_price = result.get('enhanced_ml', {}).get('current_price', 0)
            
            # Store Enhanced ML Engine predictions (one set per day)
            if enhanced_predictions:
                for pred in enhanced_predictions:
                    tracker.store_prediction(
                        engine_name="enhanced_ml",
                        timeframe=pred['timeframe'],
                        predicted_price=pred['predicted_price'],
                        current_price=current_price,
                        change_percent=pred['change_percent'],
                        direction=pred['direction'],
                        confidence=pred['confidence']
                    )
                print(f"✅ Stored Enhanced ML predictions: {len(enhanced_predictions)} timeframes")
            
            # Store Intelligent ML Engine predictions (one set per day)
            if intelligent_predictions:
                for pred in intelligent_predictions:
                    tracker.store_prediction(
                        engine_name="intelligent_ml",
                        timeframe=pred['timeframe'],
                        predicted_price=pred['predicted_price'],
                        current_price=current_price,
                        change_percent=pred['change_percent'],
                        direction=pred['direction'],
                        confidence=pred['confidence']
                    )
                print(f"✅ Stored Intelligent ML predictions: {len(intelligent_predictions)} timeframes")
            
            # Show summary
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
            total_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT engine_name, COUNT(*) FROM ml_engine_predictions GROUP BY engine_name")
            engine_counts = cursor.fetchall()
            
            print(f"\n📊 Daily Tracking Summary:")
            print(f"   Total predictions: {total_count}")
            for engine, count in engine_counts:
                display_name = "Enhanced ML" if engine == "enhanced_ml" else "Intelligent ML"
                print(f"   {display_name}: {count} timeframes")
            
            conn.close()
            
        else:
            print("❌ Failed to generate predictions")
            
    except Exception as e:
        print(f"❌ Error setting up daily tracking: {e}")

def show_tracking_status():
    """Show current tracking status"""
    print("\n📈 Current ML Tracking Status:")
    
    db_path = 'goldgpt_ml_tracking.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
        total = cursor.fetchone()[0]
        
        # By engine
        cursor.execute("""
            SELECT engine_name, COUNT(*), 
                   MIN(created_at) as first_prediction,
                   MAX(created_at) as last_prediction
            FROM ml_engine_predictions 
            GROUP BY engine_name
        """)
        
        engine_stats = cursor.fetchall()
        
        print(f"🎯 Total Predictions: {total}")
        print("📊 By Engine:")
        
        for engine, count, first, last in engine_stats:
            display_name = "Enhanced ML" if engine == "enhanced_ml" else "Intelligent ML"
            print(f"   {display_name}: {count} predictions")
            print(f"     First: {first}")
            print(f"     Last: {last}")
        
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
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error showing status: {e}")

if __name__ == "__main__":
    print("🚀 ML Prediction Daily Tracking Setup")
    print("=" * 50)
    
    # Step 1: Clear all existing predictions
    clear_all_predictions()
    
    # Step 2: Setup daily tracking with today's predictions
    setup_daily_prediction_tracking()
    
    # Step 3: Show the clean status
    show_tracking_status()
    
    print("\n✅ Daily tracking setup complete!")
    print("📝 From now on, only one prediction set per day per engine will be tracked.")
