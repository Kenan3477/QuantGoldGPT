#!/usr/bin/env python3
"""
Generate Fresh Daily Predictions
Creates one set of predictions per engine for today only
"""

import sqlite3
import asyncio
from datetime import datetime
from dual_ml_prediction_system import DualMLPredictionSystem
from ml_engine_tracker import MLEngineTracker

async def generate_fresh_daily_predictions():
    """Generate exactly one set of predictions per engine for today"""
    print("🆕 Generating Fresh Daily Predictions...")
    
    # Clear today's predictions first
    db_path = 'goldgpt_ml_tracking.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Delete all predictions from today
    cursor.execute("DELETE FROM ml_engine_predictions WHERE DATE(created_at) = ?", (today,))
    deleted = cursor.rowcount
    print(f"🗑️  Cleared {deleted} existing predictions from today")
    
    conn.commit()
    conn.close()
    
    # Initialize systems
    dual_system = DualMLPredictionSystem()
    tracker = MLEngineTracker(db_path)
    
    # Generate fresh predictions
    print("🤖 Generating new predictions...")
    try:
        # Get dual predictions (this should generate fresh ones)
        result = await dual_system.get_dual_predictions()
        
        if result.get('success'):
            print("✅ Fresh predictions generated successfully!")
            
            # Show what was generated
            enhanced = result.get('enhanced_ml', {})
            intelligent = result.get('intelligent_ml', {})
            
            print(f"\n📅 Today's Fresh Predictions ({today}):")
            
            # Enhanced ML
            if enhanced.get('predictions'):
                print("   Enhanced ML:")
                for pred in enhanced['predictions']:
                    tf = pred['timeframe']
                    price = pred['predicted_price']
                    change = pred['change_percent']
                    direction = pred['direction']
                    print(f"     {tf}: ${price} ({change:+.2f}%) - {direction}")
            
            # Intelligent ML
            if intelligent.get('predictions'):
                print("   Intelligent ML:")
                for pred in intelligent['predictions']:
                    tf = pred['timeframe']
                    price = pred['predicted_price'] 
                    change = pred['change_percent']
                    direction = pred['direction']
                    print(f"     {tf}: ${price} ({change:+.2f}%) - {direction}")
            
            # Verify database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions WHERE DATE(created_at) = ?", (today,))
            total_today = cursor.fetchone()[0]
            print(f"\n📊 Total predictions stored for today: {total_today}")
            conn.close()
            
        else:
            print("❌ Failed to generate predictions")
            print(result)
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(generate_fresh_daily_predictions())
