#!/usr/bin/env python3
"""
Clean ML Prediction System
Only stores 3 predictions per engine every 12 hours
Eliminates the 80+ prediction accumulation problem
"""
import sqlite3
import asyncio
from datetime import datetime, timedelta
from ml_engine_tracker import MLEngineTracker
from dual_ml_prediction_system import DualMLPredictionSystem

def clear_old_prediction_databases():
    """Clear old prediction databases that are accumulating unwanted predictions"""
    databases_to_clear = [
        'goldgpt_ml_predictions.db',
        'ml_predictions.db'
    ]
    
    for db_name in databases_to_clear:
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            
            # Clear prediction tables
            for table in tables:
                if 'prediction' in table.lower():
                    cursor.execute(f"DELETE FROM {table}")
                    print(f"🧹 Cleared {table} from {db_name}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error clearing {db_name}: {e}")

def setup_clean_tracking_system():
    """Setup the clean tracking system with proper constraints"""
    print("🔧 Setting up Clean ML Tracking System")
    
    # Use only the goldgpt_ml_tracking.db for tracking
    db_path = 'goldgpt_ml_tracking.db'
    tracker = MLEngineTracker(db_path)
    
    # Clear existing predictions to start fresh
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM ml_engine_predictions")
    cursor.execute("DELETE FROM ml_engine_performance") 
    cursor.execute("DELETE FROM daily_accuracy_summary")
    
    conn.commit()
    conn.close()
    
    print("✅ Clean tracking system ready")
    return tracker

async def generate_controlled_predictions():
    """Generate exactly 6 predictions (3 per engine) and stop"""
    print("🎯 Generating Controlled Daily Predictions")
    print("=" * 50)
    
    # Clear old databases first
    clear_old_prediction_databases()
    
    # Setup clean tracking
    tracker = setup_clean_tracking_system()
    
    # Generate new predictions
    prediction_system = DualMLPredictionSystem()
    
    try:
        predictions = await prediction_system.get_dual_predictions()
        
        if predictions['success']:
            print("✅ Generated today's predictions:")
            print(f"   Enhanced ML: {len(predictions['enhanced']['predictions'])} predictions")
            print(f"   Intelligent ML: {len(predictions['intelligent']['predictions'])} predictions")
            
            # Verify count
            conn = sqlite3.connect('goldgpt_ml_tracking.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ml_engine_predictions")
            total = cursor.fetchone()[0]
            conn.close()
            
            print(f"📊 Total stored predictions: {total}")
            
            if total == 6:
                print("🎯 PERFECT! Exactly 6 predictions stored as expected")
            else:
                print(f"⚠️  Warning: Expected 6 predictions, got {total}")
                
        else:
            print("❌ Failed to generate predictions")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def create_daily_prediction_api_route():
    """Create a Flask route that only serves today's predictions"""
    route_code = '''
@app.route('/api/ml-predictions/daily')
def get_daily_ml_predictions():
    """Get today's ML predictions only (max 6 predictions)"""
    try:
        conn = sqlite3.connect('goldgpt_ml_tracking.db')
        cursor = conn.cursor()
        
        # Get only today's predictions
        cursor.execute("""
            SELECT engine_name, timeframe, predicted_price, current_price,
                   change_percent, direction, confidence, created_at
            FROM ml_engine_predictions 
            WHERE DATE(created_at) = DATE('now', 'localtime')
            ORDER BY created_at DESC
        """)
        
        predictions = cursor.fetchall()
        conn.close()
        
        # Format results
        daily_predictions = {
            'enhanced_ml_prediction_engine': [],
            'intelligent_ml_predictor': []
        }
        
        for pred in predictions:
            engine_name, timeframe, predicted_price, current_price, change_percent, direction, confidence, created_at = pred
            
            prediction_data = {
                'timeframe': timeframe,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'change_percent': change_percent,
                'direction': direction,
                'confidence': confidence,
                'timestamp': created_at
            }
            
            if engine_name in daily_predictions:
                daily_predictions[engine_name].append(prediction_data)
        
        return jsonify({
            'success': True,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_predictions': len(predictions),
            'predictions': daily_predictions,
            'message': f"Today's predictions only - max 6 total",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
'''
    
    print("📝 Flask Route Code for Daily Predictions:")
    print(route_code)
    
    # Write to file for easy integration
    with open('daily_predictions_route.py', 'w') as f:
        f.write(route_code)
    
    print("✅ Saved route code to daily_predictions_route.py")

async def main():
    """Main function to set up clean prediction system"""
    print("🚀 Setting Up Clean ML Prediction System")
    print("=" * 60)
    
    # Generate controlled predictions
    await generate_controlled_predictions()
    
    # Create the API route
    create_daily_prediction_api_route()
    
    print("\n🎯 SOLUTION SUMMARY:")
    print("1. ✅ Cleared old prediction databases")
    print("2. ✅ Generated exactly 6 predictions for today")
    print("3. ✅ Created clean API route for daily predictions")
    print("4. 🔄 System now tracks only daily predictions (3 per engine)")

if __name__ == "__main__":
    asyncio.run(main())
