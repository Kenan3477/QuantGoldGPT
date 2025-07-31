#!/usr/bin/env python3
"""
ML Engine Tracking Database Initializer
Sets up the prediction tracking database and starts monitoring engines
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine_tracker import MLEngineTracker
from dual_ml_prediction_system import DualMLPredictionSystem

class MLTrackingInitializer:
    def __init__(self, db_path='goldgpt_ml_tracking.db'):
        self.db_path = db_path
        self.schema_path = 'ml_engine_tracking_schema.sql'
        
    def initialize_database(self):
        """Initialize the tracking database with proper schema"""
        print("🔧 Initializing ML Engine Tracking Database...")
        
        # Check if schema file exists
        if not os.path.exists(self.schema_path):
            print(f"❌ Schema file not found: {self.schema_path}")
            return False
            
        try:
            # Read schema
            with open(self.schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Create database and apply schema
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute schema (split by semicolons to handle multiple statements)
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                cursor.execute(statement)
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database initialized successfully: {self.db_path}")
            
            # Verify tables were created
            self.verify_database_structure()
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            return False
    
    def verify_database_structure(self):
        """Verify that all required tables exist"""
        print("🔍 Verifying database structure...")
        
        required_tables = [
            'ml_engine_predictions',
            'ml_engine_performance',
            'daily_accuracy_summary'
        ]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                print(f"⚠️ Missing tables: {missing_tables}")
                return False
            else:
                print(f"✅ All required tables exist: {existing_tables}")
                
                # Check if views exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                views = [row[0] for row in cursor.fetchall()]
                print(f"📊 Available views: {views}")
                
                return True
                
        except Exception as e:
            print(f"❌ Error verifying database: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def test_tracking_system(self):
        """Test the ML tracking system with sample data"""
        print("🧪 Testing ML Engine Tracking System...")
        
        try:
            # Initialize tracker
            tracker = MLEngineTracker(self.db_path)
            
            # Import the MLPrediction class
            from ml_engine_tracker import MLPrediction
            
            # Test storing a prediction
            test_prediction = MLPrediction(
                engine_name='test_engine',
                symbol='XAUUSD',
                current_price=2045.00,
                timeframe='1H',
                predicted_price=2050.00,
                change_percent=0.244,
                direction='bullish',
                confidence=0.75,
                market_conditions={
                    'rsi': 45.5,
                    'volume_trend': 'increasing'
                },
                prediction_factors={
                    'sentiment_score': 0.6,
                    'technical_score': 0.8
                }
            )
            
            # Store test prediction
            prediction_id = tracker.store_prediction(test_prediction)
            print(f"✅ Test prediction stored with ID: {prediction_id}")
            
            # Test getting dashboard stats
            stats = tracker.get_dashboard_stats()
            print(f"📊 Dashboard stats retrieved: {len(stats.get('engines', []))} engines")
            
            # Clean up test data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ml_engine_predictions WHERE engine_name = 'test_engine'")
            cursor.execute("DELETE FROM ml_engine_performance WHERE engine_name = 'test_engine'")
            conn.commit()
            conn.close()
            
            print("✅ ML tracking system test completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ ML tracking system test failed: {e}")
            return False
    
    def test_dual_prediction_system(self):
        """Test the dual ML prediction system"""
        print("🤖 Testing Dual ML Prediction System...")
        
        try:
            # Initialize dual system
            dual_system = DualMLPredictionSystem(self.db_path)
            
            # Get dual predictions (handle async)
            import asyncio
            predictions = asyncio.run(dual_system.get_dual_predictions())
            
            if predictions['success']:
                print(f"✅ Dual predictions retrieved successfully")
                print(f"📊 Active engines: {len([e for e in predictions['engines'] if e['status'] == 'active'])}")
                
                # Check if predictions were stored
                if any(e['status'] == 'active' and e['predictions'] for e in predictions['engines']):
                    print("✅ Predictions stored in tracking database")
                else:
                    print("⚠️ No active predictions to store")
                
                return True
            else:
                print(f"❌ Failed to get dual predictions: {predictions.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Dual prediction system test failed: {e}")
            return False
    
    def display_current_stats(self):
        """Display current tracking statistics"""
        print("\n📈 Current ML Engine Statistics:")
        print("=" * 50)
        
        try:
            tracker = MLEngineTracker(self.db_path)
            stats = tracker.get_dashboard_stats()
            
            if stats:
                print(f"Total Predictions: {stats.get('total_predictions', 0)}")
                print(f"Best Performer: {stats.get('best_performer', 'N/A')}")
                print(f"Last Updated: {stats.get('last_updated', 'N/A')}")
                
                print("\nEngine Performance:")
                for engine in stats.get('engines', []):
                    print(f"  {engine['display_name']}: {engine['overall_accuracy']:.1f}% accuracy")
                    print(f"    Predictions: {engine['total_predictions']}")
                    print(f"    Badge: {engine['badge']['label']}")
                    print()
            else:
                print("No statistics available yet")
                
        except Exception as e:
            print(f"❌ Error displaying stats: {e}")

def main():
    """Main initialization function"""
    print("🚀 Starting ML Engine Tracking Initialization")
    print("=" * 60)
    
    initializer = MLTrackingInitializer()
    
    # Step 1: Initialize database
    if not initializer.initialize_database():
        print("❌ Database initialization failed. Aborting.")
        return False
    
    # Step 2: Test tracking system
    if not initializer.test_tracking_system():
        print("❌ Tracking system test failed. Aborting.")
        return False
    
    # Step 3: Test dual prediction system
    if not initializer.test_dual_prediction_system():
        print("❌ Dual prediction system test failed. Aborting.")
        return False
    
    # Step 4: Display current stats
    initializer.display_current_stats()
    
    print("\n🎉 ML Engine Tracking System Initialized Successfully!")
    print("=" * 60)
    print("✅ Database schema applied")
    print("✅ Tracking system tested")
    print("✅ Dual prediction system tested")
    print("✅ Ready to track ML engine accuracy")
    print("\nNext steps:")
    print("1. Start the Flask application: python app.py")
    print("2. Visit the dashboard to see dual predictions")
    print("3. Let the system collect prediction data over time")
    print("4. Accuracy statistics will improve with more data")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
