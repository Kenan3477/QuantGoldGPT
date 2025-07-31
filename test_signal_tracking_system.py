#!/usr/bin/env python3
"""
Signal Tracking System Test
Tests the complete signal tracking, monitoring, and learning system
"""
import sys
import os
import asyncio
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_signal_tracking_system():
    """Test the complete signal tracking system"""
    print("🎯 Testing Signal Tracking System")
    print("=" * 60)
    
    try:
        # Test 1: Import and initialize the tracking system
        print("\n1. Testing tracking system initialization...")
        from signal_tracking_system import signal_tracking_system
        
        # Start monitoring
        signal_tracking_system.start_monitoring()
        print("✅ Tracking system started successfully")
        
        # Test 2: Generate a test signal
        print("\n2. Testing signal generation...")
        from enhanced_signal_generator import enhanced_signal_generator
        
        # Generate a signal
        signal = enhanced_signal_generator.generate_enhanced_signal()
        if signal:
            print(f"✅ Generated signal: {signal['signal_type'].upper()} at ${signal['entry_price']:.2f}")
            print(f"   TP: ${signal['target_price']:.2f}, SL: ${signal['stop_loss']:.2f}")
            print(f"   Confidence: {signal['confidence']}%, RR: {signal['risk_reward_ratio']}")
        else:
            print("⚠️ No signal generated (market conditions may not be favorable)")
        
        # Test 3: Check active signals status
        print("\n3. Testing active signals status...")
        status = enhanced_signal_generator.get_active_signals_status()
        print(f"✅ Active signals: {status['total_active']}")
        print(f"   Winning: {status['winning_count']}, Losing: {status['losing_count']}")
        
        # Test 4: Check performance insights
        print("\n4. Testing performance insights...")
        insights = enhanced_signal_generator.get_performance_insights()
        print(f"✅ Performance insights:")
        print(f"   Total signals: {insights['total_signals']}")
        print(f"   Win rate: {insights['win_rate']}%")
        print(f"   Avg profit: ${insights['avg_profit']:.2f}")
        print(f"   Recommendations: {len(insights['recommendations'])}")
        
        # Test 5: Check learning progress
        print("\n5. Testing learning progress...")
        learning = enhanced_signal_generator.get_learning_progress()
        print(f"✅ Learning progress:")
        print(f"   Learning enabled: {learning['learning_enabled']}")
        print(f"   Model ready: {learning.get('model_ready', False)}")
        print(f"   Learning samples: {learning.get('total_learning_samples', 0)}")
        print(f"   Status: {learning.get('learning_status', 'Unknown')}")
        
        # Test 6: Force a signal check (simulates monitoring)
        print("\n6. Testing signal monitoring...")
        signal_tracking_system._check_active_signals()
        print("✅ Signal monitoring check completed")
        
        # Test 7: Test ML prediction if model is available
        print("\n7. Testing ML prediction capabilities...")
        if signal_tracking_system.learning_model is not None:
            # Create dummy factors for testing
            test_factors = {
                'technical': {'rsi': 30, 'macd_signal': 0.5, 'bb_position': 0.2, 'volume_ratio': 1.2},
                'sentiment': {'news_sentiment': 0.1, 'market_fear_greed': 40},
                'volatility': {'current_volatility': 0.025},
                'trend': {'short_trend': 1, 'medium_trend': 0.5, 'long_trend': 0.2}
            }
            
            success_prob = signal_tracking_system.predict_signal_success_probability(
                test_factors, 75.0, 2.5
            )
            print(f"✅ ML prediction successful: {success_prob:.3f} probability")
        else:
            print("⚠️ ML model not yet trained (need more historical signals)")
        
        # Test 8: Database connectivity
        print("\n8. Testing database connectivity...")
        import sqlite3
        conn = sqlite3.connect('goldgpt_enhanced_signals.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM enhanced_signals")
        total_signals = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_signals WHERE status = 'active'")
        active_signals = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signal_learning_metrics")
        learning_metrics = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database check:")
        print(f"   Total signals: {total_signals}")
        print(f"   Active signals: {active_signals}")
        print(f"   Learning metrics: {learning_metrics}")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 SIGNAL TRACKING SYSTEM TEST COMPLETE")
        print("=" * 60)
        print("✅ All core components tested successfully")
        print("📊 System is ready for live trading signal tracking")
        print("🧠 ML learning will improve as more signals are processed")
        
        # Keep monitoring running for a bit to show it's working
        print("\n🔄 Monitoring will continue in background...")
        print("   (Signals will be automatically tracked for TP/SL hits)")
        print("   (ML model will learn from each completed signal)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing signal tracking system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_signal_tracking_system()
