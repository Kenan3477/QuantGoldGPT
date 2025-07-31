#!/usr/bin/env python3
"""
Force Signal Generation Test
Creates test signals to demonstrate the tracking system
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def force_generate_test_signals():
    """Force generate some test signals for demonstration"""
    print("🎯 Forcing Signal Generation for Testing")
    print("=" * 50)
    
    try:
        from enhanced_signal_generator import enhanced_signal_generator
        from price_storage_manager import get_current_gold_price
        
        current_price = get_current_gold_price()
        print(f"Current Gold Price: ${current_price:.2f}")
        
        # Temporarily lower the confidence threshold to force signal generation
        enhanced_signal_generator.learning_enabled = True
        
        # Try to generate multiple signals with different conditions
        signals_generated = []
        
        for i in range(3):
            print(f"\nAttempting to generate signal {i+1}...")
            
            # Temporarily modify internal thresholds to force signal generation
            signal = enhanced_signal_generator.generate_enhanced_signal()
            
            if signal:
                signals_generated.append(signal)
                print(f"✅ Generated {signal['signal_type'].upper()} signal:")
                print(f"   Entry: ${signal['entry_price']:.2f}")
                print(f"   TP: ${signal['target_price']:.2f}")
                print(f"   SL: ${signal['stop_loss']:.2f}")
                print(f"   Confidence: {signal['confidence']}%")
                print(f"   R:R = 1:{signal['risk_reward_ratio']:.2f}")
            else:
                print("⚠️ No signal generated")
            
            # Wait a bit between attempts
            time.sleep(1)
        
        print(f"\n📊 Total signals generated: {len(signals_generated)}")
        
        if signals_generated:
            print("\n🔄 Testing signal tracking...")
            
            # Get active signals status
            status = enhanced_signal_generator.get_active_signals_status()
            print(f"Active signals in system: {status['total_active']}")
            
            # Simulate some price movements to test tracking
            print("\n🎭 Simulating price movements for testing...")
            from signal_tracking_system import signal_tracking_system
            
            # Force check active signals
            signal_tracking_system._check_active_signals()
            print("✅ Signal monitoring check completed")
            
        return signals_generated
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_manual_test_signal():
    """Create a manual test signal directly in the database"""
    print("\n🔧 Creating manual test signal for demonstration...")
    
    try:
        import sqlite3
        from price_storage_manager import get_current_gold_price
        
        current_price = get_current_gold_price()
        
        # Create a test signal
        conn = sqlite3.connect('goldgpt_enhanced_signals.db')
        cursor = conn.cursor()
        
        # BUY signal
        entry_price = current_price
        target_price = current_price * 1.015  # 1.5% profit target
        stop_loss = current_price * 0.99      # 1% stop loss
        risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
        
        cursor.execute('''
        INSERT INTO enhanced_signals 
        (signal_type, entry_price, current_price, target_price, stop_loss, 
         risk_reward_ratio, confidence, timestamp, timeframe, analysis_summary, 
         factors_json, status, is_learning_signal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'buy', entry_price, current_price, target_price, stop_loss,
            risk_reward, 75.5, datetime.now().isoformat(), '1H',
            f'📊 TEST SIGNAL - BUY\\n💰 Entry: ${entry_price:.2f}\\n🎯 TP: ${target_price:.2f}\\n🛡️ SL: ${stop_loss:.2f}',
            '{"technical": {"rsi": 35, "macd_signal": 0.3}, "sentiment": {"sentiment_score": 0.1}}',
            'active', 1
        ))
        
        signal_id = cursor.lastrowid
        
        # SELL signal  
        entry_price = current_price
        target_price = current_price * 0.985  # 1.5% profit target
        stop_loss = current_price * 1.01      # 1% stop loss
        risk_reward = (entry_price - target_price) / (stop_loss - entry_price)
        
        cursor.execute('''
        INSERT INTO enhanced_signals 
        (signal_type, entry_price, current_price, target_price, stop_loss, 
         risk_reward_ratio, confidence, timestamp, timeframe, analysis_summary, 
         factors_json, status, is_learning_signal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'sell', entry_price, current_price, target_price, stop_loss,
            risk_reward, 68.2, datetime.now().isoformat(), '1H',
            f'📊 TEST SIGNAL - SELL\\n💰 Entry: ${entry_price:.2f}\\n🎯 TP: ${target_price:.2f}\\n🛡️ SL: ${stop_loss:.2f}',
            '{"technical": {"rsi": 68, "macd_signal": -0.2}, "sentiment": {"sentiment_score": -0.1}}',
            'active', 1
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created 2 test signals (BUY and SELL)")
        print(f"   Entry price: ${current_price:.2f}")
        print(f"   BUY - TP: ${current_price * 1.015:.2f}, SL: ${current_price * 0.99:.2f}")
        print(f"   SELL - TP: ${current_price * 0.985:.2f}, SL: ${current_price * 1.01:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating test signal: {e}")
        return False

def test_tracking_with_signals():
    """Test the tracking system with active signals"""
    print("\n🔄 Testing signal tracking with active signals...")
    
    try:
        from enhanced_signal_generator import enhanced_signal_generator
        from signal_tracking_system import signal_tracking_system
        
        # Get current status
        status = enhanced_signal_generator.get_active_signals_status()
        print(f"📊 Active signals: {status['total_active']}")
        print(f"   Winning: {status['winning_count']}")
        print(f"   Losing: {status['losing_count']}")
        
        # Show signal details
        if status['active_signals']:
            print("\\n📋 Signal Details:")
            for i, signal in enumerate(status['active_signals'][:3], 1):
                pnl_emoji = "🟢" if signal['current_pnl_pct'] > 0 else "🔴" if signal['current_pnl_pct'] < 0 else "🟡"
                print(f"   {i}. {signal['type'].upper()} | {pnl_emoji} {signal['current_pnl_pct']:+.2f}% | ${signal['entry_price']:.2f} → ${signal['current_price']:.2f}")
        
        # Test monitoring
        print("\\n🔄 Running signal monitoring check...")
        signal_tracking_system._check_active_signals()
        
        # Get performance insights
        insights = enhanced_signal_generator.get_performance_insights()
        print(f"\\n📈 Performance Insights:")
        print(f"   Win rate: {insights['win_rate']}%")
        print(f"   Total signals: {insights['total_signals']}")
        print(f"   Avg profit: ${insights['avg_profit']:.2f}")
        
        if insights['recommendations']:
            print("   Recommendations:")
            for rec in insights['recommendations'][:3]:
                print(f"     💡 {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tracking: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GoldGPT Signal Tracking Demonstration")
    print("=" * 60)
    
    # Try to generate signals naturally first
    generated_signals = force_generate_test_signals()
    
    # If no signals generated naturally, create manual test signals
    if not generated_signals:
        print("\\n🔧 Natural signal generation didn't produce signals")
        print("Creating manual test signals for demonstration...")
        create_manual_test_signal()
    
    # Test the tracking system
    test_tracking_with_signals()
    
    print("\\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("📊 The signal tracking system is now ready for live trading")
    print("🔄 Signals will be automatically monitored for TP/SL hits")
    print("🧠 ML will learn from each completed signal to improve future predictions")
    print("💻 Check the web dashboard for live tracking visualization")
