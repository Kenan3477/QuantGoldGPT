"""
Quick test of advanced signal system
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print("🎯 Testing Advanced Signal Generation...")

try:
    from advanced_trading_signal_manager import generate_trading_signal
    
    print("📊 Generating signal...")
    result = generate_trading_signal("GOLD", "1h")
    
    print(f"✅ Result: {result}")
    
    if result.get('success') and result.get('signal_generated'):
        print(f"🎉 SUCCESS! Generated {result['signal_type']} signal")
        print(f"   Entry: ${result['entry_price']:.2f}")
        print(f"   TP: ${result['take_profit']:.2f}")
        print(f"   SL: ${result['stop_loss']:.2f}")
        print(f"   ROI: {result['expected_roi']:.2f}%")
        print(f"   Confidence: {result['confidence']:.1%}")
    else:
        print(f"ℹ️ No signal: {result.get('reason', 'Unknown')}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Testing Signal Tracking...")

try:
    from auto_signal_tracker import auto_tracker
    
    print("📡 Checking tracking system...")
    print(f"   Running: {auto_tracker.is_running}")
    
    # Get stats
    stats = auto_tracker.get_performance_stats()
    print(f"   Total signals: {stats.get('total_signals', 0)}")
    print(f"   Win rate: {stats.get('win_rate', 0):.1f}%")
    
except Exception as e:
    print(f"❌ Tracking error: {e}")

print("\n✅ Test completed!")
