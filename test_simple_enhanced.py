#!/usr/bin/env python3
"""
Simple test of enhanced signal generator
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_enhanced_signal_generator():
    print("🧪 Testing Enhanced Signal Generator...")
    print("=" * 50)
    
    try:
        from enhanced_signal_generator import enhanced_signal_generator
        print("✅ Enhanced Signal Generator imported successfully")
        
        # Test 1: Signal Generation
        print("\n🎯 Test 1: Signal Generation")
        signal = enhanced_signal_generator.generate_enhanced_signal()
        if signal:
            print(f"✅ Generated {signal['signal_type'].upper()} signal:")
            print(f"   Entry: ${signal['entry_price']:.2f}")
            print(f"   TP: ${signal['target_price']:.2f}")
            print(f"   SL: ${signal['stop_loss']:.2f}")
            print(f"   Confidence: {signal['confidence']:.1f}%")
            print(f"   R:R: {signal['risk_reward_ratio']:.1f}:1")
        else:
            print("ℹ️ No signal generated - market conditions not suitable")
            
        # Test 2: Monitoring
        print("\n🔄 Test 2: Signal Monitoring")
        monitoring = enhanced_signal_generator.monitor_active_signals()
        if monitoring and not monitoring.get('error'):
            print(f"✅ Monitoring active:")
            print(f"   Active signals: {monitoring.get('active_signals', 0)}")
            print(f"   Current price: ${monitoring.get('current_price', 0):.2f}")
            print(f"   Updates: {len(monitoring.get('updates', []))}")
            print(f"   Closed signals: {len(monitoring.get('closed_signals', []))}")
        else:
            print(f"❌ Monitoring error: {monitoring.get('error', 'Unknown')}")
            
        # Test 3: Performance
        print("\n📈 Test 3: Performance Tracking")
        performance = enhanced_signal_generator.get_performance_summary()
        if performance and not performance.get('error'):
            print(f"✅ Performance data:")
            print(f"   Total signals: {performance.get('total_signals', 0)}")
            print(f"   Success rate: {performance.get('success_rate', 0):.1f}%")
            print(f"   Avg P&L: {performance.get('avg_profit_loss_pct', 0):.2f}%")
            print(f"   Best trade: {performance.get('best_profit_pct', 0):.2f}%")
        else:
            print(f"❌ Performance error: {performance.get('error', 'Unknown')}")
            
        print("\n🎉 Enhanced Signal Generator test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_signal_generator()
