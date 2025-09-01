#!/usr/bin/env python3
"""
IMMEDIATE TARGET PRICE DIVERSITY TEST
=====================================
This script will test the fixed ML engine to prove target prices are now diverse and realistic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_ml_trading_engine import RealMLTradingEngine
import time

def test_target_diversity():
    """Test that our ML engine now generates diverse target prices"""
    print("🧪 TESTING FIXED ML TRADING ENGINE")
    print("=" * 50)
    
    engine = RealMLTradingEngine()
    
    target_prices = []
    signals = []
    
    # Generate multiple signals across different timeframes
    timeframes = ['5m', '15m', '30m', '1h', '4h']
    
    for i, timeframe in enumerate(timeframes):
        print(f"\n📊 Generating signal {i+1}/5 for {timeframe} timeframe...")
        
        try:
            signal = engine.generate_real_signal("GOLD", timeframe)
            
            if signal['success']:
                data = signal['data']
                target_price = data['target_price']
                signal_type = data['signal_type']
                entry_price = data['entry_price']
                
                target_prices.append(target_price)
                signals.append({
                    'timeframe': timeframe,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'change_pct': ((target_price - entry_price) / entry_price) * 100,
                    'confidence': data['confidence']
                })
                
                print(f"   ✅ {signal_type}: ${entry_price:.2f} → ${target_price:.2f} ({((target_price - entry_price) / entry_price) * 100:+.2f}%)")
            else:
                print(f"   ❌ Failed: {signal.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ⚠️  Error: {str(e)}")
        
        # Small delay to allow for different calculations
        time.sleep(0.5)
    
    # Analysis
    print("\n" + "="*50)
    print("📈 TARGET PRICE ANALYSIS")
    print("="*50)
    
    if len(signals) > 0:
        unique_targets = len(set(target_prices))
        total_signals = len(signals)
        
        print(f"🎯 Total signals generated: {total_signals}")
        print(f"🎯 Unique target prices: {unique_targets}")
        print(f"🎯 Diversity ratio: {unique_targets/total_signals*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for signal in signals:
            print(f"   {signal['timeframe']}: {signal['signal_type']} - ${signal['target_price']:.2f} ({signal['change_pct']:+.2f}%) [Conf: {signal['confidence']:.2f}]")
        
        # Check if all targets are different
        if unique_targets == total_signals:
            print("\n🎉 SUCCESS! All target prices are UNIQUE and DIVERSE!")
            print("✅ The 'bullshit identical targets' issue has been FIXED!")
        elif unique_targets > 1:
            print(f"\n✅ PARTIAL SUCCESS! {unique_targets} out of {total_signals} targets are unique")
            print("🔧 Some diversity achieved, but could be improved further")
        else:
            print("\n❌ STILL BROKEN! All targets are identical")
            print("🔧 Need more aggressive fixes")
            
        # Check signal type diversity
        signal_types = [s['signal_type'] for s in signals]
        unique_types = len(set(signal_types))
        print(f"\n📊 Signal type diversity: {unique_types} different types: {set(signal_types)}")
        
        if 'HOLD' in signal_types and len(set(signal_types)) == 1:
            print("⚠️  WARNING: Only generating HOLD signals - thresholds might still be too high")
        elif len(set(signal_types)) > 1:
            print("✅ Good signal type diversity!")
    
    else:
        print("❌ NO SIGNALS GENERATED - System completely broken!")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_target_diversity()
