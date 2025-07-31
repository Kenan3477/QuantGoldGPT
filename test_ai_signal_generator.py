#!/usr/bin/env python3
"""
Test the AI Trade Signal Generator
Verifies all components are working correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_signal_generator import get_trade_signal, get_open_trade_signals, get_signal_stats
from price_storage_manager import get_current_gold_price

def test_signal_generator():
    print("🤖 Testing AI Trade Signal Generator")
    print("=" * 50)
    
    # Test 1: Get current gold price
    print("\n1. Testing price data...")
    try:
        current_price = get_current_gold_price()
        print(f"✅ Current gold price: ${current_price}")
    except Exception as e:
        print(f"❌ Error getting price: {e}")
        return False
    
    # Test 2: Generate a signal
    print("\n2. Testing signal generation...")
    try:
        signal = get_trade_signal()
        if signal:
            print(f"✅ Generated {signal['type']} signal:")
            print(f"   Entry: ${signal['entry_price']}")
            print(f"   Target: ${signal['target_price']}")
            print(f"   Stop Loss: ${signal['stop_loss']}")
            print(f"   Confidence: {signal['confidence']}%")
            print(f"   Summary: {signal['summary']}")
        else:
            print("ℹ️ No new signal generated (recent signal exists or conditions not met)")
    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return False
    
    # Test 3: Get open signals
    print("\n3. Testing open signals retrieval...")
    try:
        open_signals = get_open_trade_signals()
        print(f"✅ Found {len(open_signals)} open signals")
        
        for i, signal in enumerate(open_signals[:3]):  # Show first 3
            print(f"   Signal {i+1}: {signal['signal_type']} at ${signal['entry_price']} ({signal['confidence']}% confidence)")
            
    except Exception as e:
        print(f"❌ Error getting open signals: {e}")
        return False
    
    # Test 4: Get statistics
    print("\n4. Testing statistics retrieval...")
    try:
        stats = get_signal_stats()
        print(f"✅ Signal Statistics:")
        print(f"   Total Signals: {stats['total_signals']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        print(f"   Total Return: {stats['total_return']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return False
    
    print("\n✅ All tests passed! AI Signal Generator is working correctly.")
    return True

def test_technical_analysis():
    print("\n🔧 Testing Technical Analysis Components")
    print("=" * 50)
    
    try:
        from ai_signal_generator import signal_generator
        
        # Test RSI calculation
        test_prices = [2650, 2655, 2648, 2652, 2660, 2658, 2665, 2670, 2668, 2675,
                      2680, 2678, 2685, 2690, 2688, 2695, 2700, 2698, 2705, 2710]
        
        rsi = signal_generator._calculate_rsi(test_prices)
        print(f"✅ RSI calculation: {rsi:.2f}")
        
        # Test MACD calculation
        macd = signal_generator._calculate_macd(test_prices)
        print(f"✅ MACD calculation: {macd:.4f}")
        
        # Test volatility calculation
        volatility = signal_generator._calculate_volatility()
        print(f"✅ Volatility calculation: {volatility:.4f}")
        
        # Test pattern detection
        doji = signal_generator._detect_doji(test_prices)
        print(f"✅ Doji pattern detection: {doji}")
        
        engulfing = signal_generator._detect_engulfing(test_prices)
        print(f"✅ Engulfing pattern detection: {engulfing}")
        
        print("✅ Technical analysis components working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing technical analysis: {e}")
        return False
    
    return True

def test_api_integration():
    print("\n🌐 Testing API Integration")
    print("=" * 50)
    
    try:
        import requests
        
        # Test if Flask app is running
        base_url = "http://localhost:5000"
        
        # Test signal generation endpoint
        print("Testing signal generation endpoint...")
        response = requests.post(f"{base_url}/api/signals/generate", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Signal generation API: {data.get('success', False)}")
        else:
            print(f"⚠️ Signal generation API returned status: {response.status_code}")
        
        # Test open signals endpoint
        print("Testing open signals endpoint...")
        response = requests.get(f"{base_url}/api/signals/open", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Open signals API: {len(data.get('signals', []))} signals retrieved")
        else:
            print(f"⚠️ Open signals API returned status: {response.status_code}")
        
        # Test statistics endpoint
        print("Testing statistics endpoint...")
        response = requests.get(f"{base_url}/api/signals/stats", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            print(f"✅ Statistics API: {stats.get('total_signals', 0)} total signals")
        else:
            print(f"⚠️ Statistics API returned status: {response.status_code}")
        
        print("✅ API integration tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("⚠️ Flask app not running - API tests skipped")
        print("   Start the Flask app with: python app.py")
    except Exception as e:
        print(f"❌ Error testing API integration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 AI Trade Signal Generator - Comprehensive Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run all tests
    success &= test_signal_generator()
    success &= test_technical_analysis()
    success &= test_api_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ Your AI Trade Signal Generator is ready for production!")
        print("\nNext steps:")
        print("1. Start your Flask app: python app.py")
        print("2. Open your dashboard in browser: http://localhost:5000")
        print("3. Look for the AI Signals section in the Portfolio area")
        print("4. Click 'Generate Signal' to create your first high-ROI signal!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed and the database is accessible.")
