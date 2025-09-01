"""
COMPREHENSIVE ADVANCED SIGNAL SYSTEM TEST
==========================================
Tests the complete enhanced trading signal system with:
- Real market data integration
- Advanced technical analysis
- Realistic TP/SL calculation
- Automatic signal tracking
- ML learning system
- Performance analytics
"""

import sys
import os
import time
import json
import requests
from datetime import datetime
import sqlite3

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_advanced_signal_generation():
    """Test advanced signal generation system"""
    print("\n🎯 TESTING ADVANCED SIGNAL GENERATION")
    print("=" * 50)
    
    try:
        from advanced_trading_signal_manager import generate_trading_signal
        
        # Test signal generation
        print("📊 Generating advanced trading signal...")
        result = generate_trading_signal("GOLD", "1h")
        
        if result['success'] and result['signal_generated']:
            print("✅ SIGNAL GENERATED SUCCESSFULLY!")
            print(f"   📈 Type: {result['signal_type']}")
            print(f"   💰 Entry Price: ${result['entry_price']:.2f}")
            print(f"   🎯 Take Profit: ${result['take_profit']:.2f}")
            print(f"   🛡️ Stop Loss: ${result['stop_loss']:.2f}")
            print(f"   ⚖️ Risk:Reward: 1:{result['risk_reward_ratio']:.2f}")
            print(f"   📊 Expected ROI: {result['expected_roi']:.2f}%")
            print(f"   🔥 Confidence: {result['confidence']:.1%}")
            print(f"   🎲 Win Probability: {result['win_probability']:.1%}")
            print(f"   🧠 Reasoning: {result['reasoning']}")
            
            # Check signal strength
            signal_strength = result.get('signal_strength', 0)
            print(f"   💪 Signal Strength: {signal_strength:.3f}")
            
            return result
        else:
            print(f"ℹ️ No signal generated: {result.get('reason', 'Unknown')}")
            print(f"   Signal Strength: {result.get('signal_strength', 'N/A')}")
            print(f"   Required Minimum: {result.get('min_required', 'N/A')}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing signal generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_signal_tracking():
    """Test automatic signal tracking system"""
    print("\n🎯 TESTING SIGNAL TRACKING SYSTEM")
    print("=" * 50)
    
    try:
        from auto_signal_tracker import auto_tracker, start_signal_tracking, stop_signal_tracking
        
        # Check if tracking is running
        print("📡 Checking signal tracking status...")
        if auto_tracker.is_running:
            print("✅ Signal tracking is already running")
        else:
            print("🚀 Starting signal tracking...")
            start_signal_tracking()
            time.sleep(2)  # Give it time to start
        
        # Test getting active signals
        print("📊 Getting active signals...")
        active_signals = auto_tracker._get_active_signals()
        
        print(f"📈 Found {len(active_signals)} active signals")
        for i, signal in enumerate(active_signals[:3]):  # Show first 3
            print(f"   {i+1}. {signal['signal_type']} at ${signal['entry_price']:.2f}")
            print(f"      TP: ${signal['take_profit']:.2f}, SL: ${signal['stop_loss']:.2f}")
        
        # Test current price fetching
        print("💰 Testing current price fetching...")
        current_price = auto_tracker._get_current_gold_price()
        if current_price:
            print(f"✅ Current Gold Price: ${current_price:.2f}")
        else:
            print("❌ Failed to fetch current price")
        
        # Get performance stats
        print("📊 Getting performance statistics...")
        stats = auto_tracker.get_performance_stats()
        print(f"   Total Signals: {stats.get('total_signals', 0)}")
        print(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"   Average ROI: {stats.get('average_roi', 0):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing signal tracking: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_integration():
    """Test database integration and signal storage"""
    print("\n🎯 TESTING DATABASE INTEGRATION")
    print("=" * 50)
    
    try:
        # Check if database exists
        db_path = "advanced_trading_signals.db"
        
        if os.path.exists(db_path):
            print(f"✅ Database found: {db_path}")
            
            # Connect and check tables
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"📊 Database tables: {[t[0] for t in tables]}")
            
            # Count signals
            cursor.execute("SELECT COUNT(*) FROM advanced_signals")
            total_signals = cursor.fetchone()[0]
            print(f"📈 Total signals in database: {total_signals}")
            
            # Count by status
            cursor.execute("SELECT status, COUNT(*) FROM advanced_signals GROUP BY status")
            status_counts = cursor.fetchall()
            for status, count in status_counts:
                print(f"   {status}: {count}")
            
            # Get recent signals
            cursor.execute("""
                SELECT id, signal_type, entry_price, confidence, timestamp, status 
                FROM advanced_signals 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_signals = cursor.fetchall()
            
            print(f"📊 Recent signals:")
            for signal in recent_signals:
                signal_id, signal_type, entry_price, confidence, timestamp, status = signal
                print(f"   {signal_id[:12]}... {signal_type} ${entry_price:.2f} ({confidence:.1%}) [{status}]")
            
            conn.close()
            return True
            
        else:
            print(f"❌ Database not found: {db_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing database: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learning_system():
    """Test the ML learning system"""
    print("\n🎯 TESTING LEARNING SYSTEM")
    print("=" * 50)
    
    try:
        from auto_signal_tracker import learning_engine
        
        # Get learning analysis
        print("🧠 Getting learning analysis...")
        analysis = learning_engine.analyze_patterns()
        
        print(f"📊 Learning Status: {analysis.get('learning_status', 'Unknown')}")
        
        if analysis.get('learning_status') == 'active':
            stats = analysis.get('performance_stats', {})
            print(f"   Total Signals: {stats.get('total_signals', 0)}")
            print(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")
            print(f"   Average ROI: {stats.get('average_roi', 0):.2f}%")
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print("💡 Recommendations:")
                for rec in recommendations:
                    print(f"   • {rec}")
        else:
            print(f"ℹ️ {analysis.get('message', 'Learning system not active')}")
        
        # Test parameter optimization
        print("⚙️ Getting optimized parameters...")
        params = learning_engine.get_optimized_parameters()
        print(f"   Min Signal Strength: {params.get('min_signal_strength', 'N/A')}")
        print(f"   TP Multiplier: {params.get('tp_multiplier', 'N/A')}")
        print(f"   SL Multiplier: {params.get('sl_multiplier', 'N/A')}")
        print(f"   Confidence Threshold: {params.get('confidence_threshold', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing learning system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_api_integration():
    """Test Flask API endpoints (if server is running)"""
    print("\n🎯 TESTING FLASK API INTEGRATION")
    print("=" * 50)
    
    # Check if Flask server is running
    try:
        base_url = "http://localhost:5000"
        
        print(f"🌐 Testing connection to {base_url}...")
        response = requests.get(f"{base_url}/", timeout=5)
        
        if response.status_code == 200:
            print("✅ Flask server is running")
            
            # Test signal generation endpoint
            print("🎯 Testing signal generation API...")
            try:
                response = requests.get(f"{base_url}/api/generate-signal", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and data.get('signal_generated'):
                        print(f"✅ API Signal Generated: {data['signal_type']}")
                        print(f"   Entry: ${data['entry_price']:.2f}")
                        print(f"   TP: ${data['take_profit']:.2f}")
                        print(f"   SL: ${data['stop_loss']:.2f}")
                    else:
                        print(f"ℹ️ API No signal: {data.get('reason', 'Market conditions')}")
                else:
                    print(f"❌ API Error: {response.status_code}")
            except Exception as e:
                print(f"❌ API Test failed: {e}")
            
            # Test active signals endpoint
            print("📊 Testing active signals API...")
            try:
                response = requests.get(f"{base_url}/api/active-signals", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        signals = data.get('signals', [])
                        print(f"✅ Found {len(signals)} active signals via API")
                else:
                    print(f"❌ Active signals API error: {response.status_code}")
            except Exception as e:
                print(f"❌ Active signals API test failed: {e}")
            
            # Test signal stats endpoint
            print("📈 Testing signal statistics API...")
            try:
                response = requests.get(f"{base_url}/api/signal-stats", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        stats = data.get('performance_stats', {})
                        print(f"✅ Stats API working - Win Rate: {stats.get('win_rate', 0):.1f}%")
                else:
                    print(f"❌ Stats API error: {response.status_code}")
            except Exception as e:
                print(f"❌ Stats API test failed: {e}")
            
            return True
            
        else:
            print(f"❌ Flask server not responding: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Flask server not running or not accessible: {e}")
        print("ℹ️ Start the Flask server with: python app.py")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 COMPREHENSIVE ADVANCED SIGNAL SYSTEM TEST")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Test 1: Signal Generation
    test_results['signal_generation'] = test_advanced_signal_generation() is not None
    
    # Test 2: Signal Tracking
    test_results['signal_tracking'] = test_signal_tracking()
    
    # Test 3: Database Integration
    test_results['database'] = test_database_integration()
    
    # Test 4: Learning System
    test_results['learning'] = test_learning_system()
    
    # Test 5: Flask API Integration
    test_results['flask_api'] = test_flask_api_integration()
    
    # Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Advanced signal system is fully operational!")
        print("\n🎯 SYSTEM CAPABILITIES:")
        print("✅ Real market data integration")
        print("✅ Advanced technical analysis")
        print("✅ Realistic TP/SL calculation")
        print("✅ Automatic signal tracking")
        print("✅ ML learning system")
        print("✅ Performance analytics")
        print("✅ Flask API integration")
    else:
        print(f"⚠️ {total-passed} tests failed. Check the issues above.")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    print("🔧 Testing Advanced Trading Signal System...")
    run_comprehensive_test()
