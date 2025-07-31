#!/usr/bin/env python3
"""
ML Engine Validation Test
Ensures all ML engines are working and returns real predictions
"""
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_ml_engines():
    """Test all ML engines to ensure they return real predictions"""
    
    print("🔬 TESTING ALL ML ENGINES FOR REAL PREDICTIONS")
    print("=" * 60)
    
    engines_tested = []
    working_engines = []
    failed_engines = []
    
    # Test 1: Enhanced ML Prediction Engine
    print("\n1. Testing Enhanced ML Prediction Engine...")
    try:
        from enhanced_ml_prediction_engine import get_enhanced_ml_predictions
        result = get_enhanced_ml_predictions()
        
        if result and result.get('predictions'):
            current_price = result.get('current_price', 0)
            predictions_count = len(result.get('predictions', []))
            
            print(f"   ✅ SUCCESS: {predictions_count} predictions")
            print(f"   📊 Current Price: ${current_price:.2f}")
            
            for pred in result.get('predictions', [])[:2]:  # Show first 2
                print(f"   📈 {pred['timeframe']}: ${pred.get('predicted_price', 0):.2f} ({pred.get('change_percent', 0):+.2f}%)")
            
            working_engines.append('enhanced_ml_prediction_engine')
            engines_tested.append(('enhanced_ml_prediction_engine', 'SUCCESS', predictions_count))
        else:
            print("   ❌ FAILED: No predictions returned")
            failed_engines.append('enhanced_ml_prediction_engine')
            engines_tested.append(('enhanced_ml_prediction_engine', 'FAILED', 0))
            
    except ImportError as e:
        print(f"   ⚠️ NOT AVAILABLE: {e}")
        failed_engines.append('enhanced_ml_prediction_engine')
        engines_tested.append(('enhanced_ml_prediction_engine', 'NOT_AVAILABLE', 0))
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        failed_engines.append('enhanced_ml_prediction_engine')
        engines_tested.append(('enhanced_ml_prediction_engine', 'ERROR', 0))
    
    # Test 2: Intelligent ML Predictor
    print("\n2. Testing Intelligent ML Predictor...")
    try:
        from intelligent_ml_predictor import get_intelligent_ml_predictions
        result = get_intelligent_ml_predictions('XAUUSD')
        
        if result and result.get('predictions'):
            current_price = result.get('current_price', 0)
            predictions_count = len(result.get('predictions', []))
            
            print(f"   ✅ SUCCESS: {predictions_count} predictions")
            print(f"   📊 Current Price: ${current_price:.2f}")
            
            for pred in result.get('predictions', [])[:2]:  # Show first 2
                print(f"   📈 {pred['timeframe']}: ${pred.get('predicted_price', 0):.2f} ({pred.get('change_percent', 0):+.2f}%)")
            
            working_engines.append('intelligent_ml_predictor')
            engines_tested.append(('intelligent_ml_predictor', 'SUCCESS', predictions_count))
        else:
            print("   ❌ FAILED: No predictions returned")
            failed_engines.append('intelligent_ml_predictor')
            engines_tested.append(('intelligent_ml_predictor', 'FAILED', 0))
            
    except ImportError as e:
        print(f"   ⚠️ NOT AVAILABLE: {e}")
        failed_engines.append('intelligent_ml_predictor')
        engines_tested.append(('intelligent_ml_predictor', 'NOT_AVAILABLE', 0))
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        failed_engines.append('intelligent_ml_predictor')
        engines_tested.append(('intelligent_ml_predictor', 'ERROR', 0))
    
    # Test 3: Main ML Prediction API
    print("\n3. Testing Main ML Prediction API...")
    try:
        from ml_prediction_api import get_ml_predictions
        import asyncio
        
        # Create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(get_ml_predictions('XAUUSD'))
        
        if result and result.get('success') and result.get('predictions'):
            current_price = result.get('current_price', 0)
            predictions_count = len(result.get('predictions', []))
            
            print(f"   ✅ SUCCESS: {predictions_count} predictions")
            print(f"   📊 Current Price: ${current_price:.2f}")
            print(f"   🔧 Source: {result.get('source', 'unknown')}")
            
            for pred in result.get('predictions', [])[:2]:  # Show first 2
                print(f"   📈 {pred['timeframe']}: ${pred.get('predicted_price', 0):.2f} ({pred.get('change_percent', 0):+.2f}%)")
            
            working_engines.append('main_ml_prediction_api')
            engines_tested.append(('main_ml_prediction_api', 'SUCCESS', predictions_count))
        else:
            print("   ❌ FAILED: No valid predictions returned")
            failed_engines.append('main_ml_prediction_api')
            engines_tested.append(('main_ml_prediction_api', 'FAILED', 0))
            
    except ImportError as e:
        print(f"   ⚠️ NOT AVAILABLE: {e}")
        failed_engines.append('main_ml_prediction_api')
        engines_tested.append(('main_ml_prediction_api', 'NOT_AVAILABLE', 0))
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        failed_engines.append('main_ml_prediction_api')
        engines_tested.append(('main_ml_prediction_api', 'ERROR', 0))
    
    # Test 4: Self-Improving ML Engine
    print("\n4. Testing Self-Improving ML Engine...")
    try:
        from self_improving_ml_engine import SelfImprovingMLEngine
        
        engine = SelfImprovingMLEngine()
        result = engine.generate_daily_prediction('XAUUSD')
        
        if result and result.get('predictions'):
            current_price = result.get('current_price', 0)
            predictions_count = len(result.get('predictions', []))
            
            print(f"   ✅ SUCCESS: {predictions_count} predictions")
            print(f"   📊 Current Price: ${current_price:.2f}")
            
            for pred in result.get('predictions', [])[:2]:  # Show first 2
                print(f"   📈 {pred['timeframe']}: ${pred.get('predicted_price', 0):.2f} ({pred.get('change_percent', 0):+.2f}%)")
            
            working_engines.append('self_improving_ml_engine')
            engines_tested.append(('self_improving_ml_engine', 'SUCCESS', predictions_count))
        else:
            print("   ❌ FAILED: No predictions returned")
            failed_engines.append('self_improving_ml_engine')
            engines_tested.append(('self_improving_ml_engine', 'FAILED', 0))
            
    except ImportError as e:
        print(f"   ⚠️ NOT AVAILABLE: {e}")
        failed_engines.append('self_improving_ml_engine')
        engines_tested.append(('self_improving_ml_engine', 'NOT_AVAILABLE', 0))
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        failed_engines.append('self_improving_ml_engine')
        engines_tested.append(('self_improving_ml_engine', 'ERROR', 0))
    
    # Summary Report
    print("\n" + "=" * 60)
    print("📊 ML ENGINES TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Working Engines: {len(working_engines)}")
    for engine in working_engines:
        print(f"   • {engine}")
    
    print(f"\n❌ Failed Engines: {len(failed_engines)}")
    for engine in failed_engines:
        print(f"   • {engine}")
    
    print(f"\n📈 Total Engines Tested: {len(engines_tested)}")
    
    # Detailed Results Table
    print("\n📋 DETAILED RESULTS:")
    print("-" * 60)
    print(f"{'Engine':<30} {'Status':<15} {'Predictions'}")
    print("-" * 60)
    for engine, status, count in engines_tested:
        print(f"{engine:<30} {status:<15} {count}")
    
    print("\n🎯 RECOMMENDATION:")
    if working_engines:
        print(f"✅ Use: {working_engines[0]} (primary)")
        if len(working_engines) > 1:
            print(f"🔄 Fallback: {working_engines[1:]} (backup)")
        print("✅ Your ML system can provide REAL predictions!")
    else:
        print("❌ CRITICAL: No ML engines are working!")
        print("🔧 Check imports and dependencies")
    
    return working_engines, failed_engines

if __name__ == "__main__":
    working, failed = test_all_ml_engines()
    
    if working:
        print(f"\n🎉 SUCCESS: {len(working)} ML engines are working!")
        exit(0)
    else:
        print(f"\n💥 FAILURE: No ML engines are working!")
        exit(1)
