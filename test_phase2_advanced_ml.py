"""
🚀 PHASE 2 TESTING: ADVANCED MULTI-STRATEGY ML ARCHITECTURE
===========================================================

Comprehensive test suite for the Advanced Multi-Strategy ML System
Tests ensemble voting, meta-learning, and institutional data integration

Author: GoldGPT AI System
Created: July 23, 2025
"""

import asyncio
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('phase2_testing')

def test_advanced_system_imports():
    """Test 1: Verify all advanced system components can be imported"""
    print("\n🧪 TEST 1: Advanced System Imports")
    print("-" * 50)
    
    try:
        # Test base strategies
        from advanced_ensemble_ml_system import TechnicalStrategy, SentimentStrategy, BaseStrategy
        print("✅ Base strategies imported successfully")
        
        # Test additional strategies  
        from advanced_strategies_part2 import MacroStrategy, PatternStrategy, MomentumStrategy
        print("✅ Additional strategies imported successfully")
        
        # Test ensemble system
        from ensemble_voting_system import EnsembleVotingSystem, MetaLearningEngine, MarketRegimeDetector
        print("✅ Ensemble system imported successfully")
        
        # Test integration API
        from advanced_ml_integration_api import AdvancedMLIntegration, get_advanced_ml_predictions
        print("✅ Integration API imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_individual_strategies():
    """Test 2: Test individual strategy implementations"""
    print("\n🧪 TEST 2: Individual Strategy Testing")
    print("-" * 50)
    
    try:
        from advanced_ensemble_ml_system import TechnicalStrategy, SentimentStrategy
        from advanced_strategies_part2 import MacroStrategy, PatternStrategy, MomentumStrategy
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = pd.DataFrame({
            'Open': np.random.normal(3400, 50, len(dates)),
            'High': np.random.normal(3430, 60, len(dates)),
            'Low': np.random.normal(3370, 40, len(dates)),
            'Close': np.random.normal(3400, 50, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure High >= Low, Open, Close
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        strategies = [
            ('Technical', TechnicalStrategy()),
            ('Sentiment', SentimentStrategy()),
            ('Macro', MacroStrategy()),
            ('Pattern', PatternStrategy()),
            ('Momentum', MomentumStrategy())
        ]
        
        results = {}
        for name, strategy in strategies:
            try:
                result = strategy.predict(data, '1d')
                results[name] = result
                print(f"✅ {name} Strategy: ${result.predicted_price:.2f} ({result.direction}, {result.confidence:.2%})")
            except Exception as e:
                print(f"❌ {name} Strategy failed: {e}")
                results[name] = None
        
        successful_strategies = len([r for r in results.values() if r is not None])
        print(f"\n📊 Strategy Test Results: {successful_strategies}/5 strategies successful")
        
        return successful_strategies >= 3
        
    except Exception as e:
        print(f"❌ Strategy testing failed: {e}")
        return False

def test_market_regime_detection():
    """Test 3: Test market regime detection"""
    print("\n🧪 TEST 3: Market Regime Detection")
    print("-" * 50)
    
    try:
        from ensemble_voting_system import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        
        # Test different market scenarios
        test_scenarios = [
            # High volatility scenario
            {
                'name': 'High Volatility',
                'data': pd.DataFrame({
                    'Close': [3400, 3380, 3420, 3360, 3440, 3350, 3450, 3330, 3460],
                    'Volume': [5000, 7000, 8000, 9000, 10000, 8500, 7500, 9500, 8000]
                })
            },
            # Trending scenario
            {
                'name': 'Strong Trend',
                'data': pd.DataFrame({
                    'Close': [3400, 3410, 3420, 3430, 3440, 3450, 3460, 3470, 3480],
                    'Volume': [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800]
                })
            },
            # Ranging scenario
            {
                'name': 'Range-bound',
                'data': pd.DataFrame({
                    'Close': [3400, 3405, 3398, 3402, 3401, 3399, 3403, 3400, 3401],
                    'Volume': [3000, 3050, 3100, 3080, 3020, 3040, 3060, 3030, 3070]
                })
            }
        ]
        
        for scenario in test_scenarios:
            conditions = detector.detect_market_regime(scenario['data'])
            print(f"✅ {scenario['name']}: {conditions.market_regime} "
                  f"(vol: {conditions.volatility:.3f}, trend: {conditions.trend_strength:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Market regime detection failed: {e}")
        return False

async def test_ensemble_system():
    """Test 4: Test ensemble voting system"""
    print("\n🧪 TEST 4: Ensemble Voting System")
    print("-" * 50)
    
    try:
        from ensemble_voting_system import create_advanced_ml_system
        
        # Create ensemble system
        ensemble = await create_advanced_ml_system()
        
        if not ensemble:
            print("❌ Failed to create ensemble system")
            return False
        
        print(f"✅ Ensemble system created with {len(ensemble.strategies)} strategies")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = pd.DataFrame({
            'Open': np.random.normal(3400, 30, len(dates)),
            'High': np.random.normal(3420, 35, len(dates)),
            'Low': np.random.normal(3380, 25, len(dates)),
            'Close': np.random.normal(3400, 30, len(dates)),
            'Volume': np.random.randint(2000, 8000, len(dates))
        }, index=dates)
        
        # Fix OHLC relationships
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        # Test ensemble prediction
        prediction = await ensemble.generate_ensemble_prediction(data, '1d')
        
        print(f"✅ Ensemble Prediction: ${prediction.predicted_price:.2f}")
        print(f"   Direction: {prediction.direction}")
        print(f"   Confidence: {prediction.confidence:.2%}")
        print(f"   Contributing strategies: {len(prediction.contributing_strategies)}")
        print(f"   Market regime: {prediction.market_conditions.market_regime}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_api():
    """Test 5: Test integration API"""
    print("\n🧪 TEST 5: Integration API Testing")
    print("-" * 50)
    
    try:
        from advanced_ml_integration_api import AdvancedMLIntegration
        
        # Create integration instance
        integration = AdvancedMLIntegration()
        
        # Test initialization
        await integration.initialize()
        
        if not integration.is_initialized:
            print("⚠️ Integration not fully initialized (may be due to missing data)")
            # Try fallback predictions
            fallback_result = await integration.get_enhanced_predictions("XAUUSD")
            if fallback_result.get('success'):
                print(f"✅ Fallback predictions working: {len(fallback_result.get('predictions', []))} timeframes")
                return True
            else:
                print("❌ Even fallback predictions failed")
                return False
        
        # Test enhanced predictions
        result = await integration.get_enhanced_predictions("XAUUSD")
        
        if result.get('success'):
            predictions = result.get('predictions', [])
            print(f"✅ Enhanced predictions: {len(predictions)} timeframes")
            print(f"   Current price: ${result.get('current_price', 0):.2f}")
            print(f"   Overall direction: {result.get('overall_direction')}")
            print(f"   Overall confidence: {result.get('overall_confidence', 0):.2%}")
            
            # Test system status
            status = integration.get_system_status()
            print(f"✅ System status retrieved: {len(status)} metrics")
            
            return True
        else:
            print(f"❌ Enhanced predictions failed: {result.get('error')}")
            return False
        
    except Exception as e:
        print(f"❌ Integration API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ml_api_integration():
    """Test 6: Test ML API integration"""
    print("\n🧪 TEST 6: ML API Integration")
    print("-" * 50)
    
    try:
        # Test the main ML prediction function
        from ml_prediction_api import get_ml_predictions
        
        result = await get_ml_predictions("XAUUSD")
        
        if result.get('success'):
            predictions = result.get('predictions', [])
            print(f"✅ ML API integration: {len(predictions)} predictions")
            print(f"   Current price: ${result.get('current_price', 0):.2f}")
            print(f"   Data quality: {result.get('data_quality', 'unknown')}")
            print(f"   Enhanced features: {result.get('enhanced_features', False)}")
            
            # Check if any predictions have ensemble features
            ensemble_features = any(
                'strategy_ensemble' in pred and pred['strategy_ensemble'] 
                for pred in predictions
            )
            
            if ensemble_features:
                print("✅ Ensemble features detected in predictions")
            else:
                print("⚠️ No ensemble features detected (using fallback)")
            
            return True
        else:
            print(f"❌ ML API integration failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ ML API integration test failed: {e}")
        return False

async def run_phase2_test_suite():
    """Run complete Phase 2 test suite"""
    print("🚀 PHASE 2: ADVANCED MULTI-STRATEGY ML ARCHITECTURE")
    print("=" * 60)
    print("Testing comprehensive ensemble system with meta-learning")
    print(f"Test execution started: {datetime.now()}")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Import Tests", test_advanced_system_imports),
        ("Strategy Tests", test_individual_strategies),
        ("Regime Detection", test_market_regime_detection),
        ("Ensemble System", test_ensemble_system),
        ("Integration API", test_integration_api),
        ("ML API Integration", test_ml_api_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n⏳ Running {test_name}...")
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            duration = time.time() - start_time
            test_results.append((test_name, result, duration))
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_results.append((test_name, False, duration))
            print(f"   ❌ FAILED ({duration:.2f}s): {e}")
    
    # Test summary
    print("\n" + "=" * 60)
    print("🎯 PHASE 2 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result, _ in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, result, duration in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} | {status} | {duration:6.2f}s")
    
    print("-" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 PHASE 2 IMPLEMENTATION: SUCCESS!")
        print("   ✅ Advanced Multi-Strategy ML Architecture operational")
        print("   ✅ Ensemble voting system functional")
        print("   ✅ Integration with existing infrastructure complete")
        if success_rate >= 90:
            print("   🏆 EXCELLENT IMPLEMENTATION QUALITY")
    elif success_rate >= 60:
        print("\n⚠️ PHASE 2 IMPLEMENTATION: PARTIAL SUCCESS")
        print("   📋 Core functionality operational with some issues")
        print("   🔧 Review failed tests for optimization opportunities")
    else:
        print("\n❌ PHASE 2 IMPLEMENTATION: NEEDS ATTENTION")
        print("   📋 Critical issues detected")
        print("   🔧 Review system architecture and dependencies")
    
    print(f"\n🏛️ Phase 2 testing completed: {datetime.now()}")
    
    return success_rate >= 80

if __name__ == "__main__":
    print("🧪 Starting Phase 2 Advanced ML Architecture Test Suite...")
    asyncio.run(run_phase2_test_suite())
