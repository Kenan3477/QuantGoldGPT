#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced ML Prediction Engine
Tests all strategies, ensemble methods, and validation systems
"""

import asyncio
import time
import sys
import os
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_ml_prediction_engine import (
        AdvancedMLPredictionEngine, 
        get_advanced_ml_predictions,
        TechnicalStrategy,
        SentimentStrategy,
        MacroStrategy,
        PatternStrategy,
        MomentumStrategy,
        PredictionValidator,
        EnsembleVotingSystem,
        StrategyPerformanceTracker
    )
    from data_integration_engine import DataManager, DataIntegrationEngine
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Engine import failed: {e}")
    ML_ENGINE_AVAILABLE = False

class AdvancedMLTestSuite:
    """Comprehensive test suite for advanced ML system"""
    
    def __init__(self):
        self.test_results = []
        self.data_manager = None
        self.ml_engine = None
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.test_results.append(result)
        
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_symbol} {test_name}: {status}")
        if details:
            print(f"    {details}")
    
    async def test_data_manager_initialization(self):
        """Test data manager initialization"""
        try:
            integration_engine = DataIntegrationEngine()
            self.data_manager = DataManager(integration_engine)
            
            # Test data fetching
            dataset = await self.data_manager.get_ml_ready_dataset()
            
            if dataset and 'features' in dataset:
                feature_count = len(dataset['features'])
                self.log_test(
                    "Data Manager Initialization",
                    "PASS",
                    f"Dataset loaded with {feature_count} features"
                )
                return True
            else:
                self.log_test(
                    "Data Manager Initialization", 
                    "FAIL",
                    "Dataset empty or malformed"
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Data Manager Initialization",
                "FAIL", 
                f"Exception: {e}"
            )
            return False
    
    async def test_individual_strategies(self):
        """Test each prediction strategy individually"""
        if not self.data_manager:
            self.log_test("Individual Strategies", "SKIP", "Data manager not available")
            return False
        
        strategies = [
            TechnicalStrategy(self.data_manager),
            SentimentStrategy(self.data_manager),
            MacroStrategy(self.data_manager),
            PatternStrategy(self.data_manager),
            MomentumStrategy(self.data_manager)
        ]
        
        strategy_results = []
        
        for strategy in strategies:
            try:
                start_time = time.time()
                prediction = await strategy.predict("1H")
                execution_time = time.time() - start_time
                
                # Validate prediction structure
                if not prediction:
                    self.log_test(
                        f"{strategy.name} Strategy",
                        "FAIL",
                        "No prediction returned"
                    )
                    continue
                
                # Check required fields
                required_fields = [
                    'strategy_name', 'timeframe', 'current_price', 'predicted_price',
                    'direction', 'confidence', 'support_level', 'resistance_level',
                    'stop_loss', 'take_profit'
                ]
                
                missing_fields = [field for field in required_fields 
                                if not hasattr(prediction, field)]
                
                if missing_fields:
                    self.log_test(
                        f"{strategy.name} Strategy",
                        "FAIL",
                        f"Missing fields: {missing_fields}"
                    )
                    continue
                
                # Validate prediction logic
                validation_errors = []
                
                if prediction.current_price <= 0:
                    validation_errors.append("Invalid current price")
                
                if prediction.predicted_price <= 0:
                    validation_errors.append("Invalid predicted price")
                
                if not (0 <= prediction.confidence <= 1):
                    validation_errors.append("Confidence not in [0,1] range")
                
                if prediction.direction not in ["bullish", "bearish", "neutral"]:
                    validation_errors.append("Invalid direction")
                
                if validation_errors:
                    self.log_test(
                        f"{strategy.name} Strategy",
                        "FAIL",
                        f"Validation errors: {validation_errors}"
                    )
                    continue
                
                # Success
                self.log_test(
                    f"{strategy.name} Strategy",
                    "PASS",
                    f"Prediction generated in {execution_time:.3f}s, "
                    f"confidence: {prediction.confidence:.3f}, "
                    f"direction: {prediction.direction}"
                )
                strategy_results.append(True)
                
            except Exception as e:
                self.log_test(
                    f"{strategy.name} Strategy",
                    "FAIL",
                    f"Exception: {e}"
                )
                strategy_results.append(False)
        
        success_rate = sum(strategy_results) / len(strategy_results) if strategy_results else 0
        
        self.log_test(
            "Overall Strategy Performance",
            "PASS" if success_rate >= 0.6 else "FAIL",
            f"Success rate: {success_rate:.1%} ({sum(strategy_results)}/{len(strategy_results)})"
        )
        
        return success_rate >= 0.6
    
    def test_prediction_validator(self):
        """Test the prediction validation system"""
        from advanced_ml_prediction_engine import PredictionResult
        
        validator = PredictionValidator()
        
        # Create test predictions
        test_cases = [
            {
                'name': 'Valid Bullish Prediction',
                'prediction': PredictionResult(
                    strategy_name='Test',
                    timeframe='1H',
                    current_price=2000.0,
                    predicted_price=2010.0,
                    price_change=10.0,
                    price_change_percent=0.5,
                    direction='bullish',
                    confidence=0.7,
                    support_level=1990.0,
                    resistance_level=2020.0,
                    stop_loss=1980.0,
                    take_profit=2030.0,
                    timestamp=datetime.now(timezone.utc),
                    features_used=['test_feature'],
                    reasoning='Test prediction'
                ),
                'should_pass': True
            },
            {
                'name': 'Invalid Stop Loss',
                'prediction': PredictionResult(
                    strategy_name='Test',
                    timeframe='1H',
                    current_price=2000.0,
                    predicted_price=2010.0,
                    price_change=10.0,
                    price_change_percent=0.5,
                    direction='bullish',
                    confidence=0.7,
                    support_level=1990.0,
                    resistance_level=2020.0,
                    stop_loss=2010.0,  # Invalid: stop loss above current price for bullish
                    take_profit=2030.0,
                    timestamp=datetime.now(timezone.utc),
                    features_used=['test_feature'],
                    reasoning='Test prediction'
                ),
                'should_pass': False
            },
            {
                'name': 'Low Confidence',
                'prediction': PredictionResult(
                    strategy_name='Test',
                    timeframe='1H',
                    current_price=2000.0,
                    predicted_price=2005.0,
                    price_change=5.0,
                    price_change_percent=0.25,
                    direction='bullish',
                    confidence=0.1,  # Very low confidence
                    support_level=1990.0,
                    resistance_level=2020.0,
                    stop_loss=1980.0,
                    take_profit=2030.0,
                    timestamp=datetime.now(timezone.utc),
                    features_used=['test_feature'],
                    reasoning='Test prediction'
                ),
                'should_pass': False
            }
        ]
        
        validator_results = []
        
        for test_case in test_cases:
            try:
                is_valid, message, score = validator.validate_prediction(test_case['prediction'])
                
                if is_valid == test_case['should_pass']:
                    self.log_test(
                        f"Validator: {test_case['name']}",
                        "PASS",
                        f"Correctly identified as {'valid' if is_valid else 'invalid'}, score: {score:.3f}"
                    )
                    validator_results.append(True)
                else:
                    self.log_test(
                        f"Validator: {test_case['name']}",
                        "FAIL",
                        f"Expected {'valid' if test_case['should_pass'] else 'invalid'}, got {'valid' if is_valid else 'invalid'}"
                    )
                    validator_results.append(False)
                    
            except Exception as e:
                self.log_test(
                    f"Validator: {test_case['name']}",
                    "FAIL",
                    f"Exception: {e}"
                )
                validator_results.append(False)
        
        success_rate = sum(validator_results) / len(validator_results) if validator_results else 0
        return success_rate >= 0.8
    
    def test_performance_tracker(self):
        """Test the strategy performance tracking system"""
        tracker = StrategyPerformanceTracker()
        
        try:
            # Record some performance data
            test_data = [
                ('Strategy_A', 2000, 2005, 0.7),  # Good prediction
                ('Strategy_A', 2010, 2015, 0.8),  # Good prediction
                ('Strategy_B', 2020, 2010, 0.6),  # Bad prediction
                ('Strategy_A', 2030, 2035, 0.9),  # Good prediction
                ('Strategy_B', 2040, 2045, 0.7),  # Good prediction
            ]
            
            for strategy, predicted, actual, confidence in test_data:
                tracker.record_performance(strategy, predicted, actual, confidence)
            
            # Get weights
            weights = tracker.get_all_weights(['Strategy_A', 'Strategy_B'])
            
            # Strategy A should have higher weight (better performance)
            if weights['Strategy_A'] > weights['Strategy_B']:
                self.log_test(
                    "Performance Tracker",
                    "PASS", 
                    f"Correctly weighted strategies: A={weights['Strategy_A']:.3f}, B={weights['Strategy_B']:.3f}"
                )
                return True
            else:
                self.log_test(
                    "Performance Tracker",
                    "FAIL",
                    f"Incorrect weighting: A={weights['Strategy_A']:.3f}, B={weights['Strategy_B']:.3f}"
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Performance Tracker",
                "FAIL",
                f"Exception: {e}"
            )
            return False
    
    async def test_ensemble_system(self):
        """Test the ensemble voting system"""
        if not self.data_manager:
            self.log_test("Ensemble System", "SKIP", "Data manager not available")
            return False
        
        try:
            self.ml_engine = AdvancedMLPredictionEngine(self.data_manager)
            
            # Test ensemble prediction generation
            start_time = time.time()
            ensemble_prediction = await self.ml_engine.generate_prediction("1H")
            execution_time = time.time() - start_time
            
            if ensemble_prediction is None:
                self.log_test(
                    "Ensemble System",
                    "FAIL",
                    "No ensemble prediction generated"
                )
                return False
            
            # Validate ensemble prediction structure
            required_fields = [
                'timeframe', 'current_price', 'predicted_price', 'direction',
                'ensemble_confidence', 'strategy_votes', 'validation_score'
            ]
            
            missing_fields = [field for field in required_fields 
                            if not hasattr(ensemble_prediction, field)]
            
            if missing_fields:
                self.log_test(
                    "Ensemble System",
                    "FAIL",
                    f"Missing ensemble fields: {missing_fields}"
                )
                return False
            
            # Validate strategy votes
            if not ensemble_prediction.strategy_votes:
                self.log_test(
                    "Ensemble System",
                    "FAIL",
                    "No strategy votes recorded"
                )
                return False
            
            # Check vote weights sum approximately to 1
            vote_sum = sum(ensemble_prediction.strategy_votes.values())
            if not (0.8 <= vote_sum <= 1.2):  # Allow some tolerance
                self.log_test(
                    "Ensemble System",
                    "FAIL",
                    f"Vote weights don't sum to ~1.0: {vote_sum:.3f}"
                )
                return False
            
            # Performance test: should complete in under 5 seconds
            if execution_time > 5.0:
                self.log_test(
                    "Ensemble System",
                    "WARN",
                    f"Slow execution: {execution_time:.2f}s > 5s target"
                )
            
            self.log_test(
                "Ensemble System",
                "PASS",
                f"Generated ensemble prediction in {execution_time:.2f}s, "
                f"confidence: {ensemble_prediction.ensemble_confidence:.3f}, "
                f"strategies: {len(ensemble_prediction.strategy_votes)}"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "Ensemble System",
                "FAIL",
                f"Exception: {e}"
            )
            return False
    
    async def test_multi_timeframe_predictions(self):
        """Test multi-timeframe prediction generation"""
        if not self.ml_engine:
            self.log_test("Multi-Timeframe", "SKIP", "ML engine not available")
            return False
        
        try:
            timeframes = ["1H", "4H", "1D"]
            start_time = time.time()
            predictions = await self.ml_engine.generate_multi_timeframe_predictions(timeframes)
            execution_time = time.time() - start_time
            
            # Check all timeframes were processed
            generated_timeframes = list(predictions.keys())
            
            if len(generated_timeframes) == 0:
                self.log_test(
                    "Multi-Timeframe",
                    "FAIL",
                    "No predictions generated for any timeframe"
                )
                return False
            
            # Performance test: should complete in under 10 seconds
            if execution_time > 10.0:
                self.log_test(
                    "Multi-Timeframe",
                    "WARN",
                    f"Slow execution: {execution_time:.2f}s > 10s target"
                )
            
            success_rate = len(generated_timeframes) / len(timeframes)
            
            self.log_test(
                "Multi-Timeframe",
                "PASS" if success_rate >= 0.6 else "FAIL",
                f"Generated predictions for {generated_timeframes} "
                f"({success_rate:.1%} success rate) in {execution_time:.2f}s"
            )
            
            return success_rate >= 0.6
            
        except Exception as e:
            self.log_test(
                "Multi-Timeframe",
                "FAIL",
                f"Exception: {e}"
            )
            return False
    
    async def test_api_integration(self):
        """Test the main API function"""
        try:
            start_time = time.time()
            result = await get_advanced_ml_predictions(['1H', '4H'])
            execution_time = time.time() - start_time
            
            if result['status'] != 'success':
                self.log_test(
                    "API Integration",
                    "FAIL",
                    f"API returned error: {result.get('error', 'Unknown')}"
                )
                return False
            
            # Check response structure
            required_keys = ['status', 'timestamp', 'predictions', 'performance', 'system_info']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                self.log_test(
                    "API Integration",
                    "FAIL", 
                    f"Missing response keys: {missing_keys}"
                )
                return False
            
            # Check predictions
            prediction_count = len(result['predictions'])
            
            self.log_test(
                "API Integration",
                "PASS",
                f"API returned {prediction_count} predictions in {execution_time:.2f}s"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "API Integration",
                "FAIL",
                f"Exception: {e}"
            )
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("🧪 ADVANCED ML PREDICTION ENGINE TEST SUMMARY")
        print("="*70)
        
        pass_count = sum(1 for result in self.test_results if result['status'] == 'PASS')
        fail_count = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        warn_count = sum(1 for result in self.test_results if result['status'] == 'WARN')
        skip_count = sum(1 for result in self.test_results if result['status'] == 'SKIP')
        total_count = len(self.test_results)
        
        print(f"📊 Test Results: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARN, {skip_count} SKIP")
        print(f"📈 Success Rate: {pass_count/total_count*100:.1f}%" if total_count > 0 else "No tests run")
        
        if fail_count > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"   • {result['test']}: {result['details']}")
        
        if warn_count > 0:
            print(f"\n⚠️  Warnings:")
            for result in self.test_results:
                if result['status'] == 'WARN':
                    print(f"   • {result['test']}: {result['details']}")
        
        print(f"\n🏆 Overall Status: {'PASS' if fail_count == 0 else 'FAIL'}")
        
        return fail_count == 0

async def run_comprehensive_tests():
    """Run the complete test suite"""
    print("🚀 Advanced ML Prediction Engine - Comprehensive Test Suite")
    print("="*70)
    
    if not ML_ENGINE_AVAILABLE:
        print("❌ ML Engine not available - cannot run tests")
        return False
    
    test_suite = AdvancedMLTestSuite()
    
    # Run all tests
    test_sequence = [
        ("Data Manager", test_suite.test_data_manager_initialization),
        ("Individual Strategies", test_suite.test_individual_strategies),
        ("Prediction Validator", lambda: test_suite.test_prediction_validator()),
        ("Performance Tracker", lambda: test_suite.test_performance_tracker()),
        ("Ensemble System", test_suite.test_ensemble_system),
        ("Multi-Timeframe", test_suite.test_multi_timeframe_predictions),
        ("API Integration", test_suite.test_api_integration),
    ]
    
    print(f"\n🧪 Running {len(test_sequence)} test categories...\n")
    
    for test_name, test_func in test_sequence:
        print(f"🔍 Testing {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
        except Exception as e:
            test_suite.log_test(test_name, "FAIL", f"Test execution failed: {e}")
        
        print()  # Add spacing between test categories
    
    # Print final summary
    return test_suite.print_summary()

def quick_functionality_test():
    """Quick test that doesn't require full system"""
    print("⚡ Quick Functionality Test")
    print("-" * 30)
    
    try:
        # Test imports
        from advanced_ml_prediction_engine import (
            PredictionResult, EnsemblePrediction, 
            StrategyPerformanceTracker, PredictionValidator
        )
        print("✅ Core classes import successfully")
        
        # Test data structures
        prediction = PredictionResult(
            strategy_name="Test",
            timeframe="1H",
            current_price=2000.0,
            predicted_price=2010.0,
            price_change=10.0,
            price_change_percent=0.5,
            direction="bullish",
            confidence=0.7,
            support_level=1990.0,
            resistance_level=2020.0,
            stop_loss=1980.0,
            take_profit=2030.0,
            timestamp=datetime.now(timezone.utc),
            features_used=["test"],
            reasoning="test"
        )
        print("✅ PredictionResult creates successfully")
        
        # Test performance tracker
        tracker = StrategyPerformanceTracker()
        tracker.record_performance("TestStrategy", 2000, 2005, 0.8)
        weights = tracker.get_all_weights(["TestStrategy"])
        print(f"✅ PerformanceTracker works: weight = {weights['TestStrategy']:.3f}")
        
        # Test validator
        validator = PredictionValidator()
        is_valid, msg, score = validator.validate_prediction(prediction)
        print(f"✅ Validator works: valid = {is_valid}, score = {score:.3f}")
        
        print("\n🎉 Quick functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Advanced ML Prediction Engine Test Suite")
    print("="*70)
    
    # Check if we can run full tests
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_functionality_test()
    elif ML_ENGINE_AVAILABLE:
        try:
            success = asyncio.run(run_comprehensive_tests())
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            sys.exit(1)
    else:
        print("⚠️  Full ML engine not available, running quick test...")
        success = quick_functionality_test()
        sys.exit(0 if success else 1)
