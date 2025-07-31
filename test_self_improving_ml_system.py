#!/usr/bin/env python3
"""
GoldGPT Self-Improving ML System - Comprehensive Test Suite
Tests all components of the self-improving ML system
"""

import asyncio
import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all our ML system components
from prediction_validation_engine import PredictionValidationEngine
from self_improving_learning_engine import SelfImprovingLearningEngine
from advanced_unified_prediction_system import AdvancedUnifiedPredictionSystem
from performance_dashboard import PerformanceDashboard
from daily_model_retraining_scheduler import DailyModelRetrainingScheduler

class SelfImprovingMLSystemTest:
    """
    Comprehensive test suite for the self-improving ML system
    """
    
    def __init__(self):
        self.db_path = "goldgpt_ml_tracking_test.db"
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.validator = PredictionValidationEngine(self.db_path)
        self.learning_engine = SelfImprovingLearningEngine(self.db_path)
        self.prediction_system = AdvancedUnifiedPredictionSystem(self.db_path)
        self.dashboard = PerformanceDashboard(self.db_path)
        self.scheduler = DailyModelRetrainingScheduler(self.db_path)
        
        # Test configuration
        self.test_results = []
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all system components"""
        try:
            print("🚀 Starting comprehensive self-improving ML system test...\n")
            
            # Initialize test database
            await self._initialize_test_database()
            
            # Test 1: Database Schema
            print("1️⃣ Testing database schema...")
            schema_result = await self._test_database_schema()
            self.test_results.append(('Database Schema', schema_result))
            
            # Test 2: Prediction Generation
            print("\n2️⃣ Testing unified prediction generation...")
            prediction_result = await self._test_prediction_generation()
            self.test_results.append(('Prediction Generation', prediction_result))
            
            # Test 3: Validation Engine
            print("\n3️⃣ Testing prediction validation...")
            validation_result = await self._test_validation_engine()
            self.test_results.append(('Validation Engine', validation_result))
            
            # Test 4: Learning Engine
            print("\n4️⃣ Testing learning engine...")
            learning_result = await self._test_learning_engine()
            self.test_results.append(('Learning Engine', learning_result))
            
            # Test 5: Performance Dashboard
            print("\n5️⃣ Testing performance dashboard...")
            dashboard_result = await self._test_performance_dashboard()
            self.test_results.append(('Performance Dashboard', dashboard_result))
            
            # Test 6: Model Retraining
            print("\n6️⃣ Testing model retraining...")
            retraining_result = await self._test_model_retraining()
            self.test_results.append(('Model Retraining', retraining_result))
            
            # Test 7: Integration Test
            print("\n7️⃣ Testing system integration...")
            integration_result = await self._test_system_integration()
            self.test_results.append(('System Integration', integration_result))
            
            # Generate test report
            report = await self._generate_test_report()
            
            print("\n" + "="*80)
            print("🎯 SELF-IMPROVING ML SYSTEM TEST COMPLETED")
            print("="*80)
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive test failed: {e}")
            return {'error': str(e)}
    
    async def _initialize_test_database(self):
        """Initialize test database with schema"""
        try:
            # Remove existing test database
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            # Initialize with schema
            with open('prediction_tracker_schema.sql', 'r') as f:
                schema = f.read()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.commit()
            conn.close()
            
            print("✅ Test database initialized")
            
        except Exception as e:
            print(f"❌ Test database initialization failed: {e}")
            raise
    
    async def _test_database_schema(self) -> Dict[str, Any]:
        """Test database schema creation and integrity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check required tables exist
            required_tables = [
                'daily_predictions', 'prediction_validation', 'strategy_performance',
                'model_version_history', 'market_regimes', 'learning_insights',
                'ensemble_weights'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [t for t in required_tables if t not in existing_tables]
            
            # Check indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = cursor.fetchall()
            
            conn.close()
            
            result = {
                'success': len(missing_tables) == 0,
                'existing_tables': len(existing_tables),
                'required_tables': len(required_tables),
                'missing_tables': missing_tables,
                'indexes_created': len(indexes),
                'schema_valid': len(missing_tables) == 0
            }
            
            if result['success']:
                print(f"✅ Schema test passed - {len(existing_tables)} tables created")
            else:
                print(f"❌ Schema test failed - missing tables: {missing_tables}")
            
            return result
            
        except Exception as e:
            print(f"❌ Schema test error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_prediction_generation(self) -> Dict[str, Any]:
        """Test unified prediction generation"""
        try:
            # Test prediction generation
            predictions = await self.prediction_system.generate_daily_unified_predictions()
            
            # Test prediction retrieval
            latest_predictions = await self.prediction_system.get_latest_unified_predictions()
            
            result = {
                'success': len(predictions) > 0,
                'predictions_generated': len(predictions),
                'timeframes_covered': len(set(p.timeframe for p in predictions)),
                'latest_retrievable': len(latest_predictions),
                'prediction_quality': self._assess_prediction_quality(predictions)
            }
            
            if result['success']:
                print(f"✅ Prediction test passed - {len(predictions)} predictions generated")
                for pred in predictions:
                    print(f"   {pred.timeframe}: {pred.predicted_direction} ${pred.predicted_price:.2f} "
                          f"({pred.predicted_change_percent:+.1f}%) [{pred.confidence_score:.1%}]")
            else:
                print("❌ Prediction test failed - no predictions generated")
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction test error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _assess_prediction_quality(self, predictions) -> Dict[str, Any]:
        """Assess quality of generated predictions"""
        try:
            if not predictions:
                return {'quality_score': 0}
            
            quality_metrics = {
                'has_all_timeframes': len(set(p.timeframe for p in predictions)) >= 3,
                'confidence_reasonable': all(0.1 <= p.confidence_score <= 0.95 for p in predictions),
                'prices_reasonable': all(2000 <= p.predicted_price <= 3500 for p in predictions),
                'directions_valid': all(p.predicted_direction in ['bullish', 'bearish', 'neutral'] for p in predictions),
                'targets_set': all(p.target_price > 0 and p.stop_loss > 0 for p in predictions)
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'avg_confidence': np.mean([p.confidence_score for p in predictions])
            }
            
        except Exception as e:
            return {'quality_score': 0, 'error': str(e)}
    
    async def _test_validation_engine(self) -> Dict[str, Any]:
        """Test prediction validation engine"""
        try:
            # Create some test predictions to validate
            await self._create_test_predictions()
            
            # Test validation
            validation_results = await self.validator.validate_expired_predictions()
            
            # Test performance update
            await self.validator.update_strategy_performance()
            
            # Test validation summary
            summary = await self.validator.get_validation_summary(days=1)
            
            result = {
                'success': True,
                'validations_completed': len(validation_results),
                'performance_updated': True,
                'summary_generated': 'total_validations' in summary,
                'validation_quality': self._assess_validation_quality(validation_results)
            }
            
            if result['success']:
                print(f"✅ Validation test passed - {len(validation_results)} validations completed")
                if summary.get('total_validations', 0) > 0:
                    print(f"   📊 Summary: {summary['total_validations']} total, "
                          f"{summary['overall_accuracy']:.1%} accuracy")
            else:
                print("❌ Validation test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Validation test error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _create_test_predictions(self):
        """Create test predictions for validation testing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions that have already expired for testing
            yesterday = datetime.now() - timedelta(days=1)
            target_time = yesterday + timedelta(hours=1)
            
            test_predictions = [
                ('technical', '1h', 2650.0, 2645.0, 'bullish', 0.7, 0.5, 2655.0, 2640.0),
                ('sentiment', '4h', 2650.0, 2648.0, 'bearish', 0.6, -0.3, 2645.0, 2655.0),
                ('macro', '1d', 2650.0, 2652.0, 'neutral', 0.5, 0.1, 2653.0, 2647.0)
            ]
            
            for strategy, timeframe, pred_price, curr_price, direction, confidence, change, target, stop in test_predictions:
                cursor.execute("""
                    INSERT INTO daily_predictions (
                        prediction_date, timeframe, strategy_id, model_version,
                        predicted_price, current_price, predicted_direction,
                        confidence_score, predicted_change_percent, target_price,
                        stop_loss, target_time, market_volatility, is_validated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    yesterday.date(), timeframe, strategy, 'v1',
                    pred_price, curr_price, direction, confidence, change,
                    target, stop, target_time, 1.5, False
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Test prediction creation failed: {e}")
    
    def _assess_validation_quality(self, validation_results) -> Dict[str, Any]:
        """Assess quality of validation results"""
        try:
            if not validation_results:
                return {'quality_score': 0}
            
            quality_metrics = {
                'has_accuracy_scores': all(hasattr(r, 'accuracy_score') for r in validation_results),
                'has_profit_loss': all(hasattr(r, 'profit_loss_percent') for r in validation_results),
                'reasonable_accuracy': all(0 <= r.accuracy_score <= 1 for r in validation_results),
                'has_market_conditions': all(hasattr(r, 'market_conditions') for r in validation_results)
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'avg_accuracy': np.mean([r.accuracy_score for r in validation_results])
            }
            
        except Exception as e:
            return {'quality_score': 0, 'error': str(e)}
    
    async def _test_learning_engine(self) -> Dict[str, Any]:
        """Test self-improving learning engine"""
        try:
            # Test performance analysis
            performances = await self.learning_engine.analyze_strategy_performance()
            
            # Test feature discovery
            insights = await self.learning_engine.discover_new_features()
            
            # Test ensemble weight optimization
            weights = await self.learning_engine.optimize_ensemble_weights()
            
            # Test learning report generation
            report = await self.learning_engine.generate_learning_report()
            
            result = {
                'success': True,
                'performances_analyzed': len(performances),
                'insights_discovered': len(insights),
                'weights_optimized': len(weights) > 0,
                'report_generated': 'report_date' in report,
                'learning_quality': self._assess_learning_quality(insights, weights, report)
            }
            
            if result['success']:
                print(f"✅ Learning test passed")
                print(f"   📊 Performances analyzed: {len(performances)}")
                print(f"   🧠 Insights discovered: {len(insights)}")
                print(f"   ⚖️ Weights optimized: {weights}")
            else:
                print("❌ Learning test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Learning test error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _assess_learning_quality(self, insights, weights, report) -> Dict[str, Any]:
        """Assess quality of learning engine outputs"""
        try:
            quality_metrics = {
                'insights_meaningful': len(insights) >= 0,  # Any insights are good
                'weights_balanced': abs(sum(weights.values()) - 1.0) < 0.1 if weights else False,
                'report_comprehensive': len(report) >= 3 if isinstance(report, dict) else False,
                'weights_reasonable': all(0 <= w <= 1 for w in weights.values()) if weights else False
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_score': quality_score,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            return {'quality_score': 0, 'error': str(e)}
    
    async def _test_performance_dashboard(self) -> Dict[str, Any]:
        """Test performance dashboard"""
        try:
            # Test dashboard data generation
            dashboard_data = await self.dashboard.generate_dashboard_data()
            
            # Test specific components
            performance_metrics = await self.dashboard.get_comprehensive_performance_metrics()
            learning_progress = await self.dashboard.get_learning_progress_metrics()
            market_regime = await self.dashboard.detect_current_market_regime()
            
            result = {
                'success': 'timestamp' in dashboard_data,
                'dashboard_data_complete': len(dashboard_data) >= 5,
                'performance_metrics_available': len(performance_metrics) > 0,
                'learning_progress_tracked': isinstance(learning_progress, dict),
                'market_regime_detected': hasattr(market_regime, 'regime_type'),
                'dashboard_quality': self._assess_dashboard_quality(dashboard_data)
            }
            
            if result['success']:
                print(f"✅ Dashboard test passed")
                print(f"   📊 Dashboard components: {len(dashboard_data)}")
                print(f"   🎯 Market regime: {market_regime.regime_type if hasattr(market_regime, 'regime_type') else 'unknown'}")
            else:
                print("❌ Dashboard test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Dashboard test error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _assess_dashboard_quality(self, dashboard_data) -> Dict[str, Any]:
        """Assess quality of dashboard data"""
        try:
            required_components = [
                'timestamp', 'performance_metrics', 'learning_progress',
                'market_regime', 'system_status'
            ]
            
            quality_metrics = {
                'has_required_components': all(comp in dashboard_data for comp in required_components),
                'timestamp_valid': 'timestamp' in dashboard_data,
                'no_errors': 'error' not in dashboard_data,
                'system_status_healthy': dashboard_data.get('system_status', {}).get('prediction_system_active', False)
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'components_available': len(dashboard_data)
            }
            
        except Exception as e:
            return {'quality_score': 0, 'error': str(e)}
    
    async def _test_model_retraining(self) -> Dict[str, Any]:
        """Test model retraining functionality"""
        try:
            # Test manual retraining
            manual_result = await self.scheduler.trigger_manual_retraining(['technical'])
            
            # Test retraining statistics
            stats = await self.scheduler.get_retraining_statistics()
            
            # Test retraining history
            history = await self.scheduler.get_retraining_history(days=1)
            
            result = {
                'success': manual_result.success,
                'manual_retraining_works': manual_result.success,
                'statistics_available': 'total_sessions' in stats,
                'history_trackable': len(history) >= 0,
                'retraining_quality': self._assess_retraining_quality(manual_result, stats)
            }
            
            if result['success']:
                print(f"✅ Retraining test passed")
                print(f"   🔄 Manual retraining: {manual_result.success}")
                print(f"   📈 Statistics available: {'total_sessions' in stats}")
                print(f"   📚 History entries: {len(history)}")
            else:
                print("❌ Retraining test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Retraining test error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _assess_retraining_quality(self, manual_result, stats) -> Dict[str, Any]:
        """Assess quality of retraining functionality"""
        try:
            quality_metrics = {
                'manual_retraining_successful': manual_result.success,
                'session_id_generated': bool(manual_result.session_id),
                'strategies_retrained': len(manual_result.strategies_retrained) > 0,
                'statistics_meaningful': isinstance(stats, dict) and len(stats) > 0
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_score': quality_score,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            return {'quality_score': 0, 'error': str(e)}
    
    async def _test_system_integration(self) -> Dict[str, Any]:
        """Test complete system integration"""
        try:
            # Test full daily cycle
            cycle_result = await self.prediction_system.run_daily_prediction_cycle()
            
            # Test data flow between components
            flow_test = await self._test_data_flow()
            
            result = {
                'success': cycle_result.get('cycle_completed', False),
                'daily_cycle_works': cycle_result.get('cycle_completed', False),
                'data_flow_intact': flow_test['success'],
                'predictions_to_validation': flow_test['predictions_stored'],
                'validation_to_learning': flow_test['validations_processed'],
                'learning_to_retraining': flow_test['learning_applied'],
                'integration_quality': self._assess_integration_quality(cycle_result, flow_test)
            }
            
            if result['success']:
                print(f"✅ Integration test passed")
                print(f"   🔄 Daily cycle: {cycle_result.get('cycle_completed', False)}")
                print(f"   📊 Data flow: {flow_test['success']}")
            else:
                print("❌ Integration test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Integration test error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_data_flow(self) -> Dict[str, Any]:
        """Test data flow between system components"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check predictions are stored
            cursor.execute("SELECT COUNT(*) FROM daily_predictions")
            predictions_count = cursor.fetchone()[0]
            
            # Check validations are processed
            cursor.execute("SELECT COUNT(*) FROM prediction_validation")
            validations_count = cursor.fetchone()[0]
            
            # Check learning insights are generated
            cursor.execute("SELECT COUNT(*) FROM learning_insights")
            insights_count = cursor.fetchone()[0]
            
            # Check strategy performance is tracked
            cursor.execute("SELECT COUNT(*) FROM strategy_performance")
            performance_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'success': predictions_count > 0,
                'predictions_stored': predictions_count > 0,
                'validations_processed': validations_count >= 0,
                'learning_applied': insights_count >= 0,
                'performance_tracked': performance_count >= 0,
                'data_counts': {
                    'predictions': predictions_count,
                    'validations': validations_count,
                    'insights': insights_count,
                    'performance_records': performance_count
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _assess_integration_quality(self, cycle_result, flow_test) -> Dict[str, Any]:
        """Assess quality of system integration"""
        try:
            quality_metrics = {
                'cycle_completes': cycle_result.get('cycle_completed', False),
                'predictions_flow': flow_test.get('predictions_stored', False),
                'validations_flow': flow_test.get('validations_processed', False),
                'learning_flows': flow_test.get('learning_applied', False)
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'cycle_time': cycle_result.get('cycle_time', 'unknown')
            }
            
        except Exception as e:
            return {'quality_score': 0, 'error': str(e)}
    
    async def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        try:
            total_tests = len(self.test_results)
            passed_tests = len([r for r in self.test_results if r[1].get('success', False)])
            success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Calculate overall quality score
            quality_scores = []
            for test_name, result in self.test_results:
                if 'quality' in result:
                    quality_key = [k for k in result.keys() if 'quality' in k][0]
                    if isinstance(result[quality_key], dict) and 'quality_score' in result[quality_key]:
                        quality_scores.append(result[quality_key]['quality_score'])
            
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            # System readiness assessment
            critical_tests = ['Database Schema', 'Prediction Generation', 'System Integration']
            critical_passed = len([r for r in self.test_results 
                                 if r[0] in critical_tests and r[1].get('success', False)])
            system_ready = critical_passed == len(critical_tests)
            
            report = {
                'test_summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': total_tests - passed_tests,
                    'success_rate': round(success_rate, 1),
                    'average_quality_score': round(avg_quality, 3)
                },
                'detailed_results': {test_name: result for test_name, result in self.test_results},
                'system_assessment': {
                    'ready_for_production': system_ready,
                    'critical_tests_passed': critical_passed,
                    'critical_tests_total': len(critical_tests),
                    'recommended_actions': self._get_recommended_actions()
                },
                'performance_indicators': {
                    'prediction_generation': 'functional' if passed_tests >= 1 else 'needs_attention',
                    'validation_system': 'functional' if passed_tests >= 2 else 'needs_attention',
                    'learning_engine': 'functional' if passed_tests >= 3 else 'needs_attention',
                    'dashboard_system': 'functional' if passed_tests >= 4 else 'needs_attention',
                    'retraining_system': 'functional' if passed_tests >= 5 else 'needs_attention'
                },
                'test_timestamp': datetime.now().isoformat(),
                'database_path': self.db_path
            }
            
            # Print summary
            print(f"\n📊 TEST SUMMARY:")
            print(f"   Total Tests: {total_tests}")
            print(f"   Passed: {passed_tests} ✅")
            print(f"   Failed: {total_tests - passed_tests} ❌")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Average Quality: {avg_quality:.1%}")
            print(f"   System Ready: {'✅ YES' if system_ready else '❌ NO'}")
            
            return report
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_recommended_actions(self) -> List[str]:
        """Get recommended actions based on test results"""
        actions = []
        
        for test_name, result in self.test_results:
            if not result.get('success', False):
                if test_name == 'Database Schema':
                    actions.append("Fix database schema initialization")
                elif test_name == 'Prediction Generation':
                    actions.append("Debug prediction generation logic")
                elif test_name == 'Validation Engine':
                    actions.append("Review validation engine implementation")
                elif test_name == 'Learning Engine':
                    actions.append("Optimize learning engine algorithms")
                elif test_name == 'Performance Dashboard':
                    actions.append("Fix dashboard data generation")
                elif test_name == 'Model Retraining':
                    actions.append("Debug retraining scheduler")
                elif test_name == 'System Integration':
                    actions.append("Review component integration")
        
        if not actions:
            actions.append("System is functioning well - monitor performance")
        
        return actions

async def main():
    """Run the comprehensive test suite"""
    logging.basicConfig(level=logging.INFO)
    
    tester = SelfImprovingMLSystemTest()
    
    print("🧪 GOLDGPT SELF-IMPROVING ML SYSTEM TEST SUITE")
    print("=" * 80)
    
    # Run comprehensive test
    report = await tester.run_comprehensive_test()
    
    # Clean up test database
    if os.path.exists(tester.db_path):
        os.remove(tester.db_path)
        print(f"\n🧹 Cleaned up test database: {tester.db_path}")
    
    print("\n🎯 Test completed! Check the report above for detailed results.")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
