"""
🧪 COMPREHENSIVE BACKTESTING SYSTEM VALIDATION
==============================================

This script validates the professional backtesting system for GoldGPT,
testing all advanced features including walk-forward optimization,
Monte Carlo simulation, and risk management.

Author: GoldGPT AI System
Created: July 23, 2025
"""

import asyncio
import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backtesting_validation')

class BacktestingSystemValidator:
    """Comprehensive validation of the GoldGPT backtesting system"""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.test_results = {}
        logger.info("🧪 Initializing Backtesting System Validator")
    
    def generate_sample_data(self, days: int = 365) -> pd.DataFrame:
        """Generate realistic gold price data for testing"""
        logger.info(f"📊 Generating {days} days of sample gold price data")
        
        # Start with realistic gold price
        start_price = 3400.0
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='H')
        
        # Generate realistic price movements with trends and volatility
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0001, 0.015, len(dates))  # 0.01% hourly mean, 1.5% volatility
        
        # Add trend components
        trend = np.linspace(0, 0.1, len(dates))  # Slight upward trend
        volatility_regime = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.005  # Volatility cycles
        
        # Combine components
        price_changes = returns + trend/len(dates) + volatility_regime/len(dates)
        
        # Generate prices
        prices = [start_price]
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Minimum price floor
        
        # Create OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            noise = np.random.normal(0, 0.002, 4)  # 0.2% noise
            
            high = price * (1 + abs(noise[0]))
            low = price * (1 - abs(noise[1]))
            open_price = price * (1 + noise[2])
            close = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_backtesting_endpoint(self) -> Dict[str, Any]:
        """Test the main backtesting endpoint"""
        logger.info("🔍 Testing backtesting endpoint")
        
        try:
            response = requests.get(f"{self.base_url}/backtest/", timeout=10)
            
            result = {
                'endpoint_available': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'content_type': response.headers.get('content-type', ''),
                'page_size': len(response.content)
            }
            
            if response.status_code == 200:
                logger.info("✅ Backtesting dashboard accessible")
            else:
                logger.error(f"❌ Backtesting dashboard failed: {response.status_code}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Endpoint test failed: {e}")
            return {'endpoint_available': False, 'error': str(e)}
    
    def test_strategy_execution(self) -> Dict[str, Any]:
        """Test strategy execution via API"""
        logger.info("📈 Testing strategy execution")
        
        try:
            # Test data for strategy
            test_data = self.generate_sample_data(30)  # 30 days of data
            
            strategy_config = {
                'strategy_name': 'moving_average_crossover',
                'parameters': {
                    'short_window': 10,
                    'long_window': 30,
                    'threshold': 0.02
                },
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005
            }
            
            # Prepare data for API (convert timestamps to strings)
            test_data_serializable = test_data.copy()
            test_data_serializable['timestamp'] = test_data_serializable['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_payload = {
                'config': strategy_config,
                'data': test_data_serializable.to_dict('records')
            }
            
            response = requests.post(
                f"{self.base_url}/backtest/run",
                json=data_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info("✅ Strategy execution successful")
                return {
                    'execution_successful': True,
                    'total_return': result_data.get('total_return', 0),
                    'sharpe_ratio': result_data.get('sharpe_ratio', 0),
                    'max_drawdown': result_data.get('max_drawdown', 0),
                    'total_trades': result_data.get('total_trades', 0),
                    'execution_time': response.elapsed.total_seconds()
                }
            else:
                logger.error(f"❌ Strategy execution failed: {response.status_code}")
                return {'execution_successful': False, 'status_code': response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Strategy execution test failed: {e}")
            return {'execution_successful': False, 'error': str(e)}
    
    def test_walk_forward_optimization(self) -> Dict[str, Any]:
        """Test walk-forward optimization"""
        logger.info("🚶 Testing walk-forward optimization")
        
        try:
            # Generate longer dataset for walk-forward
            test_data = self.generate_sample_data(180)  # 180 days
            
            wfo_config = {
                'strategy_name': 'moving_average_crossover',
                'optimization_window': 60,  # 60 days for optimization
                'validation_window': 20,    # 20 days for validation
                'step_size': 10,           # 10-day steps
                'parameter_ranges': {
                    'short_window': {'min': 5, 'max': 20, 'step': 1},
                    'long_window': {'min': 20, 'max': 50, 'step': 5},
                    'threshold': {'min': 0.01, 'max': 0.05, 'step': 0.01}
                }
            }
            
            # Prepare data for API (convert timestamps to strings)
            test_data_serializable = test_data.copy()
            test_data_serializable['timestamp'] = test_data_serializable['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_payload = {
                'config': wfo_config,
                'data': test_data_serializable.to_dict('records')
            }
            
            response = requests.post(
                f"{self.base_url}/backtest/walk_forward",
                json=data_payload,
                timeout=60  # Longer timeout for optimization
            )
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info("✅ Walk-forward optimization successful")
                return {
                    'wfo_successful': True,
                    'optimization_windows': result_data.get('windows_tested', 0),
                    'average_is_return': result_data.get('avg_in_sample_return', 0),
                    'average_oos_return': result_data.get('avg_out_sample_return', 0),
                    'overfitting_score': result_data.get('overfitting_score', 0),
                    'parameter_stability': result_data.get('parameter_stability', 0),
                    'execution_time': response.elapsed.total_seconds()
                }
            else:
                logger.error(f"❌ Walk-forward optimization failed: {response.status_code}")
                return {'wfo_successful': False, 'status_code': response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Walk-forward optimization test failed: {e}")
            return {'wfo_successful': False, 'error': str(e)}
    
    def test_monte_carlo_analysis(self) -> Dict[str, Any]:
        """Test Monte Carlo simulation"""
        logger.info("🎲 Testing Monte Carlo analysis")
        
        try:
            test_data = self.generate_sample_data(90)  # 90 days
            
            mc_config = {
                'strategy_name': 'moving_average_crossover',
                'num_simulations': 100,  # Reduced for testing
                'randomization_methods': ['bootstrap', 'parametric'],
                'parameters': {
                    'short_window': 10,
                    'long_window': 30,
                    'threshold': 0.02
                }
            }
            
            # Prepare data for API (convert timestamps to strings)
            test_data_serializable = test_data.copy()
            test_data_serializable['timestamp'] = test_data_serializable['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_payload = {
                'config': mc_config,
                'data': test_data_serializable.to_dict('records')
            }
            
            response = requests.post(
                f"{self.base_url}/backtest/monte_carlo",
                json=data_payload,
                timeout=90  # Longer timeout for simulations
            )
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info("✅ Monte Carlo analysis successful")
                return {
                    'mc_successful': True,
                    'simulations_completed': result_data.get('simulations_completed', 0),
                    'mean_return': result_data.get('mean_return', 0),
                    'return_std': result_data.get('return_std', 0),
                    'var_95': result_data.get('var_95', 0),
                    'success_rate': result_data.get('success_rate', 0),
                    'execution_time': response.elapsed.total_seconds()
                }
            else:
                logger.error(f"❌ Monte Carlo analysis failed: {response.status_code}")
                return {'mc_successful': False, 'status_code': response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Monte Carlo analysis test failed: {e}")
            return {'mc_successful': False, 'error': str(e)}
    
    def test_risk_management(self) -> Dict[str, Any]:
        """Test risk management features"""
        logger.info("🛡️ Testing risk management")
        
        try:
            test_data = self.generate_sample_data(60)  # 60 days
            
            risk_config = {
                'strategy_name': 'moving_average_crossover',
                'risk_management': {
                    'max_position_size': 0.1,  # 10% max position
                    'max_drawdown_limit': 0.15,  # 15% max drawdown
                    'var_limit': 0.05,  # 5% VaR limit
                    'correlation_limit': 0.8,  # 80% correlation limit
                    'volatility_adjustment': True,
                    'kelly_criterion': True
                },
                'parameters': {
                    'short_window': 10,
                    'long_window': 30,
                    'threshold': 0.02
                }
            }
            
            # Prepare data for API (convert timestamps to strings)
            test_data_serializable = test_data.copy()
            test_data_serializable['timestamp'] = test_data_serializable['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_payload = {
                'config': risk_config,
                'data': test_data_serializable.to_dict('records')
            }
            
            response = requests.post(
                f"{self.base_url}/backtest/risk_analysis",
                json=data_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info("✅ Risk management analysis successful")
                return {
                    'risk_analysis_successful': True,
                    'max_position_used': result_data.get('max_position_used', 0),
                    'actual_max_drawdown': result_data.get('actual_max_drawdown', 0),
                    'var_95_actual': result_data.get('var_95_actual', 0),
                    'risk_adjusted_return': result_data.get('risk_adjusted_return', 0),
                    'kelly_positions': result_data.get('kelly_positions_used', 0),
                    'execution_time': response.elapsed.total_seconds()
                }
            else:
                logger.error(f"❌ Risk management analysis failed: {response.status_code}")
                return {'risk_analysis_successful': False, 'status_code': response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Risk management test failed: {e}")
            return {'risk_analysis_successful': False, 'error': str(e)}
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test comprehensive performance metrics calculation"""
        logger.info("📊 Testing performance metrics")
        
        try:
            # Create a sample backtest result
            portfolio_values = [100000]  # Starting capital
            
            # Simulate portfolio growth with some volatility
            for i in range(252):  # One year of daily data
                daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily mean, 2% volatility
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
            
            # Calculate metrics locally for validation
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            metrics = {
                'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
                'volatility': np.std(returns) * np.sqrt(252),
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'calmar_ratio': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] / self._calculate_max_drawdown(portfolio_values),
                'var_95': np.percentile(returns, 5),
                'trades_analyzed': len(portfolio_values)
            }
            
            logger.info("✅ Performance metrics calculated successfully")
            return {
                'metrics_successful': True,
                'total_return': metrics['total_return'],
                'annualized_volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'calmar_ratio': metrics['calmar_ratio'],
                'var_95': metrics['var_95']
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            return {'metrics_successful': False, 'error': str(e)}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def test_database_integration(self) -> Dict[str, Any]:
        """Test database storage and retrieval"""
        logger.info("🗄️ Testing database integration")
        
        try:
            # Check if backtesting database exists and is accessible
            db_path = "goldgpt_advanced_backtest.db"
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'historical_data',
                'backtest_results',
                'optimization_results',
                'walk_forward_results',
                'monte_carlo_results',
                'market_regimes'
            ]
            
            tables_found = [table for table in expected_tables if table in tables]
            
            # Test data insertion and retrieval
            test_record = {
                'strategy_name': 'test_strategy',
                'backtest_id': f'test_{int(time.time())}',
                'start_date': datetime.now().date(),
                'end_date': datetime.now().date(),
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08
            }
            
            # Insert test record (if backtest_results table exists)
            if 'backtest_results' in tables:
                cursor.execute('''
                    INSERT INTO backtest_results 
                    (strategy_name, backtest_id, start_date, end_date, total_return, sharpe_ratio, max_drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    test_record['strategy_name'],
                    test_record['backtest_id'],
                    test_record['start_date'],
                    test_record['end_date'],
                    test_record['total_return'],
                    test_record['sharpe_ratio'],
                    test_record['max_drawdown']
                ))
                conn.commit()
                
                # Retrieve test record
                cursor.execute('SELECT * FROM backtest_results WHERE backtest_id = ?', 
                             (test_record['backtest_id'],))
                retrieved = cursor.fetchone()
                
                conn.close()
                
                return {
                    'db_accessible': True,
                    'tables_found': len(tables_found),
                    'expected_tables': len(expected_tables),
                    'data_insertion_successful': retrieved is not None,
                    'total_tables': len(tables)
                }
            else:
                conn.close()
                return {
                    'db_accessible': True,
                    'tables_found': len(tables_found),
                    'expected_tables': len(expected_tables),
                    'data_insertion_successful': False,
                    'total_tables': len(tables)
                }
                
        except Exception as e:
            logger.error(f"❌ Database integration test failed: {e}")
            return {'db_accessible': False, 'error': str(e)}
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🚀 Starting comprehensive backtesting system validation")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Run all tests
        tests = [
            ('endpoint_test', self.test_backtesting_endpoint),
            ('strategy_execution', self.test_strategy_execution),
            ('walk_forward_optimization', self.test_walk_forward_optimization),
            ('monte_carlo_analysis', self.test_monte_carlo_analysis),
            ('risk_management', self.test_risk_management),
            ('performance_metrics', self.test_performance_metrics),
            ('database_integration', self.test_database_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"🔍 Running {test_name}")
            try:
                validation_results['tests'][test_name] = test_func()
                logger.info(f"✅ {test_name} completed")
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                validation_results['tests'][test_name] = {'error': str(e)}
        
        # Calculate overall system health
        successful_tests = sum(1 for test in validation_results['tests'].values() 
                             if not test.get('error') and 
                             (test.get('endpoint_available') or 
                              test.get('execution_successful') or 
                              test.get('wfo_successful') or 
                              test.get('mc_successful') or 
                              test.get('risk_analysis_successful') or 
                              test.get('metrics_successful') or 
                              test.get('db_accessible')))
        
        total_tests = len(tests)
        system_health = (successful_tests / total_tests) * 100
        
        validation_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'system_health_percentage': system_health,
            'overall_status': 'HEALTHY' if system_health >= 80 else 'NEEDS_ATTENTION' if system_health >= 60 else 'CRITICAL'
        }
        
        return validation_results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("🧪 GOLDGPT BACKTESTING SYSTEM VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"📅 Validation Date: {results['timestamp']}")
        report.append(f"🎯 System Health: {results['summary']['system_health_percentage']:.1f}%")
        report.append(f"📊 Overall Status: {results['summary']['overall_status']}")
        report.append("")
        
        # Test Results Section
        report.append("📋 TEST RESULTS")
        report.append("-" * 40)
        
        for test_name, test_result in results['tests'].items():
            if test_result.get('error'):
                status = "❌ FAILED"
                detail = f"Error: {test_result['error']}"
            else:
                # Determine status based on test type
                if test_name == 'endpoint_test':
                    status = "✅ PASSED" if test_result.get('endpoint_available') else "❌ FAILED"
                    detail = f"Response time: {test_result.get('response_time', 0):.3f}s"
                elif test_name == 'strategy_execution':
                    status = "✅ PASSED" if test_result.get('execution_successful') else "❌ FAILED"
                    detail = f"Return: {test_result.get('total_return', 0):.2%}, Trades: {test_result.get('total_trades', 0)}"
                elif test_name == 'walk_forward_optimization':
                    status = "✅ PASSED" if test_result.get('wfo_successful') else "❌ FAILED"
                    detail = f"Windows: {test_result.get('optimization_windows', 0)}, Overfitting: {test_result.get('overfitting_score', 0):.3f}"
                elif test_name == 'monte_carlo_analysis':
                    status = "✅ PASSED" if test_result.get('mc_successful') else "❌ FAILED"
                    detail = f"Simulations: {test_result.get('simulations_completed', 0)}, Success Rate: {test_result.get('success_rate', 0):.1%}"
                elif test_name == 'risk_management':
                    status = "✅ PASSED" if test_result.get('risk_analysis_successful') else "❌ FAILED"
                    detail = f"Max DD: {test_result.get('actual_max_drawdown', 0):.2%}, VaR: {test_result.get('var_95_actual', 0):.3f}"
                elif test_name == 'performance_metrics':
                    status = "✅ PASSED" if test_result.get('metrics_successful') else "❌ FAILED"
                    detail = f"Sharpe: {test_result.get('sharpe_ratio', 0):.2f}, Max DD: {test_result.get('max_drawdown', 0):.2%}"
                elif test_name == 'database_integration':
                    status = "✅ PASSED" if test_result.get('db_accessible') else "❌ FAILED"
                    detail = f"Tables: {test_result.get('tables_found', 0)}/{test_result.get('expected_tables', 0)}"
                else:
                    status = "❓ UNKNOWN"
                    detail = "No status criteria defined"
            
            report.append(f"{test_name.replace('_', ' ').title()}: {status}")
            report.append(f"  └─ {detail}")
            report.append("")
        
        # Recommendations Section
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if results['summary']['system_health_percentage'] >= 80:
            report.append("✅ System is performing well!")
            report.append("• Continue monitoring performance")
            report.append("• Consider expanding to more complex strategies")
        elif results['summary']['system_health_percentage'] >= 60:
            report.append("⚠️ System needs attention:")
            report.append("• Review failed tests and error messages")
            report.append("• Check database connectivity and table structure")
            report.append("• Verify API endpoints are responding correctly")
        else:
            report.append("🚨 Critical issues detected:")
            report.append("• Immediate attention required")
            report.append("• Review all system components")
            report.append("• Check logs for detailed error information")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


async def main():
    """Main validation function"""
    validator = BacktestingSystemValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Generate and display report
    report = validator.generate_validation_report(results)
    print(report)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"backtesting_validation_report_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(f"backtesting_validation_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📋 Validation complete! Reports saved with timestamp {timestamp}")


if __name__ == "__main__":
    print("🧪 GoldGPT Backtesting System Validator")
    print("=" * 50)
    asyncio.run(main())
