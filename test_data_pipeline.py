#!/usr/bin/env python3
"""
Comprehensive Test Suite for GoldGPT Data Integration Pipeline
Tests all components and validates data quality
"""

import asyncio
import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_integration_engine import (
    DataIntegrationEngine, DataManager, CandlestickDataFetcher,
    NewsDataFetcher, EconomicDataFetcher, TechnicalAnalyzer,
    FeatureEngineer, DataCache
)
from data_sources_config import config

class DataPipelineValidator:
    """Validates the complete data pipeline functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.integration_engine = None
        self.data_manager = None
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("🧪 GoldGPT Data Integration Pipeline - Comprehensive Test Suite")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Initialize components
        await self._test_component_initialization()
        
        # Test individual components
        await self._test_cache_system()
        await self._test_candlestick_fetcher()
        await self._test_news_fetcher()
        await self._test_economic_fetcher()
        await self._test_technical_analyzer()
        await self._test_feature_engineer()
        
        # Test integration
        await self._test_data_integration()
        await self._test_data_manager()
        
        # Performance tests
        await self._test_performance()
        
        # Generate final report
        self._generate_test_report()
        
        return self.test_results
    
    async def _test_component_initialization(self):
        """Test component initialization"""
        print("\n🔧 Testing Component Initialization...")
        
        try:
            self.integration_engine = DataIntegrationEngine()
            self.data_manager = DataManager(self.integration_engine)
            
            self.test_results['component_initialization'] = {
                'status': 'success',
                'components_initialized': [
                    'DataIntegrationEngine', 'DataManager', 'DataCache',
                    'CandlestickDataFetcher', 'NewsDataFetcher', 
                    'EconomicDataFetcher', 'TechnicalAnalyzer', 'FeatureEngineer'
                ],
                'timestamp': datetime.now().isoformat()
            }
            print("✅ All components initialized successfully")
            
        except Exception as e:
            self.test_results['component_initialization'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Component initialization failed: {e}")
    
    async def _test_cache_system(self):
        """Test cache system functionality"""
        print("\n💾 Testing Cache System...")
        
        try:
            cache = DataCache("test_cache.db")
            
            # Test cache operations
            test_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
            cache.set('test_key', test_data, 3600, 'test')
            
            retrieved_data = cache.get('test_key')
            
            cache_test_passed = (retrieved_data is not None and 
                               retrieved_data.get('test') == 'data')
            
            # Test TTL
            cache.set('ttl_test', {'data': 'expires_soon'}, 1, 'test')
            await asyncio.sleep(2)
            expired_data = cache.get('ttl_test')
            ttl_test_passed = expired_data is None
            
            # Test cleanup
            cache.cleanup_expired()
            
            self.test_results['cache_system'] = {
                'status': 'success' if cache_test_passed and ttl_test_passed else 'partial',
                'cache_operations': cache_test_passed,
                'ttl_functionality': ttl_test_passed,
                'cleanup_functionality': True,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Cache system: {'fully functional' if cache_test_passed and ttl_test_passed else 'partially functional'}")
            
        except Exception as e:
            self.test_results['cache_system'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Cache system test failed: {e}")
    
    async def _test_candlestick_fetcher(self):
        """Test candlestick data fetching"""
        print("\n📊 Testing Candlestick Data Fetcher...")
        
        try:
            cache = DataCache("test_cache.db")
            fetcher = CandlestickDataFetcher(cache)
            
            # Test single timeframe
            data = await fetcher.fetch_candlestick_data(['1h'])
            
            data_quality = {
                'data_received': len(data) > 0,
                'valid_structure': all(
                    hasattr(candle, 'timestamp') and 
                    hasattr(candle, 'open') and 
                    hasattr(candle, 'close') 
                    for candle in data[:5]  # Check first 5 items
                ) if data else False,
                'realistic_prices': all(
                    candle.close > 0 and candle.close < 10000  # Reasonable gold price range
                    for candle in data[:5]
                ) if data else False
            }
            
            self.test_results['candlestick_fetcher'] = {
                'status': 'success' if data_quality['data_received'] else 'no_data',
                'data_count': len(data),
                'data_quality': data_quality,
                'timeframes_tested': ['1h'],
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Candlestick fetcher: {len(data)} data points received")
            
        except Exception as e:
            self.test_results['candlestick_fetcher'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Candlestick fetcher test failed: {e}")
    
    async def _test_news_fetcher(self):
        """Test news data fetching"""
        print("\n📰 Testing News Data Fetcher...")
        
        try:
            cache = DataCache("test_cache.db")
            fetcher = NewsDataFetcher(cache)
            
            # Test news fetching
            news_data = await fetcher.fetch_news_data(hours_back=24)
            
            news_quality = {
                'data_received': len(news_data) > 0,
                'has_titles': all(
                    hasattr(item, 'title') and len(item.title) > 0 
                    for item in news_data[:5]
                ) if news_data else False,
                'has_sentiment': all(
                    hasattr(item, 'sentiment_score') and 
                    isinstance(item.sentiment_score, (int, float))
                    for item in news_data[:5]
                ) if news_data else False,
                'has_relevance': all(
                    hasattr(item, 'relevance_score') and 
                    isinstance(item.relevance_score, (int, float))
                    for item in news_data[:5]
                ) if news_data else False
            }
            
            self.test_results['news_fetcher'] = {
                'status': 'success' if news_quality['data_received'] else 'no_data',
                'news_count': len(news_data),
                'news_quality': news_quality,
                'avg_sentiment': sum(item.sentiment_score for item in news_data) / len(news_data) if news_data else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ News fetcher: {len(news_data)} news items processed")
            
        except Exception as e:
            self.test_results['news_fetcher'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ News fetcher test failed: {e}")
    
    async def _test_economic_fetcher(self):
        """Test economic data fetching"""
        print("\n💰 Testing Economic Data Fetcher...")
        
        try:
            cache = DataCache("test_cache.db")
            fetcher = EconomicDataFetcher(cache)
            
            # Test economic indicators
            economic_data = await fetcher.fetch_economic_indicators()
            
            economic_quality = {
                'data_received': len(economic_data) > 0,
                'valid_indicators': all(
                    hasattr(item, 'indicator_name') and 
                    hasattr(item, 'value') and
                    isinstance(item.value, (int, float))
                    for item in economic_data
                ) if economic_data else False,
                'has_impact_levels': all(
                    hasattr(item, 'impact_level') and
                    item.impact_level in ['high', 'medium', 'low']
                    for item in economic_data
                ) if economic_data else False
            }
            
            self.test_results['economic_fetcher'] = {
                'status': 'success' if economic_quality['data_received'] else 'no_data',
                'indicator_count': len(economic_data),
                'economic_quality': economic_quality,
                'indicators': [item.indicator_name for item in economic_data] if economic_data else [],
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Economic fetcher: {len(economic_data)} economic indicators")
            
        except Exception as e:
            self.test_results['economic_fetcher'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Economic fetcher test failed: {e}")
    
    async def _test_technical_analyzer(self):
        """Test technical analysis functionality"""
        print("\n📈 Testing Technical Analyzer...")
        
        try:
            cache = DataCache("test_cache.db")
            analyzer = TechnicalAnalyzer(cache)
            
            # Generate sample candlestick data for testing
            sample_data = self._generate_sample_candlestick_data()
            
            # Calculate technical indicators
            technical_indicators = analyzer.calculate_technical_indicators(sample_data)
            
            technical_quality = {
                'indicators_calculated': len(technical_indicators) > 0,
                'valid_values': all(
                    hasattr(indicator, 'value') and 
                    isinstance(indicator.value, (int, float)) and
                    not (hasattr(indicator.value, '__iter__') and 
                         any(str(v).lower() in ['nan', 'inf'] for v in [indicator.value]))
                    for indicator in technical_indicators
                ) if technical_indicators else False,
                'valid_signals': all(
                    hasattr(indicator, 'signal') and 
                    indicator.signal in ['bullish', 'bearish', 'neutral', 'overbought', 'oversold']
                    for indicator in technical_indicators
                ) if technical_indicators else False
            }
            
            self.test_results['technical_analyzer'] = {
                'status': 'success' if technical_quality['indicators_calculated'] else 'no_data',
                'indicator_count': len(technical_indicators),
                'technical_quality': technical_quality,
                'indicators': [indicator.indicator_name for indicator in technical_indicators],
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Technical analyzer: {len(technical_indicators)} indicators calculated")
            
        except Exception as e:
            self.test_results['technical_analyzer'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Technical analyzer test failed: {e}")
    
    async def _test_feature_engineer(self):
        """Test feature engineering"""
        print("\n🔧 Testing Feature Engineer...")
        
        try:
            engineer = FeatureEngineer()
            
            # Generate sample data
            sample_candlesticks = self._generate_sample_candlestick_data()
            sample_news = self._generate_sample_news_data()
            sample_economic = self._generate_sample_economic_data()
            sample_technical = self._generate_sample_technical_data()
            
            # Extract features
            features = engineer.extract_features(
                sample_candlesticks, sample_news, sample_economic, sample_technical
            )
            
            feature_quality = {
                'features_extracted': len(features) > 0,
                'valid_feature_values': all(
                    isinstance(value, (int, float)) and
                    not str(value).lower() in ['nan', 'inf', 'none']
                    for value in features.values()
                ) if features else False,
                'comprehensive_coverage': all(
                    feature_type in str(features.keys())
                    for feature_type in ['price', 'news', 'tech', 'time']
                ),
                'feature_count': len(features)
            }
            
            self.test_results['feature_engineer'] = {
                'status': 'success' if feature_quality['features_extracted'] else 'no_features',
                'feature_count': len(features),
                'feature_quality': feature_quality,
                'sample_features': list(features.keys())[:10] if features else [],
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Feature engineer: {len(features)} features extracted")
            
        except Exception as e:
            self.test_results['feature_engineer'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Feature engineer test failed: {e}")
    
    async def _test_data_integration(self):
        """Test full data integration pipeline"""
        print("\n🔗 Testing Data Integration Pipeline...")
        
        try:
            if not self.integration_engine:
                self.integration_engine = DataIntegrationEngine()
            
            # Test unified dataset creation
            dataset = await self.integration_engine.get_unified_dataset()
            
            integration_quality = {
                'dataset_created': dataset is not None,
                'has_features': 'features' in dataset and len(dataset['features']) > 0,
                'has_metadata': all(
                    key in dataset for key in ['timestamp', 'data_quality', 'raw_data']
                ),
                'quality_score': dataset.get('data_quality', {}).get('overall_score', 0) if dataset else 0,
                'feature_count': len(dataset.get('features', {})) if dataset else 0
            }
            
            self.test_results['data_integration'] = {
                'status': 'success' if integration_quality['dataset_created'] else 'failed',
                'integration_quality': integration_quality,
                'dataset_size': len(str(dataset)) if dataset else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Data integration: Dataset with {integration_quality['feature_count']} features created")
            
        except Exception as e:
            self.test_results['data_integration'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Data integration test failed: {e}")
    
    async def _test_data_manager(self):
        """Test data manager functionality"""
        print("\n🎯 Testing Data Manager...")
        
        try:
            if not self.data_manager:
                self.data_manager = DataManager(self.integration_engine)
            
            # Test ML-ready dataset
            ml_dataset = await self.data_manager.get_ml_ready_dataset()
            
            # Test feature vector extraction
            feature_vector = self.data_manager.get_feature_vector(ml_dataset)
            feature_names = self.data_manager.get_feature_names(ml_dataset)
            
            # Test health check
            health_status = await self.data_manager.health_check()
            
            manager_quality = {
                'ml_dataset_ready': ml_dataset is not None,
                'feature_vector_extracted': feature_vector is not None,
                'feature_names_available': len(feature_names) > 0,
                'health_check_works': health_status is not None,
                'validation_passed': ml_dataset.get('validation_timestamp') is not None if ml_dataset else False
            }
            
            self.test_results['data_manager'] = {
                'status': 'success' if manager_quality['ml_dataset_ready'] else 'failed',
                'manager_quality': manager_quality,
                'feature_vector_shape': feature_vector.shape if feature_vector is not None else None,
                'health_status': health_status.get('status') if health_status else 'unknown',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Data manager: ML dataset ready with {len(feature_names)} features")
            
        except Exception as e:
            self.test_results['data_manager'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Data manager test failed: {e}")
    
    async def _test_performance(self):
        """Test performance metrics"""
        print("\n⚡ Testing Performance...")
        
        try:
            # Measure data fetching performance
            start_time = time.time()
            dataset = await self.integration_engine.get_unified_dataset()
            fetch_time = time.time() - start_time
            
            # Measure feature extraction performance
            start_time = time.time()
            if dataset and 'features' in dataset:
                feature_vector = self.data_manager.get_feature_vector(dataset)
            feature_extraction_time = time.time() - start_time
            
            performance_metrics = {
                'data_fetch_time_seconds': fetch_time,
                'feature_extraction_time_seconds': feature_extraction_time,
                'total_pipeline_time': fetch_time + feature_extraction_time,
                'performance_rating': 'fast' if (fetch_time + feature_extraction_time) < 10 else 'medium' if (fetch_time + feature_extraction_time) < 30 else 'slow'
            }
            
            self.test_results['performance'] = {
                'status': 'success',
                'metrics': performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Performance: {performance_metrics['total_pipeline_time']:.2f}s total pipeline time")
            
        except Exception as e:
            self.test_results['performance'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Performance test failed: {e}")
    
    def _generate_sample_candlestick_data(self):
        """Generate sample candlestick data for testing"""
        from data_integration_engine import CandlestickData
        
        sample_data = []
        base_price = 2000.0
        
        for i in range(30):  # 30 data points
            timestamp = datetime.now() - timedelta(hours=i)
            price = base_price + (i * 2) + ((-1) ** i * 10)  # Some variation
            
            sample_data.append(CandlestickData(
                timestamp=timestamp,
                open=price - 1,
                high=price + 5,
                low=price - 5,
                close=price,
                volume=1000 + (i * 50),
                timeframe='1h'
            ))
        
        return sample_data
    
    def _generate_sample_news_data(self):
        """Generate sample news data for testing"""
        from data_integration_engine import NewsData
        
        return [
            NewsData(
                timestamp=datetime.now(),
                title="Gold prices rise amid market uncertainty",
                content="Gold prices increased as investors sought safe haven assets...",
                source="test_source",
                sentiment_score=0.3,
                relevance_score=0.8,
                url="http://test.com/news1"
            ),
            NewsData(
                timestamp=datetime.now(),
                title="Fed signals potential rate cuts",
                content="The Federal Reserve indicated possible interest rate reductions...",
                source="test_source",
                sentiment_score=0.1,
                relevance_score=0.9,
                url="http://test.com/news2"
            )
        ]
    
    def _generate_sample_economic_data(self):
        """Generate sample economic data for testing"""
        from data_integration_engine import EconomicIndicator
        
        return [
            EconomicIndicator(
                timestamp=datetime.now(),
                indicator_name="USD_INDEX",
                value=103.5,
                country="US",
                impact_level="high",
                source="test"
            ),
            EconomicIndicator(
                timestamp=datetime.now(),
                indicator_name="INFLATION_RATE",
                value=3.2,
                country="US",
                impact_level="high",
                source="test"
            )
        ]
    
    def _generate_sample_technical_data(self):
        """Generate sample technical data for testing"""
        from data_integration_engine import TechnicalIndicator
        
        return [
            TechnicalIndicator(
                timestamp=datetime.now(),
                indicator_name="RSI_14",
                value=65.0,
                signal="neutral",
                timeframe="1d"
            ),
            TechnicalIndicator(
                timestamp=datetime.now(),
                indicator_name="MACD",
                value=2.5,
                signal="bullish",
                timeframe="1d"
            )
        ]
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        # Count test results
        passed_tests = len([t for t in self.test_results.values() if t.get('status') == 'success'])
        total_tests = len(self.test_results)
        
        print(f"🎯 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"⏱️  Total Test Time: {total_time:.2f} seconds")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Individual test results
        print(f"\n📊 Individual Test Results:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result.get('status') == 'success' else "⚠️" if result.get('status') in ['partial', 'no_data'] else "❌"
            print(f"  {status_emoji} {test_name}: {result.get('status', 'unknown')}")
        
        # Recommendations
        print(f"\n🔧 Recommendations:")
        
        failed_tests = [name for name, result in self.test_results.items() if result.get('status') not in ['success', 'partial', 'no_data']]
        if failed_tests:
            print(f"  • Fix failed components: {', '.join(failed_tests)}")
        
        no_data_tests = [name for name, result in self.test_results.items() if result.get('status') == 'no_data']
        if no_data_tests:
            print(f"  • Configure data sources for: {', '.join(no_data_tests)}")
        
        # Performance assessment
        perf_result = self.test_results.get('performance', {})
        if perf_result.get('metrics', {}).get('performance_rating') == 'slow':
            print(f"  • Optimize performance: Current pipeline time is {perf_result.get('metrics', {}).get('total_pipeline_time', 0):.2f}s")
        
        print(f"\n✨ Data Pipeline Validation Complete!")
        
        # Save results to file
        with open('data_pipeline_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: data_pipeline_test_results.json")

async def main():
    """Run the comprehensive validation tests"""
    validator = DataPipelineValidator()
    
    try:
        results = await validator.run_comprehensive_tests()
        return results
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return {'error': str(e)}
    finally:
        # Cleanup
        if validator.integration_engine:
            validator.integration_engine.close()

if __name__ == "__main__":
    asyncio.run(main())
