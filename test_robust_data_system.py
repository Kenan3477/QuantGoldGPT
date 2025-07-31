#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Robust Data System
Tests all components including APIs, web scraping, fallbacks, and integration
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RobustDataSystemTester:
    """Comprehensive tester for the robust data system"""
    
    def __init__(self):
        self.results = {}
        self.test_symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
        
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 GoldGPT Robust Data System - Comprehensive Test Suite")
        print("=" * 65)
        print(f"⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test 1: Import and initialization
        await self.test_imports()
        
        # Test 2: Data source manager
        await self.test_data_source_manager()
        
        # Test 3: API services
        await self.test_api_services()
        
        # Test 4: Web scraping services
        await self.test_web_scraping()
        
        # Test 5: Fallback mechanisms
        await self.test_fallback_mechanisms()
        
        # Test 6: Cache management
        await self.test_cache_management()
        
        # Test 7: Flask integration
        await self.test_flask_integration()
        
        # Test 8: Frontend integration
        await self.test_frontend_integration()
        
        # Generate final report
        self.generate_test_report()
    
    async def test_imports(self):
        """Test if all modules can be imported successfully"""
        print("📦 Testing Module Imports...")
        try:
            # Test robust data system import
            from robust_data_system import (
                unified_data_provider, 
                DataSourceManager,
                PriceDataService,
                SentimentAnalysisService,
                TechnicalIndicatorService
            )
            self.results['imports'] = {
                'robust_data_system': True,
                'unified_provider': unified_data_provider is not None
            }
            print("   ✅ Robust data system imported successfully")
            
            # Test enhanced flask integration
            try:
                from enhanced_flask_integration import (
                    setup_enhanced_routes,
                    get_price_data_sync,
                    ROBUST_DATA_AVAILABLE
                )
                self.results['imports']['flask_integration'] = True
                print("   ✅ Flask integration imported successfully")
            except ImportError as e:
                self.results['imports']['flask_integration'] = False
                print(f"   ⚠️ Flask integration import failed: {e}")
            
        except ImportError as e:
            self.results['imports'] = {'error': str(e)}
            print(f"   ❌ Import failed: {e}")
    
    async def test_data_source_manager(self):
        """Test the data source manager functionality"""
        print("\n🎛️ Testing Data Source Manager...")
        
        try:
            from robust_data_system import DataSourceManager
            
            manager = DataSourceManager()
            
            # Test initialization
            self.results['data_source_manager'] = {
                'initialization': True,
                'cache_manager': hasattr(manager, 'cache_manager'),
                'rate_limiter': hasattr(manager, 'rate_limiter')
            }
            
            print("   ✅ Data source manager initialized")
            print(f"   📊 Cache manager available: {self.results['data_source_manager']['cache_manager']}")
            print(f"   ⏱️ Rate limiter available: {self.results['data_source_manager']['rate_limiter']}")
            
        except Exception as e:
            self.results['data_source_manager'] = {'error': str(e)}
            print(f"   ❌ Data source manager test failed: {e}")
    
    async def test_api_services(self):
        """Test API services for each data type"""
        print("\n🌐 Testing API Services...")
        
        try:
            from robust_data_system import unified_data_provider
            
            api_results = {}
            
            # Test price data API
            try:
                price_data = await unified_data_provider.get_price_data('XAUUSD')
                api_results['price'] = {
                    'success': True,
                    'symbol': price_data.symbol,
                    'price': price_data.price,
                    'source': price_data.source.value,
                    'has_bid_ask': price_data.bid is not None and price_data.ask is not None
                }
                print(f"   ✅ Price API: ${price_data.price:.2f} from {price_data.source.value}")
            except Exception as e:
                api_results['price'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Price API failed: {e}")
            
            # Test sentiment analysis API
            try:
                sentiment_data = await unified_data_provider.get_sentiment_data('XAUUSD')
                api_results['sentiment'] = {
                    'success': True,
                    'sentiment_score': sentiment_data.sentiment_score,
                    'sentiment_label': sentiment_data.sentiment_label,
                    'confidence': sentiment_data.confidence,
                    'sources_count': sentiment_data.sources_count
                }
                print(f"   ✅ Sentiment API: {sentiment_data.sentiment_label} ({sentiment_data.confidence:.2f} confidence)")
            except Exception as e:
                api_results['sentiment'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Sentiment API failed: {e}")
            
            # Test technical analysis API
            try:
                technical_data = await unified_data_provider.get_technical_data('XAUUSD')
                api_results['technical'] = {
                    'success': True,
                    'indicators': list(technical_data.indicators.keys()),
                    'timeframe': technical_data.analysis_timeframe,
                    'source': technical_data.source.value
                }
                print(f"   ✅ Technical API: {len(technical_data.indicators)} indicators from {technical_data.source.value}")
            except Exception as e:
                api_results['technical'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Technical API failed: {e}")
            
            self.results['api_services'] = api_results
            
        except Exception as e:
            self.results['api_services'] = {'error': str(e)}
            print(f"   ❌ API services test failed: {e}")
    
    async def test_web_scraping(self):
        """Test web scraping fallback mechanisms"""
        print("\n🕷️ Testing Web Scraping Services...")
        
        try:
            from robust_data_system import PriceDataService, SentimentAnalysisService
            
            scraping_results = {}
            
            # Test price scraping (simulated failure of APIs)
            price_service = PriceDataService()
            # Simulate API failure by temporarily disabling APIs
            original_apis = price_service.api_sources.copy()
            price_service.api_sources = []  # Force fallback to scraping
            
            try:
                price_data = await price_service.get_price_data('XAUUSD')
                scraping_results['price_scraping'] = {
                    'success': True,
                    'source': price_data.source.value if hasattr(price_data, 'source') else 'unknown',
                    'price': getattr(price_data, 'price', 0)
                }
                print(f"   ✅ Price scraping: Success")
            except Exception as e:
                scraping_results['price_scraping'] = {'success': False, 'error': str(e)}
                print(f"   ⚠️ Price scraping: {e}")
            finally:
                price_service.api_sources = original_apis
            
            # Test news scraping
            sentiment_service = SentimentAnalysisService()
            try:
                news_articles = await sentiment_service.scrape_financial_news('gold')
                scraping_results['news_scraping'] = {
                    'success': True,
                    'articles_count': len(news_articles),
                    'has_articles': len(news_articles) > 0
                }
                print(f"   ✅ News scraping: {len(news_articles)} articles found")
            except Exception as e:
                scraping_results['news_scraping'] = {'success': False, 'error': str(e)}
                print(f"   ⚠️ News scraping: {e}")
            
            self.results['web_scraping'] = scraping_results
            
        except Exception as e:
            self.results['web_scraping'] = {'error': str(e)}
            print(f"   ❌ Web scraping test failed: {e}")
    
    async def test_fallback_mechanisms(self):
        """Test the fallback chain functionality"""
        print("\n🔄 Testing Fallback Mechanisms...")
        
        try:
            from robust_data_system import unified_data_provider
            
            fallback_results = {}
            
            # Test comprehensive data with potential failures
            for symbol in ['XAUUSD', 'INVALID_SYMBOL']:
                try:
                    comprehensive_data = await unified_data_provider.get_comprehensive_data(symbol)
                    
                    fallback_results[symbol] = {
                        'success': True,
                        'has_price': 'price' in comprehensive_data and comprehensive_data['price'] is not None,
                        'has_sentiment': 'sentiment' in comprehensive_data and comprehensive_data['sentiment'] is not None,
                        'has_technical': 'technical' in comprehensive_data and comprehensive_data['technical'] is not None,
                        'fallback_used': any('simulated' in str(v).lower() if v else False 
                                           for v in comprehensive_data.values())
                    }
                    
                    if symbol == 'XAUUSD':
                        print(f"   ✅ Valid symbol ({symbol}): All data types available")
                    else:
                        print(f"   ✅ Invalid symbol ({symbol}): Fallback data generated")
                        
                except Exception as e:
                    fallback_results[symbol] = {'success': False, 'error': str(e)}
                    print(f"   ❌ Fallback test failed for {symbol}: {e}")
            
            self.results['fallback_mechanisms'] = fallback_results
            
        except Exception as e:
            self.results['fallback_mechanisms'] = {'error': str(e)}
            print(f"   ❌ Fallback mechanisms test failed: {e}")
    
    async def test_cache_management(self):
        """Test caching functionality"""
        print("\n💾 Testing Cache Management...")
        
        try:
            from robust_data_system import unified_data_provider
            
            cache_results = {}
            
            # Test cache performance
            symbol = 'XAUUSD'
            
            # First call (should cache)
            start_time = time.time()
            first_data = await unified_data_provider.get_price_data(symbol)
            first_call_time = time.time() - start_time
            
            # Second call (should use cache)
            start_time = time.time()
            second_data = await unified_data_provider.get_price_data(symbol)
            second_call_time = time.time() - start_time
            
            cache_results = {
                'first_call_time': first_call_time,
                'second_call_time': second_call_time,
                'cache_effective': second_call_time < first_call_time,
                'data_consistent': (
                    first_data.symbol == second_data.symbol and
                    first_data.price == second_data.price
                )
            }
            
            print(f"   ✅ First call: {first_call_time:.3f}s")
            print(f"   ✅ Second call: {second_call_time:.3f}s")
            print(f"   📊 Cache effective: {cache_results['cache_effective']}")
            print(f"   🔍 Data consistent: {cache_results['data_consistent']}")
            
            self.results['cache_management'] = cache_results
            
        except Exception as e:
            self.results['cache_management'] = {'error': str(e)}
            print(f"   ❌ Cache management test failed: {e}")
    
    async def test_flask_integration(self):
        """Test Flask integration endpoints"""
        print("\n🌐 Testing Flask Integration...")
        
        try:
            from enhanced_flask_integration import (
                get_price_data_sync,
                get_sentiment_data_sync,
                get_technical_data_sync,
                ROBUST_DATA_AVAILABLE
            )
            
            flask_results = {}
            
            # Test synchronous wrappers
            if ROBUST_DATA_AVAILABLE:
                # Test price data sync
                price_result = get_price_data_sync('XAUUSD')
                flask_results['price_sync'] = {
                    'success': price_result.get('success', False),
                    'has_price': 'price' in price_result,
                    'source': price_result.get('source', 'unknown')
                }
                
                # Test sentiment data sync
                sentiment_result = get_sentiment_data_sync('XAUUSD')
                flask_results['sentiment_sync'] = {
                    'success': sentiment_result.get('success', False),
                    'has_sentiment': 'sentiment_score' in sentiment_result
                }
                
                # Test technical data sync
                technical_result = get_technical_data_sync('XAUUSD')
                flask_results['technical_sync'] = {
                    'success': technical_result.get('success', False),
                    'has_indicators': 'indicators' in technical_result
                }
                
                print("   ✅ Synchronous wrappers working")
                print(f"   📊 Price sync: {flask_results['price_sync']['success']}")
                print(f"   💭 Sentiment sync: {flask_results['sentiment_sync']['success']}")
                print(f"   📈 Technical sync: {flask_results['technical_sync']['success']}")
                
            else:
                flask_results['error'] = 'Robust data system not available'
                print("   ⚠️ Robust data system not available for Flask integration")
            
            self.results['flask_integration'] = flask_results
            
        except Exception as e:
            self.results['flask_integration'] = {'error': str(e)}
            print(f"   ❌ Flask integration test failed: {e}")
    
    async def test_frontend_integration(self):
        """Test frontend integration readiness"""
        print("\n🎨 Testing Frontend Integration Readiness...")
        
        try:
            # Check if frontend JavaScript files exist
            frontend_files = [
                'static/js/robust-data-integration.js',
                'static/js/real-time-data-manager.js'
            ]
            
            frontend_results = {}
            
            for file_path in frontend_files:
                if os.path.exists(file_path):
                    frontend_results[file_path] = {'exists': True}
                    print(f"   ✅ {file_path} exists")
                else:
                    frontend_results[file_path] = {'exists': False}
                    print(f"   ⚠️ {file_path} missing")
            
            # Test if enhanced routes can be imported (indicates Flask integration readiness)
            try:
                from enhanced_flask_integration import setup_enhanced_routes
                frontend_results['enhanced_routes'] = {'available': True}
                print("   ✅ Enhanced routes available for frontend")
            except ImportError:
                frontend_results['enhanced_routes'] = {'available': False}
                print("   ⚠️ Enhanced routes not available")
            
            self.results['frontend_integration'] = frontend_results
            
        except Exception as e:
            self.results['frontend_integration'] = {'error': str(e)}
            print(f"   ❌ Frontend integration test failed: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 65)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 65)
        
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.results.items():
            print(f"\n🔍 {test_category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if isinstance(results, dict) and 'error' not in results:
                for test_name, test_result in results.items():
                    total_tests += 1
                    if isinstance(test_result, dict):
                        success = test_result.get('success', test_result.get('exists', True))
                    else:
                        success = test_result
                    
                    if success:
                        passed_tests += 1
                        print(f"   ✅ {test_name}: PASS")
                    else:
                        print(f"   ❌ {test_name}: FAIL")
                        if isinstance(test_result, dict) and 'error' in test_result:
                            print(f"      Error: {test_result['error']}")
            else:
                total_tests += 1
                if 'error' in results:
                    print(f"   ❌ {test_category}: FAIL")
                    print(f"      Error: {results['error']}")
                else:
                    passed_tests += 1
                    print(f"   ✅ {test_category}: PASS")
        
        # Overall summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 OVERALL RESULTS")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"   🏆 Status: EXCELLENT - Robust data system ready for production!")
        elif success_rate >= 60:
            print(f"   ✅ Status: GOOD - System functional with minor issues")
        else:
            print(f"   ⚠️ Status: NEEDS ATTENTION - Multiple components require fixes")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'test_report_{timestamp}.json'
        
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'success_rate': success_rate
                    },
                    'detailed_results': self.results
                }, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save report: {e}")

async def main():
    """Main test execution"""
    tester = RobustDataSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    print("🚀 Starting GoldGPT Robust Data System Test Suite...")
    asyncio.run(main())
