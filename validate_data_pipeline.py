#!/usr/bin/env python3
"""
Simple validation script for GoldGPT Data Integration Pipeline
Tests basic functionality and imports
"""

import sys
import os

def test_basic_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test standard library imports
        import asyncio
        import json
        import sqlite3
        import time
        from datetime import datetime, timedelta
        print("✅ Standard library imports successful")
        
        # Test numpy and pandas
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas imports successful")
        
        # Test our modules
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        print("🔄 Testing data pipeline imports...")
        from data_integration_engine import DataIntegrationEngine, DataManager
        print("✅ Data integration engine imported successfully")
        
        from data_sources_config import config
        print("✅ Data sources configuration imported successfully")
        
        from data_pipeline_api import data_pipeline_bp
        print("✅ Data pipeline API imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        from data_integration_engine import DataCache, FeatureEngineer
        
        # Test cache
        cache = DataCache("test_validation.db")
        cache.set("test", {"value": 123}, 3600, "test")
        retrieved = cache.get("test")
        
        if retrieved and retrieved.get("value") == 123:
            print("✅ Cache system working")
        else:
            print("❌ Cache system failed")
            return False
        
        # Test feature engineer
        engineer = FeatureEngineer()
        test_features = engineer._extract_time_features()
        
        if test_features and len(test_features) > 0:
            print("✅ Feature engineering working")
        else:
            print("❌ Feature engineering failed")
            return False
        
        # Clean up
        if os.path.exists("test_validation.db"):
            os.remove("test_validation.db")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        from data_sources_config import config, ECONOMIC_INDICATORS, TECHNICAL_INDICATORS
        
        # Test configuration access
        sources = config.get_all_sources()
        enabled_sources = config.get_enabled_sources()
        
        print(f"✅ Total data sources: {len(sources)}")
        print(f"✅ Enabled data sources: {len(enabled_sources)}")
        print(f"✅ Economic indicators: {len(ECONOMIC_INDICATORS['primary_indicators'])}")
        print(f"✅ Technical indicators: {len(TECHNICAL_INDICATORS['trend_indicators'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def create_sample_integration():
    """Create a sample integration to show the system works"""
    print("\n🎯 Creating Sample Integration...")
    
    try:
        from data_integration_engine import CandlestickData, NewsData, EconomicIndicator
        from datetime import datetime, timezone
        
        # Create sample data
        sample_candle = CandlestickData(
            timestamp=datetime.now(timezone.utc),
            open=2000.0,
            high=2010.0,
            low=1995.0,
            close=2005.0,
            volume=1500,
            timeframe="1h"
        )
        
        sample_news = NewsData(
            timestamp=datetime.now(timezone.utc),
            title="Gold prices steady amid market conditions",
            content="Gold trading remains stable...",
            source="sample",
            sentiment_score=0.1,
            relevance_score=0.8,
            url="http://sample.com"
        )
        
        sample_economic = EconomicIndicator(
            timestamp=datetime.now(timezone.utc),
            indicator_name="USD_INDEX",
            value=103.5,
            country="US",
            impact_level="high",
            source="sample"
        )
        
        print("✅ Sample data structures created successfully")
        print(f"   • Candlestick: {sample_candle.close} USD")
        print(f"   • News sentiment: {sample_news.sentiment_score}")
        print(f"   • Economic indicator: {sample_economic.indicator_name} = {sample_economic.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample integration failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 GoldGPT Data Integration Pipeline - Validation Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration System", test_configuration),
        ("Sample Integration", create_sample_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    print(f"🎯 Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Data pipeline is ready for use.")
        print("\n🔧 Next Steps:")
        print("   1. Install external dependencies: pip install aiohttp beautifulsoup4 textblob scikit-learn")
        print("   2. Configure API keys in environment variables")
        print("   3. Run full test suite: python test_data_pipeline.py")
        print("   4. Integrate with Flask app: from data_pipeline_api import init_data_pipeline_for_app")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print(f"\n📅 Validation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
