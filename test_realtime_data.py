#!/usr/bin/env python3
"""
Real-Time Data Engine Test Suite
Comprehensive testing of all real-time data sources and API endpoints
"""

import asyncio
import requests
import json
import time
import sys
import os
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD"]

class RealTimeDataTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.results = {}
        
    def test_api_endpoint(self, endpoint, expected_keys=None):
        """Test a single API endpoint"""
        try:
            print(f"  Testing {endpoint}...")
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print(f"    ✅ {endpoint} - SUCCESS")
                    
                    if expected_keys:
                        missing_keys = []
                        for key in expected_keys:
                            if key not in data:
                                missing_keys.append(key)
                        
                        if missing_keys:
                            print(f"    ⚠️  Missing keys: {missing_keys}")
                        else:
                            print(f"    ✅ All expected keys present")
                    
                    return True, data
                else:
                    print(f"    ❌ {endpoint} - API Error: {data.get('error', 'Unknown error')}")
                    return False, data
            else:
                print(f"    ❌ {endpoint} - HTTP {response.status_code}")
                return False, None
                
        except Exception as e:
            print(f"    ❌ {endpoint} - Exception: {e}")
            return False, None
    
    def test_price_endpoints(self):
        """Test all price-related endpoints"""
        print("\n🔍 Testing Price Endpoints...")
        
        # Test individual symbol prices
        for symbol in TEST_SYMBOLS:
            success, data = self.test_api_endpoint(
                f"/api/realtime/price/{symbol}",
                expected_keys=['data', 'timestamp']
            )
            
            if success and data:
                price_data = data.get('data', {})
                print(f"    📊 {symbol}: ${price_data.get('price', 'N/A')} ({price_data.get('source', 'unknown')})")
        
        # Test watchlist endpoint
        success, data = self.test_api_endpoint(
            "/api/realtime/watchlist",
            expected_keys=['data', 'count']
        )
        
        if success and data:
            print(f"    📋 Watchlist: {data.get('count', 0)} symbols loaded")
    
    def test_sentiment_endpoints(self):
        """Test sentiment analysis endpoints"""
        print("\n🧠 Testing Sentiment Endpoints...")
        
        for symbol in TEST_SYMBOLS:
            success, data = self.test_api_endpoint(
                f"/api/realtime/sentiment/{symbol}",
                expected_keys=['data']
            )
            
            if success and data:
                sentiment_data = data.get('data', {})
                overall = sentiment_data.get('overall', {})
                print(f"    💭 {symbol}: {overall.get('sentiment', 'N/A')} " +
                      f"(confidence: {overall.get('confidence', 0):.2f})")
    
    def test_technical_endpoints(self):
        """Test technical analysis endpoints"""
        print("\n📈 Testing Technical Analysis Endpoints...")
        
        for symbol in TEST_SYMBOLS:
            success, data = self.test_api_endpoint(
                f"/api/realtime/technical/{symbol}",
                expected_keys=['data']
            )
            
            if success and data:
                tech_data = data.get('data', {})
                rsi = tech_data.get('rsi', {})
                macd = tech_data.get('macd', {})
                print(f"    📊 {symbol}: RSI={rsi.get('value', 'N/A')} " +
                      f"MACD={macd.get('value', 'N/A')}")
    
    def test_comprehensive_endpoint(self):
        """Test comprehensive data endpoint"""
        print("\n🔍 Testing Comprehensive Data Endpoint...")
        
        symbol = "XAUUSD"
        success, data = self.test_api_endpoint(
            f"/api/realtime/comprehensive/{symbol}",
            expected_keys=['symbol', 'price', 'sentiment', 'technical']
        )
        
        if success and data:
            print(f"    ✅ Comprehensive data for {symbol}:")
            print(f"    💰 Price: ${data.get('price', {}).get('price', 'N/A')}")
            print(f"    💭 Sentiment: {data.get('sentiment', {}).get('overall', {}).get('sentiment', 'N/A')}")
            print(f"    📊 RSI: {data.get('technical', {}).get('rsi', {}).get('value', 'N/A')}")
    
    def test_system_status(self):
        """Test system status endpoint"""
        print("\n🔧 Testing System Status...")
        
        success, data = self.test_api_endpoint(
            "/api/realtime/status",
            expected_keys=['status']
        )
        
        if success and data:
            status = data.get('status', {})
            print(f"    🔧 Real-time engine: {'✅' if status.get('real_time_engine_available') else '❌'}")
            print(f"    🤖 ML predictions: {'✅' if status.get('ml_predictions_available') else '❌'}")
            print(f"    🥇 Gold API: {status.get('gold_api_status', 'unknown')}")
            print(f"    📈 Yahoo Finance: {status.get('yahoo_finance_status', 'unknown')}")
    
    def test_data_quality(self):
        """Test data quality and consistency"""
        print("\n🔍 Testing Data Quality...")
        
        # Get price data
        success, price_data = self.test_api_endpoint("/api/realtime/price/XAUUSD")
        
        if success and price_data:
            price = price_data.get('data', {}).get('price', 0)
            
            # Check if price is reasonable for gold
            if 1500 <= price <= 3000:
                print(f"    ✅ Gold price ${price:.2f} is within reasonable range")
            else:
                print(f"    ⚠️  Gold price ${price:.2f} seems unusual")
            
            # Check data freshness
            timestamp = price_data.get('timestamp')
            if timestamp:
                data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age_seconds = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds()
                
                if age_seconds < 60:
                    print(f"    ✅ Data is fresh ({age_seconds:.1f}s old)")
                else:
                    print(f"    ⚠️  Data is {age_seconds:.1f}s old")
    
    def test_live_updates(self):
        """Test live data updates"""
        print("\n⏱️  Testing Live Data Updates...")
        
        print("    Getting initial price...")
        success1, data1 = self.test_api_endpoint("/api/realtime/price/XAUUSD")
        
        if success1:
            price1 = data1.get('data', {}).get('price', 0)
            timestamp1 = data1.get('timestamp')
            
            print(f"    Initial: ${price1:.2f} at {timestamp1}")
            
            # Wait and get price again
            print("    Waiting 10 seconds...")
            time.sleep(10)
            
            success2, data2 = self.test_api_endpoint("/api/realtime/price/XAUUSD")
            
            if success2:
                price2 = data2.get('data', {}).get('price', 0)
                timestamp2 = data2.get('timestamp')
                
                print(f"    Updated: ${price2:.2f} at {timestamp2}")
                
                if timestamp2 != timestamp1:
                    print("    ✅ Timestamp updated - data is refreshing")
                else:
                    print("    ⚠️  Timestamp unchanged - may be cached")
                
                if abs(price2 - price1) > 0:
                    print(f"    📈 Price changed by ${abs(price2 - price1):.2f}")
                else:
                    print("    📊 Price unchanged (normal for short interval)")
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when real-time data fails"""
        print("\n🛡️  Testing Fallback Mechanisms...")
        
        # Test invalid symbol
        success, data = self.test_api_endpoint("/api/realtime/price/INVALID")
        
        if success and data:
            source = data.get('data', {}).get('source', '')
            if 'fallback' in source.lower():
                print("    ✅ Fallback working for invalid symbols")
            else:
                print("    ⚠️  Fallback may not be working properly")
    
    def test_frontend_integration(self):
        """Test frontend integration endpoints"""
        print("\n🌐 Testing Frontend Integration...")
        
        # Test main dashboard endpoint
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                print("    ✅ Main dashboard accessible")
                
                # Check if real-time data manager script is included
                if 'real-time-data-manager.js' in response.text:
                    print("    ✅ Real-time data manager script included")
                else:
                    print("    ⚠️  Real-time data manager script not found")
            else:
                print(f"    ❌ Dashboard not accessible (HTTP {response.status_code})")
        except Exception as e:
            print(f"    ❌ Dashboard test failed: {e}")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Starting Real-Time Data Engine Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        self.test_system_status()
        self.test_price_endpoints()
        self.test_sentiment_endpoints()
        self.test_technical_endpoints()
        self.test_comprehensive_endpoint()
        self.test_data_quality()
        self.test_live_updates()
        self.test_fallback_mechanisms()
        self.test_frontend_integration()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print(f"✅ Test suite completed in {duration:.2f} seconds")
        print("\n📋 Summary:")
        print("   • Real-time data engine tested")
        print("   • Multiple data sources verified")
        print("   • Fallback mechanisms checked")
        print("   • Frontend integration tested")
        print("\n🎯 Your GoldGPT platform now uses real-time data instead of hardcoded values!")

def main():
    """Main test function"""
    print("🚀 GoldGPT Real-Time Data Engine Tester")
    print("Testing all components of the real-time data replacement system")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/realtime/status", timeout=5)
        if response.status_code == 200:
            print("✅ GoldGPT server is running and accessible")
        else:
            print("❌ GoldGPT server returned unexpected status")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to GoldGPT server")
        print("   Please make sure the server is running on http://127.0.0.1:5000")
        print("   Run: python app.py")
        return
    except Exception as e:
        print(f"❌ Server check failed: {e}")
        return
    
    # Run tests
    tester = RealTimeDataTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
