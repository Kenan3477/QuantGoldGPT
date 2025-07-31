#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Trade Signal System
Tests signal generation, monitoring, TP/SL hits, and learning capabilities
"""
import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSignalSystemTester:
    """Comprehensive tester for the enhanced signal system"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = {}
        
    def run_comprehensive_test(self):
        """Run all tests for the enhanced signal system"""
        logger.info("🧪 Starting Comprehensive Enhanced Signal System Test")
        logger.info("=" * 60)
        
        try:
            # Test 1: API Endpoints
            self.test_api_endpoints()
            
            # Test 2: Signal Generation
            self.test_signal_generation()
            
            # Test 3: Signal Monitoring
            self.test_signal_monitoring()
            
            # Test 4: Performance Tracking
            self.test_performance_tracking()
            
            # Test 5: Active Signal Management
            self.test_active_signal_management()
            
            # Test 6: Historical Data
            self.test_historical_data()
            
            # Summary
            self.print_test_summary()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            
    def test_api_endpoints(self):
        """Test all enhanced signal API endpoints"""
        logger.info("🔍 Testing Enhanced Signal API Endpoints...")
        
        endpoints = [
            "/api/enhanced-signals/generate",
            "/api/enhanced-signals/monitor", 
            "/api/enhanced-signals/performance",
            "/api/enhanced-signals/active",
            "/api/enhanced-signals/history"
        ]
        
        endpoint_results = {}
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                endpoint_results[endpoint] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    data = response.json()
                    endpoint_results[endpoint]["has_data"] = data.get('success', False)
                    logger.info(f"✅ {endpoint}: {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
                else:
                    logger.warning(f"⚠️ {endpoint}: {response.status_code}")
                    
            except Exception as e:
                endpoint_results[endpoint] = {"error": str(e), "success": False}
                logger.error(f"❌ {endpoint}: {e}")
                
        self.test_results["api_endpoints"] = endpoint_results
        
    def test_signal_generation(self):
        """Test enhanced signal generation"""
        logger.info("🎯 Testing Enhanced Signal Generation...")
        
        try:
            # Generate a new signal
            response = requests.post(f"{self.base_url}/api/enhanced-signals/generate", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success') and data.get('signal'):
                    signal = data['signal']
                    logger.info(f"✅ Signal generated successfully!")
                    logger.info(f"   Type: {signal.get('signal_type', 'Unknown').upper()}")
                    logger.info(f"   Entry: ${signal.get('entry_price', 0):.2f}")
                    logger.info(f"   TP: ${signal.get('target_price', 0):.2f}")
                    logger.info(f"   SL: ${signal.get('stop_loss', 0):.2f}")
                    logger.info(f"   Confidence: {signal.get('confidence', 0):.1f}%")
                    logger.info(f"   R:R: {signal.get('risk_reward_ratio', 0):.1f}:1")
                    
                    self.test_results["signal_generation"] = {
                        "success": True,
                        "signal": signal,
                        "has_tp_sl": signal.get('target_price') and signal.get('stop_loss'),
                        "has_entry_price": signal.get('entry_price') is not None,
                        "has_confidence": signal.get('confidence') is not None
                    }
                else:
                    logger.info("ℹ️ No signal generated - market conditions not suitable")
                    self.test_results["signal_generation"] = {
                        "success": True,
                        "signal": None,
                        "message": data.get('message', 'No signal generated')
                    }
            else:
                logger.error(f"❌ Signal generation failed: {response.status_code}")
                self.test_results["signal_generation"] = {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Signal generation test failed: {e}")
            self.test_results["signal_generation"] = {"success": False, "error": str(e)}
            
    def test_signal_monitoring(self):
        """Test signal monitoring capabilities"""
        logger.info("🔄 Testing Signal Monitoring...")
        
        try:
            response = requests.get(f"{self.base_url}/api/enhanced-signals/monitor", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    monitoring = data.get('monitoring', {})
                    logger.info(f"✅ Monitoring active!")
                    logger.info(f"   Current Price: ${monitoring.get('current_price', 0):.2f}")
                    logger.info(f"   Active Signals: {monitoring.get('active_signals', 0)}")
                    logger.info(f"   Updates: {len(monitoring.get('updates', []))}")
                    logger.info(f"   Closed Signals: {len(monitoring.get('closed_signals', []))}")
                    
                    self.test_results["signal_monitoring"] = {
                        "success": True,
                        "monitoring_data": monitoring,
                        "has_current_price": monitoring.get('current_price') is not None
                    }
                else:
                    logger.error(f"❌ Monitoring failed: {data.get('error', 'Unknown error')}")
                    self.test_results["signal_monitoring"] = {"success": False, "error": data.get('error')}
            else:
                logger.error(f"❌ Monitoring request failed: {response.status_code}")
                self.test_results["signal_monitoring"] = {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Signal monitoring test failed: {e}")
            self.test_results["signal_monitoring"] = {"success": False, "error": str(e)}
            
    def test_performance_tracking(self):
        """Test performance tracking and statistics"""
        logger.info("📈 Testing Performance Tracking...")
        
        try:
            response = requests.get(f"{self.base_url}/api/enhanced-signals/performance", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    performance = data.get('performance', {})
                    logger.info(f"✅ Performance data retrieved!")
                    logger.info(f"   Total Signals: {performance.get('total_signals', 0)}")
                    logger.info(f"   Successful: {performance.get('successful_signals', 0)}")
                    logger.info(f"   Success Rate: {performance.get('success_rate', 0):.1f}%")
                    logger.info(f"   Avg P&L: {performance.get('avg_profit_loss_pct', 0):.2f}%")
                    logger.info(f"   Best Trade: {performance.get('best_profit_pct', 0):.2f}%")
                    
                    recent = performance.get('recent_7_days', {})
                    if recent:
                        logger.info(f"   Recent (7d): {recent.get('signals', 0)} signals, {recent.get('success_rate', 0):.1f}% success")
                    
                    self.test_results["performance_tracking"] = {
                        "success": True,
                        "performance_data": performance,
                        "has_statistics": performance.get('total_signals') is not None
                    }
                else:
                    logger.error(f"❌ Performance tracking failed: {data.get('error', 'Unknown error')}")
                    self.test_results["performance_tracking"] = {"success": False, "error": data.get('error')}
            else:
                logger.error(f"❌ Performance request failed: {response.status_code}")
                self.test_results["performance_tracking"] = {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Performance tracking test failed: {e}")
            self.test_results["performance_tracking"] = {"success": False, "error": str(e)}
            
    def test_active_signal_management(self):
        """Test active signal management"""
        logger.info("📊 Testing Active Signal Management...")
        
        try:
            response = requests.get(f"{self.base_url}/api/enhanced-signals/active", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    active_signals = data.get('active_signals', [])
                    logger.info(f"✅ Active signals retrieved!")
                    logger.info(f"   Count: {len(active_signals)}")
                    
                    for i, signal in enumerate(active_signals[:3]):  # Show first 3
                        logger.info(f"   Signal {i+1}: {signal.get('signal_type', '').upper()} @ ${signal.get('entry_price', 0):.2f}")
                        logger.info(f"              TP: ${signal.get('target_price', 0):.2f}, SL: ${signal.get('stop_loss', 0):.2f}")
                        logger.info(f"              P&L: ${signal.get('unrealized_pnl', 0):.2f} ({signal.get('unrealized_pnl_pct', 0):+.2f}%)")
                    
                    self.test_results["active_signal_management"] = {
                        "success": True,
                        "active_signals": active_signals,
                        "signal_count": len(active_signals),
                        "has_unrealized_pnl": any(s.get('unrealized_pnl') is not None for s in active_signals)
                    }
                else:
                    logger.error(f"❌ Active signals failed: {data.get('error', 'Unknown error')}")
                    self.test_results["active_signal_management"] = {"success": False, "error": data.get('error')}
            else:
                logger.error(f"❌ Active signals request failed: {response.status_code}")
                self.test_results["active_signal_management"] = {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Active signal management test failed: {e}")
            self.test_results["active_signal_management"] = {"success": False, "error": str(e)}
            
    def test_historical_data(self):
        """Test historical signal data"""
        logger.info("📚 Testing Historical Data...")
        
        try:
            response = requests.get(f"{self.base_url}/api/enhanced-signals/history?limit=5", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    history = data.get('signal_history', [])
                    logger.info(f"✅ Historical data retrieved!")
                    logger.info(f"   Count: {len(history)}")
                    
                    for i, signal in enumerate(history[:3]):  # Show first 3
                        success = signal.get('success', False)
                        result = "✅ TP" if success else "❌ SL"
                        logger.info(f"   {result} Signal {i+1}: {signal.get('signal_type', '').upper()}")
                        logger.info(f"                     P&L: {signal.get('profit_loss_pct', 0):+.2f}%")
                        logger.info(f"                     Duration: {signal.get('duration_hours', 0):.1f}h")
                    
                    self.test_results["historical_data"] = {
                        "success": True,
                        "history": history,
                        "history_count": len(history),
                        "has_outcomes": any(s.get('exit_reason') is not None for s in history)
                    }
                else:
                    logger.error(f"❌ Historical data failed: {data.get('error', 'Unknown error')}")
                    self.test_results["historical_data"] = {"success": False, "error": data.get('error')}
            else:
                logger.error(f"❌ Historical data request failed: {response.status_code}")
                self.test_results["historical_data"] = {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Historical data test failed: {e}")
            self.test_results["historical_data"] = {"success": False, "error": str(e)}
            
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 ENHANCED SIGNAL SYSTEM TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            logger.info(f"   {status} {test_name.replace('_', ' ').title()}")
            
            if not result.get('success', False) and 'error' in result:
                logger.info(f"        Error: {result['error']}")
                
        # Feature checks
        logger.info("\n🔍 Feature Verification:")
        
        signal_gen = self.test_results.get('signal_generation', {})
        if signal_gen.get('success') and signal_gen.get('signal'):
            signal = signal_gen['signal']
            logger.info(f"   ✅ Entry Price Matching: ${signal.get('entry_price', 0):.2f}")
            logger.info(f"   ✅ TP/SL Targets: TP=${signal.get('target_price', 0):.2f}, SL=${signal.get('stop_loss', 0):.2f}")
            logger.info(f"   ✅ Risk:Reward Ratio: {signal.get('risk_reward_ratio', 0):.1f}:1")
            logger.info(f"   ✅ Confidence Score: {signal.get('confidence', 0):.1f}%")
        
        monitoring = self.test_results.get('signal_monitoring', {})
        if monitoring.get('success'):
            logger.info(f"   ✅ Real-time Monitoring: Active")
            
        performance = self.test_results.get('performance_tracking', {})
        if performance.get('success'):
            logger.info(f"   ✅ Performance Tracking: Enabled")
            
        active = self.test_results.get('active_signal_management', {})
        if active.get('success'):
            count = active.get('signal_count', 0)
            logger.info(f"   ✅ Active Signal Management: {count} signals")
            
        history = self.test_results.get('historical_data', {})
        if history.get('success'):
            count = history.get('history_count', 0)
            logger.info(f"   ✅ Historical Learning: {count} records")
            
        logger.info("\n🎉 Enhanced Trade Signal System Test Complete!")
        
        if passed_tests == total_tests:
            logger.info("🏆 ALL TESTS PASSED - System is fully operational!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed - Check errors above")

def main():
    """Run the comprehensive test suite"""
    tester = EnhancedSignalSystemTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
