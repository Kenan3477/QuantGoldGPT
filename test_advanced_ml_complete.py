#!/usr/bin/env python3
"""
Advanced ML Module Verification Test
Tests all aspects of the Advanced ML module to ensure it's working correctly
"""

import requests
import json
from datetime import datetime

def test_advanced_ml_complete():
    """Complete test of Advanced ML module functionality"""
    
    print("🔍 Advanced ML Module Complete Verification")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    results = []
    
    # Test 1: Dashboard Access
    print("\n1. Testing Dashboard Access...")
    try:
        dashboard_response = requests.get(f"{base_url}/multi-strategy-ml-dashboard", timeout=10)
        if dashboard_response.status_code == 200:
            print("✅ Multi-Strategy ML Dashboard: Accessible")
            results.append("Dashboard: ✅")
        else:
            print(f"❌ Dashboard Error: {dashboard_response.status_code}")
            results.append("Dashboard: ❌")
    except Exception as e:
        print(f"❌ Dashboard Connection Error: {e}")
        results.append("Dashboard: ❌")
    
    # Test 2: API Status Endpoint
    print("\n2. Testing API Status...")
    try:
        status_response = requests.get(f"{base_url}/api/advanced-ml/status", timeout=10)
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"✅ API Status: {status_data.get('status', 'unknown')}")
            print(f"   ML Engine Available: {status_data.get('ml_engine_available', False)}")
            results.append("API Status: ✅")
        else:
            print(f"❌ API Status Error: {status_response.status_code}")
            results.append("API Status: ❌")
    except Exception as e:
        print(f"❌ API Status Error: {e}")
        results.append("API Status: ❌")
    
    # Test 3: Predictions Endpoint
    print("\n3. Testing Predictions...")
    try:
        pred_response = requests.get(f"{base_url}/api/advanced-ml/predictions", timeout=10)
        if pred_response.status_code == 200:
            pred_data = pred_response.json()
            print(f"✅ Predictions Available: {len(pred_data.get('predictions', []))} predictions")
            results.append("Predictions: ✅")
        else:
            print(f"❌ Predictions Error: {pred_response.status_code}")
            results.append("Predictions: ❌")
    except Exception as e:
        print(f"❌ Predictions Error: {e}")
        results.append("Predictions: ❌")
    
    # Test 4: Health Check
    print("\n4. Testing Health Check...")
    try:
        health_response = requests.get(f"{base_url}/api/advanced-ml/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health Status: {health_data.get('status', 'unknown')}")
            results.append("Health: ✅")
        else:
            print(f"❌ Health Error: {health_response.status_code}")
            results.append("Health: ❌")
    except Exception as e:
        print(f"❌ Health Error: {e}")
        results.append("Health: ❌")
    
    # Test 5: Accuracy Stats
    print("\n5. Testing Accuracy Stats...")
    try:
        acc_response = requests.get(f"{base_url}/api/advanced-ml/accuracy-stats", timeout=10)
        if acc_response.status_code == 200:
            acc_data = acc_response.json()
            print(f"✅ Accuracy Stats Available: {len(acc_data.get('engines', []))} engines tracked")
            results.append("Accuracy: ✅")
        else:
            print(f"❌ Accuracy Error: {acc_response.status_code}")
            results.append("Accuracy: ❌")
    except Exception as e:
        print(f"❌ Accuracy Error: {e}")
        results.append("Accuracy: ❌")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 ADVANCED ML MODULE VERIFICATION SUMMARY")
    print("=" * 50)
    
    success_count = len([r for r in results if "✅" in r])
    total_tests = len(results)
    
    for result in results:
        print(f"   {result}")
    
    print(f"\n🎯 Overall Score: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ADVANCED ML MODULE: FULLY OPERATIONAL ✅")
        return True
    else:
        print("⚠️  ADVANCED ML MODULE: PARTIALLY OPERATIONAL")
        return False

if __name__ == "__main__":
    success = test_advanced_ml_complete()
    print(f"\n{'='*50}")
    if success:
        print("✅ Your Advanced ML Module is working perfectly!")
        print("🚀 You can access it via:")
        print("   • Main Dashboard → System Hub → Advanced ML")
        print("   • Direct URL: http://localhost:5000/multi-strategy-ml-dashboard")
    else:
        print("⚠️  Some issues found - check the errors above")
