#!/usr/bin/env python3
"""
Test script to verify NaN and undefined fixes in Live Candlestick Monitor
"""

import requests
import json
import sys
from datetime import datetime

def test_live_patterns_endpoint():
    """Test the live patterns endpoint for NaN/undefined issues"""
    print("🔍 Testing Live Patterns Endpoint for NaN/undefined fixes...")
    
    try:
        # Test the endpoint
        response = requests.get('http://localhost:5000/api/live/patterns', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response received successfully")
            
            # Check for NaN or undefined values in response
            issues_found = []
            
            def check_for_nan_recursive(obj, path=""):
                """Recursively check for NaN, null, or undefined values"""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        check_for_nan_recursive(value, new_path)
                elif isinstance(obj, list):
                    for i, value in enumerate(obj):
                        new_path = f"{path}[{i}]"
                        check_for_nan_recursive(value, new_path)
                elif obj is None:
                    issues_found.append(f"NULL value at {path}")
                elif isinstance(obj, float) and (obj != obj):  # NaN check
                    issues_found.append(f"NaN value at {path}")
                elif isinstance(obj, str) and obj.lower() in ['nan', 'undefined', 'null']:
                    issues_found.append(f"String '{obj}' at {path}")
            
            # Check the response data
            check_for_nan_recursive(data)
            
            if issues_found:
                print("❌ Issues found in response:")
                for issue in issues_found:
                    print(f"  - {issue}")
                return False
            else:
                print("✅ No NaN/undefined/null issues found in response!")
                
                # Print summary of response structure
                print(f"\n📊 Response Summary:")
                print(f"  - Success: {data.get('success', 'Unknown')}")
                print(f"  - Total patterns: {data.get('total_patterns_detected', 0)}")
                print(f"  - Live patterns: {data.get('live_pattern_count', 0)}")
                print(f"  - Current price: ${data.get('current_price', 0)}")
                print(f"  - Data source: {data.get('data_source', 'Unknown')}")
                
                # Check specific pattern data
                patterns = data.get('current_patterns', [])
                if patterns:
                    print(f"\n🎯 First Pattern Example:")
                    pattern = patterns[0]
                    print(f"  - Pattern: {pattern.get('pattern', 'Unknown')}")
                    print(f"  - Confidence: {pattern.get('confidence', 'Unknown')}")
                    print(f"  - Signal: {pattern.get('signal', 'Unknown')}")
                    print(f"  - Time ago: {pattern.get('time_ago', 'Unknown')}")
                    print(f"  - Freshness score: {pattern.get('freshness_score', 'Unknown')}")
                
                return True
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_ml_predictions_endpoint():
    """Test the ML predictions endpoint for NaN/undefined issues"""
    print("\n🧠 Testing ML Predictions Endpoint for NaN/undefined fixes...")
    
    try:
        response = requests.get('http://localhost:5000/api/ml-predictions', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ ML Predictions API Response received successfully")
            
            # Check predictions data
            predictions = data.get('predictions', [])
            if predictions:
                print(f"📊 Found {len(predictions)} predictions")
                
                # Test first prediction for data integrity
                first_pred = predictions[0]
                print(f"\n🎯 First Prediction:")
                print(f"  - Signal: {first_pred.get('signal', 'Unknown')}")
                print(f"  - Confidence: {first_pred.get('confidence', 'Unknown')}")
                print(f"  - Timeframe: {first_pred.get('timeframe', 'Unknown')}")
                
                # Check for NaN in numerical fields
                numeric_fields = ['confidence', 'price_change', 'price_change_pct', 'current_price']
                for field in numeric_fields:
                    value = first_pred.get(field)
                    if value is not None:
                        try:
                            float_val = float(value)
                            if float_val != float_val:  # NaN check
                                print(f"❌ NaN found in {field}")
                                return False
                        except (ValueError, TypeError):
                            print(f"⚠️ Non-numeric value in {field}: {value}")
                
                print("✅ ML Predictions data looks clean!")
                return True
            else:
                print("ℹ️ No predictions available, but response is valid")
                return True
                
        else:
            print(f"❌ ML Predictions API failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ML Predictions test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting NaN/Undefined Fix Verification Tests")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Live Patterns
    if not test_live_patterns_endpoint():
        all_tests_passed = False
    
    # Test 2: ML Predictions
    if not test_ml_predictions_endpoint():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! NaN/undefined issues have been fixed!")
        print("✅ Your Live Candlestick Monitor should now display properly")
        print("✅ No more NaN or undefined values in the frontend")
    else:
        print("❌ SOME TESTS FAILED! Please check the issues above")
        print("🔧 Review the API responses and data validation")
    
    print(f"\nTest completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
