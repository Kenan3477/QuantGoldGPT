#!/usr/bin/env python3
"""
Test script to verify the Gold API integration is working correctly
"""
import requests
from datetime import datetime

def test_app_gold_api():
    """Test the main app gold API integration"""
    print("Testing app.py gold API integration...")
    try:
        response = requests.get('https://api.gold-api.com/price/XAU', timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"API Response: {data}")
            
            price = data.get('price', 0)
            if price > 0:
                print(f"✓ Gold Price Retrieved: ${price:.2f} USD per ounce")
                return True
            else:
                print("✗ No valid price in response")
                return False
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_enhanced_ml_api():
    """Test the enhanced ML dashboard API gold integration"""
    print("\nTesting enhanced ML dashboard gold API integration...")
    try:
        from enhanced_ml_dashboard_api import ComprehensiveGoldAnalyzer
        
        analyzer = ComprehensiveGoldAnalyzer()
        gold_data = analyzer.get_current_gold_data()
        
        print(f"Gold Data Retrieved: {gold_data}")
        
        if gold_data.get('current_price', 0) > 0:
            print(f"✓ ML Dashboard Gold Price: ${gold_data['current_price']:.2f}")
            print(f"✓ Data Source: {gold_data.get('price_data', {}).get('source', 'unknown')}")
            return True
        else:
            print("✗ No valid price in ML dashboard data")
            return False
            
    except Exception as e:
        print(f"✗ Error testing ML dashboard: {e}")
        return False

def test_comprehensive_analysis():
    """Test the comprehensive analysis with real data"""
    print("\nTesting comprehensive gold analysis...")
    try:
        from enhanced_ml_dashboard_api import ComprehensiveGoldAnalyzer
        
        analyzer = ComprehensiveGoldAnalyzer()
        analysis = analyzer.generate_comprehensive_analysis()
        
        print(f"Analysis Result Keys: {list(analysis.keys())}")
        
        bias = analysis.get('bias', {})
        if bias.get('direction'):
            print(f"✓ Market Bias: {bias['direction']}")
            print(f"✓ Confidence: {bias['confidence']:.1f}%")
            
            predictions = analysis.get('predictions', {})
            if predictions:
                print(f"✓ Predictions Generated: {len(predictions)} timeframes")
                for timeframe, pred in predictions.items():
                    print(f"  - {timeframe}: ${pred.get('target', 0):.2f} (confidence: {pred.get('confidence', 0):.1f}%)")
            
            return True
        else:
            print("✗ No bias direction in analysis")
            return False
            
    except Exception as e:
        print(f"✗ Error testing comprehensive analysis: {e}")
        return False

if __name__ == "__main__":
    print("=== Gold API Integration Test ===")
    print(f"Test Time: {datetime.now()}")
    
    app_success = test_app_gold_api()
    ml_success = test_enhanced_ml_api()
    analysis_success = test_comprehensive_analysis()
    
    print("\n=== Test Summary ===")
    print(f"App Gold API: {'✓ PASS' if app_success else '✗ FAIL'}")
    print(f"ML Dashboard API: {'✓ PASS' if ml_success else '✗ FAIL'}")
    print(f"Comprehensive Analysis: {'✓ PASS' if analysis_success else '✗ FAIL'}")
    
    if app_success and ml_success and analysis_success:
        print("\n🎉 All tests passed! Gold API integration is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
