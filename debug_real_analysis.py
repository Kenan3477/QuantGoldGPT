#!/usr/bin/env python3
"""
Debug the real technical analysis to see what's happening
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_real_analysis():
    """Test the real technical analysis directly"""
    
    print("🔍 Testing real technical analysis...")
    
    try:
        from real_technical_analysis import technical_analyzer
        
        print("✅ Successfully imported technical_analyzer")
        
        # Generate analysis
        analysis = technical_analyzer.generate_comprehensive_analysis('XAUUSD')
        
        print(f"📊 Analysis keys: {list(analysis.keys())}")
        
        if 'data_quality' in analysis:
            data_quality = analysis['data_quality']
            print(f"📈 Data Quality: {data_quality}")
            
            price_points = data_quality.get('price_points', 0)
            print(f"📊 Price points available: {price_points}")
            
            if price_points < 20:
                print("⚠️ PROBLEM: Insufficient price points for real analysis")
                print("🔍 This explains why synthetic analysis is being used")
            else:
                print("✅ Sufficient price points for real analysis")
                
        if 'signal' in analysis:
            print(f"📈 Signal: {analysis['signal']}")
            print(f"🎯 Confidence: {analysis.get('confidence', 'N/A')}")
            
        if 'technical_indicators' in analysis:
            indicators = analysis['technical_indicators']
            print(f"📊 Technical Indicators: {indicators}")
            
        print("\n🔍 Full analysis structure:")
        import json
        print(json.dumps(analysis, indent=2, default=str))
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

def check_database():
    """Check database status"""
    
    print("\n🗄️ Checking database status...")
    
    try:
        import sqlite3
        
        conn = sqlite3.connect('price_storage.db')
        cursor = conn.cursor()
        
        # Check candlestick data
        cursor.execute("SELECT COUNT(*) FROM candlestick_data")
        candlestick_count = cursor.fetchone()[0]
        print(f"📊 Candlestick records: {candlestick_count}")
        
        # Check recent data
        cursor.execute("SELECT * FROM candlestick_data ORDER BY timestamp DESC LIMIT 5")
        recent_data = cursor.fetchall()
        print(f"📈 Recent data samples: {len(recent_data)}")
        
        for row in recent_data[:2]:
            print(f"   {row}")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    print("🚀 Real Analysis Debug Test")
    print("=" * 50)
    
    check_database()
    test_real_analysis()
    
    print("\n✅ Debug test completed!")
