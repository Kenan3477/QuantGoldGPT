#!/usr/bin/env python3
"""
Test script to verify AI Analysis navigation fix
"""

import requests
import time
from urllib.parse import urljoin

def test_local_navigation():
    """Test the local application navigation"""
    base_url = "http://127.0.0.1:5000"
    
    print("🔍 Testing GoldGPT AI Analysis Navigation Fix...")
    print("=" * 50)
    
    try:
        # Test main dashboard
        print("1. Testing main dashboard...")
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ Main dashboard accessible")
            
            # Check if showAIAnalysisSection function exists in the HTML
            if 'showAIAnalysisSection' in response.text:
                print("   ✅ showAIAnalysisSection function found in HTML")
            else:
                print("   ❌ showAIAnalysisSection function NOT found")
                
            # Check if the System Hub link has the correct href for route-based navigation
            if 'href="/ai-analysis"' in response.text and 'class="system-link"' in response.text:
                print("   ✅ System Hub AI Analysis Center link properly configured")
            else:
                print("   ❌ System Hub link configuration issue")
                
            # Check for LEFT PANEL AI Analysis button (should be REMOVED)
            if 'data-section="ai-analysis"' not in response.text:
                print("   ✅ Left panel AI Analysis button properly removed")
            else:
                print("   ❌ Left panel AI Analysis button still exists (should be removed)")
                
            # Check if AI Tools section was removed
            if 'AI Tools' not in response.text or response.text.count('AI Tools') <= 1:
                print("   ✅ Duplicate AI Tools section removed")
            else:
                print("   ❌ AI Tools section still exists (might be duplicate)")
                
            # Check for daily/weekly analysis toggle
            if 'Day Trading' in response.text and 'Weekly Analysis' in response.text:
                print("   ✅ Daily/Weekly analysis toggles found")
            else:
                print("   ❌ Daily/Weekly analysis toggles missing")
                
        else:
            print(f"   ❌ Main dashboard returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to local server. Is it running?")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test AI Analysis route
    print("\n2. Testing AI Analysis route...")
    try:
        ai_analysis_url = urljoin(base_url, '/ai-analysis')
        response = requests.get(ai_analysis_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ /ai-analysis route accessible")
        else:
            print(f"   ❌ /ai-analysis route returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ AI Analysis route error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ AI Analysis navigation test completed")
    print("\n💡 Only System Hub AI Analysis Center should work now!")
    print("🎯 System Hub → AI Analysis Center → Daily/Weekly Analysis Suite")
    
    return True

if __name__ == "__main__":
    test_local_navigation()
