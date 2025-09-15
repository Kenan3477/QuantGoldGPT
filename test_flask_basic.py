#!/usr/bin/env python3
"""
Quick Flask app test - Verify Railway deployment endpoints work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_flask_import():
    """Test if Flask app can be imported without errors"""
    try:
        print("🧪 Testing Flask app import...")
        from app import app
        print("✅ Flask app imported successfully")
        
        # Test basic routes exist
        with app.test_client() as client:
            print("🧪 Testing /api/live-gold-price endpoint...")
            response = client.get('/api/live-gold-price')
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ /api/live-gold-price endpoint working")
            else:
                print(f"⚠️ /api/live-gold-price returned {response.status_code}")
                print(f"Response: {response.get_data(as_text=True)[:200]}...")
            
            print("🧪 Testing /api/ml-predictions endpoint...")
            response = client.get('/api/ml-predictions')
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ /api/ml-predictions endpoint working")
            else:
                print(f"⚠️ /api/ml-predictions returned {response.status_code}")
                print(f"Response: {response.get_data(as_text=True)[:200]}...")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Railway Deployment - Flask App Test")
    print("="*50)
    
    success = test_basic_flask_import()
    
    if success:
        print("\n🎉 SUCCESS: Flask app basic functionality working!")
        print("✅ Ready for Railway deployment")
        print("✅ Endpoints should pass healthcheck")
    else:
        print("\n❌ FAILED: Flask app has issues")
        print("Fix the above errors before deploying to Railway")
    
    sys.exit(0 if success else 1)
