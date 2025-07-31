#!/usr/bin/env python3
"""
Test Advanced ML Module Loading
Quick verification that the Advanced ML module loads correctly
"""

def test_advanced_ml_loading():
    """Test if Advanced ML module loads correctly"""
    
    print("🔍 Testing Advanced ML Module Loading...")
    
    # Test 1: Import the module
    try:
        from simplified_advanced_ml_api import integrate_advanced_ml_api
        print("✅ Module imported successfully")
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False
    
    # Test 2: Create Flask app and integrate
    try:
        from flask import Flask
        app = Flask(__name__)
        
        result = integrate_advanced_ml_api(app)
        print(f"✅ Integration result: {result}")
        
        if result:
            # Test 3: Check routes
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            ml_routes = [r for r in routes if 'advanced-ml' in r]
            print(f"✅ Advanced ML routes found: {len(ml_routes)}")
            for route in ml_routes:
                print(f"   • {route}")
            
            # Test 4: Test a simple endpoint
            with app.test_client() as client:
                response = client.get('/api/advanced-ml/status')
                print(f"✅ Status endpoint test: {response.status_code}")
                if response.status_code == 200:
                    data = response.get_json()
                    print(f"   Response: {data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"   Error: {response.data}")
                    return False
        else:
            print("❌ Integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_advanced_ml_loading()
    if success:
        print("\n🎉 Advanced ML Module Loading: SUCCESS")
    else:
        print("\n💥 Advanced ML Module Loading: FAILED")
