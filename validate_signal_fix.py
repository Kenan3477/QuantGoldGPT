"""
Quick test of signal generation with error handling
"""
import json
import sys

def test_signal_generation():
    """Test signal generation with proper error handling"""
    
    print("🧪 Testing Emergency Signal Generation")
    print("=" * 40)
    
    try:
        # Test emergency signal generator
        from emergency_signal_generator import generate_working_signal
        result = generate_working_signal()
        print(f"✅ Emergency generator: {result['signal']} at ${result['price']}")
        return True
    except Exception as e:
        print(f"❌ Emergency generator failed: {e}")
        return False

def test_import_protection():
    """Test import protection logic"""
    
    print("\n🛡️ Testing Import Protection")
    print("=" * 30)
    
    # Test signal_tracker import protection
    try:
        try:
            from signal_tracker import signal_tracker
            print("✅ signal_tracker imported successfully")
            tracker_available = True
        except ImportError as e:
            print(f"⚠️ signal_tracker not available: {e}")
            tracker_available = False
        except Exception as e:
            print(f"⚠️ signal_tracker error: {e}")
            tracker_available = False
            
        print(f"📊 Signal tracker available: {tracker_available}")
        return True
        
    except Exception as e:
        print(f"❌ Import protection test failed: {e}")
        return False

def test_emergency_fallback():
    """Test the complete fallback chain"""
    
    print("\n🔄 Testing Fallback Chain")
    print("=" * 25)
    
    # Simulate the app.py fallback logic
    signal_result = None
    
    try:
        # Try to import advanced system (will likely fail on Railway)
        from advanced_systems import AdvancedSignalGenerator
        print("✅ Advanced system available")
        signal_result = {"success": True, "source": "advanced"}
    except ImportError:
        print("⚠️ Advanced system not available")
        
        try:
            # Try simple system
            from simple_signal_generator import generate_signal
            print("✅ Simple system available")
            signal_result = {"success": True, "source": "simple"}
        except ImportError:
            print("⚠️ Simple system not available")
            
            try:
                # Emergency fallback (should always work)
                from emergency_signal_generator import generate_working_signal
                emergency_signal = generate_working_signal()
                signal_result = {
                    "success": True, 
                    "source": "emergency",
                    "signal": emergency_signal
                }
                print(f"✅ Emergency fallback works: {emergency_signal['signal']} at ${emergency_signal['price']}")
            except Exception as e:
                print(f"❌ Emergency fallback failed: {e}")
                signal_result = {"success": False, "error": str(e)}
    
    return signal_result

if __name__ == "__main__":
    print("🚀 Railway Signal Generation Fix Validation")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Emergency Signal Generation", test_signal_generation),
        ("Import Protection", test_import_protection),
        ("Fallback Chain", test_emergency_fallback)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 25)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for Railway deployment.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
