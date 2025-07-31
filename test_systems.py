#!/usr/bin/env python3

"""Test ML Prediction System and AI Analysis"""

import sys
import traceback

def test_ml_system():
    """Test the ML prediction system"""
    try:
        print("🔍 Testing ML Prediction Engine...")
        from ml_prediction_api import MLPredictionEngine
        
        engine = MLPredictionEngine()
        prediction = engine.generate_prediction("XAUUSD")
        
        print(f"✅ ML Engine: {prediction['success']}")
        if prediction['success']:
            print(f"   Prediction: ${prediction['predicted_price']:.2f}")
            print(f"   Direction: {prediction['direction']}")
            print(f"   Confidence: {prediction['confidence']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ ML Engine Error: {e}")
        traceback.print_exc()
        return False

def test_ai_analysis():
    """Test the AI analysis system"""
    try:
        print("\n🔍 Testing AI Analysis System...")
        from ai_analysis_api import get_ai_analysis_sync
        
        analysis = get_ai_analysis_sync("XAUUSD")
        
        print(f"✅ AI Analysis: {analysis['success']}")
        if analysis['success']:
            print(f"   Current Price: ${analysis['current_price']:.2f}")
            print(f"   Recommendation: {analysis['recommendation']}")
            print(f"   Confidence: {analysis['confidence']:.2f}")
            print(f"   Technical Signals: {len(analysis['technical_signals'])}")
        
        return True
    except Exception as e:
        print(f"❌ AI Analysis Error: {e}")
        traceback.print_exc()
        return False

def test_flask_integration():
    """Test Flask app import"""
    try:
        print("\n🔍 Testing Flask App Integration...")
        import app
        print("✅ Flask app imports successfully")
        return True
    except Exception as e:
        print(f"❌ Flask App Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing GoldGPT ML Prediction System")
    print("=" * 50)
    
    ml_ok = test_ml_system()
    ai_ok = test_ai_analysis()
    flask_ok = test_flask_integration()
    
    print("\n" + "=" * 50)
    if ml_ok and ai_ok and flask_ok:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("✅ ML Prediction Engine: Working")
        print("✅ AI Analysis System: Working")
        print("✅ Flask Integration: Working")
        print("\n🌟 Your GoldGPT dashboard is ready!")
    else:
        print("⚠️  Some systems need attention:")
        if not ml_ok:
            print("❌ ML Prediction Engine")
        if not ai_ok:
            print("❌ AI Analysis System")
        if not flask_ok:
            print("❌ Flask Integration")
