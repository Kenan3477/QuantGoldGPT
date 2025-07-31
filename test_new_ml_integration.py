#!/usr/bin/env python3
"""
Quick integration test for Advanced Multi-Strategy ML Engine
Tests all major integration points and API endpoints
"""

import requests
import json
import time
from datetime import datetime

def test_ml_integration():
    """Test the ML engine integration with Flask app"""
    
    print("🚀 Testing Advanced Multi-Strategy ML Engine Integration")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Strategy Performance
    print("\n📊 Test 1: Strategy Performance")
    try:
        response = requests.get(f"{base_url}/api/ml/strategies/performance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Strategy performance endpoint working")
            if data.get('success'):
                performance = data.get('performance', {})
                for strategy, perf in performance.items():
                    accuracy = perf.get('accuracy', 0) * 100
                    total = perf.get('total_predictions', 0)
                    print(f"   {strategy}: {accuracy:.1f}% accuracy ({total} predictions)")
            else:
                print(f"❌ Strategy performance failed: {data.get('error')}")
        else:
            print(f"❌ Strategy performance endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Strategy performance test failed: {e}")
    
    # Test 2: Enhanced AI Signal Generation
    print("\n🧠 Test 2: Enhanced AI Signal Generation")
    try:
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "1h"
        }
        response = requests.post(
            f"{base_url}/api/ai-signals/generate", 
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced AI signal generation working")
            if data.get('success'):
                signal = data.get('signal', {})
                print(f"   Direction: {signal.get('direction', 'N/A')}")
                print(f"   Confidence: {signal.get('confidence', 0):.2f}")
                print(f"   Strength: {signal.get('strength', 0):.2f}")
                
                # Show ensemble voting if available
                ensemble = signal.get('ensemble', {})
                voting = ensemble.get('voting_breakdown', {})
                if voting:
                    print(f"   Ensemble Voting: BUY={voting.get('BUY', 0)}, HOLD={voting.get('HOLD', 0)}, SELL={voting.get('SELL', 0)}")
                
                # Show individual strategies
                analysis = signal.get('analysis', {})
                if analysis:
                    for strategy_name, strategy_data in analysis.items():
                        if isinstance(strategy_data, dict):
                            pred = strategy_data.get('signal', 'N/A')
                            conf = strategy_data.get('confidence', 0)
                            print(f"   {strategy_name.title()}: {pred} ({conf:.2f})")
            else:
                print(f"❌ AI signal generation failed: {data.get('error')}")
        else:
            print(f"❌ AI signal generation endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ AI signal generation test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Complete - {datetime.now().strftime('%H:%M:%S')}")
    print("🔗 Access the ML dashboard at: http://localhost:5000/multi-strategy-ml-dashboard")
    print("📊 Check strategy performance at: http://localhost:5000/api/ml/strategies/performance")

if __name__ == "__main__":
    test_ml_integration()
