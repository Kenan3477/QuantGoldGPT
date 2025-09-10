#!/usr/bin/env python3
"""
Quick test of the Advanced ML Prediction System
"""

print("🧪 Testing Advanced ML Prediction System...")

try:
    print("📦 Importing dependencies...")
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    print("✅ Core ML dependencies imported successfully")
    
    print("🤖 Testing ML Prediction Engine...")
    from advanced_ml_predictions import get_ml_price_predictions, get_ml_analysis_summary
    print("✅ ML prediction module imported successfully")
    
    print("🎯 Generating test predictions...")
    predictions = get_ml_price_predictions(['1H', '4H'])
    print(f"✅ Generated predictions for {len(predictions)} timeframes")
    
    print("📊 Testing analysis summary...")
    summary = get_ml_analysis_summary()
    print(f"✅ Analysis summary status: {summary.get('status', 'Unknown')}")
    
    # Display sample prediction
    if predictions:
        sample_timeframe = list(predictions.keys())[0]
        sample_pred = predictions[sample_timeframe]
        print(f"\n📈 Sample {sample_timeframe} prediction:")
        print(f"   Signal: {sample_pred['signal']}")
        print(f"   Confidence: {sample_pred['confidence']}")
        print(f"   Current Price: ${sample_pred['current_price']:,.2f}")
        print(f"   Target 1: ${sample_pred['targets']['target_1']:,.2f}")
        print(f"   Stop Loss: ${sample_pred['stop_loss']:,.2f}")
    
    print("\n🎉 ML Prediction System Test PASSED!")
    
except Exception as e:
    print(f"❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
