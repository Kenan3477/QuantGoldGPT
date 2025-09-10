#!/usr/bin/env python3
"""
Quick test of ML endpoint
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import json

print("🧪 Testing ML Predictions Endpoint...")

with app.test_client() as client:
    print("📡 Making request to /api/ml-predictions...")
    response = client.get('/api/ml-predictions')
    
    print(f"📊 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.get_json()
        print("✅ SUCCESS! ML Predictions working")
        print(f"📈 Number of predictions: {len(data.get('predictions', []))}")
        print(f"🎯 Model status: {data.get('model_status')}")
        
        # Show first prediction
        if data.get('predictions'):
            first_pred = data['predictions'][0]
            print(f"🔥 Sample: {first_pred.get('signal')} - {first_pred.get('prediction')}")
    else:
        print("❌ ERROR Response:")
        try:
            data = response.get_json()
            print(json.dumps(data, indent=2))
        except:
            print(response.get_data(as_text=True))
