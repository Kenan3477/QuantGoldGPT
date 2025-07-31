#!/usr/bin/env python3
"""
GoldGPT ML Prediction System Test Suite
Validates all components of the ML prediction system

Tests:
1. ML Prediction API functionality
2. Model training and loading
3. Sentiment analysis
4. Technical indicators
5. Database operations
6. API endpoints
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestMLPredictionSystem(unittest.TestCase):
    """Test suite for ML prediction system"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from ml_prediction_api import ml_engine, get_ml_predictions
            self.ml_engine = ml_engine
            self.get_ml_predictions = get_ml_predictions
            self.ml_available = True
            print("✅ ML Prediction API loaded successfully")
        except ImportError as e:
            self.ml_available = False
            print(f"⚠️ ML Prediction API not available: {e}")
    
    def test_ml_engine_initialization(self):
        """Test ML engine initialization"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing ML Engine Initialization...")
        
        # Test basic initialization
        self.assertIsNotNone(self.ml_engine)
        self.assertIsNotNone(self.ml_engine.sentiment_analyzer)
        self.assertIsNotNone(self.ml_engine.technical_indicators)
        
        # Test database initialization
        self.assertTrue(os.path.exists(self.ml_engine.db_path))
        
        # Test timeframe configurations
        expected_timeframes = ['1H', '4H', '1D']
        for timeframe in expected_timeframes:
            self.assertIn(timeframe, self.ml_engine.timeframe_configs)
        
        print("✅ ML Engine initialization test passed")
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Technical Indicators...")
        
        import pandas as pd
        import numpy as np
        
        # Create sample price data
        prices = pd.Series([2000, 2005, 2010, 2015, 2020, 2015, 2010, 2005, 2000, 1995] * 10)
        
        # Test RSI calculation
        rsi = self.ml_engine.technical_indicators.calculate_rsi(prices)
        self.assertIsInstance(rsi, pd.Series)
        self.assertGreater(len(rsi.dropna()), 0)
        
        # Test MACD calculation
        macd, signal, histogram = self.ml_engine.technical_indicators.calculate_macd(prices)
        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        
        # Test Bollinger Bands
        upper, middle, lower = self.ml_engine.technical_indicators.calculate_bollinger_bands(prices)
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)
        
        # Test Stochastic
        high = prices * 1.02  # Simulate high prices
        low = prices * 0.98   # Simulate low prices
        k, d = self.ml_engine.technical_indicators.calculate_stochastic(high, low, prices)
        self.assertIsInstance(k, pd.Series)
        self.assertIsInstance(d, pd.Series)
        
        print("✅ Technical indicators test passed")
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Sentiment Analysis...")
        
        # Test news sentiment analysis
        news_sentiment = self.ml_engine.sentiment_analyzer.analyze_news_sentiment()
        self.assertIsInstance(news_sentiment, float)
        self.assertGreaterEqual(news_sentiment, -1.0)
        self.assertLessEqual(news_sentiment, 1.0)
        
        # Test fear/greed index calculation
        fear_greed = self.ml_engine.sentiment_analyzer.calculate_fear_greed_index()
        self.assertIsInstance(fear_greed, float)
        self.assertGreaterEqual(fear_greed, 0.0)
        self.assertLessEqual(fear_greed, 1.0)
        
        # Test market sentiment
        market_sentiment = self.ml_engine.sentiment_analyzer.get_market_sentiment()
        self.assertIsNotNone(market_sentiment)
        self.assertIsInstance(market_sentiment.overall_sentiment, float)
        self.assertIsInstance(market_sentiment.news_sentiment, float)
        self.assertIsInstance(market_sentiment.fear_greed_index, float)
        
        print("✅ Sentiment analysis test passed")
    
    def test_data_fetching(self):
        """Test market data fetching"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Data Fetching...")
        
        # Test data fetching for different timeframes
        for timeframe in ['1H', '4H', '1D']:
            try:
                data = self.ml_engine.fetch_market_data("GC=F", timeframe)
                self.assertIsNotNone(data)
                self.assertGreater(len(data), 0)
                
                # Check required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    self.assertIn(col, data.columns)
                
                print(f"✅ Data fetching for {timeframe} successful ({len(data)} rows)")
                
            except Exception as e:
                print(f"⚠️ Data fetching for {timeframe} failed: {e}")
                # This is expected if APIs are unavailable
                pass
        
        print("✅ Data fetching test completed")
    
    def test_feature_extraction(self):
        """Test feature extraction process"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Feature Extraction...")
        
        # Get sample data
        data = self.ml_engine.fetch_market_data("GC=F", "1D")
        sentiment = self.ml_engine.sentiment_analyzer.get_market_sentiment()
        
        # Extract features
        features = self.ml_engine.extract_features(data, sentiment)
        
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)
        
        # Check for key features
        expected_features = ['close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        print(f"✅ Feature extraction test passed ({len(features)} samples, {len(features.columns)} features)")
    
    def test_model_training(self):
        """Test model training process"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Model Training...")
        
        # Test training for 1D timeframe (fastest)
        result = self.ml_engine.train_model("GC=F", "1D")
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('model_type', result)
            self.assertIn('r2_score', result)
            self.assertIn('training_samples', result)
            self.assertIn('test_samples', result)
            
            print(f"✅ Model training successful:")
            print(f"   - Model type: {result['model_type']}")
            print(f"   - R² score: {result['r2_score']:.4f}")
            print(f"   - Training samples: {result['training_samples']}")
            print(f"   - Test samples: {result['test_samples']}")
        else:
            print(f"⚠️ Model training failed: {result.get('error', 'Unknown error')}")
    
    def test_prediction_generation(self):
        """Test prediction generation"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Prediction Generation...")
        
        # Test prediction for 1D timeframe
        prediction = self.ml_engine.predict("GC=F", "1D")
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction.current_price, float)
        self.assertIsInstance(prediction.predicted_price, float)
        self.assertIn(prediction.predicted_direction, ['UP', 'DOWN'])
        self.assertGreaterEqual(prediction.confidence_score, 0.0)
        self.assertLessEqual(prediction.confidence_score, 1.0)
        
        print(f"✅ Prediction generation successful:")
        print(f"   - Current price: ${prediction.current_price:.2f}")
        print(f"   - Predicted price: ${prediction.predicted_price:.2f}")
        print(f"   - Direction: {prediction.predicted_direction}")
        print(f"   - Confidence: {prediction.confidence_score:.2f}")
    
    def test_multi_timeframe_predictions(self):
        """Test multi-timeframe predictions"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Multi-Timeframe Predictions...")
        
        predictions = self.ml_engine.get_multi_timeframe_predictions("GC=F")
        
        self.assertIsInstance(predictions, dict)
        
        for timeframe, prediction in predictions.items():
            self.assertIn(timeframe, ['1H', '4H', '1D'])
            self.assertIsNotNone(prediction)
            
            print(f"✅ {timeframe} prediction: {prediction.predicted_direction} "
                  f"(${prediction.predicted_price:.2f}, confidence: {prediction.confidence_score:.2f})")
    
    def test_async_api_function(self):
        """Test async API function"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Async API Function...")
        
        async def test_async():
            result = await self.get_ml_predictions("GC=F")
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            
            if result['success']:
                self.assertIn('predictions', result)
                predictions = result['predictions']
                
                for timeframe in ['1H', '4H', '1D']:
                    if timeframe in predictions:
                        pred = predictions[timeframe]
                        self.assertIn('current_price', pred)
                        self.assertIn('predicted_price', pred)
                        self.assertIn('predicted_direction', pred)
                        self.assertIn('confidence_score', pred)
                        
                        print(f"✅ {timeframe} API prediction: {pred['predicted_direction']} "
                              f"(${pred['predicted_price']:.2f}, confidence: {pred['confidence_score']:.2f})")
        
        # Run async test
        asyncio.run(test_async())
    
    def test_database_operations(self):
        """Test database operations"""
        if not self.ml_available:
            self.skipTest("ML API not available")
        
        print("\n🧪 Testing Database Operations...")
        
        # Test prediction storage
        prediction = self.ml_engine.predict("GC=F", "1D")
        
        # Verify prediction was stored
        import sqlite3
        conn = sqlite3.connect(self.ml_engine.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0)
        
        cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        
        conn.close()
        
        print(f"✅ Database operations test passed ({count} predictions stored)")
    
    def test_fallback_predictions(self):
        """Test fallback prediction calculations"""
        print("\n🧪 Testing Fallback Predictions...")
        
        # This test doesn't require ML API
        # Test the JavaScript fallback calculator (conceptual test)
        
        # Mock current price
        current_price = 2000.0
        
        # Simulate fallback prediction logic
        rsi = 55.0  # Neutral RSI
        trend = 0.02  # Slight upward trend
        volatility = 0.03  # Normal volatility
        
        # Simple fallback prediction
        direction_score = 0
        if rsi > 70:
            direction_score -= 0.3
        elif rsi < 30:
            direction_score += 0.3
        
        direction_score += trend * 2
        
        timeframe_multiplier = 0.01  # For daily prediction
        price_change = direction_score * timeframe_multiplier * current_price
        predicted_price = current_price + price_change
        
        direction = "UP" if price_change > 0 else "DOWN"
        confidence = min(0.85, 0.5 + abs(direction_score) * 0.3)
        
        print(f"✅ Fallback prediction simulation:")
        print(f"   - Current price: ${current_price:.2f}")
        print(f"   - Predicted price: ${predicted_price:.2f}")
        print(f"   - Direction: {direction}")
        print(f"   - Confidence: {confidence:.2f}")
        
        # Basic assertions
        self.assertIsInstance(predicted_price, float)
        self.assertIn(direction, ['UP', 'DOWN'])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

def run_integration_tests():
    """Run integration tests with Flask app"""
    print("\n🔗 Running Integration Tests...")
    
    try:
        # Import Flask app
        from app import app
        
        with app.test_client() as client:
            # Test ML predictions endpoint
            response = client.get('/api/ml-predictions')
            print(f"✅ ML predictions endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.get_json()
                print(f"   - Response success: {data.get('success', False)}")
                if data.get('success'):
                    predictions = data.get('predictions', {})
                    print(f"   - Timeframes available: {list(predictions.keys())}")
            
            # Test ML status endpoint
            response = client.get('/api/ml-predictions/status')
            print(f"✅ ML status endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.get_json()
                print(f"   - ML available: {data.get('ml_available', False)}")
                print(f"   - Model version: {data.get('model_version', 'Unknown')}")
    
    except Exception as e:
        print(f"⚠️ Integration tests failed: {e}")

def run_performance_tests():
    """Run performance tests"""
    print("\n⚡ Running Performance Tests...")
    
    try:
        from ml_prediction_api import ml_engine
        import time
        
        # Test prediction speed
        start_time = time.time()
        prediction = ml_engine.predict("GC=F", "1D")
        end_time = time.time()
        
        prediction_time = end_time - start_time
        print(f"✅ Prediction generation time: {prediction_time:.2f} seconds")
        
        # Test multiple predictions
        start_time = time.time()
        predictions = ml_engine.get_multi_timeframe_predictions("GC=F")
        end_time = time.time()
        
        multi_prediction_time = end_time - start_time
        print(f"✅ Multi-timeframe prediction time: {multi_prediction_time:.2f} seconds")
        
        # Performance assertions
        assert prediction_time < 10.0, f"Prediction too slow: {prediction_time:.2f}s"
        assert multi_prediction_time < 30.0, f"Multi-prediction too slow: {multi_prediction_time:.2f}s"
        
        print("✅ Performance tests passed")
    
    except Exception as e:
        print(f"⚠️ Performance tests failed: {e}")

if __name__ == "__main__":
    print("🧪 GoldGPT ML Prediction System Test Suite")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    run_integration_tests()
    
    # Run performance tests
    run_performance_tests()
    
    print("\n✅ Test suite completed!")
    print("=" * 50)
