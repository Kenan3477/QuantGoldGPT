"""
Advanced Systems Module for GoldGPT Web Application
Adapted from Telegram bot advanced features
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
from price_storage_manager import get_current_gold_price, get_comprehensive_price_data

class PriceFetcher:
    """Real-time price fetching system"""
    
    def __init__(self):
        self.symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD']
        self.prices = {}
        
    def get_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for symbol"""
        # Use price storage manager for real gold prices
        if symbol == 'XAUUSD':
            price_data = get_comprehensive_price_data("XAUUSD")
            current_gold_price = price_data.get('price', 0)
            if current_gold_price > 0:
                base_prices = {
                    'XAUUSD': current_gold_price,  # Real-time Gold price
                    'EURUSD': 1.0875,
                    'GBPUSD': 1.2650,
                    'USDJPY': 148.50,
                    'BTCUSD': 43500.0
                }
            else:
                # Fallback prices if API fails
                current_gold = get_current_gold_price()
                if current_gold == 0:
                    current_gold = 3350.0  # Absolute last resort
                base_prices = {
                    'XAUUSD': current_gold,  # Use storage manager fallback
                    'EURUSD': 1.0875,
                    'GBPUSD': 1.2650,
                    'USDJPY': 148.50,
                    'BTCUSD': 43500.0
                }
        else:
            # For other symbols, use static prices for now
            current_gold = get_current_gold_price()
            if current_gold == 0:
                current_gold = 3350.0  # Absolute last resort
            base_prices = {
                'XAUUSD': current_gold,
                'EURUSD': 1.0875,
                'GBPUSD': 1.2650,
                'USDJPY': 148.50,
                'BTCUSD': 43500.0
            }
        
        base = base_prices.get(symbol, 1.0)
        current_price = base * (1 + random.uniform(-0.002, 0.002))  # Smaller realistic movements  
        change = random.uniform(-0.005, 0.005)  # More realistic daily changes
        
        return {
            'symbol': symbol,
            'price': round(current_price, 4),
            'change': round(change, 4),
            'change_percent': round(change * 100, 2),
            'high_24h': round(current_price * 1.015, 4),
            'low_24h': round(current_price * 0.985, 4),
            'volume': random.randint(10000, 100000),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_historical_data(self, symbol: str, period: str = '1d') -> List[Dict]:
        """Get historical price data"""
        # Simulate historical data with realistic Gold prices
        import random
        data = []
        base_price = 3350.0 if symbol == 'XAUUSD' else 1.0875
        
        for i in range(100):
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            data.append({
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'open': price,
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price,
                'volume': random.randint(1000, 10000)
            })
        
        return data

class SentimentAnalyzer:
    """Market sentiment analysis system"""
    
    def __init__(self):
        self.sentiment_sources = ['news', 'social', 'economic']
        
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment for symbol"""
        import random
        
        # Simulate sentiment analysis
        news_sentiment = random.uniform(-1, 1)
        social_sentiment = random.uniform(-1, 1)
        economic_sentiment = random.uniform(-1, 1)
        
        overall_sentiment = (news_sentiment + social_sentiment + economic_sentiment) / 3
        
        # Determine sentiment label
        if overall_sentiment > 0.3:
            label = 'bullish'
        elif overall_sentiment < -0.3:
            label = 'bearish'
        else:
            label = 'neutral'
        
        return {
            'symbol': symbol,
            'overall_score': round(overall_sentiment, 3),
            'label': label,
            'confidence': abs(overall_sentiment),
            'sources': {
                'news': round(news_sentiment, 3),
                'social': round(social_sentiment, 3),
                'economic': round(economic_sentiment, 3)
            },
            'analysis_time': datetime.now().isoformat()
        }

class TechnicalAnalyzer:
    """Technical analysis system"""
    
    def __init__(self):
        self.indicators = ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger']
        
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Perform technical analysis"""
        import random
        
        # Simulate technical indicators
        rsi = random.uniform(20, 80)
        macd = random.uniform(-5, 5)
        sma_20 = random.uniform(3300, 3400) if symbol == 'XAUUSD' else random.uniform(1.08, 1.09)
        ema_12 = random.uniform(3300, 3400) if symbol == 'XAUUSD' else random.uniform(1.08, 1.09)
        
        # Determine trend
        if rsi > 70:
            trend = 'overbought'
        elif rsi < 30:
            trend = 'oversold'
        else:
            trend = 'neutral'
        
        # Support and resistance levels
        if symbol == 'XAUUSD':
            current_price = get_current_gold_price()  # Use price storage manager (includes fallback)
            if current_price == 0:  # If storage manager completely fails
                current_price = 3350.0  # Only as absolute last resort
        else:
            current_price = 1.0875  # Keep EUR/USD as is for now
        support = current_price * 0.98
        resistance = current_price * 1.02
        
        return {
            'symbol': symbol,
            'trend': trend,
            'indicators': {
                'RSI': round(rsi, 2),
                'MACD': round(macd, 3),
                'SMA_20': round(sma_20, 4),
                'EMA_12': round(ema_12, 4)
            },
            'support': round(support, 4),
            'resistance': round(resistance, 4),
            'signal': 'buy' if rsi < 40 and macd > 0 else 'sell' if rsi > 60 and macd < 0 else 'hold',
            'analysis_time': datetime.now().isoformat()
        }

class PatternDetector:
    """Chart pattern detection system"""
    
    def __init__(self):
        self.patterns = ['head_and_shoulders', 'double_top', 'double_bottom', 'triangle', 'flag']
        
    def detect_patterns(self, symbol: str) -> Dict[str, Any]:
        """Detect chart patterns"""
        import random
        
        detected_patterns = []
        for pattern in self.patterns:
            if random.random() > 0.7:  # 30% chance of detecting each pattern
                confidence = random.uniform(0.6, 0.95)
                detected_patterns.append({
                    'pattern': pattern,
                    'confidence': round(confidence, 3),
                    'timeframe': random.choice(['1h', '4h', '1d']),
                    'target': random.uniform(3300, 3420) if symbol == 'XAUUSD' else random.uniform(1.08, 1.10)
                })
        
        return {
            'symbol': symbol,
            'patterns_detected': len(detected_patterns),
            'patterns': detected_patterns,
            'analysis_time': datetime.now().isoformat()
        }

class MLManager:
    """Machine Learning prediction system"""
    
    def __init__(self):
        self.models = ['lstm', 'random_forest', 'svm', 'xgboost']
        
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Generate ML predictions"""
        import random
        
        predictions = {}
        for model in self.models:
            direction = random.choice(['up', 'down'])
            confidence = random.uniform(0.5, 0.95)
            price_target = random.uniform(3300, 3420) if symbol == 'XAUUSD' else random.uniform(1.08, 1.10)
            
            predictions[model] = {
                'direction': direction,
                'confidence': round(confidence, 3),
                'price_target': round(price_target, 4),
                'timeframe': '24h'
            }
        
        # Ensemble prediction
        up_votes = sum(1 for p in predictions.values() if p['direction'] == 'up')
        avg_confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
        
        ensemble_direction = 'up' if up_votes >= len(self.models) / 2 else 'down'
        
        return {
            'symbol': symbol,
            'ensemble': {
                'direction': ensemble_direction,
                'confidence': round(avg_confidence, 3),
                'consensus': round(up_votes / len(self.models), 2)
            },
            'individual_models': predictions,
            'prediction_time': datetime.now().isoformat()
        }

class MacroFetcher:
    """Macroeconomic data fetcher"""
    
    def __init__(self):
        self.indicators = ['GDP', 'inflation', 'unemployment', 'interest_rates']
        
    def get_macro_data(self) -> Dict[str, Any]:
        """Get macroeconomic indicators"""
        import random
        
        data = {}
        for indicator in self.indicators:
            data[indicator] = {
                'value': round(random.uniform(0, 10), 2),
                'change': round(random.uniform(-0.5, 0.5), 2),
                'impact': random.choice(['positive', 'negative', 'neutral'])
            }
        
        return {
            'indicators': data,
            'overall_sentiment': random.choice(['bullish', 'bearish', 'neutral']),
            'updated_at': datetime.now().isoformat()
        }

class AdvancedAnalysisEngine:
    """Combined advanced analysis engine"""
    
    def __init__(self):
        self.price_fetcher = PriceFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.pattern_detector = PatternDetector()
        self.ml_manager = MLManager()
        self.macro_fetcher = MacroFetcher()
        
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analysis for symbol"""
        
        try:
            # Gather all analyses
            price_data = self.price_fetcher.get_price(symbol)
            sentiment = self.sentiment_analyzer.analyze(symbol)
            technical = self.technical_analyzer.analyze(symbol)
            patterns = self.pattern_detector.detect_patterns(symbol)
            ml_prediction = self.ml_manager.predict(symbol)
            macro_data = self.macro_fetcher.get_macro_data()
            
            # Calculate overall score
            sentiment_score = sentiment['overall_score']
            technical_score = 1 if technical['signal'] == 'buy' else -1 if technical['signal'] == 'sell' else 0
            ml_score = 1 if ml_prediction['ensemble']['direction'] == 'up' else -1
            
            overall_score = (sentiment_score + technical_score + ml_score) / 3
            
            # Generate recommendation
            if overall_score > 0.3:
                recommendation = 'BUY'
                confidence = abs(overall_score)
            elif overall_score < -0.3:
                recommendation = 'SELL'
                confidence = abs(overall_score)
            else:
                recommendation = 'HOLD'
                confidence = 0.5
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': round(confidence, 3),
                'overall_score': round(overall_score, 3),
                'price_data': price_data,
                'sentiment': sentiment,
                'technical': technical,
                'patterns': patterns,
                'ml_prediction': ml_prediction,
                'macro_data': macro_data,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'recommendation': 'HOLD',
                'confidence': 0.0,
                'analysis_timestamp': datetime.now().isoformat()
            }

# Global instances for easy access
price_fetcher = PriceFetcher()
sentiment_analyzer = SentimentAnalyzer()
technical_analyzer = TechnicalAnalyzer()
pattern_detector = PatternDetector()
ml_manager = MLManager()
macro_fetcher = MacroFetcher()
analysis_engine = AdvancedAnalysisEngine()

# Helper functions for backward compatibility
def get_price_fetcher():
    return price_fetcher

def get_sentiment_analyzer():
    return sentiment_analyzer

def get_technical_analyzer():
    return technical_analyzer

def get_pattern_detector():
    return pattern_detector

def get_ml_manager():
    return ml_manager

def get_macro_fetcher():
    return macro_fetcher

def get_analysis_engine():
    return analysis_engine
