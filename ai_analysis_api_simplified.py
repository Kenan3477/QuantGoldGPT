"""
Simplified AI Analysis API Module for GoldGPT
Python 3.12+ compatible version with manual technical analysis
"""

import os
import json
import ssl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Fix SSL compatibility for Python 3.12+
try:
    if not hasattr(ssl, 'wrap_socket'):
        import ssl
        ssl.wrap_socket = ssl.SSLSocket
except Exception:
    pass

warnings.filterwarnings('ignore')

@dataclass
class TechnicalSignal:
    """Data class for technical analysis signals"""
    indicator: str
    value: float
    signal: str
    confidence: float
    timestamp: datetime

@dataclass
class SentimentData:
    """Data class for sentiment analysis"""
    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    sentiment_score: float
    confidence: float
    timestamp: datetime

@dataclass
class MLPrediction:
    """Data class for ML predictions"""
    symbol: str
    predicted_price: float
    direction: str
    confidence: float
    timeframe: str
    timestamp: datetime

@dataclass
class ComprehensiveAnalysis:
    """Complete analysis result"""
    symbol: str
    current_price: float
    technical_signals: List[TechnicalSignal]
    sentiment_data: SentimentData
    ml_prediction: MLPrediction
    overall_recommendation: str
    confidence_score: float
    timestamp: datetime

class SimplifiedDataFetcher:
    """Simplified data fetcher without problematic dependencies"""
    
    def get_price_data(self, symbol: str = "XAUUSD", period_days: int = 30) -> Optional[pd.DataFrame]:
        """Get synthetic price data for testing"""
        try:
            # Use real-time price as base for synthetic data
            from price_storage_manager import get_current_gold_price
            
            # Generate realistic synthetic data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            
            if 'XAU' in symbol:
                base_price = get_current_gold_price() or 3350.0  # Use real gold price
            elif 'EUR' in symbol:
                base_price = 1.0875
            else:
                base_price = 100.0
                
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                change = np.random.normal(0, base_price * 0.002)  # 0.2% volatility
                current_price += change
                current_price = max(base_price * 0.9, min(base_price * 1.1, current_price))
                prices.append(current_price)
            
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                high = price + np.random.uniform(0, base_price * 0.005)
                low = price - np.random.uniform(0, base_price * 0.005)
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'Open': price,
                    'High': high,
                    'Low': low,
                    'Close': price,
                    'Volume': volume
                })
            
            return pd.DataFrame(data, index=dates)
        except Exception as e:
            print(f"Error generating price data: {e}")
            return None

class SimplifiedTechnicalAnalyzer:
    """Simplified technical analysis without external dependencies"""
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI manually"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception:
            return 50.0
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD manually"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
        except Exception:
            return 0.0, 0.0
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands manually"""
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1])
        except Exception:
            from price_storage_manager import get_current_gold_price
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else get_current_gold_price() or 3350.0
            return current_price * 1.02, current_price, current_price * 0.98
    
    def analyze_technical_indicators(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate technical analysis signals"""
        signals = []
        
        try:
            prices = data['Close']
            current_price = float(prices.iloc[-1])
            
            # RSI Analysis
            rsi = self.calculate_rsi(prices)
            rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            signals.append(TechnicalSignal(
                indicator="RSI",
                value=rsi,
                signal=rsi_signal,
                confidence=0.8 if rsi_signal != "neutral" else 0.5,
                timestamp=datetime.now()
            ))
            
            # MACD Analysis
            macd_line, macd_signal = self.calculate_macd(prices)
            macd_trend = "bullish" if macd_line > macd_signal else "bearish"
            signals.append(TechnicalSignal(
                indicator="MACD",
                value=macd_line - macd_signal,
                signal=macd_trend,
                confidence=0.7,
                timestamp=datetime.now()
            ))
            
            # Bollinger Bands Analysis
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(prices)
            if current_price > bb_upper:
                bb_signal = "overbought"
                bb_confidence = 0.8
            elif current_price < bb_lower:
                bb_signal = "oversold"
                bb_confidence = 0.8
            else:
                bb_signal = "neutral"
                bb_confidence = 0.5
            
            signals.append(TechnicalSignal(
                indicator="Bollinger_Bands",
                value=(current_price - bb_lower) / (bb_upper - bb_lower),
                signal=bb_signal,
                confidence=bb_confidence,
                timestamp=datetime.now()
            ))
            
            # Moving Average Analysis
            sma_20 = prices.rolling(window=20).mean().iloc[-1]
            ma_signal = "bullish" if current_price > sma_20 else "bearish"
            signals.append(TechnicalSignal(
                indicator="SMA_20",
                value=float(sma_20),
                signal=ma_signal,
                confidence=0.6,
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            print(f"Error in technical analysis: {e}")
        
        return signals

class SimplifiedSentimentAnalyzer:
    """Simplified sentiment analysis"""
    
    def analyze_sentiment(self, symbol: str) -> SentimentData:
        """Generate simplified sentiment analysis"""
        try:
            # Simulate sentiment analysis with realistic but random data
            base_sentiment = np.random.normal(0, 0.3)  # Slightly bearish to bullish range
            
            # Add some symbol-specific bias
            if 'XAU' in symbol or 'GOLD' in symbol:
                base_sentiment += 0.1  # Gold tends to be slightly positive
            
            overall_sentiment = max(-1, min(1, base_sentiment))
            news_sentiment = overall_sentiment + np.random.normal(0, 0.2)
            social_sentiment = overall_sentiment + np.random.normal(0, 0.3)
            
            sentiment_score = (overall_sentiment + 1) * 50  # Convert to 0-100 scale
            confidence = 0.6 + abs(overall_sentiment) * 0.3  # Higher confidence for extreme sentiments
            
            return SentimentData(
                overall_sentiment=overall_sentiment,
                news_sentiment=max(-1, min(1, news_sentiment)),
                social_sentiment=max(-1, min(1, social_sentiment)),
                sentiment_score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return SentimentData(
                overall_sentiment=0.0,
                news_sentiment=0.0,
                social_sentiment=0.0,
                sentiment_score=50.0,
                confidence=0.5,
                timestamp=datetime.now()
            )

class SimplifiedMLPredictor:
    """Simplified ML predictions"""
    
    def generate_prediction(self, symbol: str, data: pd.DataFrame) -> MLPrediction:
        """Generate simplified ML prediction"""
        try:
            current_price = float(data['Close'].iloc[-1])
            
            # Simple trend analysis
            short_ma = data['Close'].rolling(window=5).mean().iloc[-1]
            long_ma = data['Close'].rolling(window=20).mean().iloc[-1]
            
            # Price change prediction based on trend
            trend_factor = (short_ma - long_ma) / long_ma
            volatility = data['Close'].pct_change().std()
            
            # Predict price change
            predicted_change = trend_factor + np.random.normal(0, volatility)
            predicted_price = current_price * (1 + predicted_change)
            
            # Determine direction
            if predicted_change > 0.005:
                direction = "bullish"
                confidence = min(0.9, 0.6 + abs(predicted_change) * 5)
            elif predicted_change < -0.005:
                direction = "bearish" 
                confidence = min(0.9, 0.6 + abs(predicted_change) * 5)
            else:
                direction = "neutral"
                confidence = 0.5
            
            return MLPrediction(
                symbol=symbol,
                predicted_price=predicted_price,
                direction=direction,
                confidence=confidence,
                timeframe="1D",
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            from price_storage_manager import get_current_gold_price
            current_price = get_current_gold_price() or 3350.0  # Real-time gold price with fallback
            return MLPrediction(
                symbol=symbol,
                predicted_price=current_price,
                direction="neutral",
                confidence=0.5,
                timeframe="1D",
                timestamp=datetime.now()
            )

class AdvancedAIAnalyzer:
    """Main AI analyzer class"""
    
    def __init__(self):
        self.data_fetcher = SimplifiedDataFetcher()
        self.technical_analyzer = SimplifiedTechnicalAnalyzer()
        self.sentiment_analyzer = SimplifiedSentimentAnalyzer()
        self.ml_predictor = SimplifiedMLPredictor()
    
    def get_comprehensive_analysis(self, symbol: str = "XAUUSD") -> ComprehensiveAnalysis:
        """Get complete AI analysis for a symbol"""
        try:
            # Get price data
            data = self.data_fetcher.get_price_data(symbol)
            if data is None or len(data) < 20:
                raise ValueError("Insufficient data for analysis")
            
            current_price = float(data['Close'].iloc[-1])
            
            # Perform analyses
            technical_signals = self.technical_analyzer.analyze_technical_indicators(data)
            sentiment_data = self.sentiment_analyzer.analyze_sentiment(symbol)
            ml_prediction = self.ml_predictor.generate_prediction(symbol, data)
            
            # Calculate overall recommendation
            bullish_signals = sum(1 for signal in technical_signals if signal.signal in ["bullish", "oversold"])
            bearish_signals = sum(1 for signal in technical_signals if signal.signal in ["bearish", "overbought"])
            
            if sentiment_data.overall_sentiment > 0.2:
                bullish_signals += 1
            elif sentiment_data.overall_sentiment < -0.2:
                bearish_signals += 1
            
            if ml_prediction.direction == "bullish":
                bullish_signals += 1
            elif ml_prediction.direction == "bearish":
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                overall_recommendation = "BUY"
                confidence_score = min(0.9, 0.6 + (bullish_signals - bearish_signals) * 0.1)
            elif bearish_signals > bullish_signals:
                overall_recommendation = "SELL"
                confidence_score = min(0.9, 0.6 + (bearish_signals - bullish_signals) * 0.1)
            else:
                overall_recommendation = "HOLD"
                confidence_score = 0.5
            
            return ComprehensiveAnalysis(
                symbol=symbol,
                current_price=current_price,
                technical_signals=technical_signals,
                sentiment_data=sentiment_data,
                ml_prediction=ml_prediction,
                overall_recommendation=overall_recommendation,
                confidence_score=confidence_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            # Return fallback analysis
            from price_storage_manager import get_current_gold_price
            fallback_price = get_current_gold_price() or 3350.0  # Real-time gold price with fallback
            return ComprehensiveAnalysis(
                symbol=symbol,
                current_price=fallback_price,
                technical_signals=[],
                sentiment_data=self.sentiment_analyzer.analyze_sentiment(symbol),
                ml_prediction=self.ml_predictor.generate_prediction(symbol, pd.DataFrame()),
                overall_recommendation="HOLD",
                confidence_score=0.5,
                timestamp=datetime.now()
            )

# Global analyzer instance
ai_analyzer = AdvancedAIAnalyzer()

def get_ai_analysis_sync(symbol: str = "XAUUSD") -> Dict:
    """Synchronous wrapper for AI analysis"""
    try:
        analysis = ai_analyzer.get_comprehensive_analysis(symbol)
        return {
            "success": True,
            "symbol": analysis.symbol,
            "current_price": analysis.current_price,
            "recommendation": analysis.overall_recommendation,
            "confidence": analysis.confidence_score,
            "technical_signals": [asdict(signal) for signal in analysis.technical_signals],
            "sentiment": asdict(analysis.sentiment_data),
            "ml_prediction": asdict(analysis.ml_prediction),
            "timestamp": analysis.timestamp.isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def get_ai_analysis(symbol: str = "XAUUSD") -> Dict:
    """Asynchronous AI analysis"""
    return get_ai_analysis_sync(symbol)

print("✅ Simplified AI Analysis API loaded successfully")
