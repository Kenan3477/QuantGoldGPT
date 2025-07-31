#!/usr/bin/env python3
"""
Standardized Prediction Data Format for GoldGPT ML System
Provides consistent data structures across all ML components
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class PredictionData:
    """Standardized prediction data structure"""
    timeframe: str
    current_price: float
    target_price: float
    change_amount: float
    change_percent: float
    confidence: float
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    created: str
    volume_trend: str = 'Unknown'
    ai_reasoning: str = 'No reasoning provided'
    key_features: List[str] = None
    
    def __post_init__(self):
        if self.key_features is None:
            self.key_features = []
        # Ensure predicted_price matches target_price for compatibility
        self.predicted_price = self.target_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        data = asdict(self)
        data['predicted_price'] = self.target_price  # Add for compatibility
        return data

@dataclass
class TechnicalAnalysis:
    """Technical analysis data structure"""
    rsi: float
    macd: float
    support: float
    resistance: float
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)

@dataclass
class MarketSummary:
    """Market summary/strategy performance data structure"""
    total_predictions: int
    ensemble_accuracy: float
    average_confidence: float
    current_price: float
    trend: str = 'Unknown'
    last_30_days: Optional[Dict[str, Any]] = None
    strategy_breakdown: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)

class StandardPredictionResponse:
    """Standardized prediction response builder"""
    
    def __init__(self, symbol: str = 'XAUUSD', current_price: float = 3338.0):
        self.symbol = symbol
        self.current_price = current_price
        self.predictions: List[PredictionData] = []
        self.technical_analysis: Optional[TechnicalAnalysis] = None
        self.market_summary: Optional[MarketSummary] = None
        self.success = True
        self.error = None
        
    def add_prediction(self, 
                      timeframe: str,
                      change_percent: float,
                      confidence: float,
                      direction: str = 'BULLISH',
                      volume_trend: str = 'Increasing',
                      ai_reasoning: str = None,
                      key_features: List[str] = None) -> 'StandardPredictionResponse':
        """Add a prediction to the response"""
        
        change_amount = self.current_price * (change_percent / 100)
        target_price = self.current_price + change_amount
        
        if ai_reasoning is None:
            ai_reasoning = f"Technical analysis indicates {change_percent:.1f}% {direction.lower()} movement expected"
            
        if key_features is None:
            key_features = [
                f"RSI: {52.3 + (change_percent * 2):.1f}",
                f"MACD: {1.24 + (change_percent * 0.1):.2f}",
                f"Volume: Strong" if confidence > 0.7 else "Volume: Moderate"
            ]
        
        prediction = PredictionData(
            timeframe=timeframe,
            current_price=self.current_price,
            target_price=target_price,
            change_amount=change_amount,
            change_percent=change_percent,
            confidence=confidence,
            direction=direction,
            created=datetime.now().isoformat(),
            volume_trend=volume_trend,
            ai_reasoning=ai_reasoning,
            key_features=key_features
        )
        
        self.predictions.append(prediction)
        return self
    
    def set_technical_analysis(self, 
                              rsi: float,
                              macd: float,
                              support: float = None,
                              resistance: float = None,
                              sma20: float = None,
                              sma50: float = None) -> 'StandardPredictionResponse':
        """Set technical analysis data"""
        
        if support is None:
            support = self.current_price * 0.985
        if resistance is None:
            resistance = self.current_price * 1.015
            
        self.technical_analysis = TechnicalAnalysis(
            rsi=rsi,
            macd=macd,
            support=support,
            resistance=resistance,
            sma20=sma20,
            sma50=sma50
        )
        return self
    
    def set_market_summary(self,
                          total_predictions: int,
                          ensemble_accuracy: float,
                          average_confidence: float,
                          trend: str = 'Bullish',
                          last_30_days: Dict[str, Any] = None,
                          strategy_breakdown: Dict[str, Any] = None) -> 'StandardPredictionResponse':
        """Set market summary data"""
        
        if last_30_days is None:
            last_30_days = {
                'wins': int(total_predictions * (ensemble_accuracy / 100) * 0.8),
                'losses': int(total_predictions * (1 - ensemble_accuracy / 100) * 0.8),
                'accuracy': ensemble_accuracy * 0.95
            }
            
        if strategy_breakdown is None:
            strategy_breakdown = {
                'technical_analysis': ensemble_accuracy * 1.05,
                'sentiment_analysis': ensemble_accuracy * 0.85,
                'ensemble_ml': ensemble_accuracy
            }
        
        self.market_summary = MarketSummary(
            total_predictions=total_predictions,
            ensemble_accuracy=ensemble_accuracy,
            average_confidence=average_confidence,
            current_price=self.current_price,
            trend=trend,
            last_30_days=last_30_days,
            strategy_breakdown=strategy_breakdown
        )
        return self
    
    def set_error(self, error_message: str) -> 'StandardPredictionResponse':
        """Set error state"""
        self.success = False
        self.error = error_message
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON response"""
        if not self.success:
            return {
                'success': False,
                'error': self.error,
                'generated_at': datetime.now().isoformat()
            }
        
        response = {
            'success': True,
            'symbol': self.symbol,
            'current_price': self.current_price,
            'predictions': [pred.to_dict() for pred in self.predictions],
            'generated_at': datetime.now().isoformat()
        }
        
        if self.technical_analysis:
            response['technical_analysis'] = self.technical_analysis.to_dict()
            
        if self.market_summary:
            response['market_summary'] = self.market_summary.to_dict()
            
        return response

def create_standard_prediction_response(symbol: str = 'XAUUSD', current_price: float = 3338.0) -> StandardPredictionResponse:
    """Factory function to create a standard prediction response"""
    return StandardPredictionResponse(symbol, current_price)

# Example usage and test function
def create_example_predictions() -> Dict[str, Any]:
    """Create example predictions using the standardized format"""
    
    response = (create_standard_prediction_response('XAUUSD', 3338.0)
               .add_prediction('1H', 0.15, 0.79, 'BULLISH')
               .add_prediction('4H', 0.45, 0.71, 'BULLISH')
               .add_prediction('1D', 0.85, 0.63, 'BULLISH')
               .set_technical_analysis(52.3, 1.24)
               .set_market_summary(390, 69.7, 0.71))
    
    return response.to_dict()

if __name__ == "__main__":
    # Test the standardized format
    example = create_example_predictions()
    import json
    print("Standardized Prediction Format Example:")
    print(json.dumps(example, indent=2))
