#!/usr/bin/env python3
"""
Force Bearish Predictions Test
Temporarily modify market conditions to generate bearish predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def force_bearish_predictions():
    """Generate predictions with forced bearish market conditions"""
    print("🔴 Forcing Bearish Market Conditions for Testing...")
    
    # Import the enhanced ML engine
    from enhanced_ml_prediction_engine import EnhancedGoldPredictor, MarketFactors
    
    predictor = EnhancedGoldPredictor()
    
    # Create bearish market conditions
    bearish_market = MarketFactors(
        dollar_index=105.0,      # High DXY (bad for gold)
        inflation_rate=2.0,      # Low inflation 
        fed_funds_rate=5.5,      # High interest rates
        unemployment_rate=3.5,   # Low unemployment
        vix_level=25.0,          # Moderate fear (but high rates dominate)
        oil_price=75.0,          # Moderate oil
        bond_yield_10y=4.5,      # High bond yields (bad for gold)
        news_sentiment=-0.3,     # Negative sentiment
        geopolitical_tension=0.2 # Low tension
    )
    
    print(f"📊 Bearish Market Factors:")
    print(f"   DXY: {bearish_market.dollar_index} (High - Bad for Gold)")
    print(f"   Fed Funds: {bearish_market.fed_funds_rate}% (High - Bad for Gold)")
    print(f"   Bond Yield: {bearish_market.bond_yield_10y}% (High - Bad for Gold)")
    print(f"   Sentiment: {bearish_market.news_sentiment} (Negative)")
    
    # Generate predictions with bearish conditions
    current_price = 3350.70
    
    predictions = []
    for timeframe in ['1H', '4H', '1D']:
        pred = predictor._generate_enhanced_prediction(
            timeframe, current_price, bearish_market
        )
        predictions.append(pred)
        
        direction_emoji = "🔴" if pred.direction == "bearish" else "🟡" if pred.direction == "neutral" else "🟢"
        print(f"   {direction_emoji} {timeframe}: ${pred.predicted_price:.2f} ({pred.change_percent:+.3f}%) - {pred.direction.upper()}")
    
    print("\n✅ Bearish predictions generated successfully!")
    return predictions

if __name__ == "__main__":
    import asyncio
    asyncio.run(force_bearish_predictions())
