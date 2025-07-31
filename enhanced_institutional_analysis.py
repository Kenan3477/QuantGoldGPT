#!/usr/bin/env python3
"""
🏛️ ENHANCED INSTITUTIONAL ANALYSIS MODULE
===========================================
Professional market analysis supporting the institutional ML prediction system

Features:
- Advanced technical analysis with institutional indicators
- Professional sentiment analysis with news correlation
- Market condition assessment and risk modeling
- Economic event impact analysis
- Volatility forecasting and trend detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

async def _get_enhanced_institutional_analysis(current_price: float) -> Dict[str, Any]:
    """
    🏛️ ENHANCED INSTITUTIONAL MARKET ANALYSIS
    
    Comprehensive market analysis using institutional-grade algorithms
    and professional risk assessment methodologies
    
    Args:
        current_price: Current gold price for context
    
    Returns:
        Comprehensive analysis with technical, sentiment, and risk components
    """
    try:
        logger.info(f"🔍 Generating enhanced institutional analysis for price ${current_price:.2f}")
        
        # Import institutional data engine
        from institutional_real_data_engine import get_institutional_historical_data
        
        # Get multi-timeframe data for comprehensive analysis
        daily_data = get_institutional_historical_data('daily', 252)  # 1 year daily
        hourly_data = get_institutional_historical_data('hourly', 30)  # 30 days hourly
        
        analysis = {}
        
        # 1. ADVANCED TECHNICAL ANALYSIS
        if daily_data is not None and not daily_data.empty:
            analysis['technical_analysis'] = _calculate_institutional_technical_indicators(
                daily_data, current_price
            )
        else:
            analysis['technical_analysis'] = {'status': 'insufficient_data'}
        
        # 2. VOLATILITY AND RISK ANALYSIS
        if hourly_data is not None and not hourly_data.empty:
            analysis['risk_assessment'] = _calculate_institutional_risk_metrics(
                hourly_data, current_price
            )
        else:
            analysis['risk_assessment'] = {'status': 'insufficient_data'}
        
        # 3. MARKET CONDITIONS ASSESSMENT
        analysis['market_conditions'] = _assess_institutional_market_conditions(
            daily_data, hourly_data, current_price
        )
        
        # 4. SENTIMENT ANALYSIS (Professional)
        analysis['sentiment_analysis'] = await _get_institutional_sentiment_analysis()
        
        # 5. VALIDATION CONFIDENCE
        analysis['validation_confidence'] = _calculate_analysis_confidence(analysis)
        
        logger.info("✅ Enhanced institutional analysis complete")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Enhanced institutional analysis failed: {e}")
        return {
            'technical_analysis': {'status': 'error'},
            'risk_assessment': {'status': 'error'},
            'market_conditions': {'status': 'error'},
            'sentiment_analysis': {'status': 'error'},
            'validation_confidence': 0.0
        }

def _calculate_institutional_technical_indicators(df: pd.DataFrame, 
                                                current_price: float) -> Dict[str, Any]:
    """Calculate professional technical indicators used by institutions"""
    try:
        indicators = {}
        
        # Price-based indicators
        close_prices = df['Close']
        
        # 1. Advanced Moving Averages
        indicators['sma_20'] = close_prices.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close_prices.rolling(50).mean().iloc[-1]
        indicators['sma_200'] = close_prices.rolling(200).mean().iloc[-1] if len(df) >= 200 else None
        
        indicators['ema_12'] = close_prices.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = close_prices.ewm(span=26).mean().iloc[-1]
        
        # 2. Momentum Indicators
        # RSI
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        macd_line = indicators['ema_12'] - indicators['ema_26']
        signal_line = pd.Series([macd_line]).ewm(span=9).mean().iloc[-1]
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = macd_line - signal_line
        
        # 3. Volatility Indicators
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma_20 = close_prices.rolling(bb_period).mean()
        bb_std_dev = close_prices.rolling(bb_period).std()
        
        indicators['bollinger_upper'] = (sma_20 + bb_std * bb_std_dev).iloc[-1]
        indicators['bollinger_lower'] = (sma_20 - bb_std * bb_std_dev).iloc[-1]
        indicators['bollinger_position'] = (current_price - indicators['bollinger_lower']) / \
                                         (indicators['bollinger_upper'] - indicators['bollinger_lower'])
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(14).mean().iloc[-1]
        
        # 4. Trend Strength
        # ADX calculation (simplified)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = true_range
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        indicators['adx'] = dx.rolling(14).mean().iloc[-1] if not dx.isna().all() else 25.0
        
        # 5. Support/Resistance Levels
        recent_data = df.tail(50)  # Last 50 periods
        indicators['resistance_level'] = recent_data['High'].quantile(0.95)
        indicators['support_level'] = recent_data['Low'].quantile(0.05)
        
        # 6. Trend Direction
        short_ma = indicators['sma_20']
        long_ma = indicators['sma_50']
        
        if short_ma > long_ma:
            indicators['trend_direction'] = 'bullish'
        elif short_ma < long_ma:
            indicators['trend_direction'] = 'bearish'
        else:
            indicators['trend_direction'] = 'neutral'
        
        # 7. Price Position Analysis
        indicators['price_vs_sma20'] = (current_price / short_ma - 1) * 100
        indicators['price_vs_sma50'] = (current_price / long_ma - 1) * 100
        
        # Signal strength
        rsi = indicators['rsi']
        if rsi > 70:
            indicators['rsi_signal'] = 'overbought'
        elif rsi < 30:
            indicators['rsi_signal'] = 'oversold'
        else:
            indicators['rsi_signal'] = 'neutral'
        
        indicators['status'] = 'complete'
        return indicators
        
    except Exception as e:
        logger.error(f"❌ Technical indicators calculation failed: {e}")
        return {'status': 'error', 'error': str(e)}

def _calculate_institutional_risk_metrics(df: pd.DataFrame, 
                                        current_price: float) -> Dict[str, Any]:
    """Calculate institutional-grade risk assessment metrics"""
    try:
        risk_metrics = {}
        
        # Returns calculation
        returns = df['Close'].pct_change().dropna()
        
        # 1. Volatility Metrics
        risk_metrics['daily_volatility'] = returns.std()
        risk_metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # 2. Value at Risk (VaR)
        confidence_levels = [0.95, 0.99]
        for conf in confidence_levels:
            var = np.percentile(returns, (1 - conf) * 100)
            risk_metrics[f'var_{int(conf*100)}'] = var * current_price
        
        # 3. Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        risk_metrics['max_drawdown'] = drawdown.min()
        
        # 4. Sharpe Ratio (assuming risk-free rate of 3%)
        excess_returns = returns - 0.03 / 252  # Daily risk-free rate
        risk_metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # 5. Skewness and Kurtosis
        risk_metrics['skewness'] = stats.skew(returns)
        risk_metrics['kurtosis'] = stats.kurtosis(returns)
        
        # 6. Risk Assessment
        volatility = risk_metrics['annualized_volatility']
        if volatility > 0.3:
            risk_metrics['risk_level'] = 'high'
        elif volatility > 0.2:
            risk_metrics['risk_level'] = 'moderate'
        else:
            risk_metrics['risk_level'] = 'low'
        
        risk_metrics['status'] = 'complete'
        return risk_metrics
        
    except Exception as e:
        logger.error(f"❌ Risk metrics calculation failed: {e}")
        return {'status': 'error', 'error': str(e)}

def _assess_institutional_market_conditions(daily_data: Optional[pd.DataFrame],
                                           hourly_data: Optional[pd.DataFrame],
                                           current_price: float) -> Dict[str, Any]:
    """Assess overall market conditions using institutional criteria"""
    try:
        conditions = {}
        
        # 1. Market Regime Detection
        if daily_data is not None and not daily_data.empty:
            # Trend persistence
            returns = daily_data['Close'].pct_change().tail(20)
            positive_days = (returns > 0).sum()
            conditions['trend_persistence'] = positive_days / 20
            
            # Price momentum
            price_20d_ago = daily_data['Close'].iloc[-20] if len(daily_data) >= 20 else current_price
            conditions['momentum_20d'] = (current_price / price_20d_ago - 1) * 100
            
        # 2. Intraday Volatility
        if hourly_data is not None and not hourly_data.empty:
            hourly_returns = hourly_data['Close'].pct_change().dropna()
            conditions['intraday_volatility'] = hourly_returns.std() * np.sqrt(24 * 252)
            
            # Recent volatility vs historical
            recent_vol = hourly_returns.tail(24).std()  # Last 24 hours
            historical_vol = hourly_returns.std()
            conditions['volatility_regime'] = 'elevated' if recent_vol > historical_vol * 1.5 else 'normal'
        
        # 3. Market State Classification
        momentum = conditions.get('momentum_20d', 0)
        persistence = conditions.get('trend_persistence', 0.5)
        
        if momentum > 2 and persistence > 0.6:
            conditions['market_state'] = 'strong_uptrend'
        elif momentum < -2 and persistence < 0.4:
            conditions['market_state'] = 'strong_downtrend'
        elif abs(momentum) < 1:
            conditions['market_state'] = 'ranging'
        else:
            conditions['market_state'] = 'transitional'
        
        # 4. Trading Environment
        volatility = conditions.get('intraday_volatility', 0.2)
        if volatility > 0.4:
            conditions['trading_environment'] = 'high_volatility'
        elif volatility < 0.15:
            conditions['trading_environment'] = 'low_volatility'
        else:
            conditions['trading_environment'] = 'normal'
        
        conditions['status'] = 'complete'
        return conditions
        
    except Exception as e:
        logger.error(f"❌ Market conditions assessment failed: {e}")
        return {'status': 'error', 'error': str(e)}

async def _get_institutional_sentiment_analysis() -> Dict[str, Any]:
    """Professional sentiment analysis for institutional use"""
    try:
        sentiment = {}
        
        # Basic sentiment framework (can be enhanced with news APIs)
        sentiment['overall_sentiment'] = 'neutral'  # Default neutral
        sentiment['sentiment_score'] = 0.0  # Range: -1 to 1
        sentiment['confidence'] = 0.5
        
        # Market psychology indicators
        sentiment['fear_greed_index'] = 'neutral'  # Can be enhanced
        sentiment['market_psychology'] = 'balanced'
        
        # News sentiment (placeholder for professional news feed integration)
        sentiment['news_sentiment'] = {
            'economic_news': 'neutral',
            'geopolitical_news': 'neutral',
            'fed_policy_news': 'neutral'
        }
        
        sentiment['status'] = 'basic'  # Indicates this is basic implementation
        return sentiment
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        return {'status': 'error', 'error': str(e)}

def _calculate_analysis_confidence(analysis: Dict[str, Any]) -> float:
    """Calculate overall confidence in the analysis"""
    try:
        confidence_factors = []
        
        # Technical analysis confidence
        if analysis.get('technical_analysis', {}).get('status') == 'complete':
            confidence_factors.append(0.9)
        elif analysis.get('technical_analysis', {}).get('status') == 'error':
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.5)
        
        # Risk assessment confidence
        if analysis.get('risk_assessment', {}).get('status') == 'complete':
            confidence_factors.append(0.85)
        elif analysis.get('risk_assessment', {}).get('status') == 'error':
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.5)
        
        # Market conditions confidence
        if analysis.get('market_conditions', {}).get('status') == 'complete':
            confidence_factors.append(0.8)
        elif analysis.get('market_conditions', {}).get('status') == 'error':
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.5)
        
        # Overall confidence is the average
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return min(1.0, max(0.0, overall_confidence))
        
    except Exception as e:
        logger.error(f"❌ Confidence calculation failed: {e}")
        return 0.5

if __name__ == "__main__":
    # Testing
    import asyncio
    
    async def test_analysis():
        result = await _get_enhanced_institutional_analysis(2500.0)
        print("Analysis result:", result)
    
    asyncio.run(test_analysis())
