# GoldGPT ML Prediction System Documentation

## Overview

The GoldGPT ML Prediction System provides real-time, machine learning-driven gold price forecasts using multiple data sources and advanced ensemble methods. This system delivers actionable insights through multi-timeframe predictions with confidence metrics.

## Architecture

### Backend Components

#### 1. ML Prediction Engine (`ml_prediction_api.py`)
- **Purpose**: Core ML processing and model management
- **Features**:
  - Multi-timeframe predictions (1H, 4H, 1D)
  - Ensemble models (RandomForest + GradientBoosting)
  - Self-contained training (<100MB total model size)
  - Real-time sentiment analysis integration
  - Automatic model retraining during off-hours

#### 2. Technical Indicators Module
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility-based bands
- **Stochastic Oscillator**: Momentum indicator
- **Moving Averages**: Trend identification

#### 3. Sentiment Analysis Engine
- **News Sentiment**: RSS feed analysis from financial sources
- **Fear/Greed Index**: VIX and market volatility-based calculations
- **Social Sentiment**: Placeholder for Twitter API integration
- **Composite Scoring**: Weighted sentiment combination

### Frontend Components

#### 1. ML Prediction Manager (`gold-ml-prediction-manager.js`)
- **Purpose**: Frontend integration and UI management
- **Features**:
  - Real-time prediction display
  - Fallback calculations for offline operation
  - Interactive confidence visualization
  - Seamless UI integration

#### 2. Prediction Panel UI
- **Multi-timeframe cards**: 1H, 4H, 1D predictions
- **Confidence scoring**: Visual progress bars
- **Direction indicators**: Bullish/Bearish signals
- **Factor analysis**: Detailed prediction reasoning

## License-Compliant Data Sources

### Primary Data Sources
1. **Yahoo Finance API** (Free tier)
   - Historical OHLCV data
   - Real-time price feeds
   - Gold futures (GC=F) data

2. **Public RSS Feeds**
   - Reuters Business News
   - Bloomberg Markets
   - CNN Money
   - MarketWatch

3. **Market Indicators**
   - VIX (Volatility Index)
   - SPY volatility calculations
   - Put/call ratio approximations

### Data Processing Pipeline
1. **Raw Data Collection**: Automated fetching from multiple sources
2. **Data Validation**: Range checking and anomaly detection
3. **Feature Engineering**: Technical indicator calculations
4. **Sentiment Processing**: News content analysis
5. **Model Input Preparation**: Feature scaling and window creation

## Model Training Process

### 1. Data Preparation
```python
# Feature extraction from market data
features = extract_features(market_data, sentiment_data)

# Technical indicators
- RSI (14-period)
- MACD (12,26,9)
- Bollinger Bands (20,2)
- Stochastic (14,3)

# Price momentum
- 1-day, 5-day, 20-day price changes
- Moving averages (5,20,50)
- Volatility measures

# Sentiment features
- News sentiment score
- Fear/greed index
- Overall market sentiment
```

### 2. Model Architecture
```python
# Ensemble approach
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
}

# Model selection based on R² score
best_model = max(models, key=lambda x: r2_score(test_data, models[x].predict(test_data)))
```

### 3. Training Schedule
- **Initial Training**: On first run with 2 years of data
- **Retraining**: Daily during off-hours (2 AM - 6 AM UTC)
- **Validation**: 80/20 train/test split with time-based splitting
- **Model Persistence**: Joblib serialization for quick loading

## API Endpoints

### 1. Get ML Predictions
```http
GET /api/ml-predictions
```
**Response:**
```json
{
  "success": true,
  "predictions": {
    "1H": {
      "current_price": 2000.50,
      "predicted_price": 2005.25,
      "predicted_direction": "UP",
      "confidence_score": 0.75,
      "prediction_factors": {
        "rsi": 55.2,
        "macd_signal": "BULLISH",
        "sentiment_score": 0.15
      }
    },
    "4H": { ... },
    "1D": { ... }
  },
  "model_version": "v1.0"
}
```

### 2. Train Models
```http
POST /api/ml-predictions/train
```
**Response:**
```json
{
  "success": true,
  "message": "Model training started in background",
  "status": "training"
}
```

### 3. Get System Status
```http
GET /api/ml-predictions/status
```
**Response:**
```json
{
  "success": true,
  "ml_available": true,
  "model_version": "v1.0",
  "models": {
    "1H": {"trained": true, "last_prediction": "2025-07-18T10:30:00Z"},
    "4H": {"trained": true, "last_prediction": "2025-07-18T10:30:00Z"},
    "1D": {"trained": true, "last_prediction": "2025-07-18T10:30:00Z"}
  }
}
```

## Prediction Factors

### Technical Analysis Factors
- **RSI (0-100)**: Momentum indicator
  - <30: Oversold (potential buy signal)
  - >70: Overbought (potential sell signal)
- **MACD Signal**: Trend direction
  - BULLISH: MACD line above signal line
  - BEARISH: MACD line below signal line
- **Bollinger Band Position**: Price relative to bands
  - Near upper band: Resistance level
  - Near lower band: Support level

### Sentiment Factors
- **News Sentiment (-1 to 1)**: Media sentiment analysis
- **Fear/Greed Index (0-1)**: Market psychology
- **Overall Sentiment**: Weighted combination of all factors

### Confidence Scoring
```python
# Base confidence from model uncertainty
base_confidence = 0.6  # Default for ensemble models

# Sentiment boost (up to 30%)
sentiment_boost = min(0.3, abs(sentiment_score) * 0.3)

# Final confidence (capped at 95%)
final_confidence = min(0.95, base_confidence + sentiment_boost)
```

## Fallback Mechanisms

### 1. API Outage Handling
- **Local Calculations**: JavaScript-based technical analysis
- **Synthetic Data**: Historical pattern simulation
- **Graceful Degradation**: Reduced feature set with appropriate confidence adjustment

### 2. Model Unavailability
- **Simplified Models**: Linear regression fallback
- **Rule-Based Predictions**: Technical indicator signals
- **Conservative Confidence**: Lower confidence scores for fallback predictions

## Performance Metrics

### Model Evaluation
- **R² Score**: Explained variance (target: >0.6)
- **Direction Accuracy**: Percentage of correct direction predictions
- **Mean Absolute Error**: Average prediction error
- **Confidence Calibration**: Relationship between confidence and accuracy

### System Monitoring
- **API Response Time**: Target <2 seconds
- **Model Loading Time**: Target <5 seconds
- **Memory Usage**: Target <100MB per model
- **Update Frequency**: 5-minute intervals

## Security Considerations

### Data Protection
- **No Personal Data**: Only public market data processed
- **API Rate Limiting**: Prevent abuse of prediction endpoints
- **Input Validation**: Sanitize all user inputs
- **Error Handling**: No sensitive information in error messages

### Model Security
- **Local Training**: No external ML service dependencies
- **Model Versioning**: Track model changes and rollback capability
- **Access Control**: Admin-only endpoints for training and configuration

## Installation and Setup

### 1. Dependencies
```bash
pip install -r requirements.txt
```

### 2. Database Setup
```bash
sqlite3 goldgpt_ml_predictions.db < ml_predictions_schema.sql
```

### 3. Model Training
```python
# Initial model training
python -c "from ml_prediction_api import train_all_models; train_all_models()"
```

### 4. API Integration
```python
# In app.py
from ml_prediction_api import get_ml_predictions, train_all_models, ml_engine
```

## Troubleshooting

### Common Issues
1. **Import Error**: ML prediction API not found
   - Check if `ml_prediction_api.py` is in the correct directory
   - Verify all dependencies are installed

2. **No Data Available**: Yahoo Finance API failures
   - Check internet connectivity
   - Verify symbol formatting (GC=F for gold futures)
   - Fallback to synthetic data generation

3. **Model Training Failures**: Insufficient data
   - Ensure at least 50 data points for training
   - Check for data quality issues
   - Verify feature extraction process

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
1. **Social Sentiment Integration**: Twitter API for social sentiment
2. **Additional Symbols**: Silver, platinum, other precious metals
3. **Options Indicators**: Put/call ratios, implied volatility
4. **Economic Calendar**: News event impact prediction
5. **Portfolio Integration**: Risk assessment based on predictions

### Model Improvements
1. **Deep Learning**: LSTM networks for time series
2. **Attention Mechanisms**: Transformer-based models
3. **Multi-Asset Correlation**: Cross-asset prediction factors
4. **Real-time Learning**: Online model updates

## License and Attribution

This ML prediction system uses only MIT/BSD/Apache 2.0 licensed components:
- **scikit-learn**: BSD license
- **pandas**: BSD license
- **numpy**: BSD license
- **yfinance**: Apache 2.0 license
- **requests**: Apache 2.0 license

All data sources are publicly available and used within their terms of service.

---

*Generated for GoldGPT v1.0 - July 2025*
