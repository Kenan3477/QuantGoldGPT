# Self-Improving ML Prediction System

## Database Schema for Prediction Tracking

```sql
-- Predictions table
CREATE TABLE IF NOT EXISTS daily_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    current_price DECIMAL(10,2) NOT NULL,
    
    -- Multi-timeframe predictions
    prediction_1h DECIMAL(5,3) NOT NULL,
    prediction_4h DECIMAL(5,3) NOT NULL,
    prediction_1d DECIMAL(5,3) NOT NULL,
    prediction_3d DECIMAL(5,3) NOT NULL,
    prediction_7d DECIMAL(5,3) NOT NULL,
    
    predicted_price_1h DECIMAL(10,2) NOT NULL,
    predicted_price_4h DECIMAL(10,2) NOT NULL,
    predicted_price_1d DECIMAL(10,2) NOT NULL,
    predicted_price_3d DECIMAL(10,2) NOT NULL,
    predicted_price_7d DECIMAL(10,2) NOT NULL,
    
    confidence_1h DECIMAL(3,3) NOT NULL,
    confidence_4h DECIMAL(3,3) NOT NULL,
    confidence_1d DECIMAL(3,3) NOT NULL,
    confidence_3d DECIMAL(3,3) NOT NULL,
    confidence_7d DECIMAL(3,3) NOT NULL,
    
    -- Strategy used
    strategy_id INTEGER NOT NULL,
    reasoning TEXT NOT NULL,
    technical_indicators TEXT NOT NULL,
    sentiment_data TEXT NOT NULL,
    market_conditions TEXT NOT NULL,
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending', -- pending, partial, completed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction validation table
CREATE TABLE IF NOT EXISTS prediction_validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    timeframe VARCHAR(5) NOT NULL, -- 1h, 4h, 1d, 3d, 7d
    validation_date TIMESTAMP NOT NULL,
    actual_price DECIMAL(10,2) NOT NULL,
    predicted_price DECIMAL(10,2) NOT NULL,
    predicted_change DECIMAL(5,3) NOT NULL,
    actual_change DECIMAL(5,3) NOT NULL,
    
    -- Accuracy metrics
    price_accuracy DECIMAL(5,3) NOT NULL, -- How close was price prediction
    direction_correct BOOLEAN NOT NULL, -- Was direction (up/down) correct
    confidence_score DECIMAL(3,3) NOT NULL,
    error_margin DECIMAL(5,3) NOT NULL,
    
    -- Performance rating
    performance_score DECIMAL(3,3) NOT NULL, -- 0-1 overall score
    
    FOREIGN KEY (prediction_id) REFERENCES daily_predictions(id)
);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_description TEXT NOT NULL,
    
    -- Performance metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,3) DEFAULT 0,
    avg_confidence DECIMAL(3,3) DEFAULT 0,
    avg_error_margin DECIMAL(5,3) DEFAULT 0,
    
    -- Timeframe-specific performance
    accuracy_1h DECIMAL(5,3) DEFAULT 0,
    accuracy_4h DECIMAL(5,3) DEFAULT 0,
    accuracy_1d DECIMAL(5,3) DEFAULT 0,
    accuracy_3d DECIMAL(5,3) DEFAULT 0,
    accuracy_7d DECIMAL(5,3) DEFAULT 0,
    
    -- Status
    is_active BOOLEAN DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Strategy Evolution Framework

### Base Strategies:
1. **Technical Analysis Heavy** - RSI, MACD, Bollinger Bands focus
2. **Sentiment Analysis Heavy** - News, social media, fear/greed focus  
3. **Pattern Recognition Heavy** - Chart patterns, support/resistance
4. **Hybrid Ensemble** - Balanced combination of all approaches
5. **Momentum Following** - Trend continuation focus
6. **Mean Reversion** - Contrarian approach focus

### Adaptive Learning Rules:
- If accuracy < 60% for 5 consecutive predictions → Switch strategy
- If confidence was high but prediction wrong → Reduce confidence weighting
- If low confidence predictions are often correct → Increase confidence threshold
- Track which market conditions (volatility, volume, news) each strategy works best in
