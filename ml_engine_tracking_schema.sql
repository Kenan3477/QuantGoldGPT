-- GoldGPT ML Engine Accuracy Tracking Database Schema
-- Tracks predictions from multiple ML engines and their accuracy over time

-- Table for storing predictions from different ML engines
CREATE TABLE IF NOT EXISTS ml_engine_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    engine_name TEXT NOT NULL,                    -- 'enhanced_ml_engine', 'intelligent_ml_predictor', etc.
    symbol TEXT NOT NULL DEFAULT 'XAUUSD',        -- Trading symbol
    prediction_date DATE NOT NULL,                -- Date prediction was made
    prediction_time TIME NOT NULL,                -- Time prediction was made
    
    -- Prediction data
    current_price REAL NOT NULL,                  -- Price when prediction was made
    timeframe TEXT NOT NULL,                      -- '1H', '4H', '1D'
    predicted_price REAL NOT NULL,               -- Predicted target price
    change_percent REAL NOT NULL,                -- Predicted % change
    direction TEXT NOT NULL,                     -- 'bullish', 'bearish', 'neutral'
    confidence REAL NOT NULL,                    -- Confidence score 0-1
    
    -- Validation data (filled when prediction period expires)
    actual_price REAL,                           -- Actual price at target time
    actual_change_percent REAL,                  -- Actual % change
    actual_direction TEXT,                       -- Actual direction
    
    -- Accuracy metrics (calculated when validated)
    price_accuracy REAL,                         -- How close predicted price was (%)
    direction_correct BOOLEAN,                   -- Was direction prediction correct
    accuracy_score REAL,                         -- Overall accuracy score 0-100
    
    -- Status tracking
    status TEXT DEFAULT 'pending',               -- 'pending', 'validated', 'expired'
    validation_date TIMESTAMP,                   -- When prediction was validated
    target_validation_time TIMESTAMP NOT NULL,   -- When prediction should be validated
    
    -- Additional metadata
    market_conditions TEXT,                      -- JSON of market conditions at time of prediction
    prediction_factors TEXT                      -- JSON of factors that influenced prediction
);

-- Create indexes for ml_engine_predictions
CREATE INDEX IF NOT EXISTS idx_engine_name ON ml_engine_predictions(engine_name);
CREATE INDEX IF NOT EXISTS idx_prediction_date ON ml_engine_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_timeframe ON ml_engine_predictions(timeframe);
CREATE INDEX IF NOT EXISTS idx_status ON ml_engine_predictions(status);
CREATE INDEX IF NOT EXISTS idx_target_validation_time ON ml_engine_predictions(target_validation_time);

-- Table for storing ML engine performance statistics
CREATE TABLE IF NOT EXISTS ml_engine_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    engine_name TEXT NOT NULL UNIQUE,
    
    -- Overall statistics
    total_predictions INTEGER DEFAULT 0,
    validated_predictions INTEGER DEFAULT 0,
    pending_predictions INTEGER DEFAULT 0,
    
    -- Accuracy metrics by timeframe
    accuracy_1h REAL DEFAULT 0,                 -- 1 hour prediction accuracy %
    accuracy_4h REAL DEFAULT 0,                 -- 4 hour prediction accuracy %
    accuracy_1d REAL DEFAULT 0,                 -- 1 day prediction accuracy %
    overall_accuracy REAL DEFAULT 0,            -- Overall accuracy %
    
    -- Direction accuracy
    direction_accuracy REAL DEFAULT 0,          -- % of correct direction predictions
    
    -- Performance trends (30-day rolling)
    recent_accuracy_30d REAL DEFAULT 0,         -- 30-day accuracy
    accuracy_trend TEXT DEFAULT 'neutral',      -- 'improving', 'declining', 'neutral'
    
    -- Confidence vs accuracy correlation
    high_confidence_accuracy REAL DEFAULT 0,    -- Accuracy when confidence > 0.7
    low_confidence_accuracy REAL DEFAULT 0,     -- Accuracy when confidence < 0.5
    
    -- Best/worst performances
    best_accuracy_period TEXT,                  -- Date range of best performance
    worst_accuracy_period TEXT,                 -- Date range of worst performance
    
    -- Ranking among engines
    rank_position INTEGER DEFAULT 1             -- 1 = best performing engine
);

-- Create indexes for ml_engine_performance
CREATE INDEX IF NOT EXISTS idx_engine_perf_name ON ml_engine_performance(engine_name);
CREATE INDEX IF NOT EXISTS idx_overall_accuracy ON ml_engine_performance(overall_accuracy);
CREATE INDEX IF NOT EXISTS idx_rank_position ON ml_engine_performance(rank_position);

-- Table for storing daily accuracy summaries
CREATE TABLE IF NOT EXISTS daily_accuracy_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_date DATE NOT NULL,
    engine_name TEXT NOT NULL,
    
    predictions_made INTEGER DEFAULT 0,
    predictions_validated INTEGER DEFAULT 0,
    daily_accuracy REAL DEFAULT 0,
    
    -- Best and worst predictions of the day
    best_prediction_id INTEGER,
    worst_prediction_id INTEGER,
    
    UNIQUE(summary_date, engine_name)
);

-- Create indexes for daily_accuracy_summary
CREATE INDEX IF NOT EXISTS idx_summary_date ON daily_accuracy_summary(summary_date);
CREATE INDEX IF NOT EXISTS idx_summary_engine ON daily_accuracy_summary(engine_name);

-- View for easy accuracy comparison
CREATE VIEW IF NOT EXISTS ml_engine_comparison AS
SELECT 
    engine_name,
    overall_accuracy,
    direction_accuracy,
    total_predictions,
    validated_predictions,
    recent_accuracy_30d,
    accuracy_trend,
    rank_position,
    CASE 
        WHEN overall_accuracy >= 70 THEN 'Excellent'
        WHEN overall_accuracy >= 60 THEN 'Good'
        WHEN overall_accuracy >= 50 THEN 'Average'
        ELSE 'Poor'
    END as performance_rating
FROM ml_engine_performance
ORDER BY overall_accuracy DESC;
