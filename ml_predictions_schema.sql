
-- GoldGPT ML Predictions Database Schema
-- License: MIT - Compatible with SQLite and PostgreSQL

-- Table for storing ML predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    current_price REAL NOT NULL,
    predicted_price REAL NOT NULL,
    predicted_direction TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    prediction_factors TEXT NOT NULL, -- JSON string
    timestamp DATETIME NOT NULL,
    model_version TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_timeframe (symbol, timeframe),
    INDEX idx_timestamp (timestamp)
);

-- Table for storing sentiment analysis history
CREATE TABLE IF NOT EXISTS sentiment_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    overall_sentiment REAL NOT NULL,
    news_sentiment REAL NOT NULL,
    social_sentiment REAL NOT NULL,
    fear_greed_index REAL NOT NULL,
    sentiment_confidence REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp)
);

-- Table for storing model training history
CREATE TABLE IF NOT EXISTS model_training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_type TEXT NOT NULL,
    r2_score REAL NOT NULL,
    training_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    training_duration REAL, -- in seconds
    model_path TEXT,
    timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_timeframe (symbol, timeframe),
    INDEX idx_timestamp (timestamp)
);

-- Table for storing prediction accuracy tracking
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    actual_price REAL NOT NULL,
    prediction_error REAL NOT NULL,
    direction_correct BOOLEAN NOT NULL,
    evaluation_timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES ml_predictions(id),
    INDEX idx_prediction_id (prediction_id),
    INDEX idx_evaluation_timestamp (evaluation_timestamp)
);

-- Table for storing feature importance scores
CREATE TABLE IF NOT EXISTS feature_importance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    importance_score REAL NOT NULL,
    model_version TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_timeframe (symbol, timeframe),
    INDEX idx_timestamp (timestamp)
);

-- Create views for easy querying
CREATE VIEW IF NOT EXISTS latest_predictions AS
SELECT 
    p.*,
    ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn
FROM ml_predictions p
WHERE rn = 1;

CREATE VIEW IF NOT EXISTS prediction_performance AS
SELECT 
    p.symbol,
    p.timeframe,
    p.model_version,
    COUNT(*) as total_predictions,
    AVG(a.prediction_error) as avg_error,
    AVG(CASE WHEN a.direction_correct THEN 1.0 ELSE 0.0 END) as direction_accuracy,
    AVG(p.confidence_score) as avg_confidence
FROM ml_predictions p
JOIN prediction_accuracy a ON p.id = a.prediction_id
GROUP BY p.symbol, p.timeframe, p.model_version;

-- Insert sample data (for testing)
INSERT OR IGNORE INTO ml_predictions (
    symbol, timeframe, current_price, predicted_price, predicted_direction,
    confidence_score, prediction_factors, timestamp, model_version
) VALUES 
    ('GC=F', '1H', 2000.50, 2005.25, 'UP', 0.75, '{"rsi": 55.2, "macd": "bullish"}', datetime('now'), 'v1.0'),
    ('GC=F', '4H', 2000.50, 1995.80, 'DOWN', 0.68, '{"rsi": 62.1, "macd": "bearish"}', datetime('now'), 'v1.0'),
    ('GC=F', '1D', 2000.50, 2020.00, 'UP', 0.82, '{"rsi": 48.5, "macd": "bullish"}', datetime('now'), 'v1.0');

-- Sample sentiment data
INSERT OR IGNORE INTO sentiment_history (
    overall_sentiment, news_sentiment, social_sentiment, fear_greed_index,
    sentiment_confidence, timestamp
) VALUES 
    (0.15, 0.10, 0.20, 0.65, 0.75, datetime('now')),
    (0.05, 0.00, 0.10, 0.60, 0.70, datetime('now', '-1 hour')),
    (-0.10, -0.15, -0.05, 0.45, 0.80, datetime('now', '-2 hours'));
