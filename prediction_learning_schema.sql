-- GoldGPT Advanced Prediction Learning System Database Schema
-- Comprehensive tracking and self-learning system for ML prediction engine
-- Compatible with SQLite and PostgreSQL

-- =====================================================
-- Table: prediction_records
-- Stores all predictions with complete metadata
-- =====================================================
CREATE TABLE IF NOT EXISTS prediction_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    
    -- Timing Information
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_time TIMESTAMP NOT NULL,
    target_time TIMESTAMP NOT NULL,
    validation_time TIMESTAMP NULL,
    
    -- Prediction Details
    symbol TEXT NOT NULL DEFAULT 'XAUUSD',
    timeframe TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    
    -- Market Data at Prediction Time
    current_price REAL NOT NULL,
    predicted_price REAL NOT NULL,
    price_change_percent REAL NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('bullish', 'bearish')),
    
    -- Risk Management
    stop_loss REAL,
    take_profit REAL,
    risk_reward_ratio REAL,
    position_size_suggested REAL,
    
    -- Confidence and Quality Metrics
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    signal_strength REAL,
    model_uncertainty REAL,
    ensemble_agreement REAL,
    
    -- Feature Importance (JSON format)
    features_used TEXT, -- JSON array of feature names
    feature_weights TEXT, -- JSON object with feature importance scores
    technical_indicators TEXT, -- JSON object with indicator values
    market_conditions TEXT, -- JSON object with market state
    
    -- Strategy Voting (for ensemble methods)
    strategy_votes TEXT, -- JSON object with individual strategy predictions
    strategy_weights TEXT, -- JSON object with strategy weights used
    
    -- Validation Results (filled after target_time)
    actual_price REAL NULL,
    actual_change_percent REAL NULL,
    prediction_error REAL NULL,
    direction_correct BOOLEAN NULL,
    hit_take_profit BOOLEAN NULL,
    hit_stop_loss BOOLEAN NULL,
    max_favorable_excursion REAL NULL,
    max_adverse_excursion REAL NULL,
    
    -- Performance Metrics
    is_validated BOOLEAN DEFAULT FALSE,
    is_winner BOOLEAN NULL,
    profit_loss_pips REAL NULL,
    profit_loss_percent REAL NULL,
    
    -- Learning Metadata
    market_regime TEXT, -- bull/bear/sideways/volatile
    news_impact_score REAL,
    volatility_regime TEXT, -- low/medium/high
    trend_strength REAL,
    
    -- System Performance
    execution_time_ms REAL,
    memory_usage_mb REAL,
    cpu_usage_percent REAL
);

-- =====================================================
-- Table: strategy_performance
-- Tracks accuracy metrics for each strategy
-- =====================================================
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Strategy Identification
    strategy_name TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    
    -- Time Period
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Basic Metrics
    total_predictions INTEGER DEFAULT 0,
    validated_predictions INTEGER DEFAULT 0,
    winning_predictions INTEGER DEFAULT 0,
    losing_predictions INTEGER DEFAULT 0,
    
    -- Accuracy Metrics
    win_rate REAL DEFAULT 0.0,
    loss_rate REAL DEFAULT 0.0,
    accuracy_score REAL DEFAULT 0.0,
    direction_accuracy REAL DEFAULT 0.0,
    
    -- Profit Metrics
    total_profit_loss_pips REAL DEFAULT 0.0,
    total_profit_loss_percent REAL DEFAULT 0.0,
    average_win_pips REAL DEFAULT 0.0,
    average_loss_pips REAL DEFAULT 0.0,
    profit_factor REAL DEFAULT 0.0,
    
    -- Risk Metrics
    max_consecutive_wins INTEGER DEFAULT 0,
    max_consecutive_losses INTEGER DEFAULT 0,
    max_drawdown_percent REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0,
    sortino_ratio REAL DEFAULT 0.0,
    
    -- Statistical Metrics
    mean_prediction_error REAL DEFAULT 0.0,
    std_prediction_error REAL DEFAULT 0.0,
    mean_absolute_error REAL DEFAULT 0.0,
    root_mean_squared_error REAL DEFAULT 0.0,
    
    -- Confidence Calibration
    mean_confidence REAL DEFAULT 0.0,
    confidence_vs_accuracy_correlation REAL DEFAULT 0.0,
    overconfidence_rate REAL DEFAULT 0.0,
    underconfidence_rate REAL DEFAULT 0.0,
    
    -- Market Regime Performance (JSON format)
    bull_market_performance TEXT, -- JSON with metrics
    bear_market_performance TEXT, -- JSON with metrics
    sideways_market_performance TEXT, -- JSON with metrics
    volatile_market_performance TEXT, -- JSON with metrics
    
    -- Feature Importance Analysis (JSON format)
    most_important_features TEXT, -- JSON array
    feature_performance_correlation TEXT, -- JSON object
    
    -- Temporal Performance Patterns
    hourly_performance_pattern TEXT, -- JSON with hour-of-day performance
    daily_performance_pattern TEXT, -- JSON with day-of-week performance
    monthly_performance_pattern TEXT, -- JSON with month performance
    
    -- Learning Metrics
    learning_rate REAL DEFAULT 0.0,
    adaptation_score REAL DEFAULT 0.0,
    stability_score REAL DEFAULT 0.0,
    
    -- Unique constraint for strategy-timeframe-period combinations
    UNIQUE(strategy_name, timeframe, period_start, period_end)
);

-- =====================================================
-- Table: validation_results
-- Detailed outcome validation when predictions expire
-- =====================================================
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    
    -- Validation Timing
    validation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    time_to_validation_hours REAL NOT NULL,
    
    -- Market Data at Validation
    actual_price REAL NOT NULL,
    price_path TEXT, -- JSON array of price points during prediction period
    volume_profile TEXT, -- JSON object with volume analysis
    
    -- Outcome Analysis
    prediction_error_percent REAL NOT NULL,
    direction_correct BOOLEAN NOT NULL,
    magnitude_accuracy_score REAL NOT NULL,
    
    -- Risk Management Outcomes
    hit_stop_loss BOOLEAN DEFAULT FALSE,
    hit_take_profit BOOLEAN DEFAULT FALSE,
    stop_loss_hit_time TIMESTAMP NULL,
    take_profit_hit_time TIMESTAMP NULL,
    
    -- Excursion Analysis
    max_favorable_excursion REAL NOT NULL,
    max_adverse_excursion REAL NOT NULL,
    mfe_time TIMESTAMP NULL,
    mae_time TIMESTAMP NULL,
    
    -- Trade Simulation Results
    simulated_pnl_pips REAL,
    simulated_pnl_percent REAL,
    holding_period_return REAL,
    risk_adjusted_return REAL,
    
    -- Market Context During Prediction Period
    volatility_realized REAL,
    trend_strength_realized REAL,
    news_events_count INTEGER DEFAULT 0,
    major_news_impact TEXT, -- JSON array of significant news
    
    -- Learning Insights
    prediction_quality_score REAL NOT NULL,
    confidence_calibration_error REAL,
    feature_attribution_accuracy TEXT, -- JSON with feature performance
    market_regime_classification_accuracy REAL,
    
    -- Validation Quality
    validation_confidence REAL DEFAULT 1.0,
    data_quality_score REAL DEFAULT 1.0,
    external_factors_impact TEXT, -- JSON with external events
    
    FOREIGN KEY (prediction_id) REFERENCES prediction_records(prediction_id)
);

-- =====================================================
-- Table: market_conditions
-- Records market state during predictions for pattern analysis
-- =====================================================
CREATE TABLE IF NOT EXISTS market_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Timing
    timestamp TIMESTAMP NOT NULL,
    date_key DATE,
    hour_key INTEGER,
    
    -- Basic Market Data
    symbol TEXT NOT NULL DEFAULT 'XAUUSD',
    price REAL NOT NULL,
    volume REAL,
    bid REAL,
    ask REAL,
    spread REAL,
    
    -- Technical Indicators
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL,
    ema_12 REAL,
    ema_26 REAL,
    rsi_14 REAL,
    macd_line REAL,
    macd_signal REAL,
    macd_histogram REAL,
    bollinger_upper REAL,
    bollinger_lower REAL,
    bollinger_width REAL,
    atr_14 REAL,
    stochastic_k REAL,
    stochastic_d REAL,
    williams_r REAL,
    cci_20 REAL,
    
    -- Volatility Metrics
    realized_volatility_24h REAL,
    implied_volatility REAL,
    volatility_regime TEXT, -- low/medium/high
    vix_level REAL,
    
    -- Trend Analysis
    trend_direction TEXT, -- up/down/sideways
    trend_strength REAL, -- 0-1 scale
    support_level REAL,
    resistance_level REAL,
    fibonacci_level REAL,
    
    -- Market Regime Classification
    market_regime TEXT, -- bull/bear/sideways/volatile
    regime_confidence REAL,
    regime_duration_days INTEGER,
    
    -- Momentum Indicators
    momentum_1h REAL,
    momentum_4h REAL,
    momentum_1d REAL,
    rate_of_change REAL,
    
    -- Market Microstructure
    order_flow_imbalance REAL,
    market_depth_ratio REAL,
    tick_direction_ratio REAL,
    
    -- Sentiment Indicators
    news_sentiment_score REAL,
    social_sentiment_score REAL,
    fear_greed_index REAL,
    put_call_ratio REAL,
    
    -- Macroeconomic Context
    usd_index REAL,
    ten_year_yield REAL,
    fed_funds_rate REAL,
    inflation_expectation REAL,
    economic_surprise_index REAL,
    
    -- Market Sessions
    session_type TEXT, -- asian/european/american/overlap
    is_high_impact_news_day BOOLEAN DEFAULT FALSE,
    is_holiday BOOLEAN DEFAULT FALSE,
    is_month_end BOOLEAN DEFAULT FALSE,
    is_quarter_end BOOLEAN DEFAULT FALSE,
    
    -- Correlation Analysis
    correlation_with_stocks REAL,
    correlation_with_bonds REAL,
    correlation_with_oil REAL,
    correlation_with_crypto REAL
);

-- =====================================================
-- Table: learning_insights
-- Stores derived insights from the learning engine
-- =====================================================
CREATE TABLE IF NOT EXISTS learning_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Insight Identification
    insight_type TEXT NOT NULL, -- feature_importance/regime_pattern/strategy_optimization
    insight_category TEXT NOT NULL,
    
    -- Timing
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_from TIMESTAMP NOT NULL,
    valid_until TIMESTAMP NULL,
    
    -- Insight Content
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    insight_data TEXT NOT NULL, -- JSON with detailed analysis
    
    -- Statistical Significance
    confidence_level REAL NOT NULL,
    sample_size INTEGER NOT NULL,
    p_value REAL,
    effect_size REAL,
    
    -- Implementation Status
    is_implemented BOOLEAN DEFAULT FALSE,
    implementation_date TIMESTAMP NULL,
    implementation_impact TEXT, -- JSON with before/after metrics
    
    -- Performance Impact
    expected_improvement REAL,
    actual_improvement REAL NULL,
    roi_estimate REAL,
    
    -- Validation
    backtesting_results TEXT, -- JSON with backtest performance
    validation_period_days INTEGER,
    validation_status TEXT DEFAULT 'pending', -- pending/validated/rejected
    
    -- Learning Metadata
    algorithm_used TEXT,
    hyperparameters TEXT, -- JSON
    training_data_period TEXT
);

-- =====================================================
-- Indices for Performance Optimization
-- =====================================================

-- Prediction Records Indices
CREATE INDEX IF NOT EXISTS idx_prediction_time ON prediction_records(prediction_time);
CREATE INDEX IF NOT EXISTS idx_target_time ON prediction_records(target_time);
CREATE INDEX IF NOT EXISTS idx_strategy_name ON prediction_records(strategy_name);
CREATE INDEX IF NOT EXISTS idx_timeframe ON prediction_records(timeframe);
CREATE INDEX IF NOT EXISTS idx_is_validated ON prediction_records(is_validated);
CREATE INDEX IF NOT EXISTS idx_created_at ON prediction_records(created_at);

-- Market Conditions Indices
CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_conditions(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_date ON market_conditions(date_key);
CREATE INDEX IF NOT EXISTS idx_market_regime ON market_conditions(market_regime);
CREATE INDEX IF NOT EXISTS idx_trend_direction ON market_conditions(trend_direction);

-- Learning Insights Indices
CREATE INDEX IF NOT EXISTS idx_insight_type ON learning_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_discovered_at ON learning_insights(discovered_at);
CREATE INDEX IF NOT EXISTS idx_validation_status ON learning_insights(validation_status);

-- =====================================================
-- Views for Common Queries
-- =====================================================

-- Recent Performance Summary
CREATE VIEW IF NOT EXISTS recent_performance_summary AS
SELECT 
    strategy_name,
    timeframe,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(CASE WHEN is_winner = 1 THEN 100.0 ELSE 0 END), 2) as win_rate,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(prediction_error), 4) as avg_error,
    MIN(created_at) as first_prediction,
    MAX(created_at) as last_prediction
FROM prediction_records 
WHERE is_validated = 1 
AND created_at >= datetime('now', '-30 days')
GROUP BY strategy_name, timeframe
ORDER BY win_rate DESC;

-- Market Regime Performance
CREATE VIEW IF NOT EXISTS regime_performance AS
SELECT 
    market_regime,
    COUNT(*) as predictions,
    ROUND(AVG(CASE WHEN is_winner = 1 THEN 100.0 ELSE 0 END), 2) as win_rate,
    ROUND(AVG(profit_loss_percent), 2) as avg_return,
    ROUND(AVG(confidence), 3) as avg_confidence
FROM prediction_records 
WHERE is_validated = 1 
AND market_regime IS NOT NULL
GROUP BY market_regime
ORDER BY win_rate DESC;

-- Feature Performance Analysis
CREATE VIEW IF NOT EXISTS hourly_performance AS
SELECT 
    CAST(strftime('%H', prediction_time) AS INTEGER) as hour,
    COUNT(*) as predictions,
    ROUND(AVG(CASE WHEN is_winner = 1 THEN 100.0 ELSE 0 END), 2) as win_rate,
    ROUND(AVG(profit_loss_percent), 2) as avg_return
FROM prediction_records 
WHERE is_validated = 1
GROUP BY hour
ORDER BY hour;

-- =====================================================
-- Triggers for Automatic Updates
-- =====================================================

-- Update strategy performance when predictions are validated
CREATE TRIGGER IF NOT EXISTS update_strategy_performance_on_validation
AFTER UPDATE ON prediction_records
WHEN NEW.is_validated = 1 AND OLD.is_validated = 0
BEGIN
    INSERT OR REPLACE INTO strategy_performance (
        strategy_name, timeframe, engine_version,
        period_start, period_end, last_updated,
        total_predictions, validated_predictions, winning_predictions,
        win_rate, accuracy_score
    )
    SELECT 
        NEW.strategy_name,
        NEW.timeframe,
        NEW.engine_version,
        DATE('now', 'start of month'),
        DATE('now', 'start of month', '+1 month', '-1 day'),
        CURRENT_TIMESTAMP,
        COUNT(*),
        SUM(CASE WHEN is_validated = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END),
        ROUND(AVG(CASE WHEN is_winner = 1 THEN 100.0 ELSE 0 END), 2),
        ROUND(AVG(CASE WHEN direction_correct = 1 THEN 100.0 ELSE 0 END), 2)
    FROM prediction_records
    WHERE strategy_name = NEW.strategy_name 
    AND timeframe = NEW.timeframe
    AND created_at >= DATE('now', 'start of month');
END;
