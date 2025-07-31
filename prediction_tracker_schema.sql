-- GoldGPT Self-Improving ML System Database Schema
-- Tracks prediction accuracy and enables automatic strategy evolution

-- Daily predictions storage with comprehensive metadata
CREATE TABLE IF NOT EXISTS daily_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date DATE NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1h', '4h', '1d', '1w'
    strategy_id VARCHAR(50) NOT NULL, -- 'ensemble', 'technical', 'sentiment', etc.
    model_version VARCHAR(20) NOT NULL,
    
    -- Prediction details
    predicted_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4) NOT NULL,
    predicted_direction VARCHAR(10) NOT NULL, -- 'bullish', 'bearish', 'neutral'
    confidence_score DECIMAL(3,2) NOT NULL, -- 0.00 to 1.00
    predicted_change_percent DECIMAL(5,2) NOT NULL,
    
    -- Target information
    target_price DECIMAL(10,4),
    stop_loss DECIMAL(10,4),
    target_time TIMESTAMP NOT NULL,
    
    -- Market context
    market_volatility DECIMAL(5,2),
    market_trend VARCHAR(10), -- 'trending', 'ranging', 'volatile'
    economic_events TEXT, -- JSON array of upcoming events
    
    -- Feature importance (JSON)
    feature_weights TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_validated BOOLEAN DEFAULT FALSE,
    
    UNIQUE(prediction_date, timeframe, strategy_id)
);

-- Prediction validation results when timeframes expire
CREATE TABLE IF NOT EXISTS prediction_validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Actual outcomes
    actual_price DECIMAL(10,4) NOT NULL,
    actual_direction VARCHAR(10) NOT NULL,
    actual_change_percent DECIMAL(5,2) NOT NULL,
    
    -- Accuracy metrics
    direction_correct BOOLEAN NOT NULL,
    price_accuracy_percent DECIMAL(5,2) NOT NULL, -- How close price prediction was
    profit_loss_percent DECIMAL(5,2), -- If traded, what would P&L be
    
    -- Performance scoring
    accuracy_score DECIMAL(3,2) NOT NULL, -- Overall accuracy 0-1
    confidence_calibration DECIMAL(3,2), -- How well confidence matched outcome
    
    -- Market context at validation
    market_conditions TEXT, -- JSON of market state when validated
    volatility_during_period DECIMAL(5,2),
    
    FOREIGN KEY (prediction_id) REFERENCES daily_predictions(id)
);

-- Strategy performance tracking and evolution
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    measurement_date DATE NOT NULL,
    
    -- Performance metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,2) DEFAULT 0.00,
    avg_confidence DECIMAL(3,2) DEFAULT 0.00,
    confidence_accuracy_correlation DECIMAL(3,2) DEFAULT 0.00,
    
    -- Financial performance
    total_profit_loss DECIMAL(10,4) DEFAULT 0.00,
    win_rate DECIMAL(5,2) DEFAULT 0.00,
    avg_profit_per_trade DECIMAL(8,4) DEFAULT 0.00,
    max_drawdown DECIMAL(5,2) DEFAULT 0.00,
    sharpe_ratio DECIMAL(4,2) DEFAULT 0.00,
    
    -- Market condition performance
    trending_market_accuracy DECIMAL(5,2) DEFAULT 0.00,
    ranging_market_accuracy DECIMAL(5,2) DEFAULT 0.00,
    volatile_market_accuracy DECIMAL(5,2) DEFAULT 0.00,
    
    -- Feature importance evolution
    top_features TEXT, -- JSON array of most important features
    feature_stability_score DECIMAL(3,2) DEFAULT 0.00,
    
    -- Strategy health
    is_active BOOLEAN DEFAULT TRUE,
    performance_trend VARCHAR(20) DEFAULT 'stable', -- 'improving', 'declining', 'stable'
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(strategy_id, model_version, timeframe, measurement_date)
);

-- Model version history and evolution tracking
CREATE TABLE IF NOT EXISTS model_version_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Model details
    algorithm_type VARCHAR(50) NOT NULL, -- 'RandomForest', 'XGBoost', 'LSTM', etc.
    hyperparameters TEXT, -- JSON of model parameters
    feature_set TEXT, -- JSON array of features used
    training_data_period INTEGER, -- Days of data used for training
    
    -- Performance at creation
    initial_validation_accuracy DECIMAL(5,2),
    cross_validation_score DECIMAL(5,2),
    feature_importance TEXT, -- JSON of feature importance scores
    
    -- Evolution triggers
    created_reason VARCHAR(100), -- 'scheduled_retrain', 'performance_decline', 'new_features'
    parent_version VARCHAR(20), -- Previous version this evolved from
    improvement_over_parent DECIMAL(5,2), -- Performance improvement %
    
    -- Model artifacts
    model_path VARCHAR(255), -- Path to saved model file
    feature_scaler_path VARCHAR(255), -- Path to feature scaler
    
    -- Status
    is_production BOOLEAN DEFAULT FALSE,
    is_archived BOOLEAN DEFAULT FALSE,
    retirement_date TIMESTAMP NULL,
    retirement_reason TEXT
);

-- Market regime detection and optimal strategy mapping
CREATE TABLE IF NOT EXISTS market_regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_date DATE NOT NULL,
    regime_type VARCHAR(30) NOT NULL, -- 'bull_trending', 'bear_trending', 'high_volatility', 'low_volatility', 'ranging'
    confidence DECIMAL(3,2) NOT NULL,
    
    -- Market characteristics
    volatility_level VARCHAR(20), -- 'low', 'medium', 'high', 'extreme'
    trend_strength DECIMAL(3,2),
    momentum_score DECIMAL(3,2),
    
    -- Optimal strategies for this regime
    best_strategy_1h VARCHAR(50),
    best_strategy_4h VARCHAR(50),
    best_strategy_1d VARCHAR(50),
    best_strategy_1w VARCHAR(50),
    
    -- Performance in this regime
    regime_accuracy_1h DECIMAL(5,2),
    regime_accuracy_4h DECIMAL(5,2),
    regime_accuracy_1d DECIMAL(5,2),
    regime_accuracy_1w DECIMAL(5,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(detection_date)
);

-- Learning insights and feature discovery
CREATE TABLE IF NOT EXISTS learning_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    insight_type VARCHAR(50) NOT NULL, -- 'feature_discovery', 'pattern_recognition', 'regime_insight'
    
    -- Insight details
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    
    -- Quantitative impact
    accuracy_improvement DECIMAL(5,2),
    profit_improvement DECIMAL(5,2),
    affected_timeframes TEXT, -- JSON array
    affected_strategies TEXT, -- JSON array
    
    -- Implementation status
    is_implemented BOOLEAN DEFAULT FALSE,
    implementation_date TIMESTAMP NULL,
    implementation_results TEXT, -- JSON of results after implementation
    
    -- Discovery metadata
    discovered_by VARCHAR(50), -- 'automated_analysis', 'performance_review', 'feature_engineering'
    data_period_analyzed VARCHAR(100),
    statistical_significance DECIMAL(4,3),
    
    -- Insight data
    supporting_data TEXT -- JSON of evidence supporting this insight
);

-- Ensemble prediction weighting system
CREATE TABLE IF NOT EXISTS ensemble_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calculation_date DATE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Strategy weights (sum to 1.0)
    technical_weight DECIMAL(4,3) DEFAULT 0.25,
    sentiment_weight DECIMAL(4,3) DEFAULT 0.25,
    macro_weight DECIMAL(4,3) DEFAULT 0.25,
    pattern_weight DECIMAL(4,3) DEFAULT 0.25,
    
    -- Performance-based adjustments
    recent_accuracy_factor DECIMAL(3,2) DEFAULT 1.00,
    market_condition_factor DECIMAL(3,2) DEFAULT 1.00,
    volatility_adjustment DECIMAL(3,2) DEFAULT 1.00,
    
    -- Validation metrics
    ensemble_accuracy DECIMAL(5,2),
    individual_best_accuracy DECIMAL(5,2),
    ensemble_improvement DECIMAL(5,2),
    
    -- Meta information
    rebalance_reason VARCHAR(100),
    next_rebalance_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(calculation_date, timeframe)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_daily_predictions_date_timeframe ON daily_predictions(prediction_date, timeframe);
CREATE INDEX IF NOT EXISTS idx_daily_predictions_strategy ON daily_predictions(strategy_id, model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_validation_date ON prediction_validation(validation_date);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_timeframe ON strategy_performance(strategy_id, timeframe);
CREATE INDEX IF NOT EXISTS idx_model_version_history_strategy ON model_version_history(strategy_id, is_production);
CREATE INDEX IF NOT EXISTS idx_market_regimes_date ON market_regimes(detection_date);
CREATE INDEX IF NOT EXISTS idx_learning_insights_type_date ON learning_insights(insight_type, discovery_date);
CREATE INDEX IF NOT EXISTS idx_ensemble_weights_date_timeframe ON ensemble_weights(calculation_date, timeframe);
