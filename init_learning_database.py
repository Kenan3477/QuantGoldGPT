#!/usr/bin/env python3
"""
Simple database initialization and validation script
"""

import sqlite3
import os

def initialize_learning_database():
    """Initialize the learning system database with schema"""
    try:
        # Remove any existing test database
        if os.path.exists('goldgpt_learning_system.db'):
            os.remove('goldgpt_learning_system.db')
        
        # Create new database
        with sqlite3.connect('goldgpt_learning_system.db') as conn:
            # Create simplified schema that works with SQLite
            schema = """
            -- Prediction Records Table
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction_time TIMESTAMP NOT NULL,
                target_time TIMESTAMP NOT NULL,
                validation_time TIMESTAMP NULL,
                symbol TEXT NOT NULL DEFAULT 'XAUUSD',
                timeframe TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                price_change_percent REAL NOT NULL,
                direction TEXT NOT NULL CHECK (direction IN ('bullish', 'bearish')),
                stop_loss REAL,
                take_profit REAL,
                risk_reward_ratio REAL,
                position_size_suggested REAL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                signal_strength REAL,
                model_uncertainty REAL,
                ensemble_agreement REAL,
                features_used TEXT,
                feature_weights TEXT,
                technical_indicators TEXT,
                market_conditions TEXT,
                strategy_votes TEXT,
                strategy_weights TEXT,
                actual_price REAL NULL,
                actual_change_percent REAL NULL,
                prediction_error REAL NULL,
                direction_correct BOOLEAN NULL,
                hit_take_profit BOOLEAN NULL,
                hit_stop_loss BOOLEAN NULL,
                max_favorable_excursion REAL NULL,
                max_adverse_excursion REAL NULL,
                is_validated BOOLEAN DEFAULT FALSE,
                is_winner BOOLEAN NULL,
                profit_loss_pips REAL NULL,
                profit_loss_percent REAL NULL,
                market_regime TEXT,
                news_impact_score REAL,
                volatility_regime TEXT,
                trend_strength REAL,
                execution_time_ms REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL
            );

            -- Strategy Performance Table
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_predictions INTEGER DEFAULT 0,
                validated_predictions INTEGER DEFAULT 0,
                winning_predictions INTEGER DEFAULT 0,
                losing_predictions INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                loss_rate REAL DEFAULT 0.0,
                accuracy_score REAL DEFAULT 0.0,
                direction_accuracy REAL DEFAULT 0.0,
                total_profit_loss_pips REAL DEFAULT 0.0,
                total_profit_loss_percent REAL DEFAULT 0.0,
                average_win_pips REAL DEFAULT 0.0,
                average_loss_pips REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                max_consecutive_wins INTEGER DEFAULT 0,
                max_consecutive_losses INTEGER DEFAULT 0,
                max_drawdown_percent REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                sortino_ratio REAL DEFAULT 0.0,
                mean_prediction_error REAL DEFAULT 0.0,
                std_prediction_error REAL DEFAULT 0.0,
                mean_absolute_error REAL DEFAULT 0.0,
                root_mean_squared_error REAL DEFAULT 0.0,
                mean_confidence REAL DEFAULT 0.0,
                confidence_vs_accuracy_correlation REAL DEFAULT 0.0,
                overconfidence_rate REAL DEFAULT 0.0,
                underconfidence_rate REAL DEFAULT 0.0,
                bull_market_performance TEXT,
                bear_market_performance TEXT,
                sideways_market_performance TEXT,
                volatile_market_performance TEXT,
                most_important_features TEXT,
                feature_performance_correlation TEXT,
                hourly_performance_pattern TEXT,
                daily_performance_pattern TEXT,
                monthly_performance_pattern TEXT,
                learning_rate REAL DEFAULT 0.0,
                adaptation_score REAL DEFAULT 0.0,
                stability_score REAL DEFAULT 0.0,
                UNIQUE(strategy_name, timeframe, period_start, period_end)
            );

            -- Validation Results Table
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                validation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_to_validation_hours REAL NOT NULL,
                actual_price REAL NOT NULL,
                price_path TEXT,
                volume_profile TEXT,
                prediction_error_percent REAL NOT NULL,
                direction_correct BOOLEAN NOT NULL,
                magnitude_accuracy_score REAL NOT NULL,
                hit_stop_loss BOOLEAN DEFAULT FALSE,
                hit_take_profit BOOLEAN DEFAULT FALSE,
                stop_loss_hit_time TIMESTAMP NULL,
                take_profit_hit_time TIMESTAMP NULL,
                max_favorable_excursion REAL NOT NULL,
                max_adverse_excursion REAL NOT NULL,
                mfe_time TIMESTAMP NULL,
                mae_time TIMESTAMP NULL,
                simulated_pnl_pips REAL,
                simulated_pnl_percent REAL,
                holding_period_return REAL,
                risk_adjusted_return REAL,
                volatility_realized REAL,
                trend_strength_realized REAL,
                news_events_count INTEGER DEFAULT 0,
                major_news_impact TEXT,
                prediction_quality_score REAL NOT NULL,
                confidence_calibration_error REAL,
                feature_attribution_accuracy TEXT,
                market_regime_classification_accuracy REAL,
                validation_confidence REAL DEFAULT 1.0,
                data_quality_score REAL DEFAULT 1.0,
                external_factors_impact TEXT,
                FOREIGN KEY (prediction_id) REFERENCES prediction_records(prediction_id)
            );

            -- Market Conditions Table
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                date_key DATE,
                hour_key INTEGER,
                symbol TEXT NOT NULL DEFAULT 'XAUUSD',
                price REAL NOT NULL,
                volume REAL,
                bid REAL,
                ask REAL,
                spread REAL,
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
                realized_volatility_24h REAL,
                implied_volatility REAL,
                volatility_regime TEXT,
                vix_level REAL,
                trend_direction TEXT,
                trend_strength REAL,
                support_level REAL,
                resistance_level REAL,
                fibonacci_level REAL,
                market_regime TEXT,
                regime_confidence REAL,
                regime_duration_days INTEGER,
                momentum_1h REAL,
                momentum_4h REAL,
                momentum_1d REAL,
                rate_of_change REAL,
                order_flow_imbalance REAL,
                market_depth_ratio REAL,
                tick_direction_ratio REAL,
                news_sentiment_score REAL,
                social_sentiment_score REAL,
                fear_greed_index REAL,
                put_call_ratio REAL,
                usd_index REAL,
                ten_year_yield REAL,
                fed_funds_rate REAL,
                inflation_expectation REAL,
                economic_surprise_index REAL,
                session_type TEXT,
                is_high_impact_news_day BOOLEAN DEFAULT FALSE,
                is_holiday BOOLEAN DEFAULT FALSE,
                is_month_end BOOLEAN DEFAULT FALSE,
                is_quarter_end BOOLEAN DEFAULT FALSE,
                correlation_with_stocks REAL,
                correlation_with_bonds REAL,
                correlation_with_oil REAL,
                correlation_with_crypto REAL
            );

            -- Learning Insights Table
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT NOT NULL,
                insight_category TEXT NOT NULL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_from TIMESTAMP NOT NULL,
                valid_until TIMESTAMP NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                insight_data TEXT NOT NULL,
                confidence_level REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                p_value REAL,
                effect_size REAL,
                is_implemented BOOLEAN DEFAULT FALSE,
                implementation_date TIMESTAMP NULL,
                implementation_impact TEXT,
                expected_improvement REAL,
                actual_improvement REAL NULL,
                roi_estimate REAL,
                backtesting_results TEXT,
                validation_period_days INTEGER,
                validation_status TEXT DEFAULT 'pending',
                algorithm_used TEXT,
                hyperparameters TEXT,
                training_data_period TEXT
            );
            """
            
            # Execute schema
            conn.executescript(schema)
            
            # Create indices separately
            indices = """
            CREATE INDEX IF NOT EXISTS idx_prediction_time ON prediction_records(prediction_time);
            CREATE INDEX IF NOT EXISTS idx_target_time ON prediction_records(target_time);
            CREATE INDEX IF NOT EXISTS idx_strategy_name ON prediction_records(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_timeframe ON prediction_records(timeframe);
            CREATE INDEX IF NOT EXISTS idx_is_validated ON prediction_records(is_validated);
            CREATE INDEX IF NOT EXISTS idx_created_at ON prediction_records(created_at);
            CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_conditions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_market_regime ON market_conditions(market_regime);
            CREATE INDEX IF NOT EXISTS idx_insight_type ON learning_insights(insight_type);
            CREATE INDEX IF NOT EXISTS idx_discovered_at ON learning_insights(discovered_at);
            """
            
            conn.executescript(indices)
            
            # Create views
            views = """
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
            """
            
            conn.executescript(views)
            
            # Verify tables were created
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['prediction_records', 'strategy_performance', 'validation_results', 'market_conditions', 'learning_insights']
            
            print("✅ Database initialization completed")
            print(f"Created tables: {tables}")
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ {table}")
                else:
                    print(f"❌ {table} - missing")
            
            return True
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = initialize_learning_database()
    if success:
        print("\n🎯 Database ready for learning system integration!")
    else:
        print("\n❌ Database initialization failed")
