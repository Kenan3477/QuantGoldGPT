# QuantGold — Enterprise-Level System Scope

**Date:** 2026-08-13  
**Purpose:** Define the technical requirements, architectural improvements, and research areas needed to transform QuantGold from a research baseline (currently showing no costed edge) to an enterprise-grade, production-accurate trading system.

---

## Current State Assessment

### ✅ What We Have (M0–M9)

**Infrastructure:**
- Modular, leakage-safe research pipeline
- Chronological walk-forward validation with purge/embargo
- Triple-barrier labeling with configurable parameters
- Multi-timeframe canonical data ingestion (Yahoo/MT5/synthetic)
- Feature families: base, sessions, structure, intermarket, macro stubs
- Model abstraction layer (XGBoost, LightGBM, CatBoost)
- Fold-local regime detection and specialist routing framework
- Probability calibration (isotonic/Platt)
- Meta-label model for trade filtering
- Selective NO_TRADE decision policy
- Realistic backtesting with cost simulation
- Experiment tracking and model registry
- Paper trading runner with prediction logging
- Comprehensive leakage and unit tests

**Baseline Performance (XAUUSD D1, Yahoo data):**
- **Coverage:** ~9% (selective trading)
- **Label precision:** ~48.3% (essentially random)
- **Costed win rate:** ~37.2%
- **Profit factor:** ~0.84 (losing)
- **Sharpe:** Negative
- **Expectancy:** Negative

### ❌ Critical Gaps for Enterprise Accuracy

The current system has excellent **methodology** but **zero demonstrated edge**. Enterprise-level accuracy requires:

1. **Predictive edge** (currently missing)
2. **Production-grade infrastructure** (partially missing)
3. **Advanced feature engineering** (minimally implemented)
4. **High-frequency data and execution** (missing)
5. **Risk and portfolio management** (basic only)
6. **Operational resilience** (missing)
7. **Compliance and auditability** (partially implemented)

---

## Phase 10–25: Enterprise Roadmap

### PHASE 10: Advanced Data Infrastructure ⚠️ HIGH PRIORITY

**Problem:** Yahoo Finance D1 data is insufficient for serious production. Lacks intraday precision, bid/ask spreads, tick-level microstructure, order book dynamics.

**Requirements:**

1. **Tick/Minute Data Pipeline**
   - MT5 broker-grade M1/M5/M15 historical backfill (2015–present minimum)
   - Bid/ask spreads for realistic execution simulation
   - Volume/tick volume where available (XAU/XAG futures have better volume data than CFDs)
   - Streaming tick ingestion for live operation
   - Data quality monitoring (gap detection, anomaly flagging)

2. **Multi-Source Data Aggregation**
   - Primary: MT5 broker feeds (institutional-grade)
   - Secondary: Dukascopy/TrueFX for validation
   - Tertiary: Futures data (COMEX GC/SI) for volume/OI insights
   - Cross-validation: Flag discrepancies between sources
   - Data versioning: Track source/version per candle

3. **Alternative Data Integration**
   - **Macro calendars:** Forex Factory, TradingEconomics API (verified event times + actual vs. forecast)
   - **Sentiment:** COT (Commitments of Traders) reports for positioning
   - **Order flow proxies:** Futures CME order book depth snapshots (if accessible)
   - **News/NLP:** Reuters/Bloomberg API for gold-specific headlines (FOMC, mining, geopolitics)
   - **On-chain gold tokens:** PAXG/XAUT supply/flows as alternative positioning signal

4. **Data Storage Architecture**
   - Partitioned Parquet (by symbol/year/month) for fast time-range queries
   - Delta Lake or Iceberg for ACID transactions and time travel
   - Separate hot (recent) vs cold (historical) storage tiers
   - Automatic backup and replication (S3/GCS with versioning)

**Exit Criteria:**
- [ ] M1 XAUUSD/XAGUSD from 2015–present with bid/ask
- [ ] <0.01% missing bars after gap-fill logic
- [ ] Macro calendar with <5min timestamp error vs. official release
- [ ] Data validation suite (tick sanity, spread bounds, monotonic time)

---

### PHASE 11: Feature Engineering — Deep Dive ⚠️ HIGH PRIORITY

**Problem:** Current feature set is generic. No demonstrated OOS value. Enterprise edge requires **information asymmetry** via proprietary features.

**Requirements:**

#### 11A. Microstructure Features (Intraday)

- **Bid-ask dynamics:**
  - Spread % (bid-ask / mid)
  - Spread velocity (rate of spread widening/tightening)
  - Imbalance (bid size / ask size if available)
  
- **Intraday momentum:**
  - Opening range breakouts (first 30min of session)
  - VWAP distance and touch count
  - Volume profile (volume-by-price distribution)
  - Time-weighted average price vs. close

- **Bar patterns:**
  - Tail ratio (wick length / body)
  - Sequential candle patterns (3-bar reversal, engulfing, doji after trend)
  - High-low range percentile vs. recent 20-bar distribution

#### 11B. Multi-Timeframe (MTF) Features

- **Trend alignment:**
  - M15/H1/H4/D1 SMA/EMA alignment (all bullish = strong signal)
  - Higher timeframe pivot points (traditional + Fibonacci)
  - Support/resistance distance on D1/W1

- **Volatility regime cascade:**
  - ATR(14) normalized by 50-period ATR on each TF
  - Bollinger Band width percentile
  - Keltner Channel expansion/contraction

- **Divergence signals:**
  - Price vs. RSI divergence (higher high price, lower high RSI)
  - Price vs. volume divergence (breakout on declining volume)

#### 11C. Smart Money Concepts (SMC) — Causal Reimplementation

XAUBot's SMC features have repainting issues. Rebuild causally:

- **Order Blocks (OB):**
  - Detect bullish OB: Strong up-move after consolidation, confirm OB only after price returns and bounces
  - Bearish OB: Strong down-move, confirm on retest
  - Tag OB strength by volume/range
  - Distance to nearest untested OB

- **Fair Value Gaps (FVG):**
  - 3-candle imbalance (gap between candle 1 high and candle 3 low)
  - Confirm FVG only after gap remains unfilled for N bars (avoid immediate fill)
  - FVG fill rate as regime feature

- **Break of Structure (BOS) / Change of Character (CHoCH):**
  - Swing high/low breaks (use right-side confirmation)
  - Count bars since last BOS
  - Trend strength: consecutive BOS in same direction

**Validation:** Each feature family must prove OOS lift via ablation.

#### 11D. Intermarket Features — Expanded

- **Currency correlations:**
  - DXY (Dollar Index) momentum and divergence vs. gold
  - EUR/USD, GBP/USD (inverse gold relationship)
  - USD/JPY (risk-on proxy)

- **Rates and yields:**
  - US 10Y real yield (TIPS) — strongest inverse correlation with gold
  - Fed Funds rate and rate-of-change
  - Yield curve slope (10Y-2Y)

- **Equity market regime:**
  - S&P 500 drawdown depth (risk-off → gold bid)
  - VIX spot and term structure (VIX futures premium)
  - Nasdaq 100 momentum (tech correlation during QE)

- **Commodities:**
  - Crude oil (inflation proxy)
  - Copper (industrial demand proxy)
  - XAU/XAG ratio (relative strength, mean reversion)

- **Crypto (experimental):**
  - Bitcoin correlation (digital gold narrative)
  - Stablecoin supply growth (liquidity proxy)

#### 11E. Macro Event Features — Production Grade

- **Event proximity:**
  - Hours until next: FOMC, NFP, CPI, PCE, GDP, Jobless Claims
  - Post-event stabilization period (avoid trading 2-4h after major releases)

- **Surprise magnitude:**
  - Actual vs. consensus (Z-score vs. historical surprises)
  - Sequential surprise (2+ months of positive/negative surprises)

- **Forward guidance:**
  - Fed dot-plot changes (scrape from FOMC minutes)
  - Central bank rhetoric sentiment (dovish/hawkish classification)

#### 11F. Temporal Features

- **Session encoding:**
  - London open/close (8am-4pm GMT)
  - NY open/close (8am-4pm EST)
  - Asia session (Tokyo 9am-3pm JST)
  - Overlap periods (highest volume)

- **Day-of-week effects:**
  - Monday (gap behavior)
  - Friday (profit-taking)
  - Month-end rebalancing flows

- **Seasonality:**
  - Gold seasonality (Q1 weak, Q3/Q4 strong historically)
  - Indian wedding season (physical demand)
  - Chinese New Year gold demand

#### 11G. Regime-Specific Features

- **Volatility regimes:**
  - Low vol: Mean-reversion features dominate
  - High vol: Momentum features dominate
  - Transitional: Breakout features

- **Trend regimes:**
  - Strong trend: Moving average distance, ADX
  - Range-bound: Oscillator extremes (RSI <30 / >70)

**Exit Criteria:**
- [ ] 50+ causal features implemented and tested for leakage
- [ ] Ablation report showing OOS contribution of each family
- [ ] Feature importance stability across folds (top 10 features consistent)
- [ ] SHAP analysis for top 20 features
- [ ] Correlation matrix heatmap (remove redundant features >0.9 corr)

---

### PHASE 12: Advanced Label Engineering

**Problem:** Current triple-barrier labels may not capture optimal trade opportunities. Enterprise systems use multiple label formulations.

**Requirements:**

#### 12A. Multi-Horizon Labels

- **Short-term (scalping):** 4–8 bars (M15/H1)
- **Swing:** 12–24 bars (H4/D1)
- **Position:** 50–100 bars (D1)
- Train separate models or multi-output model for each horizon

#### 12B. Adaptive Barriers

- **Dynamic targets:**
  - ATR-multiple barriers that scale with volatility
  - Percentile-based targets (upper 75th percentile move vs. lower 25th)
  - Support/resistance distance-based targets

- **Asymmetric risk/reward:**
  - 2:1 or 3:1 R:R barriers (e.g., upper target = 2× ATR, lower stop = 1× ATR)
  - Research optimal R:R per regime

#### 12C. Alternative Label Schemes

- **Trend-following labels:**
  - Direction of N-bar move (regression-based)
  - Persistence: Will price be higher in 10 bars? (binary)

- **Mean-reversion labels:**
  - Return-to-mean within N bars (range-bound regime)

- **Breakout labels:**
  - Will price exceed recent high/low in N bars?

- **Optimal labels (hindsight):**
  - Best possible trade in next N bars (for model ceiling analysis)
  - Compare model to optimal to gauge remaining alpha

**Exit Criteria:**
- [ ] 3+ label formulations tested
- [ ] Comparative study: Which labels yield highest OOS precision?
- [ ] Label distribution analysis (class balance)

---

### PHASE 13: Model Architecture — Advanced ML ⚠️ HIGH PRIORITY

**Problem:** XGBoost/LightGBM are strong baselines but may underperform deep learning for complex temporal patterns.

**Requirements:**

#### 13A. Tabular Model Improvements

- **Hyperparameter optimization:**
  - Optuna or Ray Tune for walk-forward HPO (per fold or globally)
  - Regularization: Focus on `max_depth`, `min_child_weight`, `subsample`, `lambda`, `alpha`
  - Early stopping on validation set

- **Advanced ensembles:**
  - Stacking: Train meta-model on base model predictions
  - Blending: Weight models by recent validation performance
  - Boosting chains: Sequential models on residuals

- **CatBoost categorical features:**
  - Session, regime, day-of-week as categorical (not one-hot)
  - Ordered categoricals (volatility regime: low < med < high)

#### 13B. Deep Learning Models

- **Temporal Convolutional Networks (TCN):**
  - 1D convolutions over time series
  - Causal padding (no future leakage)
  - Residual connections
  - Fast inference

- **Transformer-based:**
  - Time series Transformer (attention over historical bars)
  - Multi-head attention for different timeframes
  - Positional encoding for time-of-day effects
  - Pre-training on large corpus, fine-tune on gold

- **Recurrent architectures:**
  - LSTM/GRU for sequence modeling
  - Bidirectional LSTM (only on historical data, not for prediction)
  - Attention mechanisms

- **Hybrid:**
  - CNN feature extractor + LSTM/Transformer
  - Tabular features (XGBoost) + sequential features (LSTM) → meta-ensemble

#### 13C. Specialized Architectures

- **Multi-task learning:**
  - Predict direction + magnitude + volatility simultaneously
  - Shared encoder, separate heads
  - Auxiliary task: Predict next bar's high/low (improves representations)

- **Meta-learning:**
  - Learn to adapt quickly to regime shifts
  - MAML (Model-Agnostic Meta-Learning) for few-shot adaptation

- **Reinforcement Learning:**
  - Deep Q-Network (DQN) for action selection (BUY/SELL/HOLD)
  - Reward: Realized PnL, Sharpe, or custom utility function
  - State: Current features + position + recent PnL
  - Action: Entry, exit, position size
  - Continuous action space: DDPG or TD3 for position sizing

- **Generative models:**
  - VAE/GAN to generate synthetic market scenarios for robustness testing
  - Conditional GAN: Generate bars given regime → test model on synthetic stress scenarios

#### 13D. Model Interpretability

- **SHAP (SHapley Additive exPlanations):**
  - Feature importance per prediction
  - Interaction effects
  - Waterfall plots for explainability

- **LIME (Local Interpretable Model-agnostic Explanations):**
  - Local linear approximations

- **Attention visualization:**
  - For Transformer models: Which historical bars does model focus on?

**Exit Criteria:**
- [ ] 5+ model architectures evaluated on same walk-forward protocol
- [ ] Comparative report: Model X vs. baseline (precision, Sharpe, stability)
- [ ] Deep learning model shows >5% OOS precision improvement over XGBoost
- [ ] Inference latency <100ms for real-time prediction
- [ ] SHAP analysis for production model

---

### PHASE 14: Probability Calibration — Production Grade

**Problem:** Current calibration is weak (ECE ~0.21). Uncalibrated probabilities mislead risk management.

**Requirements:**

#### 14A. Advanced Calibration Methods

- **Temperature scaling:** Single-parameter calibration for neural networks
- **Beta calibration:** Generalization of Platt scaling
- **Ensemble calibration:** Calibrate each model separately, then ensemble

#### 14B. Calibration Metrics

- **Expected Calibration Error (ECE):** Binned calibration
- **Maximum Calibration Error (MCE):** Worst-case bin
- **Brier score:** Probabilistic accuracy
- **Log loss:** Penalize confident wrong predictions
- **Reliability diagrams:** Visual calibration curves

#### 14C. Dynamic Recalibration

- **Online calibration:** Update calibrator incrementally as new data arrives
- **Regime-conditional calibration:** Separate calibrators per regime (trend/range/high-vol)

**Exit Criteria:**
- [ ] ECE <0.05 on validation folds
- [ ] Brier score <0.20
- [ ] Reliability diagram shows near-diagonal calibration
- [ ] Regime-conditional calibration tested

---

### PHASE 15: Meta-Model — Enhanced Trade Filtering ⚠️ HIGH PRIORITY

**Problem:** Current meta-label model is a simple trained classifier. Enterprise systems use sophisticated filters.

**Requirements:**

#### 15A. Meta-Model Features

- **Base model agreement:** Ensemble disagreement score
- **Confidence stability:** Std dev of predicted probability across ensemble
- **Recent model performance:** Win rate over last 20 predictions
- **Market quality:**
  - Spread tightness
  - Volume vs. average
  - Volatility regime
  - Time since last major news event

- **Position context:**
  - Current exposure (already long/short gold)
  - Correlation with open positions (if multi-instrument)
  - Drawdown state (avoid trading during large DD)

- **Historical signal quality:**
  - Success rate of similar setups (nearest neighbors in feature space)
  - Regime-conditional success rate

#### 15B. Meta-Model Architectures

- **Logistic regression:** Simple, interpretable
- **XGBoost on meta-features:** More complex
- **Reinforcement learning:** Learn optimal filtering policy to maximize Sharpe

#### 15C. Multi-Stage Filtering

```text
Stage 1: Base model probability threshold (e.g., >0.65 or <0.35)
Stage 2: Ensemble agreement (e.g., all models agree on direction)
Stage 3: Meta-model approval (probability of success >0.6)
Stage 4: Market quality filter (spread <X, volume >Y)
Stage 5: Risk limit check (exposure within limits)
→ BUY / SELL / NO_TRADE
```

**Exit Criteria:**
- [ ] Meta-model improves OOS Sharpe by >20% vs. base models alone
- [ ] Reduced false positives (high confidence → actual wins correlation >0.7)
- [ ] NO_TRADE rate optimized (coverage/precision trade-off)

---

### PHASE 16: Risk Management — Institutional Grade ⚠️ HIGH PRIORITY

**Problem:** Current risk engine is basic (fixed fractional + ATR stops). Enterprise systems use sophisticated risk models.

**Requirements:**

#### 16A. Position Sizing Algorithms

- **Kelly Criterion:**
  - Optimal fraction = (p × b - q) / b, where p = win prob, q = loss prob, b = win/loss ratio
  - Use fractional Kelly (e.g., 0.25× Kelly) to reduce volatility
  - Dynamic Kelly based on recent win rate and calibrated probabilities

- **Volatility-adjusted sizing:**
  - Target fixed % risk per trade (e.g., 1% account)
  - Size = (Risk $ ) / (ATR × stop distance in ATR)

- **Confidence-scaled sizing:**
  - Linear scaling: size ∝ (confidence - threshold)
  - Capped at max position size

- **Risk parity:**
  - Equal risk contribution across gold/silver (if trading both)

#### 16B. Stop-Loss and Take-Profit Logic

- **Dynamic stops:**
  - Trail stop by ATR or percentage
  - Break-even stop after X pips profit
  - Time-based stops (exit if no progress after N bars)

- **Profit targets:**
  - Fixed R:R (e.g., 2:1)
  - Support/resistance-based exits
  - Partial exits: Scale out 50% at 1:1, let 50% run to 3:1

- **Volatility breakout stops:**
  - Widen stops during high volatility (avoid noise stop-outs)

#### 16C. Portfolio Risk Management

- **Exposure limits:**
  - Max % of account in gold/silver combined
  - Max correlation exposure (if trading correlated pairs)

- **Daily/weekly loss limits:**
  - Circuit breaker: Stop trading if down X% today/this week
  - Reduced size after consecutive losses

- **Drawdown management:**
  - Reduce size by 50% if in >10% drawdown
  - Pause trading if >20% drawdown

- **Margin and leverage:**
  - Max leverage (e.g., 10:1 for gold CFDs)
  - Margin utilization <50%

#### 16D. Risk Metrics and Monitoring

- **Value at Risk (VaR):** 95% / 99% daily VaR
- **Conditional VaR (CVaR):** Expected loss beyond VaR
- **Max drawdown:** Real-time tracking
- **Sharpe / Sortino / Calmar ratios:** Rolling 30/90-day windows
- **Win rate / profit factor / expectancy:** By regime, session, timeframe

**Exit Criteria:**
- [ ] Kelly-based sizing implemented and tested
- [ ] Drawdown limits enforced (tested via synthetic losing streak)
- [ ] Risk metrics dashboard (real-time VaR, DD, Sharpe)
- [ ] Position sizing does not exceed 5% account per trade

---

### PHASE 17: Walk-Forward Validation — Enterprise Rigor

**Problem:** Current walk-forward is functional but lacks production-grade validation protocols.

**Requirements:**

#### 17A. Validation Strategies

- **Purged K-Fold CV with embargo:**
  - Already implemented, but test with multiple K values (5, 10)
  - Optimal embargo period research (10 bars? 20 bars?)

- **Combinatorial Purged CV (CPCV):**
  - Test all combinations of train/test splits
  - More robust but computationally expensive

- **Walk-forward with re-optimization:**
  - Retrain every N folds (e.g., every 500 bars)
  - Track model drift: Does old model degrade?

#### 17B. Holdout and Production Testing

- **Frozen final holdout:**
  - Reserve 2024–2026 as untouched test set
  - Touch only once, after all research is complete

- **Paper trading period:**
  - 3-6 months live paper trading before production
  - Compare paper results to backtest expectations
  - If paper Sharpe <50% of backtest → do not go live

- **Out-of-sample (OOS) time windows:**
  - Test on different market regimes (2020 COVID, 2022 rate hikes, 2023 range)

#### 17C. Robustness Testing

- **Monte Carlo simulation:**
  - Resample trades with replacement → generate 1,000 equity curves
  - Report 5th percentile Sharpe (worst-case)

- **Sensitivity analysis:**
  - Vary cost assumptions (spread ±50%)
  - Vary label parameters (barrier size ±20%)
  - If results flip sign → system is fragile

- **Stress testing:**
  - Simulate flash crashes, liquidity crises, extreme volatility
  - Does system survive or blow up?

**Exit Criteria:**
- [ ] CPCV implemented and tested
- [ ] Frozen holdout reserved and documented
- [ ] Monte Carlo 5th percentile Sharpe >0.5
- [ ] Sensitivity analysis shows <20% Sharpe variance for ±20% parameter changes

---

### PHASE 18: Live Execution — Production Infrastructure

**Problem:** Current paper broker is a stub. Enterprise systems require production-grade execution.

**Requirements:**

#### 18A. Broker Integration

- **MT5 live adapter:**
  - Real-time tick subscription
  - Order management: Market, limit, stop orders
  - Position tracking and PnL calculation
  - Error handling and retry logic

- **API redundancy:**
  - Primary: MT5
  - Fallback: Direct broker API (OANDA, FXCM, etc.)

- **Slippage and latency:**
  - Measure actual execution price vs. expected
  - Log slippage statistics
  - Optimize order routing (limit orders vs. market orders)

#### 18B. Order Execution Logic

- **Smart order routing:**
  - Use limit orders during low volatility
  - Use market orders during high volatility (avoid missing fill)

- **Partial fills:**
  - Handle partial order fills (especially for larger sizes)

- **Order timeout:**
  - Cancel limit order if not filled within N seconds
  - Re-submit at market if urgent

#### 18C. Latency Optimization

- **Low-latency infrastructure:**
  - Colocate server near broker data center (e.g., Equinix)
  - Use compiled languages for hot path (Rust, C++)
  - Sub-millisecond feature computation

- **Async execution:**
  - Non-blocking order submission
  - Event-driven architecture (asyncio, actor model)

**Exit Criteria:**
- [ ] MT5 live adapter with <50ms order latency
- [ ] Paper trading on live data for 90 days
- [ ] Slippage measurement: Avg <0.5 pips on gold M5 entries
- [ ] Zero unhandled broker API errors in 1-month test

---

### PHASE 19: Monitoring and Observability ⚠️ HIGH PRIORITY

**Problem:** Current experiment tracking is minimal. Enterprise systems require real-time monitoring and alerting.

**Requirements:**

#### 19A. Real-Time Dashboards

- **Performance metrics:**
  - Live PnL ($ and %)
  - Sharpe / Sortino (rolling 30-day)
  - Drawdown (current and max)
  - Win rate (daily, weekly, all-time)
  - Open positions and exposure

- **Model health:**
  - Prediction distribution (are predictions drifting?)
  - Feature distribution (PSI for each feature)
  - Calibration curves (live vs. expected)
  - Ensemble agreement rate

- **Execution quality:**
  - Slippage (avg, max)
  - Order fill rate
  - Latency (order to fill)

**Tech stack:** Grafana + Prometheus, or custom React dashboard

#### 19B. Alerting

- **Critical alerts (SMS/PagerDuty):**
  - Drawdown >10%
  - Daily loss >5%
  - Broker API down
  - Order execution failure

- **Warning alerts (email/Slack):**
  - Win rate drops >20% vs. backtest
  - Feature drift (PSI >0.25)
  - Model confidence distribution shift
  - Calibration ECE >0.15

#### 19C. Logging and Audit Trail

- **Prediction logs:**
  - Every prediction: timestamp, features, model version, probability, decision, reason

- **Trade logs:**
  - Every trade: entry/exit time, price, size, PnL, slippage, reason for entry/exit

- **Model registry:**
  - Version control all models (git commit SHA, training date, validation metrics)
  - Immutable model artifacts (no in-place updates)

- **Reproducibility:**
  - Every experiment logged with full config, git SHA, data version hash
  - Ability to re-run any historical experiment exactly

**Exit Criteria:**
- [ ] Real-time dashboard with <1s latency
- [ ] Critical alerts tested (manual trigger)
- [ ] All predictions and trades logged to database
- [ ] 100% experiment reproducibility (re-run 10 experiments, get identical results)

---

### PHASE 20: Model Drift Detection and Retraining ⚠️ HIGH PRIORITY

**Problem:** Models degrade over time. Enterprise systems detect drift and retrain automatically.

**Requirements:**

#### 20A. Drift Detection

- **Feature drift:**
  - PSI (Population Stability Index) for each feature
  - Alert if PSI >0.25 (significant shift)

- **Prediction drift:**
  - Compare recent prediction distribution to training distribution
  - KS test (Kolmogorov-Smirnov) for distribution shift

- **Performance drift:**
  - Rolling 30-day Sharpe vs. backtest expectation
  - Rolling win rate vs. training win rate
  - If performance <50% of backtest → trigger review

#### 20B. Automated Retraining

- **Retraining triggers:**
  - Every N days (e.g., weekly)
  - Or when drift detected
  - Or when performance degrades >X%

- **Retraining pipeline:**
  - Fetch latest data
  - Re-run walk-forward on expanded dataset
  - Compare new model to current production model on validation set
  - Promote if new model >5% better

- **Canary deployment:**
  - Deploy new model to 10% of predictions first
  - If performance ≥ current model after 1 week → full rollout
  - Else rollback

#### 20C. Model Versioning

- **Semantic versioning:** v1.2.3 (major.minor.patch)
- **Change log:** What changed (features, labels, model architecture)
- **A/B testing:** Run old vs. new model side-by-side

**Exit Criteria:**
- [ ] PSI calculated daily for top 20 features
- [ ] Automated retraining pipeline tested
- [ ] Canary deployment for model v2.0 successful

---

### PHASE 21: Regime Detection — Advanced Methods

**Problem:** Current rule-based regime detector is simplistic. Enterprise systems use sophisticated regime models.

**Requirements:**

#### 21A. Statistical Regime Models

- **Hidden Markov Models (HMM):**
  - Learn latent market states (e.g., 3 states: low-vol, high-vol, trending)
  - Fit inside each training fold only (no lookahead)
  - Gaussian emissions (returns, volatility)

- **Gaussian Mixture Models (GMM):**
  - Cluster market conditions in feature space
  - Soft assignments (probability of each regime)

- **Markov-switching autoregression:**
  - Regime-switching dynamics for gold returns

#### 21B. Machine Learning Regime Detection

- **Clustering:**
  - K-means on volatility/trend/volume features
  - DBSCAN for outlier detection (flash crashes)

- **Classification:**
  - Train classifier to predict regime (supervised if labels available)

- **Reinforcement learning:**
  - Learn optimal regime classification to maximize trading Sharpe

#### 21C. Multi-Regime Strategy

- **Regime-specific models:**
  - Trend regime: Momentum model
  - Range regime: Mean-reversion model
  - High-vol regime: Reduce size or NO_TRADE

- **Regime transition detection:**
  - Bayesian change-point detection
  - Alert when regime is shifting (increase caution)

**Exit Criteria:**
- [ ] HMM regime detector implemented and tested for leakage
- [ ] Regime-specific models show >10% Sharpe improvement vs. single model
- [ ] Regime transition alerts tested

---

### PHASE 22: Multi-Instrument and Portfolio Management

**Problem:** Current system is single-instrument (XAUUSD). Enterprise systems manage portfolios.

**Requirements:**

#### 22A. Multi-Instrument Trading

- **Instruments:** XAUUSD, XAGUSD, PAXG (gold token), DXY (inverse correlation)
- **Correlation matrix:** Real-time correlation tracking
- **Exposure limits:** Max % in gold + silver combined

#### 22B. Portfolio Optimization

- **Mean-variance optimization:**
  - Markowitz efficient frontier
  - Maximize Sharpe ratio subject to constraints

- **Risk parity:**
  - Equal risk contribution from each instrument

- **Black-Litterman:**
  - Bayesian approach combining market equilibrium with ML views

#### 22C. Cross-Asset Strategies

- **Pairs trading:**
  - XAU/XAG ratio mean reversion
  - Gold vs. DXY inverse

- **Relative strength:**
  - Trade strongest instrument (XAU if outperforming XAG)

**Exit Criteria:**
- [ ] Portfolio of XAUUSD + XAGUSD with correlation-adjusted sizing
- [ ] Portfolio Sharpe >1.5× single-instrument Sharpe
- [ ] Exposure limits enforced (tested via simulation)

---

### PHASE 23: Compliance and Auditability

**Problem:** Regulatory requirements for financial trading systems.

**Requirements:**

#### 23A. Audit Trails

- **Immutable logs:** All decisions, trades, model versions in append-only database
- **Timestamps:** Microsecond precision for all events
- **Reproducibility:** Ability to replay any trading day exactly

#### 23B. Compliance Checks

- **Position limits:** Never exceed broker/regulatory limits
- **Wash trading prevention:** No offsetting trades within short timeframe
- **Best execution:** Demonstrate reasonable efforts to minimize costs

#### 23C. Reporting

- **Daily PnL reports:** For accounting
- **Monthly performance reports:** For investors/auditors
- **Model governance:** Document model changes, approvals, backtest results

**Exit Criteria:**
- [ ] Audit log with 100% coverage of all trades
- [ ] Compliance checks automated (pre-trade and post-trade)
- [ ] Monthly performance report template

---

### PHASE 24: Disaster Recovery and High Availability

**Problem:** System downtime = missed opportunities and potential losses.

**Requirements:**

#### 24A. High Availability

- **Redundant servers:**
  - Primary + hot standby
  - Automatic failover if primary down >30s

- **Database replication:**
  - Real-time replication to backup DB
  - Automatic promotion if primary DB fails

- **Broker API redundancy:**
  - Fallback to secondary broker if primary API down

#### 24B. Disaster Recovery

- **Backup strategy:**
  - Hourly snapshots of data and model registry
  - Off-site backups (S3, GCS)

- **Recovery time objective (RTO):** <5 minutes
- **Recovery point objective (RPO):** <1 hour (max data loss)

- **Runbooks:**
  - Documented procedures for common failures
  - Tested quarterly

#### 24C. Security

- **API key rotation:** Rotate broker API keys every 90 days
- **Encryption:** All data at rest and in transit encrypted
- **Access control:** Least-privilege principle, MFA for production systems

**Exit Criteria:**
- [ ] Failover tested (kill primary server, standby takes over <60s)
- [ ] Disaster recovery drill (restore from backup, confirm system works)
- [ ] Security audit passed

---

### PHASE 25: Continuous Research and Improvement

**Problem:** Markets evolve. One-time development is insufficient.

**Requirements:**

#### 25A. Research Roadmap

- **Quarterly research sprints:**
  - Q1: New feature families
  - Q2: New model architectures
  - Q3: Label engineering
  - Q4: Risk management improvements

- **Academic collaboration:**
  - Monitor latest research (arXiv, SSRN)
  - Implement promising techniques (e.g., new Transformer variants)

#### 25B. Feedback Loop

- **Live performance analysis:**
  - Which predictions were wrong? Why?
  - Feature importance: Are models using sensible features?

- **Failure case studies:**
  - Analyze large losses
  - Root cause analysis
  - Implement safeguards

#### 25C. A/B Testing Framework

- **Hypothesis-driven development:**
  - Hypothesis: Adding feature X will improve Sharpe by Y%
  - Test: Deploy model with/without feature X
  - Measure: Statistical significance test

**Exit Criteria:**
- [ ] Quarterly research sprint framework defined
- [ ] Failure case study template
- [ ] A/B testing infrastructure (random assignment of predictions to model variants)

---

## Critical Path to First Production Edge

Given current **negative expectancy** baseline, the absolute priorities are:

### 🔴 Critical (Must-have for ANY edge)

1. **Phase 11: Feature Engineering (Deep Dive)**
   - Proprietary features are the #1 source of alpha
   - Without information advantage, no model will save us
   - Focus: Microstructure, intraday, intermarket, SMC (causal)

2. **Phase 13: Model Architecture (Advanced ML)**
   - Current XGBoost may be too simple for gold's complexity
   - Test deep learning (TCN, Transformer) and RL
   - Ensemble of diverse architectures

3. **Phase 10: Data Infrastructure (High-frequency)**
   - D1 Yahoo data is too coarse
   - Need M5/M15 broker-grade data with bid/ask
   - Tick data for microstructure features

4. **Phase 14: Calibration (Production Grade)**
   - ECE 0.21 → 0.05 is mandatory for risk management
   - Miscalibrated probabilities → bad sizing → losses

5. **Phase 15: Meta-Model (Enhanced Filtering)**
   - Improve NO_TRADE decision quality
   - Better filtering = higher precision even if coverage drops

### 🟡 High Priority (Required for enterprise, but can wait for basic edge)

6. **Phase 16: Risk Management (Institutional Grade)**
7. **Phase 19: Monitoring (Real-time)**
8. **Phase 20: Drift Detection**

### 🟢 Medium Priority (Post-production)

9. **Phase 17: Walk-Forward Validation (Enterprise Rigor)**
10. **Phase 21: Regime Detection (Advanced Methods)**
11. **Phase 12: Label Engineering**

### 🔵 Low Priority (Optimization)

12. **Phase 18: Live Execution**
13. **Phase 22: Multi-Instrument**
14. **Phase 23: Compliance**
15. **Phase 24: Disaster Recovery**
16. **Phase 25: Continuous Research**

---

## Recommended Development Order

### Sprint 1: Data + Features (8-12 weeks of implementation)

1. **Ingest M5/M15 MT5 data** (XAUUSD 2015-2026)
2. **Implement 50+ causal features** (microstructure, MTF, SMC, intermarket, macro)
3. **Ablation study:** Which features add OOS value?
4. **Feature selection:** Remove redundant/noisy features
5. **Goal:** Achieve >55% label precision on validation (up from 48%)

### Sprint 2: Models + Calibration (6-8 weeks)

1. **Deep learning models:** TCN, Transformer, hybrid
2. **Ensemble of 5-10 diverse models**
3. **Advanced calibration:** Temperature scaling, beta calibration
4. **Goal:** ECE <0.05, OOS Sharpe >0.5

### Sprint 3: Meta-Model + Risk (4-6 weeks)

1. **Enhanced meta-model** with market quality + recent performance features
2. **Kelly sizing with capped confidence scaling**
3. **Drawdown circuit breakers**
4. **Goal:** OOS Sharpe >1.0, max DD <15%

### Sprint 4: Validation + Production Prep (4-6 weeks)

1. **CPCV and Monte Carlo robustness tests**
2. **Frozen final holdout test** (touch once only)
3. **Paper trading 90 days** on live data
4. **Real-time monitoring dashboard**
5. **Goal:** Paper Sharpe ≥80% of backtest expectation

### Sprint 5: Production Launch (2-4 weeks)

1. **MT5 live broker adapter**
2. **Canary deployment (10% capital)**
3. **Disaster recovery testing**
4. **Go-live on full capital if canary successful**

---

## Success Criteria for Enterprise-Level System

| Metric | Baseline (Now) | Enterprise Target |
|--------|----------------|-------------------|
| **OOS Sharpe Ratio** | Negative | >1.5 |
| **OOS Sortino Ratio** | Negative | >2.0 |
| **Win Rate (costed)** | 37% | >55% |
| **Profit Factor** | 0.84 | >1.8 |
| **Max Drawdown** | N/A (paper) | <15% |
| **Calmar Ratio** | N/A | >1.0 |
| **Label Precision** | 48% | >60% |
| **Calibration (ECE)** | 0.21 | <0.05 |
| **Coverage** | 9% | 10-20% (selective) |
| **Inference Latency** | N/A | <100ms |
| **System Uptime** | N/A | >99.9% |
| **Data Quality** | N/A | <0.01% missing bars |
| **Paper-to-Live Gap** | N/A | <20% Sharpe degradation |

---

## Budget and Resources (Indicative)

### Infrastructure Costs (Monthly)

- **Cloud compute (AWS/GCP):** $500-2000 (GPU instances for deep learning training)
- **Data feeds:** $200-1000 (MT5 VPS, futures data subscriptions)
- **Monitoring (Grafana Cloud, PagerDuty):** $50-200
- **Backup/storage (S3):** $50-200
- **Total:** ~$800-3400/month

### Team (Full-time equivalents)

- **Quant researcher:** 1-2 FTE (feature engineering, model research)
- **ML engineer:** 1 FTE (training pipelines, model serving)
- **Data engineer:** 0.5 FTE (data ingestion, quality monitoring)
- **DevOps/SRE:** 0.5 FTE (infrastructure, monitoring, alerts)
- **Quant trader:** 0.5 FTE (strategy validation, live oversight)

### Timeline (Aggressive)

- **Sprints 1-3 (to first edge):** 18-26 weeks
- **Sprint 4 (validation):** 4-6 weeks + 90 days paper trading
- **Sprint 5 (production):** 2-4 weeks
- **Total to production-ready:** ~9-12 months assuming full-time focus

---

## Known Unknowns and Risks

### Technical Risks

1. **No guaranteed edge:** Even with all improvements, gold may be too efficient to profit after costs
2. **Overfitting:** More complex models (deep learning) risk overfitting despite validation
3. **Regime shifts:** Model trained on 2015-2023 may fail in 2026+ regime
4. **Execution slippage:** Real slippage may exceed backtest assumptions
5. **Data quality:** Broker data may have gaps, errors, or look-ahead bias (e.g., revised quotes)

### Operational Risks

1. **Broker API downtime:** Critical during volatile periods
2. **Model drift faster than detection:** Losses before retraining
3. **Human error:** Incorrect config deployment
4. **Security breach:** API keys compromised

### Mitigation Strategies

- **Diversification:** Trade multiple uncorrelated strategies (not just ML gold)
- **Conservative position sizing:** Never risk >1-2% per trade
- **Continuous monitoring:** Detect issues before large losses
- **Frequent retraining:** Adapt to regime shifts
- **Paper trading first:** Validate before risking real capital

---

## Conclusion

Transforming QuantGold from a leakage-safe research baseline (currently no edge) to an enterprise-level accurate trading system requires:

1. **Deep feature engineering** (proprietary signals)
2. **Advanced ML models** (deep learning, RL, ensembles)
3. **High-frequency data** (M5/M15 with bid/ask)
4. **Rigorous calibration** (ECE <0.05)
5. **Sophisticated risk management** (Kelly, drawdown limits)
6. **Production infrastructure** (monitoring, drift detection, high availability)

**Critical path:** Phases 10-11-13-14-15 (data, features, models, calibration, meta-model) are mandatory to achieve first positive expectancy. All other phases are infrastructure and operational improvements.

**Realistic timeline:** 9-12 months to production-ready system with full-time team, assuming research breakthroughs in feature engineering and modeling.

**No guarantees:** Even with all improvements, there is no certainty of a profitable edge. Gold markets are highly competitive, and this roadmap is necessary but not sufficient for success.

---

**Next Action:** Choose Sprint 1 priority (data + features) and begin systematic implementation of Phase 10-11.
