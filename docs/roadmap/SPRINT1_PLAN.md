# Sprint 1: Critical Path to First Edge

**Goal:** Transform baseline from **negative expectancy** to **positive OOS Sharpe >0.5**

**Timeline:** 8-12 weeks of focused implementation

**Priority:** 🔴 CRITICAL — Without this, no amount of infrastructure will help

---

## Current Baseline Performance (XAUUSD D1)

| Metric | Value | Target |
|--------|-------|--------|
| Label Precision | 48.3% | >55% |
| Costed Win Rate | 37.2% | >50% |
| Profit Factor | 0.84 | >1.5 |
| Sharpe | Negative | >0.5 |
| ECE (calibration) | 0.21 | <0.10 |

**Root cause:** Generic features + D1-only data + weak models = no information advantage

---

## Sprint 1 Workstreams

### 🗄️ Workstream 1: High-Frequency Data (Week 1-2)

**Problem:** Yahoo Finance D1 data lacks intraday precision, bid/ask, volume quality.

**Tasks:**

1. **MT5 Historical Backfill**
   - [ ] Set up MT5 terminal + broker demo account
   - [ ] Export M1/M5/M15/H1 XAUUSD from 2015-01-01 to present
   - [ ] Export M1/M5/M15/H1 XAGUSD from 2015-01-01 to present
   - [ ] Verify data quality: <0.1% missing bars, no future timestamps
   - [ ] Store in Parquet format under `data/canonical/mt5/`

2. **Bid-Ask Spread Data**
   - [ ] If MT5 provides bid/ask, extract both
   - [ ] If not, use fixed spread assumptions: XAU=0.5 pips, XAG=3 pips
   - [ ] Validate spread is within realistic bounds (e.g., XAU spread 0.2-2.0 pips)

3. **Intermarket Data (Higher Frequency)**
   - [ ] DXY M15 (or H1 if M15 unavailable)
   - [ ] US10Y yields M15/H1
   - [ ] VIX M15/H1
   - [ ] SPX M15/H1
   - [ ] Store alongside gold/silver data

4. **Data Quality Suite**
   - [ ] Script to detect gaps >1 bar
   - [ ] Script to detect duplicate timestamps
   - [ ] Script to detect anomalous prices (e.g., gold <$1000 or >$5000)
   - [ ] Run quality checks daily if ingesting live data

**Exit Criteria:**
- [ ] M5 XAUUSD 2015-2026 with <0.1% missing bars
- [ ] Bid/ask or spread assumptions documented
- [ ] All data quality tests green

**Estimated Time:** 1-2 weeks (depends on MT5 setup and export speed)

---

### 🧮 Workstream 2: Feature Engineering — Phase 1 (Week 2-5)

**Problem:** Current features are generic. Need proprietary signals.

**Implementation Priority:** Build in this order (easiest → hardest)

#### 2A. Base Features (Enhancements) — Week 2

Already have basic returns/ATR. Add:

- [ ] **Log returns** (better for fat-tailed distributions)
- [ ] **Realized volatility** (5-min returns summed over 1 hour)
- [ ] **Parkinson volatility** (high-low range estimator)
- [ ] **Garman-Klass volatility** (OHLC-based, more efficient)
- [ ] **Volume features** (if available):
  - Volume MA ratio (current / 20-bar MA)
  - Volume percentile (vs. 100-bar distribution)
- [ ] **Candle patterns:**
  - Body size % of total range
  - Upper wick % vs. lower wick %
  - Inside bar flag (high < prev high AND low > prev low)

**File:** `quantgold/features/base.py` (extend existing)

**Test:** Unit test for no lookahead, add to feature registry

#### 2B. Multi-Timeframe (MTF) Features — Week 3

- [ ] **Trend alignment:**
  - For M5 prediction: Fetch H1, H4, D1 closes
  - SMA(20) on each TF: Is price above/below?
  - Count # of TFs bullish (0-3 scale)
  
- [ ] **Support/Resistance:**
  - D1 swing high/low (5-bar left + 5-bar right confirmation)
  - Distance to nearest D1 swing high/low in pips
  
- [ ] **Volatility cascade:**
  - ATR(14) on M5, H1, H4, D1
  - Normalize each by its 50-period MA
  - Is volatility expanding or contracting across TFs?

**File:** `quantgold/features/multitimeframe.py` (new)

**Critical:** Use `align_higher_timeframe` from `quantgold/data/timestamps.py` to avoid lookahead

**Test:** `tests/leakage/test_mtf_alignment.py`

#### 2C. Microstructure Features (M5/M15) — Week 3

- [ ] **Spread dynamics:**
  - Current spread % vs. 20-bar avg spread
  - Spread velocity (rate of change)
  
- [ ] **Intraday momentum:**
  - First 30min of session: high/low
  - Is current price above/below opening range?
  - Distance from day open in ATR units
  
- [ ] **Bar quality:**
  - Consecutive up/down bars (streak counter)
  - Gap % vs. prev close

**File:** `quantgold/features/microstructure.py` (new)

#### 2D. Smart Money Concepts (Causal) — Week 4

Rebuild XAUBot's SMC features **without repainting**:

- [ ] **Order Blocks (OB):**
  ```python
  # Bullish OB: Strong up-move (>1.5 ATR) from consolidation
  # Confirm OB only AFTER price returns and bounces (right-side confirmation)
  # Tag OB with: bar index, strength (range/volume), tested (bool)
  # Feature: Pips to nearest untested bullish OB
  ```
  
- [ ] **Fair Value Gaps (FVG):**
  ```python
  # Gap = candle[i-2].high < candle[i].low (bullish FVG)
  # Confirm only if gap unfilled for 5+ bars
  # Feature: Pips to nearest unfilled FVG
  ```
  
- [ ] **Break of Structure (BOS):**
  ```python
  # Detect swing highs with 3-bar left + 3-bar right confirmation
  # BOS = close above previous swing high
  # Feature: Bars since last bullish BOS
  ```
  
- [ ] **Change of Character (CHoCH):**
  ```python
  # Trend direction based on higher highs / lower lows
  # CHoCH = first lower low after uptrend (or vice versa)
  # Feature: Bars since last CHoCH
  ```

**File:** `quantgold/features/smc.py` (rewrite from `scaffold/xaubot/smc_polars.py`)

**Test:** `tests/leakage/test_smc_no_repainting.py` (critical!)

#### 2E. Intermarket Features (Enhanced) — Week 5

Already have basic DXY/VIX/yields. Add:

- [ ] **DXY momentum:**
  - DXY returns 1-bar, 5-bar, 20-bar
  - DXY RSI(14)
  - DXY above/below SMA(50)?
  
- [ ] **Real yields:**
  - US10Y - inflation expectations (if available)
  - Or proxy: US10Y - 2% constant
  
- [ ] **VIX term structure:**
  - VIX spot vs. VIX futures (if available)
  - Or proxy: VIX rate of change
  
- [ ] **Equity risk-off:**
  - SPX returns (negative correlation with gold during crashes)
  - SPX drawdown % from recent high
  
- [ ] **XAU/XAG ratio:**
  - Current ratio vs. 50-day MA
  - Z-score for mean-reversion signals

**File:** `quantgold/features/intermarket.py` (extend existing)

#### 2F. Macro Event Features (Production) — Week 5

Currently just stubs. Implement real calendar:

- [ ] **Manual event calendar** (CSV for 2015-2026):
  - FOMC dates, NFP dates, CPI dates
  - At minimum: 50 major events per year × 11 years = 550 events
  
- [ ] **Proximity features:**
  - Hours until next FOMC, NFP, CPI
  - Hours since last FOMC, NFP, CPI
  - Boolean: Is major event within 24 hours?
  
- [ ] **Event window flags:**
  - Pre-event (4h before): Reduce size or NO_TRADE
  - Post-event (2h after): Avoid whipsaw

**File:** `quantgold/features/macro.py` (replace stubs)

**Data:** `data/events/macro_calendar_2015_2026.csv`

---

### 🧪 Workstream 3: Feature Validation (Week 5-6)

**Problem:** Don't know which features actually help OOS.

**Tasks:**

1. **Ablation Study**
   - [ ] Run walk-forward with ONLY base features → baseline Sharpe S_base
   - [ ] Add MTF features → Sharpe S_mtf
   - [ ] Add microstructure → S_micro
   - [ ] Add SMC → S_smc
   - [ ] Add intermarket → S_inter
   - [ ] Add macro → S_macro
   - [ ] Report: Which family adds >10% Sharpe improvement?

2. **Feature Importance**
   - [ ] XGBoost feature importance (gain, split count)
   - [ ] Permutation importance (shuffle feature → measure Sharpe drop)
   - [ ] SHAP analysis for top 20 features
   - [ ] Document: Which features are actually used?

3. **Correlation Analysis**
   - [ ] Correlation matrix heatmap for all features
   - [ ] Remove features with >0.9 correlation (redundant)
   - [ ] Check feature-target correlation (sanity check)

4. **Leakage Audit**
   - [ ] Run shuffled-label test: Shuffle labels → should get ~50% precision
   - [ ] If shuffled-label precision >52% → leakage detected
   - [ ] Inspect leaked features and fix

**Exit Criteria:**
- [ ] 50-100 causal features implemented
- [ ] Ablation report shows at least 2 families improve OOS Sharpe >10%
- [ ] Top 20 features by SHAP are interpretable (not noise)
- [ ] Shuffled-label test confirms no leakage

**Deliverable:** `docs/research/FEATURE_ABLATION_REPORT.md`

---

### 🤖 Workstream 4: Advanced Models (Week 6-8)

**Problem:** XGBoost alone may be insufficient.

**Implementation:**

#### 4A. Tabular Model Improvements — Week 6

- [ ] **Hyperparameter optimization:**
  - Use Optuna for walk-forward HPO
  - Search space: max_depth (3-7), learning_rate (0.01-0.1), subsample (0.6-1.0)
  - Optimize on first 5 folds, freeze params, evaluate on remaining folds
  
- [ ] **CatBoost with categorical features:**
  - Session (Asia/London/NY) as categorical
  - Regime (low-vol/high-vol/trending) as categorical
  - Day of week as categorical
  
- [ ] **LightGBM tuning:**
  - Try `num_leaves` vs. `max_depth` variations
  - Test `feature_fraction` for regularization

**File:** `quantgold/models/hyperopt.py` (new)

#### 4B. Ensemble Improvements — Week 7

- [ ] **Train 5 diverse base models:**
  1. XGBoost (default)
  2. LightGBM (fast)
  3. CatBoost (categorical-aware)
  4. RandomForest (decorrelated)
  5. ExtraTrees (high variance, low bias)
  
- [ ] **Ensemble strategies:**
  - Simple average of probabilities
  - Weighted by recent validation performance
  - Stacking: Train logistic regression on base model outputs
  
- [ ] **Disagreement threshold:**
  - If models disagree on direction → NO_TRADE
  - Measure: Std dev of predicted probabilities >0.15 → reject

**File:** `quantgold/models/ensemble.py` (extend existing)

#### 4C. Deep Learning (Optional, Week 8)

If time permits and tabular models plateau:

- [ ] **Temporal Convolutional Network (TCN):**
  - Input: 50-bar sequence of features (50 × num_features tensor)
  - 3-5 causal conv layers with dilation
  - Output: 3-class (BUY / SELL / NO_TRADE)
  - Framework: PyTorch or TensorFlow
  
- [ ] **Attention-based:**
  - Simple Transformer encoder
  - Positional encoding for time-of-day
  - Self-attention over historical bars

**File:** `quantgold/models/deep/` (new package)

**Test:** Validate causal padding (no future leakage in conv)

**Exit Criteria:**
- [ ] Ensemble of 5 models implemented
- [ ] Hyperparameter search completes for XGBoost/LightGBM
- [ ] OOS Sharpe improvement >20% vs. single XGBoost
- [ ] (Stretch) TCN model matches or beats tabular ensemble

---

### 📊 Workstream 5: Calibration + Meta-Model (Week 8-9)

**Problem:** ECE=0.21 is too high. Miscalibrated probabilities → bad risk decisions.

#### 5A. Advanced Calibration — Week 8

- [ ] **Temperature scaling** (for neural nets, if using TCN)
- [ ] **Beta calibration** (generalization of Platt scaling)
- [ ] **Ensemble-level calibration:**
  - Calibrate each base model separately
  - Then ensemble calibrated probabilities
  
- [ ] **Regime-conditional calibration:**
  - Separate calibrator for low-vol vs. high-vol regimes
  - Test: Does regime-specific calibration improve ECE?

**File:** `quantgold/models/calibration.py` (extend existing)

**Metrics:**
- [ ] ECE <0.10 (stretch goal: <0.05)
- [ ] Brier score <0.20
- [ ] Reliability diagram near-diagonal

#### 5B. Meta-Model Enhancements — Week 9

Current meta-model is basic trained sklearn. Add features:

- [ ] **Market quality features:**
  - Spread tightness (current spread / avg spread)
  - Volume ratio (current volume / avg volume)
  - Bars since last macro event
  
- [ ] **Recent model performance:**
  - Win rate over last 20 predictions
  - Avg PnL over last 20 trades
  - Current drawdown state
  
- [ ] **Base model agreement:**
  - Ensemble disagreement score (std dev of probs)
  - Confidence gap (max prob - second max prob)

**File:** `quantgold/meta_models/trained.py` (extend existing)

**Train:** Use only validation fold data (never training data)

**Goal:** Meta-model filters out false positives, improves precision by >10%

---

### ✅ Workstream 6: Walk-Forward Re-Evaluation (Week 9-10)

**Run full pipeline with all improvements:**

1. **Dataset:** MT5 M5/M15 XAUUSD 2015-2026
2. **Features:** All families (base, MTF, micro, SMC, intermarket, macro)
3. **Labels:** Triple-barrier (test 2-3 parameter sets)
4. **Models:** Ensemble of 5 calibrated models
5. **Meta-model:** Enhanced with market quality features
6. **Decision:** Selective with disagreement gates
7. **Validation:** 22-fold walk-forward (same as baseline)

**Run commands:**
```bash
python -m quantgold.cli build-datasets --source mt5 --timeframes M5,M15,H1,H4,D1
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --config configs/sprint1.yaml
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M15 --config configs/sprint1.yaml
```

**Compare to baseline:**

| Metric | Baseline (D1 Yahoo) | Sprint 1 Target (M5/M15 MT5) |
|--------|---------------------|-------------------------------|
| Label Precision | 48.3% | >55% |
| Costed Win Rate | 37.2% | >50% |
| Profit Factor | 0.84 | >1.5 |
| Sharpe | Negative | >0.5 |
| ECE | 0.21 | <0.10 |

**Document:** `docs/research/SPRINT1_RESULTS.md`

---

### 📝 Workstream 7: Documentation + Handoff (Week 10-12)

1. **Feature documentation:**
   - [ ] `docs/features/FEATURE_CATALOG.md` — All features with formulas, lookback, rationale
   - [ ] Add docstrings to all feature builders

2. **Ablation report:**
   - [ ] `docs/research/FEATURE_ABLATION_REPORT.md` — OOS contribution of each family

3. **Model comparison:**
   - [ ] `docs/research/MODEL_COMPARISON.md` — XGB vs. LGBM vs. CatBoost vs. ensemble vs. TCN

4. **Calibration analysis:**
   - [ ] `docs/research/CALIBRATION_ANALYSIS.md` — Reliability diagrams, ECE evolution

5. **Update architecture docs:**
   - [ ] `docs/architecture/QUANTGOLD_ARCHITECTURE.md` — Add Sprint 1 components

6. **Prepare for Sprint 2:**
   - [ ] If OOS Sharpe >0.5 achieved → Sprint 2 (risk, monitoring, paper trading)
   - [ ] If OOS Sharpe still <0.3 → Re-evaluate approach (maybe gold is too efficient)

---

## Definition of Done

Sprint 1 is complete when:

- [x] M5/M15 MT5 data for XAUUSD 2015-2026 ingested and validated
- [x] 50-100 causal features implemented and tested for leakage
- [x] Ablation report shows ≥2 feature families improve OOS Sharpe >10%
- [x] Ensemble of 5 models with calibration ECE <0.10
- [x] Enhanced meta-model improves precision >10% vs. base models
- [x] Walk-forward on M5/M15 achieves OOS Sharpe >0.5 (or explains why not)
- [x] All tests green (unit + leakage + integration)
- [x] Documentation updated
- [x] Experiment logs for reproducibility

**If Sharpe still negative after Sprint 1:**
- Root cause analysis: Is it data quality? Feature leakage? Model capacity? Or fundamental market efficiency?
- Decision: Continue to Sprint 2 (long-shot), or pivot to different markets/strategies

---

## Resource Allocation

**Quant Researcher (primary):**
- Feature engineering (60% time)
- Ablation analysis (20%)
- Model experiments (20%)

**ML Engineer (supporting):**
- Data pipeline (30%)
- Model training infra (30%)
- Calibration (20%)
- Testing (20%)

**Data Engineer (0.5 FTE):**
- MT5 data export and quality checks
- Data versioning

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MT5 data has gaps | Medium | High | Multiple broker sources, synthetic gap-fill |
| Features still don't work | High | Critical | Ablation → drop weak families early |
| Overfitting complex models | High | High | Strict walk-forward, shuffled-label tests |
| Time overrun (>12 weeks) | Medium | Medium | Prioritize: Data → Features → Models (skip DL if tabular works) |
| Still no edge after Sprint 1 | Medium | Critical | Honest re-evaluation: Is gold tradable with this approach? |

---

## Next Steps (Post-Sprint 1)

**If successful (Sharpe >0.5):**
→ Sprint 2: Risk management, monitoring, paper trading (Phase 16, 19, 20)

**If marginal (Sharpe 0.2-0.5):**
→ Research sprint: Alternative label formulations, RL, regime specialists

**If failure (Sharpe <0.2):**
→ Pivot: Test silver (XAGUSD), or different asset classes, or abandon discretionary ML approach

---

## Appendix: Quick Wins (Low-hanging fruit)

If time-constrained, prioritize these high-ROI tasks:

1. **MTF trend alignment** (easy, likely to help)
2. **DXY momentum features** (strong gold correlation)
3. **Macro event windows** (avoid trading pre/post major events → reduce noise)
4. **Ensemble of XGB+LGBM+CatBoost** (diversification usually helps)
5. **Better calibration** (isotonic is easy to implement)
6. **Disagreement filtering** (simple but effective)

These 6 tasks could be done in **4 weeks** and might yield 50-80% of Sprint 1's gains.
