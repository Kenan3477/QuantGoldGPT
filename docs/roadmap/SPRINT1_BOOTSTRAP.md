# Sprint 1: Bootstrap Edition (Zero-Cost)

**Goal:** Achieve first positive edge with **$0 budget** using free data and open-source tools.

**Timeline:** 8-12 weeks solo development (or faster with focus)

**Philosophy:** Build > Buy. Everything sourced from online or built from scratch.

---

## Zero-Cost Alternatives to Enterprise Plan

| Component | Enterprise Plan | Bootstrap Alternative |
|-----------|-----------------|----------------------|
| **Data** | MT5 broker feeds ($200-1000/mo) | Free sources (see below) |
| **Infrastructure** | AWS/GCP ($500-2000/mo) | Local machine (laptop/desktop) |
| **Monitoring** | Grafana Cloud ($50-200/mo) | Matplotlib + local logs |
| **Team** | 3.5-4.5 FTE | Solo (you) |
| **ML Libraries** | Commercial tools | Free: sklearn, XGBoost, LightGBM, PyTorch |
| **Total Cost** | $800-3400/mo | **$0** |

---

## Free Data Sources (Priority Order)

### 1. Yahoo Finance (Already Implemented ✅)

**Coverage:**
- XAUUSD: `GC=F` (gold futures, daily)
- XAGUSD: `SI=F` (silver futures, daily)
- DXY: `DX-Y.NYB` (dollar index)
- US10Y: `^TNX` (10-year treasury)
- VIX: `^VIX`
- SPX: `^GSPC`

**Limitations:**
- Daily data only (no intraday)
- Delayed (15-20 min)
- No bid/ask spreads
- Occasional gaps

**Verdict:** ✅ Already working, good for D1 research

### 2. Alpha Vantage (Free Tier)

**API:** https://www.alphavantage.co/  
**Free Tier:** 500 requests/day

**Coverage:**
- Forex (XAUUSD, XAGUSD intraday)
- Crypto
- Stocks

**Resolution:** 1min, 5min, 15min, 30min, 60min, daily

**How to get:**
```bash
# Get free API key: https://www.alphavantage.co/support/#api-key
# Store in env: export ALPHAVANTAGE_API_KEY="your_key"
```

**Limitations:**
- 500 requests/day (can backfill slowly over days)
- No historical deep backfill (last 1-2 years only for intraday)

**Verdict:** ✅ Good for M15/H1 intraday data (recent)

### 3. Twelve Data (Free Tier)

**API:** https://twelvedata.com/  
**Free Tier:** 800 API credits/day

**Coverage:**
- Forex (XAUUSD, XAGUSD)
- Crypto, stocks, ETFs

**Resolution:** 1min, 5min, 15min, 30min, 1h, 4h, 1day

**Limitations:**
- 800 requests/day
- 8 API calls/min rate limit

**Verdict:** ✅ Alternative to Alpha Vantage

### 4. OANDA Historical Data (Free)

**Source:** https://www1.oanda.com/fx-for-business/historical-rates  
**Coverage:** XAUUSD, major forex pairs  
**Resolution:** Daily, hourly (limited)  
**Format:** CSV download (manual)

**Verdict:** ✅ Good for validation and backfill

### 5. Dukascopy Historical Data (Free)

**Source:** https://www.dukascopy.com/swiss/english/marketwatch/historical/  
**Coverage:** Forex, gold, silver  
**Resolution:** Tick, 1min, 1h, daily  
**Format:** CSV download or API

**How to use:**
- Python library: `pip install dukascopy`
- Can download tick data for free

**Limitations:**
- Slower download
- Need to parse proprietary format

**Verdict:** ✅ BEST free source for tick/M1 gold data

### 6. Stooq (Free)

**Source:** https://stooq.com/  
**Coverage:** Daily data for many instruments  
**Format:** CSV download

**Verdict:** ✅ Good for daily validation

### 7. Macro Economic Data (Free)

**FRED (Federal Reserve Economic Data):**
- API: https://fred.stlouisfed.org/
- Free API key
- Data: CPI, PCE, GDP, unemployment, etc.

**Trading Economics:**
- Free tier with limited requests
- Calendar data for FOMC, NFP, etc.

**Verdict:** ✅ Essential for macro features

---

## Bootstrap Sprint 1 Implementation Plan

### Week 1-2: Data Pipeline (FREE Sources)

#### Task 1.1: Dukascopy Ingest (M1/M5 Gold/Silver)

**Goal:** Download 2015-2026 M1 data for XAUUSD, XAGUSD

```python
# Install dukascopy library
pip install dukascopy
```

**Implementation:**
- [ ] Create `quantgold/data/ingest/dukascopy_source.py`
- [ ] Implement daily download loop (avoid rate limits)
- [ ] Convert to canonical OHLCV format
- [ ] Store in Parquet
- [ ] Resample M1 → M5, M15, H1 as needed

**Time:** 2-3 days (download is slow)

#### Task 1.2: Alpha Vantage as Backup

- [ ] Implement `quantgold/data/ingest/alphavantage_source.py`
- [ ] Use for recent data (last 1-2 years at M15)
- [ ] Free API key from alphavantage.co

**Time:** 1 day

#### Task 1.3: FRED Macro Data

- [ ] Implement `quantgold/data/ingest/fred_source.py`
- [ ] Download: US10Y, DXY, VIX, SPX, CPI, NFP dates
- [ ] Free API key from FRED

**Time:** 1 day

#### Task 1.4: Manual Macro Calendar

- [ ] Create CSV: `data/events/macro_calendar_2015_2026.csv`
- [ ] Scrape or manually enter major events:
  - FOMC meeting dates (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
  - NFP dates (first Friday of month, verify)
  - CPI release dates (BLS.gov)
- [ ] ~50 events/year × 11 years = 550 rows

**Time:** 1 day (mostly manual, but small dataset)

**Exit Criteria:**
- [ ] M1 XAUUSD 2015-2026 from Dukascopy (gaps OK for now)
- [ ] M15 recent data from Alpha Vantage
- [ ] Macro calendar with 500+ events

---

### Week 2-5: Feature Engineering (50-100 Features)

**All features built from scratch, no paid libraries.**

#### Week 2: Microstructure + MTF Base

**Files to create:**
- `quantgold/features/microstructure.py`
- `quantgold/features/multitimeframe.py`

**Features (20-30 total):**

**Microstructure (M5/M15):**
- [ ] Spread proxy (high-low / close) — since no bid/ask
- [ ] Intraday range percentile (current range / 20-bar avg)
- [ ] Opening range breakout (first 30min of session)
- [ ] Distance from day open (in ATR units)
- [ ] Bar quality: body size %, wick ratios
- [ ] Consecutive direction counter (causal)

**Multi-timeframe:**
- [ ] Trend alignment: SMA(20) on M5, H1, H4, D1
- [ ] Count bullish TFs (0-4 scale)
- [ ] D1 swing high/low (5-bar confirmation)
- [ ] Distance to D1 swing high/low (pips)
- [ ] ATR cascade: ATR(14) on each TF, normalized

**Time:** 5-7 days

#### Week 3-4: Smart Money Concepts (Causal)

**File:** `quantgold/features/smc.py`

**SMC Features (15-20 total):**
- [ ] Order Blocks (OB):
  - Detect strong moves (>1.5 ATR in 1-3 bars)
  - Confirm OB after price returns and bounces
  - Feature: Distance to nearest untested bullish/bearish OB
- [ ] Fair Value Gaps (FVG):
  - 3-candle gap detection
  - Confirm after gap unfilled for N bars
  - Feature: Distance to nearest unfilled FVG
- [ ] Break of Structure (BOS):
  - Swing high/low with 3+3 confirmation
  - Feature: Bars since last bullish/bearish BOS
- [ ] Change of Character (CHoCH):
  - Trend breaks detected causally
  - Feature: Bars since last CHoCH

**Critical:** Test for repainting with `tests/leakage/test_smc_no_repainting.py`

**Time:** 7-10 days (careful implementation)

#### Week 4-5: Intermarket + Macro

**Files:**
- `quantgold/features/intermarket.py` (extend)
- `quantgold/features/macro.py` (replace stubs)

**Intermarket (15-20 features):**
- [ ] DXY: returns (1, 5, 20 bars), RSI(14), SMA distance
- [ ] US10Y: real yield proxy (10Y - 2%), rate of change
- [ ] VIX: spot level, rate of change, percentile
- [ ] SPX: returns, drawdown from high, correlation with gold
- [ ] XAU/XAG ratio: current vs. MA(50), z-score

**Macro (5-10 features):**
- [ ] Hours until next FOMC, NFP, CPI
- [ ] Hours since last major event
- [ ] Is major event within 24h? (boolean)
- [ ] Pre-event window (4h before)
- [ ] Post-event window (2h after)

**Time:** 5-7 days

**Week 5 Total:** 50-80 causal features implemented

---

### Week 5-6: Feature Validation (Critical)

**Ablation Study:**

Run walk-forward with feature families incrementally:

```bash
# Baseline (base features only)
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --features base

# +Microstructure
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --features base,micro

# +MTF
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --features base,micro,mtf

# +SMC
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --features base,micro,mtf,smc

# +Intermarket
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --features base,micro,mtf,smc,inter

# +Macro
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe M5 --features base,micro,mtf,smc,inter,macro
```

**Document:**
- Which families improve OOS Sharpe >10%?
- Feature importance (XGBoost gain)
- SHAP analysis

**Exit Criteria:**
- [ ] ≥2 feature families show OOS improvement
- [ ] Report: `docs/research/FEATURE_ABLATION_REPORT.md`

**Time:** 3-5 days (mostly compute time)

---

### Week 6-8: Advanced Models (FREE)

All models are open-source and free.

#### Week 6: Tabular Ensemble

**Models (all free):**
1. XGBoost (already have)
2. LightGBM (already have)
3. CatBoost (free: `pip install catboost`)
4. RandomForest (sklearn)
5. ExtraTrees (sklearn)

**Hyperparameter Optimization:**
- Use **Optuna** (free): `pip install optuna`
- Search space: max_depth, learning_rate, subsample, etc.
- Optimize on first 5 walk-forward folds

**Implementation:**
- [ ] `quantgold/models/catboost_model.py`
- [ ] `quantgold/models/hyperopt.py` (Optuna integration)
- [ ] Extend `quantgold/models/ensemble.py` to handle 5 models

**Time:** 5-7 days

#### Week 7: Deep Learning (Optional)

**If tabular ensemble plateaus, try deep learning (all free):**

**PyTorch (free, already popular):**
- [ ] Temporal Convolutional Network (TCN)
  - 1D causal convolutions
  - Dilation for long-range patterns
  - Residual connections
  
**Architecture:**
```python
# Input: [batch, sequence_len=50, features=80]
# Conv1D layers with causal padding
# Output: [batch, 3] (BUY/SELL/NO_TRADE probabilities)
```

**File:** `quantgold/models/deep/tcn.py`

**Time:** 5-7 days (if needed)

#### Week 8: Calibration + Meta-Model

**All free (sklearn-based):**

- [ ] Isotonic calibration (already have)
- [ ] Beta calibration (extend `quantgold/models/calibration.py`)
- [ ] Temperature scaling for neural nets

**Meta-model enhancements:**
- [ ] Add market quality features
- [ ] Recent performance tracking
- [ ] Train on validation fold only

**Time:** 3-4 days

---

### Week 9-10: Full Pipeline Re-Run

**Run walk-forward on Dukascopy M5/M15 data:**

```bash
# Build datasets from Dukascopy
python -m quantgold.cli build-datasets --source dukascopy --timeframes M5,M15,H1,H4,D1

# Walk-forward with all features + ensemble
python -m quantgold.cli walk-forward \
  --symbol XAUUSD \
  --timeframe M5 \
  --features base,micro,mtf,smc,inter,macro \
  --models xgboost,lightgbm,catboost,rf,extratrees \
  --calibration isotonic \
  --meta-model enhanced
```

**Compare to baseline:**

| Metric | Baseline (D1 Yahoo) | Target (M5 Dukascopy) |
|--------|---------------------|------------------------|
| Sharpe | Negative | >0.5 |
| Precision | 48.3% | >55% |
| PF | 0.84 | >1.5 |
| ECE | 0.21 | <0.10 |

**Document:** `docs/research/SPRINT1_BOOTSTRAP_RESULTS.md`

**Time:** 3-5 days (mostly compute)

---

### Week 10-12: Polish + Documentation

1. **Documentation:**
   - [ ] Feature catalog with formulas
   - [ ] Ablation report
   - [ ] Model comparison
   - [ ] Calibration analysis

2. **Tests:**
   - [ ] Leakage tests for new features
   - [ ] Unit tests for Dukascopy ingest
   - [ ] Integration tests

3. **Cleanup:**
   - [ ] Remove unused code
   - [ ] Add docstrings
   - [ ] Optimize slow features

**Time:** 5-7 days

---

## Local Infrastructure (Zero-Cost)

### Development Environment

**Hardware:** Your laptop/desktop (no cloud needed)

**Recommended specs:**
- CPU: 4+ cores (8+ preferred)
- RAM: 16GB minimum (32GB preferred)
- Storage: 50-100GB for data
- GPU: Optional (CPU is fine for tabular models; GPU helps for deep learning)

**Software (all free):**
- Python 3.9+
- VS Code or PyCharm Community Edition
- Git
- Libraries: sklearn, xgboost, lightgbm, catboost, polars, pandas, matplotlib

### Monitoring (Zero-Cost)

Instead of Grafana Cloud:

**Option 1: Matplotlib + Local HTML**
- Generate plots after each walk-forward run
- Save as PNG/HTML
- View in browser

**Option 2: Jupyter Notebooks**
- Interactive analysis
- Save notebooks with results

**Option 3: Simple Python Dashboard**
- Flask app (runs locally)
- Real-time metrics in terminal
- Logs to `artifacts/logs/`

**Implementation:**
- [ ] Create `quantgold/monitoring/local_dashboard.py`
- [ ] Generate HTML reports with charts
- [ ] No external dependencies

---

## Free Learning Resources

**ML for Trading:**
- "Advances in Financial Machine Learning" by Marcos López de Prado (PDF available online)
- Quantopian lectures (archived, free)
- QuantConnect tutorials (free tier)

**Technical Analysis:**
- Babypips (free forex education)
- TradingView (free plan for charting)
- YouTube channels (The Trading Channel, UKspreadbetting, etc.)

**Smart Money Concepts:**
- YouTube: ICT (Inner Circle Trader), The Trading Channel
- Free PDFs on SMC available online

**Python + ML:**
- Kaggle competitions and notebooks (free)
- Fast.ai (free courses)
- Scikit-learn documentation

---

## Quick Wins (4-Week Fast Track)

If you want results faster, prioritize these high-ROI tasks:

### Week 1-2: Data + Basic Features
1. Download Dukascopy M5 XAUUSD (2020-2026 only if slow)
2. Implement MTF trend alignment (easy)
3. Implement DXY momentum features

### Week 3: Models
4. Train ensemble: XGBoost + LightGBM + CatBoost
5. Implement disagreement filtering

### Week 4: Test
6. Run walk-forward on M5
7. Compare to baseline

**This 4-week subset might give 50-70% of full Sprint 1 gains.**

---

## Cost Breakdown (Final)

| Item | Cost |
|------|------|
| Dukascopy data | $0 (free download) |
| Alpha Vantage API | $0 (free tier) |
| FRED API | $0 (free) |
| Python libraries | $0 (open-source) |
| Local compute | $0 (your machine) |
| Monitoring | $0 (matplotlib) |
| Total | **$0** |

**Only costs:**
- Your time (8-12 weeks)
- Electricity (~$10-20/month if running 24/7)

---

## Risks (Zero-Cost Constraints)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Free data has gaps | Medium | Use multiple sources, synthetic fill |
| Free data delayed | Low | OK for research, not real-time trading |
| Slow downloads | Low | Download overnight, cache locally |
| Limited intraday history | Medium | Focus on 2020-2026 (6 years sufficient) |
| Local compute slower | Low | Use Polars (fast), optimize code |
| No professional support | Low | StackOverflow, GitHub issues |

---

## Success Criteria (Same as Enterprise)

- [ ] OOS Sharpe >0.5 (up from negative)
- [ ] Label precision >55% (up from 48%)
- [ ] Profit factor >1.5 (up from 0.84)
- [ ] ECE <0.10 (down from 0.21)
- [ ] Feature ablation: ≥2 families improve Sharpe >10%

**If achieved:** You've proven the approach works with $0 budget. Can then decide:
- Continue solo (paper trade, then micro-live)
- Or upgrade to paid data/infrastructure for production scale

---

## Next Steps (Immediate)

1. **Install Dukascopy library:**
   ```bash
   pip install dukascopy
   ```

2. **Get free API keys:**
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key
   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html

3. **Start data download:**
   ```bash
   python -m quantgold.cli build-datasets --source dukascopy --timeframes M1 --start 2020-01-01
   ```

4. **Build features incrementally** (microstructure → MTF → SMC → intermarket → macro)

5. **Run ablation after each feature family** to validate OOS value

---

## Decision Point (End of Sprint 1)

**If Sharpe >0.5 achieved:**
- ✅ Proven: ML edge exists in gold with free data
- Next: Paper trade on live free data (Alpha Vantage real-time or TradingView webhooks)
- Future: Consider upgrading to paid MT5 broker for live execution

**If Sharpe 0.2-0.5:**
- ⚠️ Marginal edge, needs more research
- Try: Alternative label formulations, RL, regime specialists

**If Sharpe <0.2:**
- ❌ No edge with this approach + free data
- Options: Try different markets (crypto has free real-time data), or different strategies

---

## Bonus: Free Real-Time Data (For Future Live Trading)

When ready for paper/live trading:

**Option 1: Alpha Vantage Real-Time**
- WebSocket feed (free tier)
- Last 15min delayed

**Option 2: TradingView Webhooks**
- Free alerts → webhook → your system
- Can trigger on indicators

**Option 3: Binance (if adding crypto)**
- Real-time WebSocket (free)
- PAXG (gold token) trades 24/7

**Option 4: OANDA Practice Account**
- Free demo account
- MT5/API access
- Real-time data

---

## Philosophy

**"Constraints breed creativity."**

Zero budget forces us to:
1. Focus on signal, not noise
2. Build understanding (not buy black boxes)
3. Prove edge with limited data before scaling
4. Learn every component deeply

**If you can't make it work with free data, paid data won't save you.**

The edge comes from **feature engineering** and **modeling**, not data vendor quality (for research purposes).

---

**Let's build.** 🚀
