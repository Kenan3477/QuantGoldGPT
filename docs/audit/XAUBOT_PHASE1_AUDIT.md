# QuantGold Phase 1 — XAUBot AI Architecture Audit

**Audit date:** 2026-08-13  
**Source scaffold:** [GifariKemal/xaubot-ai](https://github.com/GifariKemal/xaubot-ai) (v0.2.8)  
**Target system:** QuantGold (XAUUSD / XAGUSD research + selective signal platform)  
**Rule:** XAUBot engineering patterns may be retained; trading logic, labels, thresholds, SMC/regime assumptions, and claimed performance are **not** treated as validated.

---

## 1. Repository Architecture Map

```text
MT5 (OHLCV / orders)
        │
        ▼
┌───────────────────┐
│  DATA INGESTION   │  mt5_connector.py  (+ SimulationConnector)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  FEATURES         │  feature_eng.py (TA) + smc_polars.py (SMC)
│                   │  + ml_v2_feature_eng.py (H1 / continuous SMC)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  REGIME           │  regime_detector.py (HMM, 3-state)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  MODEL            │  TradingModelV2 (live) / TradingModel (legacy train)
│                   │  auto_trainer.py (production retrain)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  CONFIDENCE       │  raw XGB prob + SMC weighted score
│                   │  + dynamic_confidence.py (mostly not enforced)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  ENTRY FILTER     │  session_filter + news_agent + SMC gates
│                   │  + filter_config.json (partially wired)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  RISK             │  risk_engine + smart_risk_manager + kelly scaler
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  EXECUTION        │  main_live.py → MT5 orders + position_manager
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  LOGGING          │  trade_logger + PostgreSQL + Telegram + Next.js
└───────────────────┘
```

### Parallel research tracks (not production)

| Path | Role |
|------|------|
| `backtests/ml_v2/` | Multi-bar targets, 23 extra features, purged CV scaffolding |
| `backtests/ml_v3/` | Triple-barrier labeling research (bugs present) |
| `backtests/backtest_*.py` | ~40 ad-hoc experiment scripts |
| `ea-research/` | MQL5 EA research notes |

---

## 2. Important Files and Responsibilities

| File | Responsibility | Size / notes |
|------|----------------|--------------|
| `main_live.py` | Async live orchestrator | ~2880 lines; SMC-primary, ML secondary |
| `train_models.py` | Legacy V1 training CLI | Trains `TradingModel`, not live V2 |
| `src/mt5_connector.py` | MT5 + simulation connector | KEEP BUT REFACTOR |
| `src/feature_eng.py` | TA features + **target creation** | Target mixed into feature module |
| `src/smc_polars.py` | FVG / OB / BOS / CHoCH | OB assignment can repaint |
| `src/ml_model.py` | Legacy XGBoost wrapper | REPLACE |
| `src/regime_detector.py` | HMM regimes | KEEP BUT REFACTOR |
| `src/dynamic_confidence.py` | Market-quality → threshold | Computed, weakly enforced |
| `src/auto_trainer.py` | Production retrain | **URGENT REPLACE** (leakage) |
| `src/session_filter.py` | Session gates / lot multipliers | KEEP BUT REFACTOR |
| `src/risk_engine.py` | Risk validation | KEEP BUT REFACTOR |
| `src/smart_risk_manager.py` | Large dynamic risk/exit logic | KEEP BUT REFACTOR |
| `src/position_manager.py` | Open-position management | KEEP BUT REFACTOR |
| `src/config.py` | Capital modes / thresholds | KEEP as pattern |
| `src/db/*` | PostgreSQL connection/repo | KEEP BUT REFACTOR |
| `src/news_agent.py` | Economic news filter | RESEARCH REQUIRED |
| `src/telegram_*.py` | Notifications / commands | KEEP (execution layer) |
| `web-dashboard/` | Next.js monitoring UI | KEEP BUT REFACTOR |
| `docker/` | Compose + Postgres schema | KEEP |
| `backtests/ml_v2/*` | V2 research stack | RESEARCH → selective port |
| `backtests/ml_v3/*` | Triple-barrier research | RESEARCH → rewrite |

---

## 3. Current Model / Data Flow (Live)

```text
1. Fetch ~200 M15 bars from MT5
2. FeatureEngineer.calculate_all()
3. SMCAnalyzer.calculate_all()
4. Cache H1 → MLV2FeatureEngineer.add_all_v2_features()
5. HMM predict regime
6. TradingModelV2.predict() → BUY/SELL/HOLD + probability
7. SMC signal generation
8. Combine:
   - Reject CRISIS / avoid quality
   - Require SMC signal (ML disagreement often ignored)
   - SMC confidence floor ~0.55
   - If ML agrees: average confidences; else use SMC
9. Session / news / cooldown / risk gates
10. Size lot → MT5 execute → log → Telegram
```

**Critical inconsistency:** Live uses `TradingModelV2` from `backtests/ml_v2/`, while `train_models.py` trains legacy `TradingModel`. Production retrain via `auto_trainer.py` trains V2 but on a **1-bar next-close target**, not V2 multi-bar / triple-barrier labels.

---

## 4. Component Classification

| Component | Classification | Rationale |
|-----------|----------------|-----------|
| MT5 connector + sim connector | **KEEP BUT REFACTOR** | Essential infrastructure; isolate sim from research claims |
| Config / capital modes pattern | **KEEP BUT REFACTOR** | Good pattern; move to QuantGold config system |
| PostgreSQL logging / Docker | **KEEP** | Ops scaffold |
| Telegram notifications | **KEEP** | Execution monitoring |
| Next.js dashboard shell | **KEEP BUT REFACTOR** | Monitoring only; no performance claims |
| Base TA feature code | **KEEP BUT REFACTOR** | Mostly causal; separate labels from features |
| Session filter concept | **KEEP BUT REFACTOR** | Sessions matter; re-validate thresholds empirically |
| Risk engine / daily loss / lot limits | **KEEP BUT REFACTOR** | Structure useful; confidence≠size coupling must be capped |
| Position manager | **KEEP BUT REFACTOR** | Needed for execution layer |
| Kalman / exit heuristics | **RESEARCH REQUIRED** | Exit research only; not entry edge |
| HMM regime detector | **KEEP BUT REFACTOR** | Fit **inside** each train fold only |
| Dynamic confidence module | **KEEP BUT REFACTOR** | Must be enforced + calibrated |
| Filter config JSON | **KEEP BUT REFACTOR** | Wire declared filters to real gates |
| SMC analyzer | **RESEARCH REQUIRED** | Domain ideas OK; every feature must prove OOS value; fix OB timestamping |
| Legacy XGBoost (`ml_model.py`) | **REPLACE** | Noisy 1-bar target, fragile validation |
| Auto trainer | **REPLACE** | Future-return leakage into features |
| Live SMC-primary combiner | **REPLACE** | QuantGold: ML+meta → selective NO TRADE |
| M5 confirmation module | **REMOVE** | Wrong column names; unused |
| Profit momentum tracker (unused) | **REMOVE** |
| Ad-hoc `backtests/backtest_*.py` zoo | **REMOVE** from production path | Archive as historical notes |
| Claimed backtest metrics in README | **REMOVE** as evidence | Not accepted without QuantGold validation |
| V2 continuous features (safe subset) | **RESEARCH REQUIRED** | Keep ideas; drop leaking features |
| V2 `regime_duration_bars` / `consecutive_direction` | **REPLACE** | Future-length leakage |
| V3 triple-barrier idea | **KEEP BUT REFACTOR** | Right research direction; rewrite bugs |
| News / macro connector | **RESEARCH REQUIRED** | Event blocking first; edges later |

---

## 5. Leakage and Backtesting Risks (Evidence)

### 5.1 CRITICAL — `target_return` can enter training features

`FeatureEngineer.create_target()` creates both `target` and `target_return` from future close (`src/feature_eng.py`).

`auto_trainer.py` excludes `target` but **not** `target_return` when auto-selecting numeric columns:

```text
exclude_cols = {..., "target", ...}  # missing "target_return"
```

**Impact:** Production retrain can learn the future return directly. Any post-retrain performance is invalid.

### 5.2 Order-block feature repainting

In `smc_polars.py`, confirmation at bar `i` writes `ob[j]` onto an earlier bar `j`. As a row-`j` ML feature this is lookahead.

**QuantGold rule:** Emit structure features at **confirmation timestamp**, never backfill onto origin bars for training.

### 5.3 HMM fit on full sample before XGB split

Both `train_models.py` and `auto_trainer.py` fit HMM on all rows, then attach regimes before the XGB train/test split → transductive leakage of test-period distribution.

### 5.4 V2 group-count features leak future length

`regime_duration_bars` and `consecutive_direction` use `.count().over(group)`, assigning full future group length to every bar in the group.

### 5.5 Higher-timeframe asof join risk

H1 features joined with `strategy="backward"` are only safe if H1 bars are **closed** and timestamped at availability time. Forming-bar high/low/close must not be visible.

### 5.6 Validation protocol failures

- `train_models.py` walk-forward can overwrite the saved full model via `fit()` autosave.
- V2 experiment scripts can disable CV then still pick a “best” config.
- V3 Optuna tunes using the test set; class-stratified splits are not pure chronological.
- Many backtests precompute full-sample features then iterate bar-by-bar.

### 5.7 Confidence not what the README implies

Raw classifier probability is treated as “calibrated confidence.” Dynamic thresholds are computed in live but **not consistently enforced**. Live often prefers SMC over ML disagreement → NO TRADE is not first-class.

### 5.8 Claimed performance

README reports Win Rate 63.9%, PF 2.64, Sharpe 4.83, Max DD 2.2%.  
**QuantGold stance:** treat as unverified marketing until reproduced under leakage-safe walk-forward with realistic costs.

---

## 6. Components Worth Retaining (Scaffold Value)

1. **MT5 connector abstraction** (live + sim)
2. **Config-driven capital / risk dataclasses**
3. **PostgreSQL trade logging schema**
4. **Docker compose for API + DB + dashboard**
5. **Telegram alerting plumbing**
6. **Session taxonomy** (Asia / London / NY) as a research feature family
7. **Risk circuit breakers** (max daily loss, max positions, lot caps)
8. **Idea of filter config toggles** (must be actually enforced)
9. **V3 triple-barrier research direction** (rewrite, do not copy bugs)
10. **Purged CV concept** from V2 trainer (needs fold-local transformers)

---

## 7. Proposed QuantGold Architecture

```text
quantgold/
  data/           # canonical OHLCV, intermarket, events; timestamp contracts
  features/       # causal feature families; availability timestamps
  labels/         # triple-barrier / event labels (research params)
  regimes/        # HMM / GMM / rule regimes; fold-local fit
  models/         # XGB / LGBM / CatBoost common interface
  meta_models/    # trust / take-trade second stage
  strategies/     # routing: instrument × session × regime specialists
  validation/     # walk-forward, purged CV, embargo, leakage tests
  backtesting/    # realistic costs, selective thresholds, metrics
  execution/      # MT5 / paper — isolated from research
  risk/           # sizing separate from confidence
  portfolio/      # multi-instrument heat / exposure
  monitoring/     # drift, calibration, registry
  research/       # notebooks/experiments — cannot import into live hot path
  config/         # YAML/TOML driven params
```

### Decision pipeline (QuantGold)

```text
DATA → FEATURES → REGIME → SPECIALIST MODEL(S)
     → ENSEMBLE / DISAGREEMENT → CALIBRATION
     → META-LABEL → CONFIDENCE THRESHOLD
     → NO TRADE | BUY | SELL
     → RISK → EXECUTION → LOGGING / MONITORING
```

### Hard separations

| Boundary | Rule |
|----------|------|
| `quantgold/research` → `quantgold/execution` | Research cannot be imported by live runner |
| Labels vs features | Labels never appear in feature matrices |
| Confidence vs size | Confidence may gate; size uses risk engine only |
| Candidate vs production models | Registry stages: candidate → validated → paper → production → retired |

---

## 8. Migration Plan

| Stage | Action | Exit criterion |
|-------|--------|----------------|
| **M0** | This audit + QuantGold package scaffold | Structure + leakage tests green |
| **M1** | Canonical dataset builders (XAU/XAG, M1–D1) + timestamp contracts | Leakage unit tests pass; sample parquet schema locked |
| **M2** | Triple-barrier labels + baseline XGB/LGBM | Walk-forward baseline metrics logged (no holdout tuning) |
| **M3** | Session + causal structure features (ablation) | Feature family OOS contribution report |
| **M4** | Regime fold-local + specialist routing stubs | Routing config-driven; no live change yet |
| **M5** | Calibration + meta-label + selective thresholds | Coverage–precision curves |
| **M6** | Realistic backtester (spread/slippage/latency) | Full metric suite by year/session/regime |
| **M7** | Experiment tracking + model registry | Reproducible experiment_id records |
| **M8** | Paper trading adapter (MT5 demo) | Forward log of every reject/accept reason |
| **M9** | Production consideration | Only after paper + holdout policy satisfied |

XAUBot code retained under `scaffold/xaubot/` as **read-only reference adapters**, not as the QuantGold hot path.

---

## 9. First Development Milestone (M0 → M1)

**Milestone M0 (this PR):**

- [x] Phase 1 audit document
- [x] QuantGold modular package layout
- [x] Config system (instruments, timeframes, labels, validation)
- [x] Timestamp / availability contracts
- [x] Triple-barrier label interface (parameterised)
- [x] Walk-forward splitter skeleton
- [x] Leakage tests (target_return exclusion, causal regime duration, feature/label separation)
- [x] Research vs execution import boundary test
- [x] Adapter notes for MT5 / risk keepers from XAUBot

**Milestone M1 (next):**

1. Historical data ingest adapters (MT5 export + parquet store)
2. Causal base feature set (returns, ATR, vol, candle geometry)
3. Frozen simple baseline: LightGBM/XGBoost on triple-barrier labels
4. First walk-forward report artifact (no holdout optimisation)

---

## 10. Development Philosophy Reminder

1. Simpler model + better data beats unnecessary complexity.
2. Out-of-sample performance > training accuracy.
3. Calibration > impressive confidence numbers.
4. Selective high-confidence trading > predicting every bar.
5. Every feature is guilty until proven useful.
6. Never optimise against the final holdout.
7. Never claim an improvement from one backtest.
8. Do not fabricate results.
