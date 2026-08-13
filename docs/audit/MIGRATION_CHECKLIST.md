# QuantGold Migration Checklist

## M0 — Audit + Scaffold

- [x] Audit XAUBot architecture
- [x] Classify KEEP / REFACTOR / REPLACE / RESEARCH / REMOVE
- [x] Document leakage risks with evidence
- [x] Create `quantgold/` modular package
- [x] Config-driven defaults (`configs/default.yaml`)
- [x] Timestamp contracts + HTF alignment helper
- [x] Triple-barrier labeler (ambiguous same-bar policy)
- [x] Walk-forward + purge/embargo helpers
- [x] Risk engine with capped confidence sizing
- [x] Ensemble disagreement → NO_TRADE
- [x] Paper broker stub
- [x] Model registry stages
- [x] Research/execution import boundary guard
- [x] Leakage + unit tests

## M1 — Canonical data

- [x] YFinance ingest adapter (XAU/XAG + intermarket)
- [x] MT5 adapter interface (optional dependency)
- [x] Synthetic source for offline tests
- [x] Parquet store for multiple timeframes
- [x] Dataset version hashing + manifests
- [x] `quantgold build-datasets` CLI

## M2 — Labels + baseline models + walk-forward

- [x] Triple-barrier labels wired into dataset prep
- [x] Common model interface (sklearn / XGBoost / LightGBM)
- [x] Walk-forward runner with fold-local fitting
- [x] First baseline reports (see `BASELINE_RESULTS.md`)

## M3 — Feature families

- [x] Session features
- [x] Causal structure features (confirmation-time swings)
- [x] Intermarket asof-joined features
- [x] Macro-event proximity stubs + event block flag

## M4 — Regimes + routing

- [x] Fold-local rule regime detector
- [x] Specialist router stubs (config-driven)
- [x] Regime one-hots injected per fold only

## M5 — Calibration / meta / selective thresholds

- [x] Isotonic/Platt calibrator (fit on validation only)
- [x] Brier + ECE reporting
- [x] Trained meta-label model
- [x] SelectivePolicy with NO_TRADE first-class

## M6 — Realistic backtester

- [x] Spread / commission / slippage modelling
- [x] Barrier-path trade simulation
- [x] Metrics: PF, Sharpe, DD, expectancy, coverage-precision

## M7 — Experiment / registry / drift

- [x] ExperimentTracker JSON records
- [x] ModelRegistry stages
- [x] Drift helpers (PSI, precision degradation)

## M8 — Paper trading

- [x] PaperTradingRunner with prediction logs
- [x] Execution import guard
- [x] CLI `paper-once`

## M9 — Integration

- [x] CLI `run-all`
- [x] End-to-end tests (synthetic)
- [x] Real-data baseline run documented
- [x] PR updated

## Hard rules (ongoing)

- [x] Never put `target_return` / labels in feature matrices (enforced + tested)
- [x] Fit regimes/scalers inside training folds only
- [x] Do not optimise against final holdout
- [x] Do not claim performance without logged experiments
- [x] Baseline results reported honestly (currently no costed edge)

## Remaining research (not blockers for scaffold completion)

- [ ] Feature-family ablation report on frozen protocol
- [ ] MT5 broker-grade history backfill
- [ ] Verified macro event calendar integration
- [ ] Promote a model to `validated` only after multi-fold + holdout policy
