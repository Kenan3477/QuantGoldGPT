# QuantGold Migration Checklist

## M0 — Audit + Scaffold (current)

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

## M1 — Canonical data (next)

- [ ] MT5 historical export adapter (XAUUSD + XAGUSD)
- [ ] Parquet store for M1/M5/M15/H1/H4/D1
- [ ] Dataset version hashing
- [ ] Session feature family (causal)
- [ ] First baseline walk-forward smoke report (no holdout tuning)

## Hard rules (ongoing)

- [ ] Never put `target_return` / labels in feature matrices
- [ ] Fit regimes/scalers inside training folds only
- [ ] Do not optimise against final holdout
- [ ] Do not claim performance without logged experiments
