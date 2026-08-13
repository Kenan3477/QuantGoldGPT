# QuantGold Enterprise Roadmap — Executive Summary

**Date:** 2026-08-13  
**Status:** M0–M9 complete, enterprise roadmap scoped

---

## Current State

✅ **Infrastructure:** World-class leakage-safe research pipeline  
❌ **Performance:** No costed edge (negative Sharpe, 48% precision, 0.84 PF)

**Root cause:** Generic features + D1-only data + simple models = no information advantage

---

## Gap Analysis

| Component | Current | Enterprise Target |
|-----------|---------|-------------------|
| **Data** | Yahoo D1 | MT5 M5/M15 with bid/ask, 2015-2026 |
| **Features** | ~20 generic | 50-100 proprietary (microstructure, SMC, MTF) |
| **Models** | XGBoost only | Ensemble (XGB+LGBM+CatBoost+TCN+RL) |
| **Calibration** | ECE 0.21 | ECE <0.05 |
| **Sharpe** | Negative | >1.5 |
| **Win Rate** | 37% | >55% |
| **Infrastructure** | Basic | Monitoring, drift detection, HA, DR |

---

## Critical Path (3 Priorities)

### 🔴 CRITICAL: Sprint 1 (8-12 weeks)

**Must achieve first positive edge or abort.**

**Workstreams:**
1. **Data:** MT5 M5/M15 backfill (XAUUSD/XAGUSD 2015-2026)
2. **Features:** 50-100 causal features
   - Microstructure (spread, intraday momentum, bar patterns)
   - Multi-timeframe (trend alignment, S/R distance, volatility cascade)
   - Smart Money Concepts causal (OB, FVG, BOS, CHoCH — no repainting)
   - Intermarket (DXY momentum, real yields, VIX, equity risk-off, XAU/XAG ratio)
   - Macro events (FOMC/NFP/CPI proximity, avoid trading pre/post events)
3. **Models:** Ensemble (XGB+LGBM+CatBoost) + optional TCN
4. **Calibration:** ECE <0.10 (isotonic + beta calibration)
5. **Meta-model:** Enhanced with market quality + recent performance features

**Success Criteria:**
- Label precision >55% (up from 48%)
- OOS Sharpe >0.5 (up from negative)
- Profit factor >1.5 (up from 0.84)
- Feature ablation report: ≥2 families show >10% Sharpe lift

**If failed:** Re-evaluate if gold is tradable with ML approach.

---

### 🟡 HIGH PRIORITY: Sprint 2-3 (10-16 weeks + 90 days paper)

**After Sprint 1 succeeds only.**

**Sprint 2 (4-6 weeks):**
- Risk management (Kelly, drawdown limits, VaR)
- Real-time monitoring dashboard (Grafana + Prometheus)
- Drift detection (PSI, performance degradation alerts)
- Walk-forward rigor (CPCV, Monte Carlo, frozen holdout)

**Sprint 3 (2-4 weeks + 90 days):**
- MT5 live broker adapter
- Paper trading 90 days on live data
- Canary deployment (10% capital)
- Full production launch if paper Sharpe ≥80% of backtest

---

### 🟢 MEDIUM PRIORITY: Post-Production (Continuous)

**After live trading proven.**

- Advanced regimes (HMM, GMM)
- Multi-instrument portfolio (XAUUSD + XAGUSD + DXY)
- Label engineering experiments
- Deep RL (DQN, DDPG)
- Compliance and disaster recovery
- Continuous research framework

---

## Full Roadmap (25 Phases)

See [`docs/roadmap/ENTERPRISE_SCOPE.md`](docs/roadmap/ENTERPRISE_SCOPE.md) for detailed requirements:

| Phase | Priority | Topic |
|-------|----------|-------|
| 10 | 🔴 Critical | Advanced data infrastructure |
| 11 | 🔴 Critical | Feature engineering — deep dive |
| 12 | 🟢 Medium | Advanced label engineering |
| 13 | 🔴 Critical | Model architecture — advanced ML |
| 14 | 🔴 Critical | Probability calibration — production grade |
| 15 | 🔴 Critical | Meta-model — enhanced filtering |
| 16 | 🟡 High | Risk management — institutional grade |
| 17 | 🟡 High | Walk-forward validation — enterprise rigor |
| 18 | 🟢 Medium | Live execution — production infrastructure |
| 19 | 🟡 High | Monitoring and observability |
| 20 | 🟡 High | Model drift detection and retraining |
| 21 | 🟢 Medium | Regime detection — advanced methods |
| 22 | 🟢 Medium | Multi-instrument and portfolio management |
| 23 | 🟢 Medium | Compliance and auditability |
| 24 | 🟢 Medium | Disaster recovery and high availability |
| 25 | 🟢 Medium | Continuous research and improvement |

---

## Investment Required

**Infrastructure (Monthly):**
- Cloud compute (AWS/GCP): $500-2000
- Data feeds (MT5 VPS, futures): $200-1000
- Monitoring/storage: $100-400
- **Total:** ~$800-3400/month

**Team (Full-time):**
- Quant researcher: 1-2 FTE
- ML engineer: 1 FTE
- Data engineer: 0.5 FTE
- DevOps: 0.5 FTE
- Quant trader: 0.5 FTE
- **Total:** 3.5-4.5 FTE

**Timeline:**
- Sprint 1 (to first edge): 8-12 weeks
- Sprints 2-3 (to production): 10-16 weeks + 90 days paper
- **Total to live trading:** 9-12 months (assuming Sprint 1 succeeds)

---

## Key Risks

1. **No guaranteed edge:** Even with all improvements, gold may be too efficient
2. **Overfitting:** Complex models risk memorizing noise
3. **Regime shifts:** 2015-2023 training may not predict 2026+
4. **Execution slippage:** Real costs may exceed backtest assumptions
5. **Model drift:** Performance degrades faster than detection/retraining

**Mitigation:**
- Conservative sizing (1-2% risk per trade)
- Continuous monitoring and retraining
- Paper trading before live
- Honest reporting and kill switches

---

## Success Metrics

| Metric | Baseline | Sprint 1 Goal | Enterprise Target |
|--------|----------|---------------|-------------------|
| OOS Sharpe | Negative | >0.5 | >1.5 |
| Win Rate (costed) | 37% | >50% | >55% |
| Profit Factor | 0.84 | >1.5 | >1.8 |
| Max Drawdown | N/A | N/A | <15% |
| Calibration (ECE) | 0.21 | <0.10 | <0.05 |
| Label Precision | 48% | >55% | >60% |
| System Uptime | N/A | N/A | >99.9% |

---

## Decision Points

### End of Sprint 1

- **If OOS Sharpe >0.5:** Proceed to Sprint 2 (risk/monitoring)
- **If OOS Sharpe 0.2-0.5:** Research sprint (alternative labels, RL, regimes)
- **If OOS Sharpe <0.2:** Pivot or abort (gold may be too efficient)

### End of Paper Trading

- **If paper Sharpe ≥80% backtest:** Go live with 10% capital (canary)
- **If paper Sharpe 50-80% backtest:** Re-optimize or extend paper period
- **If paper Sharpe <50% backtest:** Do not go live, root cause analysis

### After 3 Months Live

- **If live Sharpe ≥60% backtest:** Scale to 100% capital
- **If live Sharpe 40-60% backtest:** Maintain canary, investigate degradation
- **If live Sharpe <40% backtest:** Halt trading, root cause analysis

---

## Next Action

**Start Sprint 1 implementation:**

1. Set up MT5 terminal + demo account
2. Export M5/M15 XAUUSD 2015-2026
3. Implement microstructure + MTF + SMC features
4. Run ablation study
5. Train ensemble + calibrate
6. Re-run walk-forward
7. Document results

**See:** [`docs/roadmap/SPRINT1_PLAN.md`](docs/roadmap/SPRINT1_PLAN.md) for detailed task breakdown.

---

## Questions?

- **Is gold tradable with ML?** Unknown. Sprint 1 will test this hypothesis.
- **Why not start with easier markets?** Gold was the original brief. Can pivot if Sprint 1 fails.
- **Can we skip Sprint 1 and go straight to production?** No. Current baseline has negative expectancy. No edge = guaranteed losses.
- **What if we never achieve positive Sharpe?** Honest re-evaluation. Not all markets are ML-predictable after costs.

---

**Bottom line:** Current system has excellent methodology but zero edge. Sprint 1 is a 8-12 week bet on whether proprietary features + advanced models can find signal in gold noise. If yes → production path clear. If no → pivot or abort.
