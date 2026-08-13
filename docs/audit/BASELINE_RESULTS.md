# QuantGold Baseline Walk-Forward Results (Research Only)

**Generated:** 2026-08-13  
**Disclaimer:** These are preliminary research metrics from the first leakage-safe baseline.  
They are **not** production performance claims. Costs are approximate (Yahoo data, placeholder spreads).

## Setup

| Item | Value |
|------|-------|
| Data source | Yahoo Finance (`GC=F`, `SI=F`, plus DXY/VIX/US10Y/SPX where available) |
| Primary TF | D1 |
| Labels | Triple-barrier (upper 1.5 ATR, lower 1.0 ATR, 12 bars, same-bar=`ambiguous`) |
| Features | base + sessions + structure + intermarket + macro stubs |
| Models | XGBoost |
| Validation | Chronological walk-forward + purge/embargo; fold-local regimes; val-only calibration/meta |
| Decision | Selective `BUY`/`SELL`/`NO_TRADE` with disagreement + meta gates |
| Execution sim | Spread + slippage + commission placeholders |

## XAUUSD D1

| Metric | Value |
|--------|------:|
| Folds | 22 |
| Predictions scored | 4839 |
| Trades taken | 441 |
| Mean coverage | ~9.0% |
| Mean precision (label success among trades) | ~48.3% |
| Mean Brier | ~0.30 |
| Mean ECE | ~0.21 |
| Backtest win rate (costed) | ~37.2% |
| Backtest expectancy | negative |
| Backtest profit factor | ~0.84 |
| Backtest Sharpe | negative |

**Interpretation:** Selective trading reduces coverage, but the current feature/model stack does **not** yet show a robust costed edge on D1 gold. Calibration is weak (ECE high). This is expected for M1–M6 scaffolding and correctly rejects the idea of shipping the baseline as production.

## XAGUSD D1

| Metric | Value |
|--------|------:|
| Trades taken | 268 |
| Backtest win rate (costed) | ~11.2% |
| Backtest profit factor | ≪ 1 |
| Backtest Sharpe | negative |

Silver baseline is worse under current costs/barriers — do not treat gold/silver as identical.

## Paper smoke

Latest paper iteration on XAUUSD D1 correctly emitted `NO_TRADE` (confidence below threshold) and logged feature snapshot + model version metadata.

## What this unlocks next

1. Feature ablations (does session/structure/intermarket help OOS?)  
2. Barrier/horizon research **without** touching a frozen final holdout  
3. Specialist session models only if baseline routing improves precision/coverage curves  
4. Better broker-grade data (MT5) before trusting execution metrics  

Artifacts: `artifacts/reports/wf_XAUUSD_D1.json`, `artifacts/reports/wf_XAGUSD_D1.json`
