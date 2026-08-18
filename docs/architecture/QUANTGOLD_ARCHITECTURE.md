# QuantGold Architecture

## Purpose

QuantGold is a modular research and selective-signal platform for **XAUUSD** and **XAGUSD**.

It uses XAUBot AI only as an engineering scaffold. Prediction logic, labels, thresholds, SMC assumptions, regime assumptions, and performance claims are rebuilt and validated independently.

## Package Map

```text
quantgold/
├── data/            Canonical datasets, stores, timestamp contracts
├── features/        Causal feature families + registries
├── labels/          Event / triple-barrier labels
├── regimes/         Regime detectors (fold-local fit)
├── models/          Tabular model interface (XGB/LGBM/CatBoost)
├── meta_models/     Second-stage take-trade models
├── strategies/      Specialist routing
├── validation/      Walk-forward, purged CV, embargo
├── backtesting/     Realistic execution simulation + metrics
├── execution/       Live/paper brokers (isolated)
├── risk/            Sizing and circuit breakers
├── portfolio/       Multi-instrument exposure
├── monitoring/      Drift, calibration, registry hooks
├── research/        Experiment-only code (not importable by live)
├── config/          Typed configuration loaders
└── utils/           Shared helpers
```

## Pipeline

```text
DATA
 ↓
FEATURES          (available_timestamp <= prediction_timestamp)
 ↓
REGIME            (fit inside training fold only)
 ↓
MODEL(S)          (specialists + ensemble disagreement)
 ↓
CALIBRATION
 ↓
META-MODEL        (should we take this trade?)
 ↓
ENTRY FILTER      (threshold → BUY / SELL / NO TRADE)
 ↓
RISK              (size independent of ML bravado)
 ↓
EXECUTION         (paper → production)
 ↓
LOGGING / MONITORING
```

## Research vs Execution Boundary

- `quantgold.execution` and live runners must **not** import `quantgold.research`.
- Research may import data/features/labels/models/validation.
- Production models are immutable registry artifacts.

## Decision Semantics

QuantGold outputs one of:

- `BUY`
- `SELL`
- `NO_TRADE`

`NO_TRADE` is a first-class outcome, not a failure mode.

## Validation Policy

- Chronological splits only
- Walk-forward / purged CV with embargo
- Untouched final holdout
- No random train/test for time series claims
- No optimisation against final holdout

## First Milestone Path

See `docs/audit/XAUBOT_PHASE1_AUDIT.md` §8–9.
