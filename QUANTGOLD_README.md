# QuantGold

Modular **XAUUSD / XAGUSD** quantitative research and selective-signal platform.

XAUBot AI is an engineering scaffold only. Its trading logic, labels, thresholds, SMC/regime assumptions, and claimed performance are **not** treated as validated.

## Priorities

1. Out-of-sample precision  
2. Probability calibration  
3. Selective high-confidence trading (`BUY` / `SELL` / `NO_TRADE`)  
4. Regime robustness  
5. Leakage prevention  
6. Realistic execution modelling  
7. Walk-forward validation  
8. Explainability & experiment tracking  

## Package layout

See `docs/architecture/QUANTGOLD_ARCHITECTURE.md` and `quantgold/`.

## Phase 1 audit

See `docs/audit/XAUBOT_PHASE1_AUDIT.md`.

## Quick start

```bash
python -m pip install -e ".[dev,ml]"
pytest tests/unit tests/leakage -q

# Build canonical datasets (Yahoo research source)
python -m quantgold.cli build-datasets --source yfinance --timeframes D1,H1

# Walk-forward + costed backtest + experiment log
python -m quantgold.cli walk-forward --symbol XAUUSD --timeframe D1

# Paper smoke (logs NO_TRADE/BUY/SELL decision)
python -m quantgold.cli paper-once --symbol XAUUSD --timeframe D1

# Full chain
python -m quantgold.cli run-all --symbol XAUUSD --timeframe D1
```

## Milestone status

| Milestone | Status |
|-----------|--------|
| M0 Audit + scaffold | Done |
| M1 Canonical datasets | Done |
| M2 Triple-barrier + walk-forward baseline | Done |
| M3 Session/structure/intermarket/macro features | Done |
| M4 Fold-local regimes + routing stubs | Done |
| M5 Calibration + meta + selective NO_TRADE | Done |
| M6 Realistic backtester | Done |
| M7 Experiment tracking + registry + drift | Done |
| M8 Paper runner | Done |
| M9 CLI + docs + baseline report | Done |

Baseline research results (honest, currently **no costed edge**): `docs/audit/BASELINE_RESULTS.md`

## Legacy note

This repository historically contained QuantGoldGPT dashboard prototypes.  
**QuantGold** (`quantgold/`) is the new research/execution architecture. Legacy dashboard code is not the production prediction path.
