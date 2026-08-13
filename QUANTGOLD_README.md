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
python -m pip install -e ".[dev]"
pytest tests/unit tests/leakage -q
```

## Milestone status

| Milestone | Status |
|-----------|--------|
| M0 Audit + scaffold | In progress (this branch) |
| M1 Canonical datasets + baseline features | Next |
| M2 Triple-barrier + walk-forward baseline | Planned |
| … | See audit §8 |

## Legacy note

This repository historically contained QuantGoldGPT dashboard prototypes.  
**QuantGold** (`quantgold/`) is the new research/execution architecture. Legacy dashboard code is not the production prediction path.
