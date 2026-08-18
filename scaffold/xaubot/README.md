# XAUBot Scaffold Reference

This directory documents which XAUBot AI components QuantGold intends to adapt.

**Do not copy trading logic, labels, thresholds, or claimed performance into QuantGold production paths without independent validation.**

Upstream: https://github.com/GifariKemal/xaubot-ai

## KEEP (adapt behind QuantGold interfaces)

| XAUBot module | QuantGold destination | Notes |
|---------------|----------------------|-------|
| `src/mt5_connector.py` | `quantgold/execution/mt5.py` (future) | Keep connection/sim patterns; rewrite API |
| `src/config.py` | `quantgold/config/` | Pattern only — new YAML settings |
| `src/db/*` + `docker/` | ops / monitoring | Logging infra |
| `src/telegram_*.py` | monitoring alerts | Optional |
| `src/risk_engine.py` concepts | `quantgold/risk/` | Rebuild with capped confidence sizing |
| `src/session_filter.py` sessions | `quantgold/features/sessions.py` (future) | Re-validate empirically |
| `web-dashboard/` | monitoring UI later | No performance claims |

## REPLACE / DO NOT PORT AS-IS

| XAUBot module | Reason |
|---------------|--------|
| `src/auto_trainer.py` | `target_return` feature leakage |
| `src/ml_model.py` | Noisy 1-bar target + fragile validation |
| `src/smc_polars.py` OB writes | Repainting / confirmation-time mismatch |
| V2 `regime_duration_bars` | Future group-length leakage |
| V2 `consecutive_direction` | Future run-length leakage |
| README backtest table | Unverified under QuantGold standards |

## RESEARCH IDEAS (rewrite, don't copy)

- Triple-barrier labeling (`backtests/ml_v3`) — bugs fixed in `quantgold/labels`
- Purged CV concept (`backtests/ml_v2/ml_v2_train.py`)
- Continuous SMC distances — only if OOS ablation helps
- HMM regimes — fold-local fit only

See `docs/audit/XAUBOT_PHASE1_AUDIT.md` for the full classification.
