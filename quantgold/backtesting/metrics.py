"""Core metric helpers — expand in M6; no fabricated performance claims."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def classification_summary(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    if len(yt) == 0:
        return {"n": 0, "accuracy": float("nan")}
    acc = float((yt == yp).mean())
    return {"n": int(len(yt)), "accuracy": acc}


def trade_summary(pnl: pd.Series) -> Dict[str, float]:
    x = pnl.dropna().astype(float)
    if x.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "profit_factor": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    wins = x[x > 0]
    losses = x[x < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    equity = x.cumsum()
    dd = equity - equity.cummax()
    sharpe = float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
    return {
        "n_trades": int(len(x)),
        "win_rate": float((x > 0).mean()),
        "expectancy": float(x.mean()),
        "profit_factor": float(pf),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
    }


def coverage_precision_table(
    probabilities: pd.Series,
    outcomes_success: pd.Series,
    thresholds: list[float],
) -> pd.DataFrame:
    """Coverage vs precision curve for selective trading research."""
    rows = []
    for thr in thresholds:
        mask = probabilities >= thr
        n = int(mask.sum())
        coverage = float(mask.mean()) if len(mask) else 0.0
        precision = float(outcomes_success[mask].mean()) if n else float("nan")
        rows.append(
            {
                "threshold": thr,
                "trades": n,
                "coverage": coverage,
                "precision": precision,
            }
        )
    return pd.DataFrame(rows)
