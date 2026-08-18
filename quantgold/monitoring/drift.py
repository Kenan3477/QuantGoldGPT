"""Simple drift monitors for prediction/feature distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class DriftAlert:
    name: str
    severity: str
    detail: str


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    quantiles = np.linspace(0, 100, buckets + 1)
    breaks = np.unique(np.percentile(expected, quantiles))
    if len(breaks) < 3:
        return float("nan")
    e_counts = np.histogram(expected, bins=breaks)[0].astype(float)
    a_counts = np.histogram(actual, bins=breaks)[0].astype(float)
    e_perc = np.clip(e_counts / e_counts.sum(), 1e-4, 1)
    a_perc = np.clip(a_counts / a_counts.sum(), 1e-4, 1)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def monitor_predictions(
    reference: pd.Series,
    current: pd.Series,
    *,
    psi_warn: float = 0.1,
    psi_alert: float = 0.25,
) -> list[DriftAlert]:
    score = psi(reference.to_numpy(dtype=float), current.to_numpy(dtype=float))
    alerts: list[DriftAlert] = []
    if np.isnan(score):
        return alerts
    if score >= psi_alert:
        alerts.append(DriftAlert("prediction_psi", "alert", f"PSI={score:.3f}"))
    elif score >= psi_warn:
        alerts.append(DriftAlert("prediction_psi", "warn", f"PSI={score:.3f}"))
    return alerts


def monitor_winrate_degradation(
    recent_success: pd.Series,
    baseline_precision: float,
    *,
    min_trades: int = 30,
    drop: float = 0.1,
) -> list[DriftAlert]:
    alerts: list[DriftAlert] = []
    s = recent_success.dropna()
    if len(s) < min_trades:
        return alerts
    p = float(s.mean())
    if p < baseline_precision - drop:
        alerts.append(
            DriftAlert(
                "precision_degradation",
                "alert",
                f"recent_precision={p:.3f} baseline={baseline_precision:.3f}",
            )
        )
    return alerts
