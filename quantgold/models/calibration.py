"""Probability calibration research utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationReport:
    method: str
    brier: float
    ece: float
    n: int
    reliability: pd.DataFrame


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1] if i < n_bins - 1 else proba <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = proba[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def reliability_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


def evaluate_calibration(y_true: pd.Series, proba: pd.Series, method: str = "raw") -> CalibrationReport:
    yt = y_true.astype(int).to_numpy()
    pr = proba.astype(float).to_numpy()
    mask = np.isfinite(pr)
    yt, pr = yt[mask], pr[mask]
    brier = float(brier_score_loss(yt, pr)) if len(yt) else float("nan")
    ece = expected_calibration_error(yt, pr) if len(yt) else float("nan")
    table = reliability_table(yt, pr) if len(yt) >= 10 else pd.DataFrame()
    return CalibrationReport(method=method, brier=brier, ece=ece, n=len(yt), reliability=table)


class ProbabilityCalibrator:
    """
    Fit isotonic/Platt on validation fold only; apply to test.
    Wraps a binary probability vector — not the full classifier — via isotonic/logistic.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "platt", "none"}:
            raise ValueError(method)
        self.method = method
        self._iso = None
        self._platt = None

    def fit(self, y_true: pd.Series, proba: pd.Series) -> "ProbabilityCalibrator":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        yt = y_true.astype(int).to_numpy()
        pr = proba.astype(float).to_numpy().reshape(-1, 1)
        if self.method == "none":
            return self
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(pr.ravel(), yt)
        else:
            self._platt = LogisticRegression(max_iter=1000)
            self._platt.fit(pr, yt)
        return self

    def transform(self, proba: pd.Series) -> pd.Series:
        pr = proba.astype(float).to_numpy()
        if self.method == "none" or (self._iso is None and self._platt is None):
            return proba.astype(float)
        if self.method == "isotonic":
            return pd.Series(self._iso.transform(pr), index=proba.index)
        return pd.Series(self._platt.predict_proba(pr.reshape(-1, 1))[:, 1], index=proba.index)
