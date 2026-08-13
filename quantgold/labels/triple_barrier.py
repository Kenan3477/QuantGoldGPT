"""
Triple-barrier style event labels.

At prediction time T (bar close):
  upper = close + upper_atr_mult * ATR
  lower = close - lower_atr_mult * ATR
  time  = T + max_holding_bars

Target:
  +1 upside barrier first
  -1 downside barrier first
   0 timeout / neither
   2 ambiguous (both touched same bar; policy-dependent)

IMPORTANT:
- Labels are research outputs, never feature inputs.
- same_bar_policy defaults to 'ambiguous' — we do not silently favour BUY.
- Barrier multipliers are configuration parameters, not proven optima.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from quantgold.config.settings import TripleBarrierConfig

LABEL_UP = 1
LABEL_DOWN = -1
LABEL_TIMEOUT = 0
LABEL_AMBIGUOUS = 2

SameBarPolicy = Literal["ambiguous", "favor_upper", "favor_lower", "no_trade"]


@dataclass
class TripleBarrierResult:
    labels: pd.Series
    upper_barrier: pd.Series
    lower_barrier: pd.Series
    touch_bar: pd.Series
    atr: pd.Series


class TripleBarrierLabeler:
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        self.config = config or TripleBarrierConfig()

    def compute_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        period = period or self.config.atr_period
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    def label(self, df: pd.DataFrame) -> TripleBarrierResult:
        """
        Label each row using only future path *after* the decision bar.

        Decision uses close[i]; path inspection starts at i+1.
        """
        if not {"high", "low", "close"}.issubset(df.columns):
            raise KeyError("DataFrame requires high/low/close")

        cfg = self.config
        n = len(df)
        atr = self.compute_atr(df)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        atr_v = atr.to_numpy(dtype=float)

        labels = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        touch = np.full(n, np.nan)

        horizon = int(cfg.max_holding_bars)
        for i in range(n):
            if np.isnan(atr_v[i]) or atr_v[i] <= 0:
                continue
            up = close[i] + cfg.upper_atr_mult * atr_v[i]
            dn = close[i] - cfg.lower_atr_mult * atr_v[i]
            upper[i] = up
            lower[i] = dn

            end = min(n - 1, i + horizon)
            decided = False
            for j in range(i + 1, end + 1):
                hit_up = high[j] >= up
                hit_dn = low[j] <= dn
                if hit_up and hit_dn:
                    labels[i] = self._resolve_same_bar(cfg.same_bar_policy)
                    touch[i] = j
                    decided = True
                    break
                if hit_up:
                    labels[i] = LABEL_UP
                    touch[i] = j
                    decided = True
                    break
                if hit_dn:
                    labels[i] = LABEL_DOWN
                    touch[i] = j
                    decided = True
                    break
            if not decided:
                labels[i] = LABEL_TIMEOUT
                touch[i] = end

        idx = df.index
        return TripleBarrierResult(
            labels=pd.Series(labels, index=idx, name="tb_label"),
            upper_barrier=pd.Series(upper, index=idx, name="tb_upper"),
            lower_barrier=pd.Series(lower, index=idx, name="tb_lower"),
            touch_bar=pd.Series(touch, index=idx, name="tb_touch_bar"),
            atr=atr.rename("tb_atr"),
        )

    @staticmethod
    def _resolve_same_bar(policy: str) -> int:
        if policy == "favor_upper":
            return LABEL_UP
        if policy == "favor_lower":
            return LABEL_DOWN
        if policy == "no_trade":
            return LABEL_TIMEOUT
        return LABEL_AMBIGUOUS

    @staticmethod
    def label_columns() -> tuple[str, ...]:
        """Columns that must never enter feature matrices."""
        return (
            "tb_label",
            "tb_upper",
            "tb_lower",
            "tb_touch_bar",
            "tb_atr",
            "target",
            "target_return",
            "multi_bar_target",
            "label",
            "y",
        )
