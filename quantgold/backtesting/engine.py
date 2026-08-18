"""
Realistic event-driven backtest over selective predictions.

Applies spread, commission, slippage. Uses triple-barrier path when OHLC provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quantgold.backtesting.metrics import coverage_precision_table, trade_summary
from quantgold.config.settings import ExecutionCostConfig, TripleBarrierConfig
from quantgold.models.base import Side


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]
    by_session: pd.DataFrame
    by_confidence: pd.DataFrame
    coverage_precision: pd.DataFrame


class RealisticBacktester:
    def __init__(
        self,
        costs: Optional[ExecutionCostConfig] = None,
        barriers: Optional[TripleBarrierConfig] = None,
        point_value: float = 1.0,
        lots: float = 0.1,
    ):
        self.costs = costs or ExecutionCostConfig()
        self.barriers = barriers or TripleBarrierConfig()
        self.point_value = point_value
        self.lots = lots

    def run(
        self,
        predictions: pd.DataFrame,
        ohlc: pd.DataFrame,
        *,
        time_col: str = "timestamp",
    ) -> BacktestResult:
        """
        predictions: side, calibrated_probability, timestamp/close/label...
        ohlc: full bar history with timestamp/high/low/close/atr optional
        """
        if predictions.empty:
            empty = pd.DataFrame()
            return BacktestResult(empty, pd.Series(dtype=float), trade_summary(pd.Series(dtype=float)), empty, empty, empty)

        bars = ohlc.copy()
        bars[time_col] = pd.to_datetime(bars[time_col], utc=True)
        bars = bars.sort_values(time_col).reset_index(drop=True)
        # map timestamp -> positional index via available_timestamp if present
        idx_col = "available_timestamp" if "available_timestamp" in bars.columns else time_col
        bars[idx_col] = pd.to_datetime(bars[idx_col], utc=True)
        time_to_i = {t: i for i, t in enumerate(bars[idx_col])}

        if "atr_14" not in bars.columns:
            prev = bars["close"].shift(1)
            tr = pd.concat(
                [
                    (bars["high"] - bars["low"]).abs(),
                    (bars["high"] - prev).abs(),
                    (bars["low"] - prev).abs(),
                ],
                axis=1,
            ).max(axis=1)
            bars["atr_14"] = tr.rolling(self.barriers.atr_period, min_periods=self.barriers.atr_period).mean()

        trade_rows = []
        for _, pred in predictions.iterrows():
            side = pred["side"]
            if side == Side.NO_TRADE.value:
                continue
            ts = pd.Timestamp(pred.get("timestamp") or pred.get(time_col))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            # find bar index
            i = time_to_i.get(ts)
            if i is None:
                # nearest previous
                candidates = bars.index[bars[idx_col] <= ts]
                if len(candidates) == 0:
                    continue
                i = int(candidates.max())
            entry_mid = float(bars.loc[i, "close"])
            atr = float(bars.loc[i, "atr_14"]) if not pd.isna(bars.loc[i, "atr_14"]) else entry_mid * 0.002
            half_spread = float(self.costs.spread_points) * 0.01  # points→price proxy
            slip = float(self.costs.slippage_points) * 0.01
            if side == Side.BUY.value:
                entry = entry_mid + half_spread + slip
                upper = entry + self.barriers.upper_atr_mult * atr
                lower = entry - self.barriers.lower_atr_mult * atr
            else:
                entry = entry_mid - half_spread - slip
                upper = entry + self.barriers.lower_atr_mult * atr  # adverse for short naming
                lower = entry - self.barriers.upper_atr_mult * atr

            exit_price = entry
            exit_reason = "timeout"
            hold = 0
            for j in range(i + 1, min(len(bars), i + 1 + self.barriers.max_holding_bars)):
                hold = j - i
                hi = float(bars.loc[j, "high"])
                lo = float(bars.loc[j, "low"])
                if side == Side.BUY.value:
                    hit_up = hi >= upper
                    hit_dn = lo <= lower
                    if hit_up and hit_dn:
                        exit_price = lower  # conservative
                        exit_reason = "ambiguous_conservative_loss"
                        break
                    if hit_up:
                        exit_price = upper - half_spread - slip
                        exit_reason = "take_profit"
                        break
                    if hit_dn:
                        exit_price = lower - half_spread - slip
                        exit_reason = "stop_loss"
                        break
                else:
                    # SELL profits when price goes down to lower barrier relative to entry
                    tp = entry - self.barriers.upper_atr_mult * atr
                    sl = entry + self.barriers.lower_atr_mult * atr
                    hit_tp = lo <= tp
                    hit_sl = hi >= sl
                    if hit_tp and hit_sl:
                        exit_price = sl
                        exit_reason = "ambiguous_conservative_loss"
                        break
                    if hit_tp:
                        exit_price = tp + half_spread + slip
                        exit_reason = "take_profit"
                        break
                    if hit_sl:
                        exit_price = sl + half_spread + slip
                        exit_reason = "stop_loss"
                        break
            else:
                j = min(len(bars) - 1, i + self.barriers.max_holding_bars)
                hold = max(1, j - i)
                exit_price = float(bars.loc[j, "close"])
                if side == Side.BUY.value:
                    exit_price -= half_spread + slip
                else:
                    exit_price += half_spread + slip
                exit_reason = "timeout"

            if side == Side.BUY.value:
                pnl = (exit_price - entry) * self.lots * self.point_value
            else:
                pnl = (entry - exit_price) * self.lots * self.point_value
            pnl -= float(self.costs.commission_per_lot) * self.lots

            trade_rows.append(
                {
                    "timestamp": ts,
                    "side": side,
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "hold_bars": hold,
                    "exit_reason": exit_reason,
                    "calibrated_probability": float(pred.get("calibrated_probability", np.nan)),
                    "hour": ts.hour,
                }
            )

        trades = pd.DataFrame(trade_rows)
        pnl = trades["pnl"] if not trades.empty else pd.Series(dtype=float)
        equity = pnl.cumsum() if not pnl.empty else pd.Series(dtype=float)
        metrics = trade_summary(pnl)
        if not trades.empty:
            metrics["avg_hold_bars"] = float(trades["hold_bars"].mean())
            metrics["timeout_rate"] = float((trades["exit_reason"] == "timeout").mean())

        by_session = (
            trades.groupby(trades["hour"].map(_session_name))["pnl"].agg(["count", "mean", "sum"])
            if not trades.empty
            else pd.DataFrame()
        )
        if not trades.empty:
            trades["conf_bucket"] = pd.cut(
                trades["calibrated_probability"],
                bins=[0, 0.55, 0.65, 0.75, 0.85, 1.01],
                labels=["<55", "55-65", "65-75", "75-85", "85+"],
            )
            by_confidence = trades.groupby("conf_bucket", observed=False)["pnl"].agg(["count", "mean", "sum"])
            cov = coverage_precision_table(
                trades["calibrated_probability"],
                (trades["pnl"] > 0).astype(float),
                thresholds=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            )
        else:
            by_confidence = pd.DataFrame()
            cov = pd.DataFrame()

        return BacktestResult(trades, equity, metrics, by_session, by_confidence, cov)


def _session_name(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 16:
        return "overlap"
    if 16 <= hour < 21:
        return "new_york"
    return "off"
