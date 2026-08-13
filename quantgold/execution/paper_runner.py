"""
Paper-trading runner.

Loads latest canonical bars, builds features, applies a frozen production candidate
model artifact if present, otherwise a freshly fit research model on historical data
for smoke testing — never silently promotes to production.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quantgold.config.settings import load_settings
from quantgold.data.store import CanonicalDataStore
from quantgold.decision.selective import SelectivePolicy
from quantgold.execution.guard import assert_no_research_imports
from quantgold.execution.paper import PaperBroker
from quantgold.features.bundle import FeatureBundle
from quantgold.labels.triple_barrier import TripleBarrierLabeler
from quantgold.models.base import Side
from quantgold.models.xgboost_model import make_model, available_model_backends
from quantgold.risk.engine import RiskEngine


class PaperTradingRunner:
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        log_dir: str | Path = "artifacts/reports/paper",
    ):
        assert_no_research_imports()
        self.settings = load_settings()
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.store = CanonicalDataStore(self.settings.data_root)
        self.broker = PaperBroker()
        self.risk = RiskEngine(self.settings.risk)
        self.policy = SelectivePolicy(
            min_calibrated_probability=self.settings.decision.min_calibrated_probability,
            require_meta=False,
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_once(self) -> Dict[str, Any]:
        df = self.store.load_ohlcv(self.symbol, self.timeframe)
        feats = FeatureBundle().transform(df)
        frame = feats.frame.copy()
        optional = set(feats.families.get("intermarket", [])) | set(feats.families.get("macro", []))
        for c in optional:
            if c in frame.columns:
                frame[c] = frame[c].fillna(0.0)
        core = [c for c in feats.feature_columns if c not in optional]
        frame = frame.dropna(subset=core)
        if len(frame) < 200:
            return {"accepted": False, "reason": "insufficient_bars", "n_bars": int(len(frame))}

        # Fit a disposable research model on history excluding last bar (paper smoke only)
        train = frame.iloc[:-1]
        live = frame.iloc[[-1]]
        labels = TripleBarrierLabeler(self.settings.triple_barrier).label(train)
        train = train.copy()
        train["tb_label"] = labels.labels
        train = train.dropna(subset=feats.feature_columns + ["tb_label"])
        train = train[train["tb_label"].isin([1, -1, 0])]

        backend = "xgboost" if "xgboost" in available_model_backends() else "sklearn_gbm_baseline"
        model = make_model(backend, random_state=self.settings.random_seed)
        model.fit(train[feats.feature_columns].fillna(0.0), train["tb_label"])
        proba = model.predict_proba(live[feats.feature_columns].fillna(0.0))[0, -1]
        side = Side.BUY if proba >= 0.5 else Side.SELL
        conf = float(proba if side == Side.BUY else 1 - proba)
        decision = self.policy.decide(
            candidate_side=side if conf >= self.settings.decision.min_calibrated_probability else Side.NO_TRADE,
            calibrated_probability=conf,
            meta_probability=conf,
        )

        atr = float(live.iloc[0].get("atr_14") or live.iloc[0]["close"] * 0.002)
        risk_dec = self.risk.size_order(
            equity=10_000,
            stop_distance_price=self.settings.triple_barrier.lower_atr_mult * atr,
            confidence=decision.calibrated_probability,
        )

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model": backend,
            "probability_buy": float(proba),
            "side": decision.side.value,
            "reason": decision.reason,
            "calibrated_probability": decision.calibrated_probability,
            "feature_snapshot": {c: float(live.iloc[0][c]) for c in feats.feature_columns[:20]},
            "risk_approved": risk_dec.approved,
            "lots": risk_dec.lots,
            "bar_available_timestamp": str(live.iloc[0]["available_timestamp"]),
            "stage": "paper",
        }

        if decision.side != Side.NO_TRADE and risk_dec.approved:
            from quantgold.execution.base import OrderRequest

            self.broker.fill_price = float(live.iloc[0]["close"])
            result = self.broker.submit(
                OrderRequest(self.symbol, decision.side, risk_dec.lots, comment="QuantGold-paper")
            )
            record["order"] = {"accepted": result.accepted, "order_id": result.order_id, "reason": result.reason}
        else:
            record["order"] = {"accepted": False, "order_id": None, "reason": decision.reason}

        out = self.log_dir / f"paper_{self.symbol}_{self.timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
