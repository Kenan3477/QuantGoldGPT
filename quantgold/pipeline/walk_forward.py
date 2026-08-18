"""
Walk-forward training/evaluation pipeline.

Rules:
- Fit regime detector inside each training fold only
- Fit calibrator on validation fold only
- Never touch final holdout for model selection
- Do not claim improvement from a single fold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantgold.config.settings import QuantGoldSettings, load_settings
from quantgold.decision.selective import SelectivePolicy
from quantgold.meta_models.trained import SklearnMetaLabelModel
from quantgold.models.base import Side
from quantgold.models.calibration import ProbabilityCalibrator, evaluate_calibration
from quantgold.models.ensemble import EnsembleAgreementFilter
from quantgold.models.xgboost_model import available_model_backends, make_model
from quantgold.pipeline.dataset import PreparedDataset
from quantgold.regimes.rules import RuleRegimeDetector
from quantgold.validation.purged import purge_embargo_mask
from quantgold.validation.walk_forward import WalkForwardSplitter


@dataclass
class FoldResult:
    fold_id: int
    n_train: int
    n_val: int
    n_test: int
    test_precision_trades: float
    test_coverage: float
    test_brier: float
    test_ece: float
    n_trades: int
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    symbol: str
    timeframe: str
    model_names: List[str]
    folds: List[FoldResult]
    predictions: pd.DataFrame
    summary: Dict[str, Any]


def run_walk_forward(
    dataset: PreparedDataset,
    *,
    settings: Optional[QuantGoldSettings] = None,
    model_names: Optional[List[str]] = None,
    use_regime: bool = True,
    use_meta: bool = True,
    use_calibration: bool = True,
) -> WalkForwardResult:
    settings = settings or load_settings()
    backends = available_model_backends()
    model_names = model_names or (["xgboost"] if "xgboost" in backends else ["sklearn_gbm_baseline"])
    model_names = [m for m in model_names if m in backends or m == "sklearn_gbm_baseline"]
    if not model_names:
        model_names = ["sklearn_gbm_baseline"]

    df = dataset.frame.copy()
    time_col = "available_timestamp"
    splitter = WalkForwardSplitter(settings.validation)
    policy = SelectivePolicy(
        min_calibrated_probability=settings.decision.min_calibrated_probability,
        min_meta_probability=0.60,
        max_disagreement=settings.decision.max_model_disagreement,
        require_meta=use_meta and settings.decision.enable_meta_label,
    )
    ensemble = EnsembleAgreementFilter(
        max_disagreement=settings.decision.max_model_disagreement,
        min_mean_probability=settings.decision.min_calibrated_probability,
    )

    fold_results: List[FoldResult] = []
    pred_rows: List[Dict[str, Any]] = []

    # If chronological year folds don't fit short Yahoo history, fall back to bar-fraction folds
    splits = list(splitter.iter_masks(df, time_col=time_col))
    if not splits:
        splits = list(_fractional_folds(df, time_col=time_col, n_folds=3))

    feature_cols = dataset.feature_columns

    for sp, train_mask, val_mask, test_mask in splits:
        train_mask = purge_embargo_mask(
            df[time_col],
            train_mask,
            test_mask,
            label_horizon_bars=settings.validation.purge_label_horizon_bars,
            embargo_bars=settings.validation.embargo_bars,
        )
        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]
        test_df = df.loc[test_mask]
        if len(train_df) < 100 or len(test_df) < 20:
            continue

        # Fold-local regime
        if use_regime:
            regime = RuleRegimeDetector().fit(train_df)
            for part_name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
                part = part.copy()
                part["regime"] = regime.predict(part)
                if part_name == "train":
                    train_df = part
                elif part_name == "val":
                    val_df = part
                else:
                    test_df = part
            # One-hot regime into features for this fold only
            for rname in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY", "TRANSITION"]:
                col = f"regime_{rname}"
                train_df[col] = (train_df["regime"] == rname).astype(float)
                val_df[col] = (val_df["regime"] == rname).astype(float)
                test_df[col] = (test_df["regime"] == rname).astype(float)
            fold_features = feature_cols + [f"regime_{r}" for r in [
                "TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY", "TRANSITION"
            ]]
        else:
            fold_features = feature_cols

        X_train = train_df[fold_features].astype(float).fillna(0.0)
        y_train = train_df[dataset.label_column]
        X_val = val_df[fold_features].astype(float).fillna(0.0)
        y_val = val_df[dataset.label_column]
        X_test = test_df[fold_features].astype(float).fillna(0.0)
        y_test = test_df[dataset.label_column]

        # Train models
        fitted = []
        for name in model_names:
            model = make_model(name, random_state=settings.random_seed)
            model.fit(X_train, y_train)
            fitted.append(model)

        # Validation probabilities for calibration / meta
        val_member_p = [m.predict_proba(X_val)[:, -1] for m in fitted]
        val_p = np.mean(np.vstack(val_member_p), axis=0)
        val_success = (y_val.astype(float) == 1).astype(int)

        calibrator = ProbabilityCalibrator("isotonic" if use_calibration else "none")
        if use_calibration and len(val_df) >= 30 and val_success.nunique() > 1:
            calibrator.fit(val_success, pd.Series(val_p, index=val_df.index))

        meta = SklearnMetaLabelModel(random_state=settings.random_seed)
        if use_meta and len(val_df) >= 30:
            meta_X = pd.DataFrame(
                {
                    "probability_buy": val_p,
                    "raw_confidence": np.maximum(val_p, 1 - val_p),
                    "atr_pct_14": val_df["atr_pct_14"].fillna(0.0).values if "atr_pct_14" in val_df else 0.0,
                    "realized_vol_20": val_df["realized_vol_20"].fillna(0.0).values if "realized_vol_20" in val_df else 0.0,
                },
                index=val_df.index,
            )
            # Success if label matches predicted side
            pred_side_up = val_p >= 0.5
            meta_y = ((pred_side_up & (y_val.astype(float) == 1)) | ((~pred_side_up) & (y_val.astype(float) == -1))).astype(int)
            meta.fit(meta_X, meta_y)

        # Test predictions
        from quantgold.models.base import ModelPrediction

        test_member_p = [m.predict_proba(X_test)[:, -1] for m in fitted]
        n_test = len(test_df)
        for i in range(n_test):
            preds_i = []
            for k, m in enumerate(fitted):
                pb = float(test_member_p[k][i])
                ps = 1.0 - pb
                side = Side.BUY if pb >= ps else Side.SELL
                preds_i.append(ModelPrediction(side, pb, ps, max(pb, ps), m.name))
            comb = ensemble.combine(preds_i)
            raw_p = comb.probability_buy
            cal_p = float(calibrator.transform(pd.Series([raw_p])).iloc[0])
            # Re-orient confidence for chosen side
            if comb.side == Side.SELL:
                cal_conf = 1.0 - cal_p
                cand = Side.SELL
            elif comb.side == Side.BUY:
                cal_conf = cal_p
                cand = Side.BUY
            else:
                cal_conf = max(cal_p, 1 - cal_p)
                cand = Side.NO_TRADE

            meta_X_row = pd.DataFrame(
                {
                    "probability_buy": [cal_p],
                    "raw_confidence": [cal_conf],
                    "atr_pct_14": [float(test_df.iloc[i].get("atr_pct_14", 0.0) or 0.0)],
                    "realized_vol_20": [float(test_df.iloc[i].get("realized_vol_20", 0.0) or 0.0)],
                }
            )
            meta_dec = meta.decide(meta_X_row, min_success_probability=0.60)[0]
            event_blocked = bool(test_df.iloc[i].get("event_block", 0) or 0)
            decision = policy.decide(
                candidate_side=cand if comb.side != Side.NO_TRADE else Side.NO_TRADE,
                calibrated_probability=cal_conf,
                meta_probability=meta_dec.success_probability,
                disagreement=float(comb.extras.get("disagreement", 0.0)) if comb.extras else 0.0,
                event_blocked=event_blocked,
            )
            y_i = float(y_test.iloc[i])
            success = None
            if decision.side == Side.BUY:
                success = y_i == 1.0
            elif decision.side == Side.SELL:
                success = y_i == -1.0

            pred_rows.append(
                {
                    "fold_id": sp.fold_id,
                    "timestamp": test_df.iloc[i][time_col],
                    "symbol": dataset.symbol,
                    "side": decision.side.value,
                    "calibrated_probability": decision.calibrated_probability,
                    "meta_probability": decision.meta_probability,
                    "reason": decision.reason,
                    "label": y_i,
                    "success": success,
                    "close": float(test_df.iloc[i]["close"]),
                    "disagreement": float(comb.extras.get("disagreement", 0.0)) if comb.extras else 0.0,
                }
            )

        trades = [r for r in pred_rows if r["fold_id"] == sp.fold_id and r["side"] != Side.NO_TRADE.value]
        successes = [r["success"] for r in trades if r["success"] is not None]
        precision = float(np.mean(successes)) if successes else float("nan")
        coverage = len(trades) / max(len(test_df), 1)
        cal_rep = evaluate_calibration(
            (y_test.astype(float) == 1).astype(int),
            pd.Series(np.mean(np.vstack(test_member_p), axis=0), index=test_df.index),
        )
        fold_results.append(
            FoldResult(
                fold_id=sp.fold_id,
                n_train=len(train_df),
                n_val=len(val_df),
                n_test=len(test_df),
                test_precision_trades=precision,
                test_coverage=coverage,
                test_brier=cal_rep.brier,
                test_ece=cal_rep.ece,
                n_trades=len(trades),
                metrics={"calibration_method": cal_rep.method},
            )
        )

    preds = pd.DataFrame(pred_rows)
    traded = preds[preds["side"] != Side.NO_TRADE.value] if not preds.empty else preds
    summary = {
        "n_folds": len(fold_results),
        "n_predictions": int(len(preds)),
        "n_trades": int(len(traded)),
        "mean_coverage": float(np.nanmean([f.test_coverage for f in fold_results])) if fold_results else float("nan"),
        "mean_precision_trades": float(np.nanmean([f.test_precision_trades for f in fold_results])) if fold_results else float("nan"),
        "mean_brier": float(np.nanmean([f.test_brier for f in fold_results])) if fold_results else float("nan"),
        "mean_ece": float(np.nanmean([f.test_ece for f in fold_results])) if fold_results else float("nan"),
        "note": "Research metrics only — not production performance claims. Costs applied in backtester.",
    }
    return WalkForwardResult(
        symbol=dataset.symbol,
        timeframe=dataset.timeframe,
        model_names=model_names,
        folds=fold_results,
        predictions=preds,
        summary=summary,
    )


def _fractional_folds(df: pd.DataFrame, time_col: str, n_folds: int = 3):
    """Fallback for short histories: expanding train / val / test by fractions."""
    from quantgold.validation.walk_forward import WalkForwardSplit

    n = len(df)
    t = pd.to_datetime(df[time_col], utc=True)
    # each fold uses 60% train, 20% val, 20% test sliding
    for fold_id in range(n_folds):
        end = n - (n_folds - fold_id - 1) * max(n // 10, 1)
        start = 0
        span = end - start
        if span < 150:
            continue
        train_end = start + int(span * 0.6)
        val_end = start + int(span * 0.8)
        train = pd.Series([False] * n)
        val = pd.Series([False] * n)
        test = pd.Series([False] * n)
        train.iloc[start:train_end] = True
        val.iloc[train_end:val_end] = True
        test.iloc[val_end:end] = True
        sp = WalkForwardSplit(
            fold_id=fold_id,
            train_start=t.iloc[start],
            train_end=t.iloc[train_end - 1],
            validation_start=t.iloc[train_end],
            validation_end=t.iloc[val_end - 1],
            test_start=t.iloc[val_end],
            test_end=t.iloc[end - 1],
        )
        yield sp, train, val, test
