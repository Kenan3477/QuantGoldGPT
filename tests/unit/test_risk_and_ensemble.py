from quantgold.config.settings import RiskConfig
from quantgold.models.base import ModelPrediction, Side
from quantgold.models.ensemble import EnsembleAgreementFilter
from quantgold.risk.engine import RiskEngine


def test_risk_engine_caps_confidence_multiplier():
    engine = RiskEngine(RiskConfig(risk_per_trade_pct=1.0, max_confidence_size_multiplier=1.25))
    low = engine.size_order(equity=10_000, stop_distance_price=10.0, confidence=0.0)
    high = engine.size_order(equity=10_000, stop_distance_price=10.0, confidence=1.0)
    assert low.approved and high.approved
    assert high.lots <= low.lots * 1.25 + 1e-9
    assert high.risk_amount <= 10_000 * 0.01 * 1.25 + 1e-9


def test_risk_engine_blocks_daily_loss():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=2.0))
    engine.update_state(daily_loss_pct=2.5)
    decision = engine.size_order(equity=10_000, stop_distance_price=10.0, confidence=0.9)
    assert decision.approved is False
    assert decision.reason == "max_daily_loss"


def test_ensemble_disagreement_forces_no_trade():
    filt = EnsembleAgreementFilter(max_disagreement=0.20, min_mean_probability=0.65)
    preds = [
        ModelPrediction(Side.BUY, 0.82, 0.18, 0.82, "xgb"),
        ModelPrediction(Side.BUY, 0.54, 0.46, 0.54, "lgbm"),
        ModelPrediction(Side.SELL, 0.48, 0.52, 0.52, "cat"),
    ]
    out = filt.combine(preds)
    assert out.side == Side.NO_TRADE
    assert out.extras["disagreement"] > 0.20


def test_ensemble_agreement_allows_trade():
    filt = EnsembleAgreementFilter(max_disagreement=0.20, min_mean_probability=0.65)
    preds = [
        ModelPrediction(Side.BUY, 0.81, 0.19, 0.81, "xgb"),
        ModelPrediction(Side.BUY, 0.79, 0.21, 0.79, "lgbm"),
        ModelPrediction(Side.BUY, 0.77, 0.23, 0.77, "cat"),
    ]
    out = filt.combine(preds)
    assert out.side == Side.BUY
