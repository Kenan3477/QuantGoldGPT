import pandas as pd

from quantgold.decision.selective import SelectivePolicy
from quantgold.models.base import Side
from quantgold.models.calibration import ProbabilityCalibrator, evaluate_calibration


def test_selective_policy_prefers_no_trade():
    policy = SelectivePolicy(min_calibrated_probability=0.7, min_meta_probability=0.7, require_meta=True)
    d = policy.decide(candidate_side=Side.BUY, calibrated_probability=0.82, meta_probability=0.55)
    assert d.side == Side.NO_TRADE
    assert d.reason == "meta_reject"


def test_calibrator_isotonic_fits():
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1, 1, 0] * 5)
    p = pd.Series([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.75, 0.9, 0.35] * 5)
    cal = ProbabilityCalibrator("isotonic").fit(y, p)
    out = cal.transform(p)
    assert len(out) == len(p)
    rep = evaluate_calibration(y, out, method="isotonic")
    assert rep.n == len(y)
