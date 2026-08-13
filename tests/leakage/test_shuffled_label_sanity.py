"""Sanity: shuffled labels should destroy walk-forward trade precision."""

import numpy as np

from quantgold.config.settings import load_settings
from quantgold.data.build import build_canonical_dataset, get_source
from quantgold.pipeline.dataset import prepare_research_dataset
from quantgold.pipeline.walk_forward import run_walk_forward


def test_shuffled_labels_near_chance(tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path / "datasets")
    settings.decision.min_calibrated_probability = 0.55
    source = get_source("synthetic", seed=1)
    build_canonical_dataset("XAUUSD", "H1", source=source, settings=settings, limit=1200)
    ds = prepare_research_dataset("XAUUSD", "H1", settings=settings)
    rng = np.random.RandomState(0)
    ds.frame = ds.frame.copy()
    ds.frame["tb_label"] = rng.permutation(ds.frame["tb_label"].to_numpy())
    wf = run_walk_forward(
        ds,
        settings=settings,
        model_names=["sklearn_gbm_baseline"],
        use_meta=False,
        use_calibration=False,
    )
    # With pure noise labels, precision among trades should not be stably excellent
    if wf.summary["n_trades"] >= 30 and not np.isnan(wf.summary["mean_precision_trades"]):
        assert wf.summary["mean_precision_trades"] < 0.75
