"""End-to-end smoke on synthetic data (no network)."""

from quantgold.config.settings import load_settings
from quantgold.data.build import build_canonical_dataset, get_source
from quantgold.pipeline.dataset import prepare_research_dataset
from quantgold.pipeline.walk_forward import run_walk_forward
from quantgold.backtesting.engine import RealisticBacktester
from quantgold.features.registry import FORBIDDEN_LABEL_COLUMNS


def test_synthetic_walk_forward_smoke(tmp_path, monkeypatch):
    settings = load_settings()
    settings.data_root = str(tmp_path / "datasets")
    settings.experiment_root = str(tmp_path / "experiments")
    source = get_source("synthetic", seed=7)
    build_canonical_dataset("XAUUSD", "H1", source=source, settings=settings, limit=800)
    build_canonical_dataset("XAGUSD", "H1", source=source, settings=settings, limit=800)
    ds = prepare_research_dataset("XAUUSD", "H1", settings=settings)
    assert not set(ds.feature_columns) & FORBIDDEN_LABEL_COLUMNS
    assert "tb_label" in ds.frame.columns
    wf = run_walk_forward(ds, settings=settings, model_names=["sklearn_gbm_baseline"], use_meta=True)
    assert wf.summary["n_folds"] >= 1
    assert "mean_precision_trades" in wf.summary
    # backtest should run even if few trades
    from quantgold.data.store import CanonicalDataStore

    ohlc = CanonicalDataStore(settings.data_root).load_ohlcv("XAUUSD", "H1")
    bt = RealisticBacktester(costs=settings.costs, barriers=settings.triple_barrier).run(wf.predictions, ohlc)
    assert "n_trades" in bt.metrics
