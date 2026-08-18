"""QuantGold command-line entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantgold import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantgold", description="QuantGold research CLI")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-datasets", help="Build canonical parquet datasets")
    p_build.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic", "mt5"])
    p_build.add_argument("--symbols", default="XAUUSD,XAGUSD")
    p_build.add_argument("--timeframes", default="M15,H1,D1")
    p_build.add_argument("--limit", type=int, default=None)

    p_run = sub.add_parser("walk-forward", help="Run walk-forward research pipeline")
    p_run.add_argument("--symbol", default="XAUUSD")
    p_run.add_argument("--timeframe", default="D1")
    p_run.add_argument("--source", default="yfinance")
    p_run.add_argument("--models", default="auto")
    p_run.add_argument("--limit", type=int, default=None)

    p_paper = sub.add_parser("paper-once", help="Run one paper-trading iteration")
    p_paper.add_argument("--symbol", default="XAUUSD")
    p_paper.add_argument("--timeframe", default="D1")

    p_all = sub.add_parser("run-all", help="Build datasets + walk-forward + backtest + paper smoke")
    p_all.add_argument("--source", default="yfinance")
    p_all.add_argument("--symbol", default="XAUUSD")
    p_all.add_argument("--timeframe", default="D1")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.cmd == "build-datasets":
        return cmd_build(args)
    if args.cmd == "walk-forward":
        return cmd_walk_forward(args)
    if args.cmd == "paper-once":
        return cmd_paper(args)
    if args.cmd == "run-all":
        return cmd_run_all(args)
    parser.print_help()
    return 1


def cmd_build(args) -> int:
    from quantgold.data.build import build_all_datasets

    built = build_all_datasets(
        source_name=args.source,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframes=[t.strip() for t in args.timeframes.split(",") if t.strip()],
        limit=args.limit,
    )
    for b in built:
        print(f"OK {b.symbol}/{b.timeframe} rows={b.n_rows} version={b.version_id} src={b.source}")
    return 0


def cmd_walk_forward(args) -> int:
    from quantgold.config.settings import load_settings
    from quantgold.data.build import build_canonical_dataset, get_source
    from quantgold.pipeline.dataset import prepare_research_dataset
    from quantgold.pipeline.walk_forward import run_walk_forward
    from quantgold.backtesting.engine import RealisticBacktester
    from quantgold.monitoring.experiments import ExperimentTracker
    from quantgold.monitoring.registry import ModelRegistry, ModelStage
    from quantgold.data.store import CanonicalDataStore

    settings = load_settings()
    source = get_source(args.source)
    build_canonical_dataset(args.symbol, args.timeframe, source=source, settings=settings, limit=args.limit)

    # peer metal for ratio features
    peer = "XAGUSD" if args.symbol.upper() == "XAUUSD" else "XAUUSD"
    try:
        build_canonical_dataset(peer, args.timeframe, source=source, settings=settings, limit=args.limit)
        peer_df = CanonicalDataStore(settings.data_root).load_ohlcv(peer, args.timeframe)
    except Exception:
        peer_df = None

    # optional externals
    externals = {}
    for ext in ("DXY", "VIX", "US10Y", "SPX"):
        try:
            build_canonical_dataset(ext, "D1", source=source, settings=settings, limit=args.limit)
            externals[ext] = CanonicalDataStore(settings.data_root).load_ohlcv(ext, "D1")
        except Exception:
            pass

    ds = prepare_research_dataset(
        args.symbol,
        args.timeframe,
        settings=settings,
        peer_metal=peer_df,
        externals=externals or None,
    )
    models = None if args.models == "auto" else [m.strip() for m in args.models.split(",")]
    wf = run_walk_forward(ds, settings=settings, model_names=models)

    store = CanonicalDataStore(settings.data_root)
    ohlc = store.load_ohlcv(args.symbol, args.timeframe)
    bt = RealisticBacktester(costs=settings.costs, barriers=settings.triple_barrier).run(wf.predictions, ohlc)

    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "walk_forward_summary": wf.summary,
        "fold_metrics": [
            {
                "fold_id": f.fold_id,
                "n_train": f.n_train,
                "n_val": f.n_val,
                "n_test": f.n_test,
                "precision": f.test_precision_trades,
                "coverage": f.test_coverage,
                "brier": f.test_brier,
                "ece": f.test_ece,
                "n_trades": f.n_trades,
            }
            for f in wf.folds
        ],
        "backtest_metrics": bt.metrics,
        "models": wf.model_names,
        "disclaimer": "Research results only. Not production performance claims.",
    }
    out = report_dir / f"wf_{args.symbol}_{args.timeframe}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not wf.predictions.empty:
        wf.predictions.to_parquet(report_dir / f"preds_{args.symbol}_{args.timeframe}.parquet", index=False)
    if not bt.trades.empty:
        bt.trades.to_parquet(report_dir / f"trades_{args.symbol}_{args.timeframe}.parquet", index=False)

    tracker = ExperimentTracker(settings.experiment_root)
    tracker.start(
        dataset_version=ds.dataset_version,
        instrument=ds.symbol,
        timeframe=ds.timeframe,
        features=ds.feature_columns,
        label_definition={
            "type": "triple_barrier",
            "upper_atr_mult": settings.triple_barrier.upper_atr_mult,
            "lower_atr_mult": settings.triple_barrier.lower_atr_mult,
            "max_holding_bars": settings.triple_barrier.max_holding_bars,
            "same_bar_policy": settings.triple_barrier.same_bar_policy,
        },
        model=wf.model_names,
        threshold=settings.decision.min_calibrated_probability,
        results={"walk_forward": wf.summary, "backtest": bt.metrics},
    )
    ModelRegistry(settings.model_registry_root).register(
        model_id=f"{ds.symbol}_{ds.timeframe}_{'_'.join(wf.model_names)}",
        artifact_path=str(out),
        stage=ModelStage.CANDIDATE,
        dataset_version=ds.dataset_version,
        metrics=wf.summary,
    )

    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out}")
    return 0


def cmd_paper(args) -> int:
    from quantgold.data.build import build_canonical_dataset, get_source
    from quantgold.execution.paper_runner import PaperTradingRunner

    build_canonical_dataset(args.symbol, args.timeframe, source=get_source("yfinance"))
    result = PaperTradingRunner(args.symbol, args.timeframe).run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_run_all(args) -> int:
    class A:
        source = args.source
        symbols = "XAUUSD,XAGUSD"
        timeframes = "D1,H1"
        limit = None

    cmd_build(A())

    class B:
        symbol = args.symbol
        timeframe = args.timeframe
        source = args.source
        models = "auto"
        limit = None

    cmd_walk_forward(B())

    class C:
        symbol = args.symbol
        timeframe = args.timeframe

    cmd_paper(C())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
