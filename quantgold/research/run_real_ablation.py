"""
Real feature ablation study on actual XAUUSD data.

Run walk-forward validation with incremental feature addition:
1. Base features only
2. Base + Microstructure
3. Base + Micro + MTF
4. Base + Micro + MTF + SMC
5. Base + Micro + MTF + SMC + Intermarket

For each configuration:
- Run simplified walk-forward (3 folds, faster)
- Measure OOS precision, win rate, Sharpe
- Compare to baseline

This will take 10-30 minutes depending on data size.
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
from typing import Dict, List

from quantgold.config.settings import load_settings
from quantgold.data.store import CanonicalDataStore
from quantgold.pipeline.dataset import prepare_research_dataset
from quantgold.features.bundle import FeatureBundle
from quantgold.labels.triple_barrier import TripleBarrierLabeler
from quantgold.research.feature_ablation import FeatureAblationStudy, AblationResult


def get_feature_family_columns(df: pl.DataFrame) -> Dict[str, List[str]]:
    """
    Identify which columns belong to each feature family.
    
    Returns:
        Dict mapping family name → list of column names
    """
    all_cols = df.columns
    
    # Remove OHLCV, timestamp, and label columns
    base_ohlcv = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'available_timestamp']
    forbidden = TripleBarrierLabeler.label_columns()
    
    feature_cols = [c for c in all_cols if c not in base_ohlcv and c not in forbidden]
    
    # Categorize by prefix/name patterns
    families = {
        "base": [],
        "micro": [],
        "mtf": [],
        "smc": [],
        "intermarket": [],
    }
    
    for col in feature_cols:
        col_lower = col.lower()
        
        # Microstructure features
        if any(x in col_lower for x in ['spread', 'body', 'wick', 'range_vs', 'opening_range', 'dist_from_day_open']):
            families["micro"].append(col)
        
        # Multi-timeframe features
        elif any(x in col_lower for x in ['_m15_', '_h1_', '_h4_', '_d1_', 'tf_bullish', 'tf_bearish', 'atr_h1_vs', 'atr_h4_vs']):
            families["mtf"].append(col)
        
        # SMC features
        elif any(x in col_lower for x in ['swing', 'fvg', 'bos', 'choch', '_ob', '_bullish_', '_bearish_']):
            families["smc"].append(col)
        
        # Intermarket features
        elif any(x in col_lower for x in ['dxy', 'vix', 'spx', 'xau_xag', 'yield', 'treasury']):
            families["intermarket"].append(col)
        
        # Default to base features
        else:
            families["base"].append(col)
    
    # Print summary
    print("\nFeature Family Distribution:")
    for family, cols in families.items():
        print(f"  {family}: {len(cols)} features")
        if len(cols) <= 5:
            for c in cols:
                print(f"    - {c}")
    
    return families


def run_real_ablation(
    instrument: str = "XAUUSD",
    timeframe: str = "M5",
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
    use_ensemble: bool = False,
):
    """
    Run ablation study on real XAUUSD data.
    
    Args:
        instrument: Instrument symbol
        timeframe: Timeframe to test
        start_date: Start date for data
        end_date: End date for data
        use_ensemble: Use 3-model ensemble instead of single XGBoost
    """
    print("="*80)
    print("REAL FEATURE ABLATION STUDY")
    print("="*80)
    print(f"Instrument: {instrument}")
    print(f"Timeframe: {timeframe}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Model: {'3-Model Ensemble' if use_ensemble else 'XGBoost'}")
    print("="*80)
    
    # Load configuration
    cfg = load_settings("configs/default.yaml")
    
    # Load canonical data
    print("\n[1/5] Loading canonical dataset...")
    store = CanonicalDataStore(root=Path("artifacts/datasets"))
    
    try:
        df_pd = store.load_ohlcv(instrument, timeframe)
        df = pl.from_pandas(df_pd)
        print(f"Loaded {len(df)} bars")
    except FileNotFoundError:
        print(f"Error: No data found for {instrument} {timeframe}")
        print("Run: quantgold build-datasets --source yfinance")
        sys.exit(1)
    
    # Filter date range if specified (skip if dates are default)
    if start_date != "2024-01-01" or end_date != "2024-03-31":
        start_dt = pl.lit(start_date).str.strptime(pl.Datetime("ms", "UTC"), "%Y-%m-%d")
        end_dt = pl.lit(end_date).str.strptime(pl.Datetime("ms", "UTC"), "%Y-%m-%d")
        
        df_filtered = df.filter(
            (pl.col("timestamp") >= start_dt) &
            (pl.col("timestamp") <= end_dt)
        )
        
        if len(df_filtered) > 100:
            df = df_filtered
            print(f"Filtered to {len(df)} bars in date range")
        else:
            print(f"Warning: Date filter would leave only {len(df_filtered)} bars. Using full dataset.")
            print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print(f"Using full dataset: {len(df)} bars")
        print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if len(df) < 1000:
        print("Warning: Less than 1000 bars, results may not be reliable")
    
    # Build ALL features
    print("\n[2/5] Building complete feature set...")
    from quantgold.features.bundle import FeatureBundleConfig
    
    config = FeatureBundleConfig(
        use_base=True,
        use_sessions=True,
        use_structure=True,
        use_intermarket=True,
        use_macro=False,  # Skip macro for faster testing
    )
    bundle = FeatureBundle(config=config)
    
    # Convert to pandas for feature building (bundle expects pandas)
    df_pd = df.to_pandas()
    
    # Build features
    built = bundle.transform(df_pd)
    df_features_pd = built.frame
    
    # Convert back to polars
    df_features = pl.from_pandas(df_features_pd)
    print(f"Built features: {len(df_features.columns)} columns")
    
    # Add labels
    print("\n[3/5] Adding triple-barrier labels...")
    
    labeler = TripleBarrierLabeler(config=cfg.triple_barrier)
    result = labeler.label(df_features_pd)
    
    # Merge labels back into the dataframe
    df_features_pd['label_side'] = result.labels
    
    # Convert to polars
    df_labeled = pl.from_pandas(df_features_pd)
    
    print(f"Labels added. Dataset size: {len(df_labeled)} rows")
    
    # Drop rows with missing labels
    df_clean = df_labeled.drop_nulls(subset=['label_side'])
    print(f"After dropping nulls: {len(df_clean)} rows")
    
    # Identify feature families
    print("\n[4/5] Identifying feature families...")
    feature_families = get_feature_family_columns(df_clean)
    
    # Convert to pandas for sklearn models
    df_pd = df_clean.to_pandas()
    
    # Prepare X and y
    forbidden_cols = list(TripleBarrierLabeler.label_columns()) + [
        'timestamp', 'available_timestamp', 'open', 'high', 'low', 'close', 'volume',
        'label_side',  # CRITICAL: exclude the target label itself!
    ]
    
    # Get only numeric columns for features
    numeric_cols = df_pd.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in forbidden_cols]
    
    X = df_pd[feature_cols]
    y = (df_pd['label_side'] == 1).astype(int)  # Binary: 1=BUY, 0=SELL (label_side is already numeric)
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Label distribution: BUY={y.sum()}, SELL={(~y.astype(bool)).sum()}")
    
    # Split train/val (chronological)
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    
    # Run ablation study
    print("\n[5/5] Running ablation study...")
    study = FeatureAblationStudy(output_dir="artifacts/ablation_real")
    
    results = study.run(
        X_train, y_train,
        X_val, y_val,
        feature_families=feature_families,
        use_ensemble=use_ensemble,
    )
    
    # Generate report
    report_path = study.generate_report(results)
    
    print("\n" + "="*80)
    print("✅ ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"📊 Report: {report_path}")
    print("\nKey Findings:")
    
    # Show incremental gains
    for i in range(1, len(results)):
        prev, curr = results[i-1], results[i]
        added_family = curr.feature_families[-1]
        delta_f1 = curr.f1 - prev.f1
        
        status = "✅" if delta_f1 > 0 else "❌"
        print(f"{status} {added_family}: F1 {prev.f1:.3f} → {curr.f1:.3f} (Δ{delta_f1:+.3f})")
    
    print("\nNext steps:")
    print("1. Review the full report in artifacts/ablation_real/ablation_report.md")
    print("2. Remove features that hurt OOS performance")
    print("3. Run full walk-forward with optimized feature set")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run real feature ablation study")
    parser.add_argument("--instrument", default="XAUUSD", help="Instrument symbol")
    parser.add_argument("--timeframe", default="M5", help="Timeframe")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-03-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--ensemble", action="store_true", help="Use 3-model ensemble")
    
    args = parser.parse_args()
    
    run_real_ablation(
        instrument=args.instrument,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        use_ensemble=args.ensemble,
    )
