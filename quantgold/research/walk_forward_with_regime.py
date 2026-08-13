"""
Run walk-forward validation with regime filtering.

This script tests the regime filter's impact on trading performance.
"""

import sys
import argparse
from quantgold.config.settings import load_settings
from quantgold.pipeline.dataset import prepare_research_dataset
from quantgold.pipeline.walk_forward import run_walk_forward
from quantgold.filters.regime_filter import RegimeFilter, RegimeConfig, apply_regime_filter_to_predictions


def run_walk_forward_with_regime_filter(
    symbol: str = "XAUUSD",
    timeframe: str = "H4",
    model_names: list = None,
    disable_choppy: bool = True,
    disable_volatile: bool = True,
):
    """
    Run walk-forward with regime filter applied.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        model_names: Models to use
        disable_choppy: Filter out choppy markets
        disable_volatile: Filter out volatile markets
    """
    print("=" * 80)
    print("WALK-FORWARD WITH REGIME FILTER")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Filter Choppy: {disable_choppy}")
    print(f"Filter Volatile: {disable_volatile}")
    print("=" * 80)
    print()
    
    # Load settings and prepare dataset
    cfg = load_settings()
    prep_ds = prepare_research_dataset(
        symbol=symbol,
        timeframe=timeframe,
        settings=cfg,
    )
    
    # Run standard walk-forward
    print("Running standard walk-forward (no regime filter)...")
    result_baseline = run_walk_forward(
        dataset=prep_ds,
        settings=cfg,
        model_names=model_names or ["xgboost"],
    )
    
    # Apply regime filter to predictions
    print("\nApplying regime filter...")
    regime_config = RegimeConfig(
        allow_choppy=not disable_choppy,
        allow_volatile=not disable_volatile,
        allow_trending=True,
        allow_normal=True,
    )
    
    regime_filter = RegimeFilter(regime_config)
    regimes = regime_filter.detect_regime(prep_ds.frame)
    
    print(f"\nRegime distribution:")
    for regime, pct in regimes.value_counts(normalize=True).items():
        print(f"  {regime}: {pct:.1%}")
    
    # Filter predictions
    filtered_preds = apply_regime_filter_to_predictions(
        predictions=result_baseline.predictions,
        ohlcv=prep_ds.frame,
        config=regime_config,
    )
    
    # Calculate metrics for filtered predictions
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print("\n📊 BASELINE (No Filter):")
    print(f"   Total Predictions: {result_baseline.summary['n_predictions']:,}")
    print(f"   Executed Trades: {result_baseline.summary['n_trades']:,}")
    print(f"   Win Rate: {result_baseline.summary.get('precision_trades', 0):.1%}")
    
    # Calculate filtered metrics
    filtered_trades = filtered_preds[filtered_preds['side'].isin(['BUY', 'SELL'])]
    filtered_successful = (filtered_trades['success'] == True).sum() if 'success' in filtered_trades.columns else 0
    filtered_win_rate = filtered_successful / len(filtered_trades) if len(filtered_trades) > 0 else 0
    
    print("\n🎯 WITH REGIME FILTER:")
    print(f"   Total Predictions: {len(filtered_preds):,}")
    print(f"   Executed Trades: {len(filtered_trades):,}")
    print(f"   Win Rate: {filtered_win_rate:.1%}")
    print(f"   Filtered Out: {result_baseline.summary['n_trades'] - len(filtered_trades)} trades")
    
    improvement = filtered_win_rate - result_baseline.summary.get('precision_trades', 0)
    print(f"\n💡 Impact: {improvement:+.1%} win rate")
    
    if improvement > 0.02:
        print("   ✅ REGIME FILTER IMPROVES PERFORMANCE")
    elif improvement < -0.02:
        print("   ⚠️  REGIME FILTER DEGRADES PERFORMANCE")
    else:
        print("   ➖ REGIME FILTER HAS MINIMAL IMPACT")
    
    print("=" * 80)
    
    return {
        'baseline': result_baseline.summary,
        'filtered_win_rate': filtered_win_rate,
        'filtered_trades': len(filtered_trades),
        'improvement': improvement,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward with regime filter")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol")
    parser.add_argument("--timeframe", default="H4", help="Timeframe")
    parser.add_argument("--model", default="xgboost", help="Model")
    parser.add_argument("--allow-choppy", action="store_true", help="Allow choppy markets")
    parser.add_argument("--allow-volatile", action="store_true", help="Allow volatile markets")
    
    args = parser.parse_args()
    
    result = run_walk_forward_with_regime_filter(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_names=[args.model],
        disable_choppy=not args.allow_choppy,
        disable_volatile=not args.allow_volatile,
    )
    
    # Return success if filter improves or doesn't degrade
    sys.exit(0 if result['improvement'] >= -0.02 else 1)


if __name__ == "__main__":
    main()
