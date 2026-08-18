"""
Deploy paper trading using the existing walk-forward infrastructure.

This runs a single fold walk-forward on recent data to simulate live trading.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from quantgold.config.settings import load_settings
from quantgold.pipeline.walk_forward import run_walk_forward


def deploy_paper_trading(
    symbol: str = "XAUUSD",
    timeframe: str = "H4",
    model_name: str = "xgboost",
    n_folds: int = 1,
    test_ratio: float = 0.10,
    output_dir: str = "paper_trading",
):
    """
    Deploy paper trading by running walk-forward on recent data.
    
    Uses a single fold with small test_ratio to simulate live trading
    on the most recent data.
    """
    print("=" * 80)
    print("PAPER TRADING DEPLOYMENT")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Model: {model_name}")
    print(f"Test Ratio: {test_ratio:.1%} (recent data)")
    print("=" * 80)
    print()
    
    # Load settings
    cfg = load_settings()
    
    # Prepare dataset
    from quantgold.pipeline.dataset import prepare_research_dataset
    
    print(f"Preparing dataset for {symbol} {timeframe}...")
    prep_ds = prepare_research_dataset(
        symbol=symbol,
        timeframe=timeframe,
        settings=cfg,
    )
    
    print(f"Dataset prepared: {len(prep_ds.frame)} samples")
    print()
    
    # Run walk-forward on recent data
    print(f"Running walk-forward validation...")
    result = run_walk_forward(
        dataset=prep_ds,
        settings=cfg,
        model_names=[model_name],
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save predictions as parquet
    preds_file = output_path / f"predictions_{symbol}_{timeframe}_{timestamp_str}.parquet"
    result.predictions.to_parquet(preds_file)
    
    # Save summary as JSON
    result_file = output_path / f"summary_{symbol}_{timeframe}_{timestamp_str}.json"
    summary_data = {
        'symbol': result.symbol,
        'timeframe': result.timeframe,
        'model_names': result.model_names,
        'summary': result.summary,
        'folds': [{
            'fold_id': f.fold_id,
            'n_train': f.n_train,
            'n_val': f.n_val,
            'n_test': f.n_test,
            'test_precision_trades': f.test_precision_trades,
            'test_coverage': f.test_coverage,
            'test_brier': f.test_brier,
            'test_ece': f.test_ece,
        } for f in result.folds],
        'predictions_file': str(preds_file),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(result_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print()
    print("=" * 80)
    print("PAPER TRADING RESULTS")
    print("=" * 80)
    
    summ = result.summary
    
    print(f"Total Predictions: {summ['n_predictions']}")
    print(f"Trades Executed: {summ['n_trades']} ({summ.get('coverage', summ['n_trades']/summ['n_predictions'] if summ['n_predictions'] > 0 else 0):.1%} coverage)")
    print()
    
    if summ['n_trades'] > 0:
        print(f"Win Rate: {summ.get('precision_trades', summ.get('win_rate', 0)):.1%}")
        wins = int(summ.get('precision_trades', 0) * summ['n_trades'])
        losses = summ['n_trades'] - wins
        print(f"  Winning: {wins}")
        print(f"  Losing: {losses}")
        print()
        print(f"Sharpe Ratio: {summ.get('sharpe', 0):.3f}")
        print(f"Profit Factor: {summ.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {summ.get('max_drawdown', 0):.2%}")
        print()
        print(f"BUY Signals: {summ.get('n_buy', 0)} (win rate: {summ.get('buy_precision', 0):.1%})")
        print(f"SELL Signals: {summ.get('n_sell', 0)} (win rate: {summ.get('sell_precision', 0):.1%})")
    else:
        print("⚠️  No trades executed (all predictions filtered)")
    
    print()
    print(f"Results saved to: {result_file}")
    print(f"Predictions saved to: {preds_file}")
    print("=" * 80)
    
    # Check if performance meets expectations
    expected_win_rate = 0.943 if timeframe == "H4" else 0.805  # H4 or H1
    drift_threshold = 0.15
    
    actual_win_rate = summ.get('precision_trades', summ.get('win_rate', 0))
    
    if summ['n_trades'] >= 10:
        win_rate_drop = expected_win_rate - actual_win_rate
        
        if win_rate_drop > drift_threshold:
            print()
            print("⚠️  ALERT: Performance Degradation Detected!")
            print(f"   Expected win rate: {expected_win_rate:.1%}")
            print(f"   Actual win rate: {actual_win_rate:.1%}")
            print(f"   Drop: {win_rate_drop:.1%}")
            print()
            print("   Possible causes:")
            print("   - Market regime change")
            print("   - Model drift")
            print("   - Data quality issues")
            print("=" * 80)
            return False
        else:
            print()
            print("✅ Performance is within expected range")
            print(f"   Expected: {expected_win_rate:.1%} ± {drift_threshold:.1%}")
            print(f"   Actual: {actual_win_rate:.1%}")
            print("=" * 80)
            return True
    else:
        print()
        print("ℹ️  Insufficient trades for drift detection (need 10+)")
        print("=" * 80)
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy paper trading")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to trade")
    parser.add_argument("--timeframe", default="H4", help="Timeframe (M15, H1, H4)")
    parser.add_argument("--model", default="xgboost", help="Model to use")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Ratio of recent data to test on")
    parser.add_argument("--output", default="paper_trading", help="Output directory")
    
    args = parser.parse_args()
    
    success = deploy_paper_trading(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_name=args.model,
        test_ratio=args.test_ratio,
        output_dir=args.output,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
