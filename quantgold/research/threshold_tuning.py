"""
Confidence threshold tuning for optimal coverage-precision trade-off.

Goal: Find the threshold that maximizes Sharpe while maintaining precision >65%.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score


def tune_threshold_from_trades(
    trades_path: str,
    threshold_range: List[float] = None,
    min_precision: float = 0.65,
) -> pd.DataFrame:
    """
    Tune confidence threshold using saved trades.
    
    Args:
        trades_path: Path to trades parquet file
        threshold_range: List of thresholds to test
        min_precision: Minimum acceptable precision
        
    Returns:
        DataFrame with threshold tuning results
    """
    if threshold_range is None:
        # Test from 0.50 (current) to 0.70 in 0.02 increments
        threshold_range = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    
    # Load trades
    df = pd.read_parquet(trades_path)
    
    print(f"Loaded {len(df)} trades")
    print(f"Columns: {df.columns.tolist()}")
    
    # Total possible predictions (we need this from predictions file to calculate coverage)
    # For now, let's use the trades as the baseline
    total_predictions = len(df)
    
    results = []
    
    for threshold in threshold_range:
        # Apply threshold filter
        # Only keep trades where calibrated_probability > threshold
        df_filtered = df[df['calibrated_probability'] > threshold].copy()
        
        if len(df_filtered) == 0:
            print(f"Threshold {threshold:.2f}: No trades")
            continue
        
        # Coverage relative to baseline (0.50 threshold)
        coverage = len(df_filtered) / total_predictions
        
        # Calculate win rate (trades with positive PNL)
        winning_trades = (df_filtered['pnl'] > 0).sum()
        losing_trades = (df_filtered['pnl'] < 0).sum()
        n_trades = len(df_filtered)
        
        win_rate = winning_trades / n_trades if n_trades > 0 else 0
        precision = win_rate  # For trading, precision = win rate
        
        # Count trades by side
        n_buy = (df_filtered['side'] == 'BUY').sum()
        n_sell = (df_filtered['side'] == 'SELL').sum()
        
        # Calculate average PNL and total PNL
        avg_pnl = df_filtered['pnl'].mean()
        total_pnl = df_filtered['pnl'].sum()
        
        # Calculate Sharpe estimate (simple: mean PNL / std PNL * sqrt(n))
        sharpe_estimate = (avg_pnl / df_filtered['pnl'].std()) * np.sqrt(n_trades) if n_trades > 1 else 0
        
        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'n_trades': n_trades,
            'n_buy': n_buy,
            'n_sell': n_sell,
            'win_rate': win_rate,
            'precision': precision,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'sharpe_estimate': sharpe_estimate,
            'passes_min_precision': precision >= min_precision,
        })
        
        print(f"Threshold {threshold:.2f}: coverage={coverage:.1%}, "
              f"trades={n_trades}, win_rate={win_rate:.1%}, "
              f"sharpe={sharpe_estimate:.3f}, "
              f"{'✅ PASS' if precision >= min_precision else '❌ FAIL'}")
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold (highest Sharpe while maintaining min precision)
    valid_results = results_df[results_df['passes_min_precision']]
    
    if len(valid_results) > 0:
        # Sort by Sharpe (descending)
        optimal = valid_results.loc[valid_results['sharpe_estimate'].idxmax()]
        print(f"\n🏆 Optimal Threshold (max Sharpe, precision ≥{min_precision:.0%}): {optimal['threshold']:.2f}")
        print(f"   Coverage: {optimal['coverage']:.1%}")
        print(f"   Win Rate: {optimal['win_rate']:.1%}")
        print(f"   Trades: {int(optimal['n_trades'])}")
        print(f"   Sharpe: {optimal['sharpe_estimate']:.3f}")
    else:
        print(f"\n⚠️ No threshold maintains win rate ≥{min_precision:.1%}")
    
    return results_df


def run_threshold_grid_search(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    trades_dir: str = "artifacts/reports",
    output_dir: str = "artifacts/threshold_tuning",
) -> Dict:
    """
    Run threshold tuning on existing trades.
    
    Args:
        symbol: Instrument symbol
        timeframe: Timeframe
        trades_dir: Directory with trades
        output_dir: Output directory for results
        
    Returns:
        Dict with tuning results
    """
    # Find trades file
    trades_path = Path(trades_dir) / f"trades_{symbol}_{timeframe}.parquet"
    
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades not found: {trades_path}")
    
    print("="*80)
    print("CONFIDENCE THRESHOLD TUNING")
    print("="*80)
    print(f"Instrument: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Trades: {trades_path}")
    print("="*80)
    
    # Run tuning
    results_df = tune_threshold_from_trades(
        str(trades_path),
        threshold_range=[0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70],
        min_precision=0.65,
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"threshold_tuning_{symbol}_{timeframe}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n📊 Results saved to: {results_file}")
    
    # Generate report
    report_file = output_path / f"threshold_tuning_{symbol}_{timeframe}.md"
    with open(report_file, 'w') as f:
        f.write(f"# Confidence Threshold Tuning Report\n\n")
        f.write(f"**Instrument:** {symbol} {timeframe}\n\n")
        f.write(f"**Goal:** Maximize Sharpe while maintaining win rate ≥65%\n\n")
        
        f.write("## Results Table\n\n")
        f.write("| Threshold | Coverage | # Trades | Win Rate | Sharpe | Status |\n")
        f.write("|-----------|----------|----------|----------|--------|--------|\n")
        
        for _, row in results_df.iterrows():
            status = "✅ PASS" if row['passes_min_precision'] else "❌ FAIL"
            f.write(f"| {row['threshold']:.2f} | {row['coverage']:.1%} | "
                   f"{int(row['n_trades'])} | {row['win_rate']:.1%} | "
                   f"{row['sharpe_estimate']:.3f} | {status} |\n")
        
        # Find optimal
        valid_results = results_df[results_df['passes_min_precision']]
        if len(valid_results) > 0:
            optimal = valid_results.loc[valid_results['sharpe_estimate'].idxmax()]
            f.write(f"\n## 🏆 Optimal Threshold\n\n")
            f.write(f"**Threshold:** {optimal['threshold']:.2f}\n\n")
            f.write(f"- **Coverage:** {optimal['coverage']:.1%}\n")
            f.write(f"- **Win Rate:** {optimal['win_rate']:.1%}\n")
            f.write(f"- **Trades:** {int(optimal['n_trades'])}\n")
            f.write(f"- **BUY trades:** {int(optimal['n_buy'])}\n")
            f.write(f"- **SELL trades:** {int(optimal['n_sell'])}\n")
            f.write(f"- **Sharpe:** {optimal['sharpe_estimate']:.3f}\n")
            f.write(f"- **Avg PNL:** {optimal['avg_pnl']:.2f}\n")
            f.write(f"- **Total PNL:** {optimal['total_pnl']:.2f}\n")
            
            # Compare to baseline (0.50)
            baseline = results_df[results_df['threshold'] == 0.50].iloc[0]
            f.write(f"\n## Comparison to Baseline (threshold=0.50)\n\n")
            f.write(f"| Metric | Baseline | Optimal | Improvement |\n")
            f.write(f"|--------|----------|---------|-------------|\n")
            f.write(f"| Coverage | {baseline['coverage']:.1%} | {optimal['coverage']:.1%} | "
                   f"{(optimal['coverage'] - baseline['coverage']):.1%} |\n")
            f.write(f"| Win Rate | {baseline['win_rate']:.1%} | {optimal['win_rate']:.1%} | "
                   f"{(optimal['win_rate'] - baseline['win_rate']):.1%} |\n")
            f.write(f"| Trades | {int(baseline['n_trades'])} | {int(optimal['n_trades'])} | "
                   f"{int(optimal['n_trades'] - baseline['n_trades'])} |\n")
            f.write(f"| Sharpe | {baseline['sharpe_estimate']:.3f} | {optimal['sharpe_estimate']:.3f} | "
                   f"{(optimal['sharpe_estimate'] - baseline['sharpe_estimate']):+.3f} |\n")
    
    print(f"📄 Report saved to: {report_file}")
    
    return {
        'results': results_df.to_dict('records'),
        'optimal_threshold': float(optimal['threshold']) if len(valid_results) > 0 else None,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune confidence thresholds")
    parser.add_argument("--symbol", default="XAUUSD", help="Instrument symbol")
    parser.add_argument("--timeframe", default="M15", help="Timeframe")
    
    args = parser.parse_args()
    
    run_threshold_grid_search(
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
