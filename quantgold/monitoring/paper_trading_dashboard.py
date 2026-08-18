"""
Paper Trading Dashboard - Real-time monitoring of QuantGold performance.

Usage:
    python3 quantgold/monitoring/paper_trading_dashboard.py --predictions paper_trading/predictions_*.parquet
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def analyze_predictions(predictions_file: str) -> dict:
    """Analyze predictions file and calculate performance metrics."""
    df = pd.read_parquet(predictions_file)
    
    # Filter to executed trades
    executed = df[df['side'].isin(['BUY', 'SELL'])].copy()
    
    if len(executed) == 0:
        return {
            'error': 'No executed trades found',
            'n_predictions': len(df),
            'n_trades': 0,
        }
    
    # Calculate metrics
    successful = (executed['success'] == True).sum()
    failed = (executed['success'] == False).sum()
    win_rate = successful / len(executed)
    
    # By side
    buy_trades = executed[executed['side'] == 'BUY']
    sell_trades = executed[executed['side'] == 'SELL']
    
    buy_win_rate = (buy_trades['success'] == True).sum() / len(buy_trades) if len(buy_trades) > 0 else 0
    sell_win_rate = (sell_trades['success'] == True).sum() / len(sell_trades) if len(sell_trades) > 0 else 0
    
    # Recent performance (last 50 trades)
    recent = executed.tail(50)
    recent_win_rate = (recent['success'] == True).sum() / len(recent) if len(recent) > 0 else 0
    
    # Calibration quality
    avg_calibrated_prob = executed['calibrated_probability'].mean()
    
    return {
        'file': predictions_file,
        'n_predictions': len(df),
        'n_trades': len(executed),
        'coverage': len(executed) / len(df),
        'win_rate': win_rate,
        'successful': successful,
        'failed': failed,
        'n_buy': len(buy_trades),
        'buy_win_rate': buy_win_rate,
        'n_sell': len(sell_trades),
        'sell_win_rate': sell_win_rate,
        'recent_win_rate_50': recent_win_rate,
        'avg_calibrated_prob': avg_calibrated_prob,
        'timestamp': datetime.now().isoformat(),
    }


def print_dashboard(metrics: dict):
    """Print formatted dashboard."""
    print()
    print("=" * 80)
    print("QUANTGOLD PAPER TRADING DASHBOARD")
    print("=" * 80)
    print(f"Updated: {metrics['timestamp']}")
    print(f"Data File: {metrics['file']}")
    print("=" * 80)
    print()
    
    if metrics.get('error'):
        print(f"⚠️  {metrics['error']}")
        print(f"Total Predictions: {metrics['n_predictions']}")
        print()
        return
    
    print(f"📊 OVERALL PERFORMANCE")
    print(f"   Total Predictions: {metrics['n_predictions']:,}")
    print(f"   Executed Trades: {metrics['n_trades']:,} ({metrics['coverage']:.1%} coverage)")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"     ✓ Successful: {metrics['successful']}")
    print(f"     ✗ Failed: {metrics['failed']}")
    print()
    
    print(f"📈 BY SIGNAL TYPE")
    print(f"   BUY Signals: {metrics['n_buy']} (win rate: {metrics['buy_win_rate']:.1%})")
    print(f"   SELL Signals: {metrics['n_sell']} (win rate: {metrics['sell_win_rate']:.1%})")
    print()
    
    print(f"🔍 QUALITY METRICS")
    print(f"   Recent Win Rate (last 50): {metrics['recent_win_rate_50']:.1%}")
    print(f"   Avg Calibrated Probability: {metrics['avg_calibrated_prob']:.3f}")
    print()
    
    # Drift detection
    expected_win_rate = 0.943  # H4 backtest performance
    drift_threshold = 0.15
    win_rate_drop = expected_win_rate - metrics['win_rate']
    
    print(f"🚨 DRIFT DETECTION")
    print(f"   Expected Win Rate: {expected_win_rate:.1%}")
    print(f"   Actual Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Difference: {win_rate_drop:+.1%}")
    
    if win_rate_drop > drift_threshold:
        print(f"   Status: ⚠️  DRIFT DETECTED (>{drift_threshold:.0%} drop)")
        print(f"   Action: Review model, check data quality, consider retraining")
    elif win_rate_drop > 0.05:
        print(f"   Status: ⚡ MINOR DEGRADATION (<{drift_threshold:.0%} drop)")
        print(f"   Action: Monitor closely")
    else:
        print(f"   Status: ✅ HEALTHY (within expected range)")
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Dashboard")
    parser.add_argument("--predictions", required=True, help="Path to predictions parquet file")
    parser.add_argument("--watch", action="store_true", help="Watch mode (refresh every 60s)")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds (watch mode)")
    
    args = parser.parse_args()
    
    if not Path(args.predictions).exists():
        print(f"❌ Error: File not found: {args.predictions}")
        return 1
    
    if args.watch:
        import time
        print("Starting dashboard in watch mode (Ctrl+C to stop)...")
        print(f"Refresh interval: {args.interval}s")
        try:
            while True:
                metrics = analyze_predictions(args.predictions)
                print("\033[2J\033[H")  # Clear screen
                print_dashboard(metrics)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
    else:
        metrics = analyze_predictions(args.predictions)
        print_dashboard(metrics)
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
