"""
Continuous paper trading with live monitoring and drift detection.

This runs the QuantGold system in paper trading mode, making predictions
on the latest data and tracking performance over time.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

from quantgold.config.settings import load_settings
from quantgold.data.store import CanonicalDataStore
from quantgold.pipeline.dataset import prepare_research_dataset
from quantgold.models.xgboost_model import make_model
from quantgold.models.calibration import ProbabilityCalibrator
from quantgold.decision.selective import SelectivePolicy
from quantgold.risk.engine import RiskEngine


class PaperTradingMonitor:
    """
    Continuous paper trading with performance tracking and drift detection.
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "H4",
        model_name: str = "ensemble",
        output_dir: str = "paper_trading",
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cfg = load_settings()
        self.store = CanonicalDataStore()
        
        # Tracking
        self.trades: List[Dict] = []
        self.predictions: List[Dict] = []
        self.start_time = datetime.now()
        
        # Load or train model
        self.model = None
        self.calibrator = None
        
        print("=" * 80)
        print("PAPER TRADING MONITOR")
        print("=" * 80)
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Model: {model_name}")
        print(f"Output: {output_dir}")
        print("=" * 80)
        print()
    
    def initialize_model(self):
        """Train model on all available historical data."""
        print("Initializing model...")
        
        # Prepare dataset
        prep_ds = prepare_research_dataset(
            symbol=self.symbol,
            timeframe=self.timeframe,
            settings=self.cfg,
        )
        
        df = prep_ds.frame
        X = df[prep_ds.feature_columns]
        y = df[prep_ds.label_column]
        
        print(f"Training on {len(X)} samples with {len(prep_ds.feature_columns)} features")
        
        # Train model
        self.model = make_model(self.model_name)
        
        # Use 80/20 split for train/calibration
        split_idx = int(len(X) * 0.8)
        X_train, X_cal = X[:split_idx], X[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Calibrate
        print("Calibrating probabilities...")
        preds_cal = self.model.predict_proba(X_cal)
        
        # Handle prediction format (List[ModelPrediction] or numpy array)
        if isinstance(preds_cal, list) and hasattr(preds_cal[0], 'side'):
            # List of ModelPrediction objects
            y_pred_cal = np.array([p.side for p in preds_cal])
            y_prob_cal = np.array([p.probability_buy if p.side == 1 else p.probability_sell for p in preds_cal])
        else:
            # Numpy array of probabilities [prob_class_0, prob_class_1]
            y_pred_cal = np.where(preds_cal[:, 1] > 0.5, 1, -1)
            y_prob_cal = np.max(preds_cal, axis=1)
        
        # Convert to success/failure (1/0) for calibrator
        val_success = (y_cal.values == y_pred_cal).astype(int)
        
        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.calibrator.fit(pd.Series(val_success, index=y_cal.index), pd.Series(y_prob_cal, index=y_cal.index))
        
        print("✅ Model initialized and calibrated")
        print()
    
    def make_prediction(self, current_data: pd.DataFrame) -> Dict:
        """Make a prediction on the latest bar."""
        # Prepare full dataset with features and labels
        prep_ds = prepare_research_dataset(
            symbol=self.symbol,
            timeframe=self.timeframe,
            settings=self.cfg,
        )
        
        # Get the prepared dataframe and extract latest row with features
        df = prep_ds.frame
        X_latest = df[prep_ds.feature_columns].iloc[[-1]]
        latest_timestamp = df['timestamp'].iloc[-1]
        latest_close = df['close'].iloc[-1]
        
        # Predict
        preds = self.model.predict_proba(X_latest)
        
        # Handle prediction format
        if isinstance(preds, list) and hasattr(preds[0], 'side'):
            # List of ModelPrediction objects
            pred = preds[0]
            side = pred.side
            raw_proba = pred.probability_buy if pred.side == 1 else pred.probability_sell
        else:
            # Numpy array of probabilities
            side = 1 if preds[0, 1] > 0.5 else -1
            raw_proba = preds[0, 1] if side == 1 else preds[0, 0]
        
        # Calibrate
        cal_proba = self.calibrator.calibrate(side, raw_proba)
        
        # Apply decision policy
        policy = SelectivePolicy(
            min_calibrated_probability=self.cfg.decision.min_calibrated_probability,
            max_disagreement=self.cfg.decision.max_model_disagreement,
        )
        
        decision = policy.decide(
            side=side,
            calibrated_proba=cal_proba,
            disagreement=0.0,  # Single model has no disagreement
            meta_proba=1.0,
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'bar_timestamp': latest_timestamp.isoformat(),
            'close': float(latest_close),
            'side': decision.final_side,
            'raw_side': int(side),
            'calibrated_probability': float(cal_proba),
            'decision': decision.final_side,
            'reason': decision.reason,
        }
    
    def calculate_metrics(self) -> Dict:
        """Calculate current performance metrics."""
        if len(self.trades) == 0:
            return {
                'n_predictions': len(self.predictions),
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'sharpe': 0.0,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate metrics
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
        
        avg_pnl = trades_df['pnl'].mean()
        total_pnl = trades_df['pnl'].sum()
        
        # Sharpe (simple: mean / std)
        sharpe = (avg_pnl / trades_df['pnl'].std()) * np.sqrt(len(trades_df)) if len(trades_df) > 1 else 0
        
        return {
            'n_predictions': len(self.predictions),
            'n_trades': len(self.trades),
            'win_rate': float(win_rate),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'avg_pnl': float(avg_pnl),
            'total_pnl': float(total_pnl),
            'sharpe': float(sharpe),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'predictions_per_hour': len(self.predictions) / ((datetime.now() - self.start_time).total_seconds() / 3600),
            'trades_per_day': len(self.trades) / ((datetime.now() - self.start_time).total_seconds() / 86400),
        }
    
    def check_drift(self) -> Dict:
        """Check for distribution drift."""
        if len(self.predictions) < 50:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Check win rate drift
        recent_trades = pd.DataFrame(self.trades[-20:]) if len(self.trades) >= 20 else pd.DataFrame(self.trades)
        
        if len(recent_trades) < 10:
            return {'drift_detected': False, 'reason': 'Insufficient trades'}
        
        recent_win_rate = (recent_trades['pnl'] > 0).sum() / len(recent_trades)
        
        # Expected win rate (from backtest)
        expected_win_rate = 0.943 if self.timeframe == "H4" else 0.805  # H4 or H1
        
        # Alert if win rate drops more than 15%
        drift_threshold = 0.15
        win_rate_drop = expected_win_rate - recent_win_rate
        
        if win_rate_drop > drift_threshold:
            return {
                'drift_detected': True,
                'reason': f'Win rate dropped from {expected_win_rate:.1%} to {recent_win_rate:.1%}',
                'recent_win_rate': float(recent_win_rate),
                'expected_win_rate': float(expected_win_rate),
                'drop': float(win_rate_drop),
            }
        
        return {
            'drift_detected': False,
            'recent_win_rate': float(recent_win_rate),
            'expected_win_rate': float(expected_win_rate),
        }
    
    def save_state(self):
        """Save current state to disk."""
        state = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_name': self.model_name,
            'start_time': self.start_time.isoformat(),
            'predictions': self.predictions,
            'trades': self.trades,
            'metrics': self.calculate_metrics(),
            'drift_check': self.check_drift(),
        }
        
        # Save to JSON
        output_file = self.output_dir / f"paper_trading_{self.symbol}_{self.timeframe}.json"
        with open(output_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save metrics separately for easy monitoring
        metrics_file = self.output_dir / f"metrics_{self.symbol}_{self.timeframe}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'last_update': datetime.now().isoformat(),
                'metrics': state['metrics'],
                'drift': state['drift_check'],
            }, f, indent=2)
    
    def print_status(self):
        """Print current status."""
        metrics = self.calculate_metrics()
        drift = self.check_drift()
        
        print("\n" + "=" * 80)
        print(f"PAPER TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"Symbol: {self.symbol} | Timeframe: {self.timeframe} | Model: {self.model_name}")
        print(f"Uptime: {metrics['uptime_hours']:.1f} hours")
        print()
        print(f"Predictions: {metrics['n_predictions']}")
        print(f"Trades: {metrics['n_trades']}")
        if metrics['n_trades'] > 0:
            print(f"Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Winning: {metrics['winning_trades']}")
            print(f"  Losing: {metrics['losing_trades']}")
            print(f"Avg PNL: {metrics['avg_pnl']:.2f}")
            print(f"Total PNL: {metrics['total_pnl']:.2f}")
            print(f"Sharpe: {metrics['sharpe']:.3f}")
        print()
        
        if drift['drift_detected']:
            print("⚠️  DRIFT DETECTED!")
            print(f"   {drift['reason']}")
        else:
            print("✅ No drift detected")
        
        print("=" * 80)
    
    def run_simulation(self, duration_hours: int = 24):
        """
        Run paper trading simulation using historical data.
        
        This simulates real-time trading by stepping through historical data
        one bar at a time.
        """
        print("Running paper trading simulation...")
        print(f"Duration: {duration_hours} hours")
        print()
        
        # Initialize model
        self.initialize_model()
        
        # Load data
        df = self.store.load_ohlcv(self.symbol, self.timeframe)
        
        # Use last N bars for simulation
        bars_per_hour = 1 if self.timeframe == "H1" else (4 if self.timeframe == "H4" else 60)
        n_bars = int(duration_hours / bars_per_hour)
        
        simulation_data = df.iloc[-n_bars:].reset_index(drop=True)
        
        print(f"Simulating on {len(simulation_data)} bars")
        print()
        
        # Step through each bar
        for i in range(len(simulation_data)):
            # Get data up to current bar
            current_data = df.iloc[:-(len(simulation_data)-i-1)] if i < len(simulation_data)-1 else df
            
            # Make prediction
            pred = self.make_prediction(current_data)
            self.predictions.append(pred)
            
            # Simulate trade execution (simplified)
            if pred['decision'] != 0:  # If not NO_TRADE
                # Simulate trade
                entry_price = pred['close']
                # Use next bar's close as exit (simplified)
                if i < len(simulation_data) - 1:
                    exit_price = float(simulation_data['close'].iloc[i + 1])
                    
                    # Calculate PNL
                    if pred['decision'] == 1:  # BUY
                        pnl = exit_price - entry_price
                    else:  # SELL
                        pnl = entry_price - exit_price
                    
                    self.trades.append({
                        'timestamp': pred['timestamp'],
                        'side': 'BUY' if pred['decision'] == 1 else 'SELL',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                    })
            
            # Print status every 10 bars
            if (i + 1) % 10 == 0:
                self.print_status()
                self.save_state()
        
        # Final status
        self.print_status()
        self.save_state()
        
        print()
        print("✅ Simulation complete!")
        print(f"📊 Results saved to: {self.output_dir}")
        
        return self.calculate_metrics()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper trading with live monitoring")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to trade")
    parser.add_argument("--timeframe", default="H4", help="Timeframe (M15, H1, H4)")
    parser.add_argument("--model", default="ensemble", help="Model to use")
    parser.add_argument("--duration", type=int, default=168, help="Simulation duration in hours (default: 1 week)")
    parser.add_argument("--output", default="paper_trading", help="Output directory")
    
    args = parser.parse_args()
    
    monitor = PaperTradingMonitor(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_name=args.model,
        output_dir=args.output,
    )
    
    monitor.run_simulation(duration_hours=args.duration)


if __name__ == "__main__":
    main()
