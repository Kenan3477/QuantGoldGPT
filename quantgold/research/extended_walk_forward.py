"""
Extended walk-forward validation with more folds for robust statistics.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from quantgold.config.settings import load_settings
from quantgold.pipeline.dataset import prepare_research_dataset
from quantgold.models.xgboost_model import make_model


def run_extended_walk_forward(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    model_name: str = "ensemble",
    n_folds: int = 10,
):
    """
    Run walk-forward with many folds for robust statistics.
    
    With 10 folds instead of 3, we get:
    - More out-of-sample test periods
    - Better statistics on win rate stability
    - Reduced variance in performance estimates
    """
    print("=" * 80)
    print(f"EXTENDED WALK-FORWARD VALIDATION ({n_folds} folds)")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Model: {model_name}")
    print(f"Folds: {n_folds}")
    print("=" * 80)
    print()
    
    cfg = load_settings()
    
    # Prepare dataset
    prep_ds = prepare_research_dataset(
        symbol=symbol,
        timeframe=timeframe,
        settings=cfg,
    )
    
    df = prep_ds.frame
    X = df[prep_ds.feature_columns]
    y = df[prep_ds.label_column]
    
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(prep_ds.feature_columns)}")
    print()
    
    # Calculate fold size
    fold_size = len(df) // n_folds
    print(f"Fold size: ~{fold_size} bars")
    print()
    
    results = []
    
    for fold in range(n_folds - 2):  # Leave 2 folds for validation + test
        # Train on all data up to this fold
        train_end = (fold + 1) * fold_size
        val_end = train_end + fold_size
        test_end = min(val_end + fold_size, len(df))
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        
        X_test = X.iloc[val_end:test_end]
        y_test = y.iloc[val_end:test_end]
        
        if len(X_test) < 50:
            print(f"Fold {fold}: Skipping (test set too small: {len(X_test)} samples)")
            continue
        
        print(f"Fold {fold}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Train model
        model = make_model(model_name)
        model.fit(X_train, y_train)
        
        # Predict on test
        preds = model.predict_proba(X_test)
        
        # Handle both list and numpy array of predictions
        if isinstance(preds, np.ndarray) and preds.dtype == object:
            preds = preds.tolist()
        
        y_pred = np.array([p.side for p in preds])
        y_prob_buy = np.array([p.probability_buy for p in preds])
        y_prob_sell = np.array([p.probability_sell for p in preds])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Only evaluate on actual trades (not NO_TRADE)
        traded_mask = y_pred != 0
        if traded_mask.sum() > 0:
            acc = accuracy_score(y_test[traded_mask], y_pred[traded_mask])
            prec = precision_score(y_test[traded_mask], y_pred[traded_mask], average='weighted', zero_division=0)
            rec = recall_score(y_test[traded_mask], y_pred[traded_mask], average='weighted', zero_division=0)
            f1 = f1_score(y_test[traded_mask], y_pred[traded_mask], average='weighted', zero_division=0)
            
            results.append({
                'fold': fold,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_trades': traded_mask.sum(),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'coverage': traded_mask.sum() / len(X_test),
            })
            
            print(f"  Trades: {traded_mask.sum()}, Acc: {acc:.1%}, Prec: {prec:.1%}, Coverage: {traded_mask.sum()/len(X_test):.1%}")
        else:
            print(f"  No trades (too selective)")
        
        print()
    
    # Summary
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Completed folds: {len(results)}")
        print()
        print("Mean ± Std:")
        for col in ['accuracy', 'precision', 'recall', 'f1', 'coverage']:
            mean = results_df[col].mean()
            std = results_df[col].std()
            print(f"  {col:15s}: {mean:.1%} ± {std:.1%}")
        
        print()
        print("Min / Max:")
        for col in ['accuracy', 'precision']:
            min_val = results_df[col].min()
            max_val = results_df[col].max()
            print(f"  {col:15s}: {min_val:.1%} / {max_val:.1%}")
        
        # Save results
        output_path = Path("artifacts/reports")
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path / f"extended_wf_{symbol}_{timeframe}_{model_name}.csv", index=False)
        print(f"\nResults saved to: {output_path / f'extended_wf_{symbol}_{timeframe}_{model_name}.csv'}")
    else:
        print("No results to summarize")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--model", default="ensemble")
    parser.add_argument("--folds", type=int, default=10)
    
    args = parser.parse_args()
    
    run_extended_walk_forward(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_name=args.model,
        n_folds=args.folds,
    )
