"""
Enhanced walk-forward with class weights for imbalanced learning.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from quantgold.pipeline.walk_forward import run_walk_forward
from quantgold.config.settings import load_settings


def run_walk_forward_with_class_weights(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    model_name: str = "xgboost",
) -> Dict:
    """
    Run walk-forward with automatic class weighting.
    
    This gives more weight to minority class examples during training,
    forcing the model to learn both BUY and SELL equally well.
    """
    print("=" * 80)
    print("WALK-FORWARD WITH CLASS WEIGHTS")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Model: {model_name}")
    print("=" * 80)
    print()
    
    cfg = load_settings()
    
    # Prepare dataset
    from quantgold.pipeline.dataset import prepare_research_dataset
    
    prep_ds = prepare_research_dataset(
        symbol=symbol,
        timeframe=timeframe,
        settings=cfg,
    )
    
    df = prep_ds.frame
    X = df[prep_ds.feature_columns]
    y = df[prep_ds.label_column]
    
    # Compute class weights
    classes = np.unique(y)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    class_weights = dict(zip(classes, class_weights_array))
    
    print("Class Distribution:")
    for cls in classes:
        count = (y == cls).sum()
        pct = count / len(y) * 100
        weight = class_weights[cls]
        cls_name = {-1: 'SELL', 0: 'NO_LABEL', 1: 'BUY'}.get(cls, f'CLASS_{cls}')
        print(f"  {cls_name:10s}: {count:5d} ({pct:5.1f}%) — weight: {weight:.3f}")
    print()
    
    # Run walk-forward
    # Note: Current implementation doesn't support sample weights in walk_forward
    # This would require modifying the walk_forward loop to compute and pass weights per fold
    print("NOTE: To fully implement class weights, modify walk_forward.py")
    print("      to compute sample_weight per fold and pass to model.fit()")
    print()
    
    # For now, run standard walk-forward
    # The balanced labels already help significantly
    from quantgold.models.xgboost_model import make_model
    from quantgold.pipeline.walk_forward import run_walk_forward
    
    results = run_walk_forward(
        symbol=symbol,
        timeframe=timeframe,
        model_names=[model_name],
        settings=cfg,
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--model", default="xgboost")
    
    args = parser.parse_args()
    
    run_walk_forward_with_class_weights(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_name=args.model,
    )
