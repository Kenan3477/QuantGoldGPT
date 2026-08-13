"""
Feature ablation study framework.

Incrementally test feature families to measure their OOS contribution.

Approach:
1. Start with base features only
2. Add one family at a time (micro, mtf, smc, intermarket)
3. For each combination, run simplified walk-forward
4. Measure OOS metrics: Sharpe, precision, coverage
5. Report which families add value

This is a simplified version for Sprint 1 Bootstrap.
For production, integrate with full walk-forward pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from quantgold.models.xgboost_model import XGBoostModel
from quantgold.models.ensemble_multi import MultiModelEnsemble, EnsembleMember
from quantgold.models.lightgbm_model import LightGBMModel
from quantgold.models.catboost_model import CatBoostModel


@dataclass
class AblationResult:
    """Results from one ablation experiment."""
    feature_set: str
    feature_families: List[str]
    n_features: int
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    # Coverage (% of predictions made)
    coverage: float
    
    # Feature importance (top 10 features)
    top_features: List[tuple[str, float]]
    
    # Training time
    train_time_seconds: float


class FeatureAblationStudy:
    """
    Run feature ablation study.
    
    Example:
        study = FeatureAblationStudy(output_dir="artifacts/ablation")
        
        results = study.run(
            X_train, y_train,
            X_val, y_val,
            feature_families={
                "base": base_feature_cols,
                "micro": microstructure_cols,
                "mtf": multitimeframe_cols,
                "smc": smc_cols,
                "intermarket": intermarket_cols,
            }
        )
        
        study.generate_report(results)
    """
    
    def __init__(self, output_dir: str | Path = "artifacts/ablation"):
        """
        Initialize ablation study.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        feature_families: Dict[str, List[str]],
        use_ensemble: bool = False,
    ) -> List[AblationResult]:
        """
        Run ablation study with incremental feature addition.
        
        Args:
            X_train: Training features (all features)
            y_train: Training labels
            X_val: Validation features (all features)
            y_val: Validation labels
            feature_families: Dict mapping family name → list of feature columns
            use_ensemble: Use 3-model ensemble instead of single XGBoost
            
        Returns:
            List of AblationResult objects (one per feature combination)
        """
        results = []
        
        # Test incrementally: base, base+micro, base+micro+mtf, etc.
        family_order = ["base", "micro", "mtf", "smc", "intermarket"]
        cumulative_families = []
        cumulative_features = []
        
        for family in family_order:
            if family not in feature_families:
                print(f"Warning: Family '{family}' not found, skipping")
                continue
            
            # Add this family
            cumulative_families.append(family)
            cumulative_features.extend(feature_families[family])
            
            # Remove duplicates while preserving order
            cumulative_features = list(dict.fromkeys(cumulative_features))
            
            # Filter to only existing columns
            available_features = [f for f in cumulative_features if f in X_train.columns]
            
            if not available_features:
                print(f"Warning: No features available for {'+'.join(cumulative_families)}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Testing: {'+'.join(cumulative_families)} ({len(available_features)} features)")
            print(f"{'='*60}")
            
            # Train and evaluate
            result = self._train_and_evaluate(
                X_train[available_features],
                y_train,
                X_val[available_features],
                y_val,
                feature_set="+".join(cumulative_families),
                feature_families=cumulative_families.copy(),
                use_ensemble=use_ensemble,
            )
            
            results.append(result)
            
            # Print summary
            print(f"Accuracy: {result.accuracy:.3f}")
            print(f"Precision: {result.precision:.3f}")
            print(f"F1: {result.f1:.3f}")
            print(f"Coverage: {result.coverage:.1%}")
            print(f"Train time: {result.train_time_seconds:.1f}s")
        
        return results
    
    def _train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        feature_set: str,
        feature_families: List[str],
        use_ensemble: bool = False,
    ) -> AblationResult:
        """Train model and evaluate on validation set."""
        import time
        
        start_time = time.time()
        
        # Train model
        if use_ensemble:
            model = MultiModelEnsemble(
                models=[
                    EnsembleMember("xgb", XGBoostModel(n_estimators=100)),
                    EnsembleMember("lgbm", LightGBMModel(n_estimators=100)),
                    EnsembleMember("cat", CatBoostModel(iterations=100, verbose=False)),
                ],
                strategy="weighted_average",
            )
        else:
            model = XGBoostModel(n_estimators=100, max_depth=6)
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred_proba = model.predict_proba(X_val)
        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Coverage (for now, just 100% since we predict everything)
        # In real system, this would be % of predictions with high confidence
        coverage = 1.0
        
        # Feature importance
        try:
            if use_ensemble:
                # Get importance from first model (XGBoost)
                importances = model.models[0].model._model.feature_importances_
            else:
                importances = model._model.feature_importances_
            
            feature_importance = list(zip(X_train.columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance[:10]
        except Exception as e:
            print(f"Warning: Could not extract feature importance: {e}")
            top_features = []
        
        return AblationResult(
            feature_set=feature_set,
            feature_families=feature_families,
            n_features=len(X_train.columns),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            coverage=coverage,
            top_features=top_features,
            train_time_seconds=train_time,
        )
    
    def generate_report(self, results: List[AblationResult]) -> str:
        """
        Generate markdown report from ablation results.
        
        Args:
            results: List of AblationResult objects
            
        Returns:
            Path to generated report file
        """
        report_path = self.output_dir / "ablation_report.md"
        
        with open(report_path, "w") as f:
            f.write("# Feature Ablation Study Report\n\n")
            f.write("**Sprint 1 Bootstrap — Zero-Cost Implementation**\n\n")
            f.write("Testing incremental feature addition to measure OOS contribution.\n\n")
            
            # Summary table
            f.write("## Summary Table\n\n")
            f.write("| Feature Set | # Features | Accuracy | Precision | F1 | Coverage | Train Time |\n")
            f.write("|-------------|------------|----------|-----------|----|---------|-----------|\n")
            
            for r in results:
                f.write(f"| {r.feature_set} | {r.n_features} | "
                       f"{r.accuracy:.3f} | {r.precision:.3f} | {r.f1:.3f} | "
                       f"{r.coverage:.1%} | {r.train_time_seconds:.1f}s |\n")
            
            # Incremental gains
            f.write("\n## Incremental Gains\n\n")
            f.write("Improvement from adding each feature family:\n\n")
            f.write("| Added Family | Δ Accuracy | Δ Precision | Δ F1 |\n")
            f.write("|--------------|------------|-------------|------|\n")
            
            for i in range(1, len(results)):
                prev = results[i-1]
                curr = results[i]
                added_family = curr.feature_families[-1]
                
                delta_acc = curr.accuracy - prev.accuracy
                delta_prec = curr.precision - prev.precision
                delta_f1 = curr.f1 - prev.f1
                
                f.write(f"| {added_family} | {delta_acc:+.3f} | {delta_prec:+.3f} | {delta_f1:+.3f} |\n")
            
            # Top features per configuration
            f.write("\n## Top Features Per Configuration\n\n")
            for r in results:
                f.write(f"### {r.feature_set} (Top 5)\n\n")
                for i, (feat, imp) in enumerate(r.top_features[:5], 1):
                    f.write(f"{i}. `{feat}`: {imp:.4f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            # Find best performing configuration
            best_f1_idx = max(range(len(results)), key=lambda i: results[i].f1)
            best_config = results[best_f1_idx]
            
            f.write(f"**Best configuration:** {best_config.feature_set}\n\n")
            f.write(f"- Features: {best_config.n_features}\n")
            f.write(f"- F1 Score: {best_config.f1:.3f}\n")
            f.write(f"- Precision: {best_config.precision:.3f}\n\n")
            
            # Check if any family hurts performance
            negative_impact = []
            for i in range(1, len(results)):
                if results[i].f1 < results[i-1].f1:
                    added_family = results[i].feature_families[-1]
                    negative_impact.append(added_family)
            
            if negative_impact:
                f.write(f"⚠️ **Warning:** These families reduced F1 score: {', '.join(negative_impact)}\n\n")
                f.write("Consider removing these features or investigating for leakage/overfitting.\n\n")
            else:
                f.write("✅ All feature families improved OOS performance.\n\n")
        
        # Also save as JSON (convert numpy types to Python types)
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, "w") as f:
            results_dict = []
            for r in results:
                r_dict = asdict(r)
                # Convert numpy floats to Python floats
                for key in ['accuracy', 'precision', 'recall', 'f1', 'coverage', 'train_time_seconds']:
                    if key in r_dict:
                        r_dict[key] = float(r_dict[key])
                # Convert feature importance tuples
                r_dict['top_features'] = [(str(name), float(imp)) for name, imp in r_dict['top_features']]
                results_dict.append(r_dict)
            json.dump(results_dict, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        print(f"Results saved to: {json_path}")
        
        return str(report_path)


def run_quick_ablation_example():
    """
    Run a quick ablation study on synthetic data.
    
    This is for testing the framework. For real ablation:
    1. Load actual XAUUSD data
    2. Build all feature families
    3. Run walk-forward with each configuration
    4. Compare OOS metrics
    """
    from sklearn.datasets import make_classification
    
    print("Generating synthetic data for ablation example...")
    
    # Generate data with 80 features (simulating our feature families)
    X, y = make_classification(
        n_samples=5000,
        n_features=80,
        n_informative=50,
        n_redundant=20,
        n_repeated=10,
        random_state=42,
    )
    
    # Split
    split = 4000
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Convert to DataFrame with feature names
    feature_names = [f"feat_{i}" for i in range(80)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    
    # Define feature families (simulated)
    feature_families = {
        "base": [f"feat_{i}" for i in range(0, 20)],  # 20 base features
        "micro": [f"feat_{i}" for i in range(20, 35)],  # 15 microstructure
        "mtf": [f"feat_{i}" for i in range(35, 50)],  # 15 multi-timeframe
        "smc": [f"feat_{i}" for i in range(50, 65)],  # 15 SMC
        "intermarket": [f"feat_{i}" for i in range(65, 80)],  # 15 intermarket
    }
    
    # Run ablation
    study = FeatureAblationStudy(output_dir="artifacts/ablation_example")
    results = study.run(
        X_train_df, y_train,
        X_val_df, y_val,
        feature_families=feature_families,
        use_ensemble=False,
    )
    
    # Generate report
    report_path = study.generate_report(results)
    
    print(f"\n✅ Ablation study complete!")
    print(f"📊 Report: {report_path}")


if __name__ == "__main__":
    run_quick_ablation_example()
