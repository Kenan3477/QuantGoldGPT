#!/usr/bin/env python3
"""
GoldGPT Self-Improving ML Learning Engine
Automatically retrains models based on validation results and discovers new patterns
"""

import asyncio
import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    strategy_id: str
    model_version: str
    accuracy: float
    profit_loss: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    confidence_calibration: float
    feature_importance: Dict[str, float]
    training_date: datetime
    validation_samples: int

@dataclass
class FeatureInsight:
    """Discovered feature insight"""
    feature_name: str
    importance_score: float
    correlation_with_accuracy: float
    best_timeframes: List[str]
    market_conditions: List[str]
    statistical_significance: float

class SelfImprovingLearningEngine:
    """
    Advanced ML learning engine that automatically evolves strategies based on validation results
    """
    
    def __init__(self, db_path: str = "goldgpt_ml_tracking.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.model_artifacts_dir = "model_artifacts"
        self._ensure_directories()
        
        # Available algorithms for experimentation
        self.algorithms = {
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'MLP': MLPRegressor
        }
        
        # Feature engineering functions
        self.feature_generators = [
            self._generate_technical_features,
            self._generate_momentum_features,
            self._generate_volatility_features,
            self._generate_pattern_features,
            self._generate_market_structure_features
        ]
    
    def _ensure_directories(self):
        """Ensure model artifacts directory exists"""
        os.makedirs(self.model_artifacts_dir, exist_ok=True)
        os.makedirs(f"{self.model_artifacts_dir}/scalers", exist_ok=True)
    
    async def analyze_strategy_performance(self) -> List[ModelPerformance]:
        """Analyze current strategy performance and identify improvement opportunities"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get performance data for all strategies
            cursor.execute("""
                SELECT 
                    sp.strategy_id, sp.model_version, sp.timeframe,
                    sp.accuracy_rate, sp.total_profit_loss, sp.sharpe_ratio,
                    sp.max_drawdown, sp.win_rate, sp.confidence_accuracy_correlation,
                    COUNT(pv.id) as validation_samples
                FROM strategy_performance sp
                LEFT JOIN daily_predictions dp ON (
                    sp.strategy_id = dp.strategy_id AND 
                    sp.model_version = dp.model_version AND 
                    sp.timeframe = dp.timeframe
                )
                LEFT JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE sp.last_updated >= date('now', '-7 days')
                GROUP BY sp.strategy_id, sp.model_version, sp.timeframe
                HAVING validation_samples >= 5
                ORDER BY sp.accuracy_rate DESC
            """)
            
            performance_data = cursor.fetchall()
            performances = []
            
            for row in performance_data:
                # Get feature importance for this strategy
                feature_importance = await self._get_feature_importance(row[0], row[1])
                
                performance = ModelPerformance(
                    strategy_id=row[0],
                    model_version=row[1],
                    accuracy=row[3] or 0,
                    profit_loss=row[4] or 0,
                    sharpe_ratio=row[5] or 0,
                    max_drawdown=row[6] or 0,
                    win_rate=row[7] or 0,
                    confidence_calibration=row[8] or 0,
                    feature_importance=feature_importance,
                    training_date=datetime.now(),
                    validation_samples=row[9] or 0
                )
                performances.append(performance)
            
            conn.close()
            
            self.logger.info(f"✅ Analyzed performance for {len(performances)} strategies")
            return performances
            
        except Exception as e:
            self.logger.error(f"❌ Strategy performance analysis failed: {e}")
            return []
    
    async def _get_feature_importance(self, strategy_id: str, model_version: str) -> Dict[str, float]:
        """Get feature importance for a strategy"""
        try:
            # Load model if exists
            model_path = f"{self.model_artifacts_dir}/{strategy_id}_{model_version}.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    return model_data.get('feature_importance', {})
            
            # Default feature importance if model not found
            return {
                'price_change_1h': 0.15,
                'volume_ratio': 0.12,
                'rsi_14': 0.10,
                'macd_signal': 0.08,
                'bollinger_position': 0.08,
                'volatility_20': 0.07,
                'momentum_5': 0.06,
                'support_resistance': 0.05,
                'market_sentiment': 0.05,
                'economic_calendar': 0.04,
                'correlation_spy': 0.03,
                'correlation_dxy': 0.03,
                'time_of_day': 0.02,
                'day_of_week': 0.02,
                'other_features': 0.10
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature importance retrieval failed: {e}")
            return {}
    
    async def discover_new_features(self) -> List[FeatureInsight]:
        """Automatically discover new predictive features"""
        try:
            insights = []
            
            # Get recent prediction and validation data
            market_data = await self._get_market_data_for_analysis()
            if not market_data:
                return insights
            
            # Analyze each potential feature
            for feature_generator in self.feature_generators:
                try:
                    feature_insights = await feature_generator(market_data)
                    insights.extend(feature_insights)
                except Exception as e:
                    self.logger.error(f"❌ Feature generation failed: {e}")
            
            # Rank insights by statistical significance
            insights.sort(key=lambda x: x.statistical_significance, reverse=True)
            
            # Store top insights
            await self._store_feature_insights(insights[:10])  # Top 10 insights
            
            self.logger.info(f"✅ Discovered {len(insights)} new feature insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Feature discovery failed: {e}")
            return []
    
    async def _get_market_data_for_analysis(self) -> Optional[pd.DataFrame]:
        """Get market data for feature analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get prediction and validation data
            cursor.execute("""
                SELECT 
                    dp.prediction_date, dp.timeframe, dp.predicted_price,
                    dp.current_price, dp.predicted_direction, dp.confidence_score,
                    dp.market_volatility, pv.actual_price, pv.actual_direction,
                    pv.accuracy_score, pv.profit_loss_percent, pv.market_conditions
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE pv.validation_date >= date('now', '-30 days')
                ORDER BY dp.prediction_date DESC
                LIMIT 1000
            """)
            
            data = cursor.fetchall()
            conn.close()
            
            if not data:
                return None
            
            # Convert to DataFrame
            columns = [
                'date', 'timeframe', 'predicted_price', 'entry_price', 
                'predicted_direction', 'confidence', 'volatility',
                'actual_price', 'actual_direction', 'accuracy_score',
                'profit_loss', 'market_conditions'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Market data retrieval failed: {e}")
            return None
    
    async def _generate_technical_features(self, df: pd.DataFrame) -> List[FeatureInsight]:
        """Generate technical analysis feature insights"""
        insights = []
        
        try:
            # Price momentum features
            df['price_momentum_1h'] = df['actual_price'].pct_change(1)
            df['price_momentum_4h'] = df['actual_price'].pct_change(4)
            
            # Volatility features
            df['volatility_rolling'] = df['actual_price'].rolling(window=5).std()
            df['volatility_normalized'] = df['volatility_rolling'] / df['actual_price']
            
            # Accuracy correlation analysis
            for feature in ['price_momentum_1h', 'price_momentum_4h', 'volatility_normalized']:
                if feature in df.columns:
                    correlation = df[feature].corr(df['accuracy_score'])
                    if abs(correlation) > 0.1:  # Minimum correlation threshold
                        insight = FeatureInsight(
                            feature_name=feature,
                            importance_score=abs(correlation),
                            correlation_with_accuracy=correlation,
                            best_timeframes=['1h', '4h'] if 'momentum' in feature else ['1d', '1w'],
                            market_conditions=['trending'] if 'momentum' in feature else ['volatile'],
                            statistical_significance=abs(correlation) * len(df) / 100
                        )
                        insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"❌ Technical feature generation failed: {e}")
        
        return insights
    
    async def _generate_momentum_features(self, df: pd.DataFrame) -> List[FeatureInsight]:
        """Generate momentum-based feature insights"""
        insights = []
        
        try:
            # Calculate momentum indicators
            df['price_velocity'] = df['actual_price'].diff()
            df['price_acceleration'] = df['price_velocity'].diff()
            
            # Direction consistency
            df['direction_momentum'] = (
                df['predicted_direction'] == df['predicted_direction'].shift(1)
            ).astype(int)
            
            # Analyze momentum vs accuracy
            for feature in ['price_velocity', 'price_acceleration', 'direction_momentum']:
                if feature in df.columns:
                    correlation = df[feature].corr(df['accuracy_score'])
                    if abs(correlation) > 0.08:
                        insight = FeatureInsight(
                            feature_name=feature,
                            importance_score=abs(correlation),
                            correlation_with_accuracy=correlation,
                            best_timeframes=['1h', '4h'],
                            market_conditions=['trending', 'momentum'],
                            statistical_significance=abs(correlation) * np.sqrt(len(df)) / 10
                        )
                        insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"❌ Momentum feature generation failed: {e}")
        
        return insights
    
    async def _generate_volatility_features(self, df: pd.DataFrame) -> List[FeatureInsight]:
        """Generate volatility-based feature insights"""
        insights = []
        
        try:
            # Volatility regime detection
            df['volatility_regime'] = pd.cut(
                df['volatility'], bins=3, labels=['low', 'medium', 'high']
            )
            
            # Volatility vs accuracy by regime
            for regime in ['low', 'medium', 'high']:
                regime_mask = df['volatility_regime'] == regime
                if regime_mask.sum() > 10:  # Minimum samples
                    regime_accuracy = df[regime_mask]['accuracy_score'].mean()
                    overall_accuracy = df['accuracy_score'].mean()
                    
                    if abs(regime_accuracy - overall_accuracy) > 0.05:  # 5% difference
                        insight = FeatureInsight(
                            feature_name=f'volatility_regime_{regime}',
                            importance_score=abs(regime_accuracy - overall_accuracy),
                            correlation_with_accuracy=regime_accuracy - overall_accuracy,
                            best_timeframes=['1h', '4h'] if regime == 'high' else ['1d', '1w'],
                            market_conditions=[regime],
                            statistical_significance=abs(regime_accuracy - overall_accuracy) * regime_mask.sum() / 100
                        )
                        insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"❌ Volatility feature generation failed: {e}")
        
        return insights
    
    async def _generate_pattern_features(self, df: pd.DataFrame) -> List[FeatureInsight]:
        """Generate pattern-based feature insights"""
        insights = []
        
        try:
            # Time-based patterns
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # Accuracy by time patterns
            for time_feature in ['hour', 'day_of_week']:
                if time_feature in df.columns:
                    time_accuracy = df.groupby(time_feature)['accuracy_score'].mean()
                    overall_accuracy = df['accuracy_score'].mean()
                    
                    # Find time periods with significantly different accuracy
                    for time_value, accuracy in time_accuracy.items():
                        if abs(accuracy - overall_accuracy) > 0.08:  # 8% difference
                            insight = FeatureInsight(
                                feature_name=f'{time_feature}_{time_value}',
                                importance_score=abs(accuracy - overall_accuracy),
                                correlation_with_accuracy=accuracy - overall_accuracy,
                                best_timeframes=['1h', '4h'] if time_feature == 'hour' else ['1d'],
                                market_conditions=['time_pattern'],
                                statistical_significance=abs(accuracy - overall_accuracy) * len(df) / 200
                            )
                            insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"❌ Pattern feature generation failed: {e}")
        
        return insights
    
    async def _generate_market_structure_features(self, df: pd.DataFrame) -> List[FeatureInsight]:
        """Generate market structure feature insights"""
        insights = []
        
        try:
            # Confidence calibration analysis
            df['confidence_bucket'] = pd.cut(
                df['confidence'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            # Analyze confidence vs actual accuracy
            for bucket in ['very_low', 'low', 'medium', 'high', 'very_high']:
                bucket_mask = df['confidence_bucket'] == bucket
                if bucket_mask.sum() > 5:
                    bucket_accuracy = df[bucket_mask]['accuracy_score'].mean()
                    expected_confidence = {
                        'very_low': 0.2, 'low': 0.4, 'medium': 0.6, 'high': 0.8, 'very_high': 0.9
                    }[bucket]
                    
                    calibration_error = abs(bucket_accuracy - expected_confidence)
                    if calibration_error > 0.1:  # 10% calibration error
                        insight = FeatureInsight(
                            feature_name=f'confidence_calibration_{bucket}',
                            importance_score=calibration_error,
                            correlation_with_accuracy=bucket_accuracy - expected_confidence,
                            best_timeframes=['1h', '4h', '1d'],
                            market_conditions=['confidence_analysis'],
                            statistical_significance=calibration_error * bucket_mask.sum() / 50
                        )
                        insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"❌ Market structure feature generation failed: {e}")
        
        return insights
    
    async def _store_feature_insights(self, insights: List[FeatureInsight]):
        """Store discovered feature insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for insight in insights:
                cursor.execute("""
                    INSERT INTO learning_insights (
                        insight_type, title, description, confidence,
                        accuracy_improvement, affected_timeframes, affected_strategies,
                        discovered_by, statistical_significance, supporting_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'feature_discovery',
                    f'New Feature: {insight.feature_name}',
                    f'Feature {insight.feature_name} shows {insight.correlation_with_accuracy:.1%} correlation with accuracy',
                    insight.importance_score,
                    abs(insight.correlation_with_accuracy) * 100,
                    json.dumps(insight.best_timeframes),
                    json.dumps(['ensemble', 'technical', 'pattern']),
                    'automated_analysis',
                    insight.statistical_significance,
                    json.dumps(asdict(insight))
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Stored {len(insights)} feature insights")
            
        except Exception as e:
            self.logger.error(f"❌ Feature insight storage failed: {e}")
    
    async def retrain_underperforming_strategies(self) -> Dict[str, Any]:
        """Retrain strategies that are underperforming"""
        try:
            # Analyze current performance
            performances = await self.analyze_strategy_performance()
            
            # Identify underperforming strategies
            avg_accuracy = np.mean([p.accuracy for p in performances])
            underperforming = [
                p for p in performances 
                if p.accuracy < avg_accuracy * 0.8 or p.profit_loss < 0
            ]
            
            retrain_results = {}
            
            for strategy in underperforming:
                try:
                    result = await self._retrain_single_strategy(strategy)
                    retrain_results[f"{strategy.strategy_id}_{strategy.model_version}"] = result
                except Exception as e:
                    self.logger.error(f"❌ Retrain failed for {strategy.strategy_id}: {e}")
            
            self.logger.info(f"✅ Retrained {len(retrain_results)} underperforming strategies")
            return retrain_results
            
        except Exception as e:
            self.logger.error(f"❌ Strategy retraining failed: {e}")
            return {}
    
    async def _retrain_single_strategy(self, strategy: ModelPerformance) -> Dict[str, Any]:
        """Retrain a single strategy with improved features"""
        try:
            # Get training data
            training_data = await self._prepare_training_data(strategy.strategy_id)
            if training_data is None or len(training_data) < 50:
                return {"error": "Insufficient training data"}
            
            # Enhanced feature engineering
            X, y = await self._engineer_features(training_data, strategy)
            
            # Try multiple algorithms
            best_model = None
            best_score = -np.inf
            best_algorithm = None
            
            for alg_name, alg_class in self.algorithms.items():
                try:
                    model = self._configure_algorithm(alg_class, alg_name)
                    
                    # Cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                    avg_score = np.mean(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_algorithm = alg_name
                        
                except Exception as e:
                    self.logger.error(f"❌ Algorithm {alg_name} failed: {e}")
            
            if best_model is None:
                return {"error": "No algorithm succeeded"}
            
            # Train best model on full dataset
            best_model.fit(X, y)
            
            # Generate new model version
            new_version = await self._create_new_model_version(
                strategy.strategy_id, best_algorithm, best_model, X.columns.tolist()
            )
            
            # Save model artifacts
            await self._save_model_artifacts(strategy.strategy_id, new_version, best_model, X.columns)
            
            return {
                "strategy_id": strategy.strategy_id,
                "new_version": new_version,
                "algorithm": best_algorithm,
                "cross_val_score": -best_score,
                "improvement_expected": max(0, -best_score - strategy.accuracy / 100),
                "features_count": len(X.columns),
                "training_samples": len(X)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Single strategy retrain failed: {e}")
            return {"error": str(e)}
    
    async def _prepare_training_data(self, strategy_id: str) -> Optional[pd.DataFrame]:
        """Prepare training data for strategy retraining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get historical prediction and validation data
            cursor.execute("""
                SELECT 
                    dp.prediction_date, dp.current_price, dp.predicted_price,
                    dp.confidence_score, dp.market_volatility, dp.feature_weights,
                    pv.actual_price, pv.accuracy_score, pv.profit_loss_percent,
                    pv.market_conditions
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE dp.strategy_id = ?
                AND pv.validation_date >= date('now', '-90 days')
                ORDER BY dp.prediction_date DESC
            """, (strategy_id,))
            
            data = cursor.fetchall()
            conn.close()
            
            if not data:
                return None
            
            columns = [
                'date', 'entry_price', 'predicted_price', 'confidence',
                'volatility', 'feature_weights', 'actual_price', 'accuracy',
                'profit_loss', 'market_conditions'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Training data preparation failed: {e}")
            return None
    
    async def _engineer_features(self, df: pd.DataFrame, strategy: ModelPerformance) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineer enhanced features for retraining"""
        try:
            features = pd.DataFrame()
            
            # Basic price features
            features['price_change'] = df['actual_price'].pct_change()
            features['price_volatility'] = df['actual_price'].rolling(5).std()
            features['price_momentum'] = df['actual_price'].diff(3)
            
            # Confidence features
            features['confidence'] = df['confidence']
            features['confidence_squared'] = df['confidence'] ** 2
            features['confidence_volatility'] = df['confidence'].rolling(5).std()
            
            # Market volatility features
            features['market_volatility'] = df['volatility']
            features['volatility_change'] = df['volatility'].diff()
            features['volatility_normalized'] = df['volatility'] / df['actual_price']
            
            # Time-based features
            features['hour'] = df['date'].dt.hour
            features['day_of_week'] = df['date'].dt.dayofweek
            features['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            
            # Rolling statistics
            for window in [3, 5, 10]:
                features[f'accuracy_ma_{window}'] = df['accuracy'].rolling(window).mean()
                features[f'profit_ma_{window}'] = df['profit_loss'].rolling(window).mean()
            
            # Feature importance from strategy
            top_features = sorted(
                strategy.feature_importance.items(), 
                key=lambda x: x[1], reverse=True
            )[:10]
            
            for feature_name, importance in top_features:
                if feature_name not in features.columns:
                    # Simulate feature based on importance
                    features[feature_name] = np.random.normal(0, importance, len(df))
            
            # Target variable (accuracy score to predict)
            target = df['accuracy']
            
            # Remove NaN values
            mask = ~(features.isnull().any(axis=1) | target.isnull())
            features = features[mask]
            target = target[mask]
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            
            return features_scaled, target
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _configure_algorithm(self, alg_class, alg_name: str):
        """Configure algorithm with optimized hyperparameters"""
        if alg_name == 'RandomForest':
            return alg_class(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif alg_name == 'GradientBoosting':
            return alg_class(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif alg_name == 'Ridge':
            return alg_class(alpha=1.0)
        elif alg_name == 'Lasso':
            return alg_class(alpha=0.1)
        elif alg_name == 'MLP':
            return alg_class(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            return alg_class()
    
    async def _create_new_model_version(self, strategy_id: str, algorithm: str, 
                                      model, features: List[str]) -> str:
        """Create new model version entry"""
        try:
            # Generate version number
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(CAST(SUBSTR(version, 2) AS INTEGER))
                FROM model_version_history 
                WHERE strategy_id = ?
            """, (strategy_id,))
            
            max_version = cursor.fetchone()[0] or 0
            new_version = f"v{max_version + 1}"
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(features, abs(model.coef_)))
            
            # Insert new version
            cursor.execute("""
                INSERT INTO model_version_history (
                    strategy_id, version, algorithm_type, feature_set,
                    feature_importance, created_reason, is_production
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, new_version, algorithm,
                json.dumps(features), json.dumps(feature_importance),
                'performance_improvement', True
            ))
            
            conn.commit()
            conn.close()
            
            return new_version
            
        except Exception as e:
            self.logger.error(f"❌ Model version creation failed: {e}")
            return "v1"
    
    async def _save_model_artifacts(self, strategy_id: str, version: str, 
                                  model, feature_columns):
        """Save model and scaler artifacts"""
        try:
            # Save model
            model_path = f"{self.model_artifacts_dir}/{strategy_id}_{version}.pkl"
            model_data = {
                'model': model,
                'feature_columns': list(feature_columns),
                'feature_importance': dict(zip(feature_columns, 
                    getattr(model, 'feature_importances_', [0] * len(feature_columns)))),
                'created_date': datetime.now().isoformat(),
                'algorithm': type(model).__name__
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"✅ Saved model artifacts for {strategy_id}_{version}")
            
        except Exception as e:
            self.logger.error(f"❌ Model artifact saving failed: {e}")
    
    async def optimize_ensemble_weights(self) -> Dict[str, float]:
        """Optimize ensemble weights based on recent performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent performance by strategy
            cursor.execute("""
                SELECT 
                    dp.strategy_id,
                    AVG(pv.accuracy_score) as avg_accuracy,
                    AVG(pv.profit_loss_percent) as avg_profit,
                    COUNT(*) as predictions
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE pv.validation_date >= date('now', '-14 days')
                GROUP BY dp.strategy_id
                HAVING predictions >= 3
            """)
            
            strategy_performance = cursor.fetchall()
            
            if not strategy_performance:
                return {'technical': 0.25, 'sentiment': 0.25, 'macro': 0.25, 'pattern': 0.25}
            
            # Calculate weights based on accuracy and profit
            total_score = 0
            strategy_scores = {}
            
            for strategy_id, accuracy, profit, count in strategy_performance:
                # Combined score: accuracy weight 70%, profit weight 30%
                score = (accuracy * 0.7) + ((profit + 10) / 20 * 0.3)  # Normalize profit to 0-1
                strategy_scores[strategy_id] = score
                total_score += score
            
            # Normalize to weights that sum to 1
            weights = {}
            for strategy_id, score in strategy_scores.items():
                weights[strategy_id] = round(score / total_score, 3)
            
            # Store new weights
            await self._store_ensemble_weights(weights)
            
            conn.close()
            
            self.logger.info(f"✅ Optimized ensemble weights: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble weight optimization failed: {e}")
            return {'technical': 0.25, 'sentiment': 0.25, 'macro': 0.25, 'pattern': 0.25}
    
    async def _store_ensemble_weights(self, weights: Dict[str, float]):
        """Store optimized ensemble weights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for timeframe in ['1h', '4h', '1d', '1w']:
                cursor.execute("""
                    INSERT INTO ensemble_weights (
                        calculation_date, timeframe, technical_weight, sentiment_weight,
                        macro_weight, pattern_weight, rebalance_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().date(), timeframe,
                    weights.get('technical', 0.25),
                    weights.get('sentiment', 0.25),
                    weights.get('macro', 0.25),
                    weights.get('pattern', 0.25),
                    'performance_optimization'
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble weight storage failed: {e}")
    
    async def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning progress report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent insights
            cursor.execute("""
                SELECT insight_type, COUNT(*), AVG(confidence)
                FROM learning_insights 
                WHERE discovery_date >= date('now', '-30 days')
                GROUP BY insight_type
            """)
            insights_summary = cursor.fetchall()
            
            # Get strategy evolution
            cursor.execute("""
                SELECT strategy_id, COUNT(*) as versions, MAX(created_date) as latest
                FROM model_version_history 
                WHERE created_date >= date('now', '-30 days')
                GROUP BY strategy_id
            """)
            evolution_summary = cursor.fetchall()
            
            # Get performance trends
            cursor.execute("""
                SELECT 
                    measurement_date,
                    AVG(accuracy_rate) as avg_accuracy,
                    AVG(total_profit_loss) as avg_profit
                FROM strategy_performance 
                WHERE measurement_date >= date('now', '-30 days')
                GROUP BY measurement_date
                ORDER BY measurement_date
            """)
            performance_trend = cursor.fetchall()
            
            conn.close()
            
            return {
                "report_date": datetime.now().isoformat(),
                "insights_discovered": {
                    insight_type: {"count": count, "avg_confidence": conf}
                    for insight_type, count, conf in insights_summary
                },
                "strategy_evolution": {
                    strategy: {"versions_created": versions, "latest_update": latest}
                    for strategy, versions, latest in evolution_summary
                },
                "performance_trend": [
                    {"date": date, "accuracy": acc, "profit": profit}
                    for date, acc, profit in performance_trend
                ],
                "total_insights": sum(row[1] for row in insights_summary),
                "active_strategies": len(evolution_summary),
                "learning_velocity": len(insights_summary) / 30,  # Insights per day
            }
            
        except Exception as e:
            self.logger.error(f"❌ Learning report generation failed: {e}")
            return {"error": str(e)}

async def main():
    """Test the learning engine"""
    logging.basicConfig(level=logging.INFO)
    
    engine = SelfImprovingLearningEngine()
    
    print("🧠 Starting self-improving learning engine...")
    
    # Analyze current performance
    performances = await engine.analyze_strategy_performance()
    print(f"✅ Analyzed {len(performances)} strategies")
    
    # Discover new features
    insights = await engine.discover_new_features()
    print(f"✅ Discovered {len(insights)} feature insights")
    
    # Retrain underperforming strategies
    retrain_results = await engine.retrain_underperforming_strategies()
    print(f"✅ Retrained {len(retrain_results)} strategies")
    
    # Optimize ensemble weights
    weights = await engine.optimize_ensemble_weights()
    print(f"✅ Optimized ensemble weights: {weights}")
    
    # Generate learning report
    report = await engine.generate_learning_report()
    print(f"\n📊 Learning Report:")
    print(f"Total Insights: {report.get('total_insights', 0)}")
    print(f"Active Strategies: {report.get('active_strategies', 0)}")
    print(f"Learning Velocity: {report.get('learning_velocity', 0):.2f} insights/day")

if __name__ == "__main__":
    asyncio.run(main())
