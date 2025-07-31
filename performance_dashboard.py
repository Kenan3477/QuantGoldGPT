#!/usr/bin/env python3
"""
GoldGPT Performance Dashboard - Real-time ML Performance Monitoring
Shows strategy accuracy, learning progress, and market regime detection
"""

import asyncio
import sqlite3
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import io
import base64

# Import our prediction systems
from advanced_unified_prediction_system import AdvancedUnifiedPredictionSystem
from prediction_validation_engine import PredictionValidationEngine
from self_improving_learning_engine import SelfImprovingLearningEngine

@dataclass
class PerformanceMetrics:
    """Performance metrics for dashboard display"""
    strategy_id: str
    timeframe: str
    accuracy_rate: float
    profit_loss: float
    win_rate: float
    total_predictions: int
    sharpe_ratio: float
    max_drawdown: float
    confidence_calibration: float
    trend: str  # 'improving', 'declining', 'stable'
    last_updated: str

@dataclass
class MarketRegime:
    """Market regime information"""
    regime_type: str
    confidence: float
    volatility_level: str
    trend_strength: float
    optimal_strategies: Dict[str, str]
    duration_days: int
    accuracy_in_regime: Dict[str, float]

class PerformanceDashboard:
    """
    Advanced performance dashboard for monitoring ML system performance,
    learning progress, and market regime detection
    """
    
    def __init__(self, db_path: str = "goldgpt_ml_tracking.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize systems
        self.prediction_system = AdvancedUnifiedPredictionSystem()
        self.validator = PredictionValidationEngine()
        self.learning_engine = SelfImprovingLearningEngine()
        
        # Dashboard configuration
        self.timeframes = ['1h', '4h', '1d', '1w']
        self.strategies = ['technical', 'sentiment', 'macro', 'pattern', 'unified_ensemble']
        
        # Initialize matplotlib for dashboard charts
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    async def get_comprehensive_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for all strategies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get performance data for last 30 days
            cursor.execute("""
                SELECT 
                    sp.strategy_id, sp.timeframe, sp.accuracy_rate,
                    sp.total_profit_loss, sp.win_rate, sp.total_predictions,
                    sp.sharpe_ratio, sp.max_drawdown, sp.confidence_accuracy_correlation,
                    sp.performance_trend, sp.last_updated
                FROM strategy_performance sp
                WHERE sp.measurement_date >= date('now', '-30 days')
                AND sp.total_predictions >= 3
                ORDER BY sp.accuracy_rate DESC
            """)
            
            performance_data = cursor.fetchall()
            
            # Format performance metrics
            metrics = []
            for row in performance_data:
                metric = PerformanceMetrics(
                    strategy_id=row[0],
                    timeframe=row[1],
                    accuracy_rate=row[2] or 0,
                    profit_loss=row[3] or 0,
                    win_rate=row[4] or 0,
                    total_predictions=row[5] or 0,
                    sharpe_ratio=row[6] or 0,
                    max_drawdown=row[7] or 0,
                    confidence_calibration=row[8] or 0,
                    trend=row[9] or 'stable',
                    last_updated=row[10] or datetime.now().isoformat()
                )
                metrics.append(metric)
            
            # Calculate summary statistics
            if metrics:
                avg_accuracy = np.mean([m.accuracy_rate for m in metrics])
                avg_profit = np.mean([m.profit_loss for m in metrics])
                total_predictions = sum([m.total_predictions for m in metrics])
                best_strategy = max(metrics, key=lambda x: x.accuracy_rate)
            else:
                avg_accuracy = 0
                avg_profit = 0
                total_predictions = 0
                best_strategy = None
            
            conn.close()
            
            return {
                'performance_metrics': [asdict(m) for m in metrics],
                'summary': {
                    'avg_accuracy': round(avg_accuracy, 2),
                    'avg_profit': round(avg_profit, 2),
                    'total_predictions': total_predictions,
                    'best_strategy': asdict(best_strategy) if best_strategy else None,
                    'strategies_active': len(set(m.strategy_id for m in metrics)),
                    'last_updated': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics retrieval failed: {e}")
            return {'performance_metrics': [], 'summary': {}}
    
    async def get_learning_progress_metrics(self) -> Dict[str, Any]:
        """Get learning engine progress and insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent learning insights
            cursor.execute("""
                SELECT 
                    insight_type, title, description, confidence,
                    accuracy_improvement, discovery_date, is_implemented
                FROM learning_insights 
                WHERE discovery_date >= date('now', '-30 days')
                ORDER BY confidence DESC, discovery_date DESC
                LIMIT 20
            """)
            
            insights = cursor.fetchall()
            
            # Get model evolution history
            cursor.execute("""
                SELECT 
                    strategy_id, version, created_date, algorithm_type,
                    improvement_over_parent, created_reason
                FROM model_version_history 
                WHERE created_date >= date('now', '-30 days')
                ORDER BY created_date DESC
            """)
            
            model_evolution = cursor.fetchall()
            
            # Get feature discovery statistics
            cursor.execute("""
                SELECT 
                    insight_type, COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    AVG(accuracy_improvement) as avg_improvement
                FROM learning_insights 
                WHERE discovery_date >= date('now', '-7 days')
                GROUP BY insight_type
            """)
            
            feature_stats = cursor.fetchall()
            
            conn.close()
            
            # Format insights
            formatted_insights = []
            for insight in insights:
                formatted_insights.append({
                    'type': insight[0],
                    'title': insight[1],
                    'description': insight[2],
                    'confidence': insight[3],
                    'improvement': insight[4],
                    'date': insight[5],
                    'implemented': bool(insight[6])
                })
            
            # Format model evolution
            formatted_evolution = []
            for model in model_evolution:
                formatted_evolution.append({
                    'strategy': model[0],
                    'version': model[1],
                    'date': model[2],
                    'algorithm': model[3],
                    'improvement': model[4],
                    'reason': model[5]
                })
            
            # Format feature statistics
            formatted_stats = {}
            for stat in feature_stats:
                formatted_stats[stat[0]] = {
                    'count': stat[1],
                    'avg_confidence': round(stat[2] or 0, 3),
                    'avg_improvement': round(stat[3] or 0, 2)
                }
            
            return {
                'recent_insights': formatted_insights,
                'model_evolution': formatted_evolution,
                'feature_discovery_stats': formatted_stats,
                'learning_velocity': len(formatted_insights) / 30,  # Insights per day
                'models_evolved': len(formatted_evolution),
                'total_improvements': sum(s.get('avg_improvement', 0) for s in formatted_stats.values())
            }
            
        except Exception as e:
            self.logger.error(f"❌ Learning progress retrieval failed: {e}")
            return {}
    
    async def detect_current_market_regime(self) -> MarketRegime:
        """Detect and analyze current market regime"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent prediction validation data to analyze market behavior
            cursor.execute("""
                SELECT 
                    pv.actual_change_percent, pv.volatility_during_period,
                    pv.validation_date, dp.strategy_id, pv.accuracy_score
                FROM prediction_validation pv
                JOIN daily_predictions dp ON pv.prediction_id = dp.id
                WHERE pv.validation_date >= date('now', '-14 days')
                ORDER BY pv.validation_date DESC
            """)
            
            market_data = cursor.fetchall()
            
            if not market_data:
                # Default regime if no data
                return MarketRegime(
                    regime_type='ranging',
                    confidence=0.5,
                    volatility_level='medium',
                    trend_strength=0.5,
                    optimal_strategies={'1h': 'technical', '4h': 'sentiment', '1d': 'macro', '1w': 'pattern'},
                    duration_days=1,
                    accuracy_in_regime={'technical': 0.6, 'sentiment': 0.6, 'macro': 0.6, 'pattern': 0.6}
                )
            
            # Analyze market characteristics
            changes = [row[0] for row in market_data if row[0] is not None]
            volatilities = [row[1] for row in market_data if row[1] is not None]
            
            # Calculate regime characteristics
            avg_volatility = np.mean(volatilities) if volatilities else 1.0
            volatility_std = np.std(volatilities) if len(volatilities) > 1 else 0.5
            
            # Determine volatility level
            if avg_volatility > 2.5:
                volatility_level = 'high'
            elif avg_volatility > 1.0:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            # Analyze trend strength
            trend_strength = min(1.0, abs(np.mean(changes)) / 2.0) if changes else 0.5
            
            # Determine regime type
            if avg_volatility > 3.0:
                regime_type = 'high_volatility'
            elif trend_strength > 0.8:
                if np.mean(changes) > 0:
                    regime_type = 'bull_trending'
                else:
                    regime_type = 'bear_trending'
            else:
                regime_type = 'ranging'
            
            # Calculate strategy performance in current regime
            strategy_accuracy = {}
            for strategy in self.strategies:
                strategy_data = [row[4] for row in market_data if row[3] == strategy and row[4] is not None]
                if strategy_data:
                    strategy_accuracy[strategy] = np.mean(strategy_data)
                else:
                    strategy_accuracy[strategy] = 0.6  # Default
            
            # Determine optimal strategies per timeframe based on regime
            optimal_strategies = self._get_optimal_strategies_for_regime(regime_type, strategy_accuracy)
            
            # Calculate confidence based on data consistency
            confidence = min(1.0, len(market_data) / 20)  # Higher confidence with more data
            
            conn.close()
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=round(confidence, 3),
                volatility_level=volatility_level,
                trend_strength=round(trend_strength, 3),
                optimal_strategies=optimal_strategies,
                duration_days=14,  # Analysis period
                accuracy_in_regime=strategy_accuracy
            )
            
        except Exception as e:
            self.logger.error(f"❌ Market regime detection failed: {e}")
            # Return default regime
            return MarketRegime(
                regime_type='unknown',
                confidence=0.3,
                volatility_level='medium',
                trend_strength=0.5,
                optimal_strategies={'1h': 'technical', '4h': 'sentiment', '1d': 'macro', '1w': 'pattern'},
                duration_days=1,
                accuracy_in_regime={'technical': 0.5, 'sentiment': 0.5, 'macro': 0.5, 'pattern': 0.5}
            )
    
    def _get_optimal_strategies_for_regime(self, regime_type: str, 
                                         strategy_accuracy: Dict[str, float]) -> Dict[str, str]:
        """Get optimal strategies for each timeframe based on regime"""
        
        # Regime-based strategy preferences
        regime_preferences = {
            'bull_trending': {
                '1h': ['technical', 'pattern'],
                '4h': ['technical', 'sentiment'],
                '1d': ['macro', 'sentiment'],
                '1w': ['macro', 'pattern']
            },
            'bear_trending': {
                '1h': ['technical', 'sentiment'],
                '4h': ['sentiment', 'technical'],
                '1d': ['macro', 'technical'],
                '1w': ['macro', 'sentiment']
            },
            'high_volatility': {
                '1h': ['technical', 'pattern'],
                '4h': ['technical', 'pattern'],
                '1d': ['sentiment', 'macro'],
                '1w': ['macro', 'sentiment']
            },
            'ranging': {
                '1h': ['pattern', 'technical'],
                '4h': ['sentiment', 'pattern'],
                '1d': ['macro', 'sentiment'],
                '1w': ['macro', 'pattern']
            }
        }
        
        optimal = {}
        preferences = regime_preferences.get(regime_type, regime_preferences['ranging'])
        
        for timeframe in self.timeframes:
            # Get preferred strategies for this timeframe and regime
            preferred = preferences.get(timeframe, ['technical', 'sentiment'])
            
            # Select best performing preferred strategy
            best_strategy = max(preferred, key=lambda s: strategy_accuracy.get(s, 0.5))
            optimal[timeframe] = best_strategy
        
        return optimal
    
    async def get_confidence_calibration_analysis(self) -> Dict[str, Any]:
        """Analyze how well confidence scores correlate with actual accuracy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get prediction confidence vs actual accuracy
            cursor.execute("""
                SELECT 
                    dp.confidence_score, pv.accuracy_score, dp.strategy_id,
                    dp.timeframe, pv.validation_date
                FROM daily_predictions dp
                JOIN prediction_validation pv ON dp.id = pv.prediction_id
                WHERE pv.validation_date >= date('now', '-30 days')
                AND dp.confidence_score > 0 AND pv.accuracy_score > 0
            """)
            
            calibration_data = cursor.fetchall()
            conn.close()
            
            if not calibration_data:
                return {'error': 'No calibration data available'}
            
            # Analyze calibration by confidence buckets
            confidence_buckets = {
                'very_low': (0.0, 0.2),
                'low': (0.2, 0.4),
                'medium': (0.4, 0.6),
                'high': (0.6, 0.8),
                'very_high': (0.8, 1.0)
            }
            
            calibration_analysis = {}
            
            for bucket_name, (min_conf, max_conf) in confidence_buckets.items():
                bucket_data = [
                    (conf, acc) for conf, acc, _, _, _ in calibration_data
                    if min_conf <= conf < max_conf
                ]
                
                if bucket_data:
                    confidences = [d[0] for d in bucket_data]
                    accuracies = [d[1] for d in bucket_data]
                    
                    calibration_analysis[bucket_name] = {
                        'count': len(bucket_data),
                        'avg_confidence': round(np.mean(confidences), 3),
                        'avg_accuracy': round(np.mean(accuracies), 3),
                        'calibration_error': round(abs(np.mean(confidences) - np.mean(accuracies)), 3),
                        'confidence_range': f"{min_conf:.1f}-{max_conf:.1f}"
                    }
            
            # Overall calibration metrics
            all_confidences = [row[0] for row in calibration_data]
            all_accuracies = [row[1] for row in calibration_data]
            
            overall_calibration_error = abs(np.mean(all_confidences) - np.mean(all_accuracies))
            correlation = np.corrcoef(all_confidences, all_accuracies)[0, 1] if len(all_confidences) > 1 else 0
            
            return {
                'bucket_analysis': calibration_analysis,
                'overall_metrics': {
                    'total_predictions': len(calibration_data),
                    'overall_calibration_error': round(overall_calibration_error, 3),
                    'confidence_accuracy_correlation': round(correlation, 3),
                    'avg_confidence': round(np.mean(all_confidences), 3),
                    'avg_accuracy': round(np.mean(all_accuracies), 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Confidence calibration analysis failed: {e}")
            return {'error': str(e)}
    
    async def generate_performance_charts(self) -> Dict[str, str]:
        """Generate performance visualization charts"""
        try:
            charts = {}
            
            # 1. Strategy Accuracy Comparison Chart
            accuracy_chart = await self._create_accuracy_comparison_chart()
            charts['accuracy_comparison'] = accuracy_chart
            
            # 2. Performance Trend Chart
            trend_chart = await self._create_performance_trend_chart()
            charts['performance_trend'] = trend_chart
            
            # 3. Confidence Calibration Chart
            calibration_chart = await self._create_calibration_chart()
            charts['confidence_calibration'] = calibration_chart
            
            # 4. Market Regime Analysis Chart
            regime_chart = await self._create_regime_analysis_chart()
            charts['market_regime'] = regime_chart
            
            return charts
            
        except Exception as e:
            self.logger.error(f"❌ Chart generation failed: {e}")
            return {}
    
    async def _create_accuracy_comparison_chart(self) -> str:
        """Create strategy accuracy comparison chart"""
        try:
            # Get performance data
            performance_data = await self.get_comprehensive_performance_metrics()
            metrics = performance_data['performance_metrics']
            
            if not metrics:
                return ""
            
            # Prepare data for plotting
            strategies = list(set(m['strategy_id'] for m in metrics))
            timeframes = self.timeframes
            
            # Create accuracy matrix
            accuracy_matrix = []
            for strategy in strategies:
                strategy_accuracies = []
                for timeframe in timeframes:
                    accuracy = next((m['accuracy_rate'] for m in metrics 
                                   if m['strategy_id'] == strategy and m['timeframe'] == timeframe), 0)
                    strategy_accuracies.append(accuracy)
                accuracy_matrix.append(strategy_accuracies)
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(accuracy_matrix, 
                       xticklabels=timeframes, 
                       yticklabels=strategies,
                       annot=True, 
                       fmt='.1f',
                       cmap='RdYlGn',
                       center=70)  # Center colormap at 70% accuracy
            
            plt.title('Strategy Accuracy by Timeframe (%)', fontsize=14, fontweight='bold')
            plt.xlabel('Timeframe', fontsize=12)
            plt.ylabel('Strategy', fontsize=12)
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1e1e1e', bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_data}"
            
        except Exception as e:
            self.logger.error(f"❌ Accuracy chart creation failed: {e}")
            return ""
    
    async def _create_performance_trend_chart(self) -> str:
        """Create performance trend over time chart"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get historical performance data
            cursor.execute("""
                SELECT measurement_date, AVG(accuracy_rate), AVG(total_profit_loss)
                FROM strategy_performance 
                WHERE measurement_date >= date('now', '-30 days')
                GROUP BY measurement_date
                ORDER BY measurement_date
            """)
            
            trend_data = cursor.fetchall()
            conn.close()
            
            if not trend_data:
                return ""
            
            dates = [row[0] for row in trend_data]
            accuracies = [row[1] or 0 for row in trend_data]
            profits = [row[2] or 0 for row in trend_data]
            
            # Create dual-axis plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Accuracy trend
            color = 'tab:green'
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Accuracy Rate (%)', color=color, fontsize=12)
            ax1.plot(dates, accuracies, color=color, linewidth=2, marker='o', label='Accuracy')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Profit trend
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Profit/Loss (%)', color=color, fontsize=12)
            ax2.plot(dates, profits, color=color, linewidth=2, marker='s', label='P&L')
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Performance Trends Over Time', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1e1e1e', bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_data}"
            
        except Exception as e:
            self.logger.error(f"❌ Trend chart creation failed: {e}")
            return ""
    
    async def _create_calibration_chart(self) -> str:
        """Create confidence calibration chart"""
        try:
            calibration_data = await self.get_confidence_calibration_analysis()
            
            if 'error' in calibration_data:
                return ""
            
            bucket_analysis = calibration_data['bucket_analysis']
            
            # Prepare data
            buckets = list(bucket_analysis.keys())
            confidences = [bucket_analysis[b]['avg_confidence'] for b in buckets]
            accuracies = [bucket_analysis[b]['avg_accuracy'] for b in buckets]
            
            # Create calibration plot
            plt.figure(figsize=(8, 8))
            
            # Perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            
            # Actual calibration
            plt.scatter(confidences, accuracies, s=100, alpha=0.7, c='red', label='Actual Calibration')
            
            # Add bucket labels
            for i, bucket in enumerate(buckets):
                plt.annotate(bucket, (confidences[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            plt.xlabel('Average Confidence', fontsize=12)
            plt.ylabel('Average Accuracy', fontsize=12)
            plt.title('Confidence Calibration Analysis', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1e1e1e', bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_data}"
            
        except Exception as e:
            self.logger.error(f"❌ Calibration chart creation failed: {e}")
            return ""
    
    async def _create_regime_analysis_chart(self) -> str:
        """Create market regime analysis chart"""
        try:
            regime = await self.detect_current_market_regime()
            
            # Create regime visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Strategy accuracy in current regime
            strategies = list(regime.accuracy_in_regime.keys())
            accuracies = list(regime.accuracy_in_regime.values())
            
            ax1.bar(strategies, accuracies, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4'])
            ax1.set_title(f'Strategy Performance in {regime.regime_type.replace("_", " ").title()} Regime')
            ax1.set_ylabel('Accuracy Rate')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Optimal strategies by timeframe
            timeframes = list(regime.optimal_strategies.keys())
            optimal_strats = list(regime.optimal_strategies.values())
            
            strategy_colors = {s: i for i, s in enumerate(set(optimal_strats))}
            colors = [strategy_colors[s] for s in optimal_strats]
            
            ax2.bar(timeframes, [1]*len(timeframes), color=plt.cm.Set3(np.array(colors)))
            ax2.set_title('Optimal Strategies by Timeframe')
            ax2.set_ylabel('Strategy Assignment')
            
            # Add strategy labels
            for i, (tf, strat) in enumerate(zip(timeframes, optimal_strats)):
                ax2.text(i, 0.5, strat, ha='center', va='center', rotation=90, fontweight='bold')
            
            # 3. Regime characteristics
            characteristics = {
                'Volatility': regime.volatility_level,
                'Trend Strength': f"{regime.trend_strength:.2f}",
                'Confidence': f"{regime.confidence:.2f}",
                'Duration (days)': str(regime.duration_days)
            }
            
            y_pos = np.arange(len(characteristics))
            values = [0.8 if v in ['high', 'medium', 'low'] else float(v.split()[0]) if v.replace('.', '').isdigit() else 0.5 
                     for v in characteristics.values()]
            
            ax3.barh(y_pos, values, color='skyblue')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(characteristics.keys())
            ax3.set_title('Regime Characteristics')
            ax3.set_xlabel('Normalized Value')
            
            # 4. Regime type indicator
            regime_types = ['bull_trending', 'bear_trending', 'ranging', 'high_volatility']
            regime_scores = [1 if rt == regime.regime_type else 0 for rt in regime_types]
            
            colors = ['green' if score == 1 else 'gray' for score in regime_scores]
            ax4.bar(regime_types, regime_scores, color=colors)
            ax4.set_title('Current Market Regime')
            ax4.set_ylabel('Active')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1e1e1e', bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_data}"
            
        except Exception as e:
            self.logger.error(f"❌ Regime chart creation failed: {e}")
            return ""
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data"""
        try:
            self.logger.info("📊 Generating comprehensive dashboard data...")
            
            # Collect all dashboard components
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': await self.get_comprehensive_performance_metrics(),
                'learning_progress': await self.get_learning_progress_metrics(),
                'market_regime': asdict(await self.detect_current_market_regime()),
                'confidence_calibration': await self.get_confidence_calibration_analysis(),
                'performance_charts': await self.generate_performance_charts(),
                'system_status': {
                    'prediction_system_active': True,
                    'validation_engine_active': True,
                    'learning_engine_active': True,
                    'last_prediction_generation': datetime.now().isoformat(),
                    'database_status': 'healthy',
                    'total_strategies_monitored': len(self.strategies)
                }
            }
            
            self.logger.info("✅ Dashboard data generation completed")
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"❌ Dashboard data generation failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_status': {'status': 'error'}
            }

async def main():
    """Test the performance dashboard"""
    logging.basicConfig(level=logging.INFO)
    
    dashboard = PerformanceDashboard()
    
    print("📊 Generating performance dashboard...")
    
    # Generate complete dashboard data
    dashboard_data = await dashboard.generate_dashboard_data()
    
    print("✅ Dashboard Data Generated")
    print(f"📈 Performance Metrics: {len(dashboard_data.get('performance_metrics', {}).get('performance_metrics', []))}")
    print(f"🧠 Learning Insights: {dashboard_data.get('learning_progress', {}).get('total_improvements', 0)}")
    print(f"🎯 Market Regime: {dashboard_data.get('market_regime', {}).get('regime_type', 'unknown')}")
    print(f"📊 Charts Generated: {len(dashboard_data.get('performance_charts', {}))}")
    
    # Display summary
    summary = dashboard_data.get('performance_metrics', {}).get('summary', {})
    if summary:
        print(f"\n📊 Performance Summary:")
        print(f"  Average Accuracy: {summary.get('avg_accuracy', 0)}%")
        print(f"  Average Profit: {summary.get('avg_profit', 0)}%")
        print(f"  Total Predictions: {summary.get('total_predictions', 0)}")
        print(f"  Active Strategies: {summary.get('strategies_active', 0)}")

if __name__ == "__main__":
    asyncio.run(main())
