#!/usr/bin/env python3
"""
Dashboard API Endpoints for GoldGPT Prediction Tracking and Learning System
Provides comprehensive API endpoints for visualizing prediction performance, learning insights, and backtesting results
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from flask import Blueprint, request, jsonify, render_template_string
import sqlite3
import pandas as pd
import numpy as np

from prediction_tracker import PredictionTracker, PredictionRecord, ValidationResult
from learning_engine import LearningEngine
from backtesting_framework import BacktestEngine, BacktestConfig, HistoricalDataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint for dashboard endpoints
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

# Global instances (will be initialized by the Flask app)
prediction_tracker = None
learning_engine = None
backtest_engine = None

def initialize_dashboard_services(tracker: PredictionTracker, engine: LearningEngine, bt_engine: BacktestEngine):
    """Initialize the dashboard services"""
    global prediction_tracker, learning_engine, backtest_engine
    prediction_tracker = tracker
    learning_engine = engine
    backtest_engine = bt_engine

@dashboard_bp.route('/api/performance/summary', methods=['GET'])
def get_performance_summary():
    """Get overall performance summary"""
    try:
        days = request.args.get('days', 30, type=int)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        summary = prediction_tracker.get_performance_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'success': True,
            'data': summary,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/performance/strategy/<strategy_name>', methods=['GET'])
def get_strategy_performance(strategy_name: str):
    """Get detailed performance metrics for a specific strategy"""
    try:
        days = request.args.get('days', 30, type=int)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        performance = prediction_tracker.get_strategy_performance(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'success': True,
            'strategy': strategy_name,
            'data': performance,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get strategy performance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/performance/trends', methods=['GET'])
def get_performance_trends():
    """Get performance trends over time"""
    try:
        days = request.args.get('days', 90, type=int)
        interval = request.args.get('interval', 'daily')  # daily, weekly, monthly
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        trends = prediction_tracker.get_performance_trends(
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        return jsonify({
            'success': True,
            'data': trends,
            'interval': interval,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/predictions/recent', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions with validation status"""
    try:
        limit = request.args.get('limit', 50, type=int)
        symbol = request.args.get('symbol')
        strategy = request.args.get('strategy')
        
        predictions = prediction_tracker.get_recent_predictions(
            limit=limit,
            symbol=symbol,
            strategy=strategy
        )
        
        return jsonify({
            'success': True,
            'data': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        logger.error(f"Failed to get recent predictions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/predictions/accuracy', methods=['GET'])
def get_prediction_accuracy():
    """Get prediction accuracy metrics by various dimensions"""
    try:
        days = request.args.get('days', 30, type=int)
        group_by = request.args.get('group_by', 'strategy')  # strategy, symbol, timeframe, confidence_range
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        accuracy_data = prediction_tracker.get_accuracy_breakdown(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        
        return jsonify({
            'success': True,
            'data': accuracy_data,
            'group_by': group_by,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get prediction accuracy: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# @dashboard_bp.route('/api/learning/insights-alt', methods=['GET'])
# def get_learning_insights_alt():
#     """Get recent learning insights and improvements (alternative endpoint)"""
#     try:
#         limit = request.args.get('limit', 20, type=int)
#         category = request.args.get('category')  # feature_importance, model_improvement, market_regime
#         
#         insights = learning_engine.get_recent_insights(
#             limit=limit,
#             category=category
#         )
#         
#         return jsonify({
#             'success': True,
#             'data': insights,
#             'count': len(insights)
#         })
#         
#     except Exception as e:
#         logger.error(f"Failed to get learning insights: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/learning/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get current feature importance analysis"""
    try:
        strategy = request.args.get('strategy')
        days = request.args.get('days', 30, type=int)
        
        feature_importance = learning_engine.get_feature_importance_analysis(
            strategy=strategy,
            days_lookback=days
        )
        
        return jsonify({
            'success': True,
            'data': feature_importance,
            'strategy': strategy,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/learning/model-performance', methods=['GET'])
def get_model_performance():
    """Get model performance evolution over time"""
    try:
        strategy = request.args.get('strategy')
        days = request.args.get('days', 90, type=int)
        
        performance_evolution = learning_engine.get_model_performance_evolution(
            strategy=strategy,
            days_lookback=days
        )
        
        return jsonify({
            'success': True,
            'data': performance_evolution,
            'strategy': strategy,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/learning/market-regimes', methods=['GET'])
def get_market_regimes():
    """Get market regime analysis and detection"""
    try:
        symbol = request.args.get('symbol', 'XAUUSD')
        days = request.args.get('days', 60, type=int)
        
        regimes = learning_engine.get_market_regime_analysis(
            symbol=symbol,
            days_lookback=days
        )
        
        return jsonify({
            'success': True,
            'data': regimes,
            'symbol': symbol,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get market regimes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/backtesting/run', methods=['POST'])
def run_backtest():
    """Run a new backtest with specified parameters"""
    try:
        config_data = request.get_json()
        
        # Parse backtest configuration
        config = BacktestConfig(
            start_date=datetime.fromisoformat(config_data['start_date']),
            end_date=datetime.fromisoformat(config_data['end_date']),
            symbols=config_data.get('symbols', ['XAUUSD']),
            timeframes=config_data.get('timeframes', ['1H']),
            strategies=config_data.get('strategies', ['technical']),
            initial_capital=config_data.get('initial_capital', 10000.0),
            max_risk_per_trade=config_data.get('max_risk_per_trade', 0.02),
            commission_per_trade=config_data.get('commission_per_trade', 2.0),
            slippage_pips=config_data.get('slippage_pips', 1.0)
        )
        
        # Get available strategies (this would normally come from your ML system)
        strategies = learning_engine.get_available_strategies()
        
        # Run backtest asynchronously (in a real app, you'd want to run this in background)
        # For demo purposes, we'll run a simplified version
        result = asyncio.run(backtest_engine.run_backtest(config, strategies))
        
        return jsonify({
            'success': True,
            'backtest_id': f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'data': {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl_percent': result.total_pnl_percent,
                'max_drawdown_percent': result.max_drawdown_percent,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'strategy_performance': result.strategy_performance
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/backtesting/results/<backtest_id>', methods=['GET'])
def get_backtest_results(backtest_id: str):
    """Get detailed backtest results"""
    try:
        # In a real implementation, you'd store backtest results and retrieve by ID
        # For now, return a placeholder response
        
        return jsonify({
            'success': True,
            'backtest_id': backtest_id,
            'data': {
                'status': 'completed',
                'message': 'Backtest results would be retrieved from storage'
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/analytics/correlation', methods=['GET'])
def get_correlation_analysis():
    """Get correlation analysis between strategies and market conditions"""
    try:
        days = request.args.get('days', 60, type=int)
        
        correlation_data = prediction_tracker.get_strategy_correlation_analysis(
            days_lookback=days
        )
        
        return jsonify({
            'success': True,
            'data': correlation_data,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get correlation analysis: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/analytics/risk-metrics', methods=['GET'])
def get_risk_metrics():
    """Get risk metrics and analysis"""
    try:
        days = request.args.get('days', 30, type=int)
        
        risk_metrics = prediction_tracker.get_risk_analysis(
            days_lookback=days
        )
        
        return jsonify({
            'success': True,
            'data': risk_metrics,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get overall system health and status"""
    try:
        # Check database connections
        db_status = prediction_tracker.check_database_health()
        
        # Check learning engine status
        learning_status = learning_engine.get_system_status()
        
        # Get recent activity summary
        activity_summary = prediction_tracker.get_activity_summary()
        
        return jsonify({
            'success': True,
            'data': {
                'database_status': db_status,
                'learning_engine_status': learning_status,
                'activity_summary': activity_summary,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/export/predictions', methods=['GET'])
def export_predictions():
    """Export prediction data in various formats"""
    try:
        days = request.args.get('days', 30, type=int)
        format_type = request.args.get('format', 'json')  # json, csv, xlsx
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        export_data = prediction_tracker.export_prediction_data(
            start_date=start_date,
            end_date=end_date,
            format_type=format_type
        )
        
        if format_type == 'json':
            return jsonify({
                'success': True,
                'data': export_data,
                'format': format_type,
                'period_days': days
            })
        else:
            # For CSV/XLSX, you'd return the file as a download
            return jsonify({
                'success': True,
                'download_url': f'/dashboard/downloads/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format_type}',
                'format': format_type
            })
        
    except Exception as e:
        logger.error(f"Failed to export predictions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Dashboard HTML Template
@dashboard_bp.route('/', methods=['GET'])
def dashboard_home():
    """Main dashboard page"""
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GoldGPT - ML Prediction Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        }
        
        .metric-title {
            font-size: 1.1em;
            color: #ffd700;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #ffffff;
        }
        
        .metric-change {
            font-size: 0.9em;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 500;
        }
        
        .positive { color: #00ff88; background: rgba(0, 255, 136, 0.1); }
        .negative { color: #ff4757; background: rgba(255, 71, 87, 0.1); }
        .neutral { color: #ffd700; background: rgba(255, 215, 0, 0.1); }
        
        .charts-section {
            margin-bottom: 40px;
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .chart-title {
            font-size: 1.4em;
            color: #ffd700;
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.1em;
            color: #cccccc;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .control-group label {
            color: #ffd700;
            font-weight: 500;
        }
        
        select, input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            padding: 8px 12px;
            color: white;
            font-size: 0.9em;
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: #ffd700;
            box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2);
        }
        
        .btn {
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .strategy-performance {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .strategy-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #ffd700;
        }
        
        .strategy-name {
            font-weight: 600;
            color: #ffd700;
            margin-bottom: 10px;
        }
        
        .strategy-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.8em;
            color: #cccccc;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .strategy-performance {
                grid-template-columns: 1fr;
            }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>🏆 GoldGPT ML Dashboard</h1>
            <p>Advanced Prediction Tracking & Learning System</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="timePeriod">Time Period:</label>
                <select id="timePeriod">
                    <option value="7">Last 7 days</option>
                    <option value="30" selected>Last 30 days</option>
                    <option value="90">Last 90 days</option>
                </select>
            </div>
            <div class="control-group">
                <label for="symbol">Symbol:</label>
                <select id="symbol">
                    <option value="all">All Symbols</option>
                    <option value="XAUUSD">XAUUSD</option>
                    <option value="EURUSD">EURUSD</option>
                    <option value="BTCUSD">BTCUSD</option>
                </select>
            </div>
            <button class="btn" onclick="refreshDashboard()">Refresh Data</button>
            <button class="btn" onclick="exportData()">Export Data</button>
        </div>
        
        <div class="metrics-grid" id="metricsGrid">
            <!-- Metrics will be loaded here -->
        </div>
        
        <div class="charts-section">
            <div class="chart-container">
                <div class="chart-title">Performance Trends</div>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div class="charts-section">
            <div class="chart-container">
                <div class="chart-title">Strategy Performance</div>
                <div id="strategyPerformance" class="strategy-performance">
                    <!-- Strategy cards will be loaded here -->
                </div>
            </div>
        </div>
        
        <div class="charts-section">
            <div class="chart-container">
                <div class="chart-title">Recent Learning Insights</div>
                <div id="learningInsights" class="loading">
                    Loading learning insights...
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Dashboard JavaScript
        let performanceChart;
        
        async function loadDashboardData() {
            try {
                const timePeriod = document.getElementById('timePeriod').value;
                const symbol = document.getElementById('symbol').value;
                
                // Load performance summary
                const summaryResponse = await fetch(`/dashboard/api/performance/summary?days=${timePeriod}`);
                const summaryData = await summaryResponse.json();
                
                if (summaryData.success) {
                    updateMetricsGrid(summaryData.data);
                }
                
                // Load performance trends
                const trendsResponse = await fetch(`/dashboard/api/performance/trends?days=${timePeriod}`);
                const trendsData = await trendsResponse.json();
                
                if (trendsData.success) {
                    updatePerformanceChart(trendsData.data);
                }
                
                // Load strategy performance
                loadStrategyPerformance(timePeriod);
                
                // Load learning insights
                loadLearningInsights();
                
            } catch (error) {
                console.error('Failed to load dashboard data:', error);
            }
        }
        
        function updateMetricsGrid(data) {
            const metricsGrid = document.getElementById('metricsGrid');
            
            // Demo data (replace with actual API data)
            const metrics = [
                {
                    title: 'Total Predictions',
                    value: data.total_predictions || '1,247',
                    change: '+23 from last period',
                    changeType: 'positive'
                },
                {
                    title: 'Accuracy Rate',
                    value: (data.accuracy_rate || 0.673 * 100).toFixed(1) + '%',
                    change: '+2.3% from last period',
                    changeType: 'positive'
                },
                {
                    title: 'Average Confidence',
                    value: (data.avg_confidence || 0.756 * 100).toFixed(1) + '%',
                    change: '+1.2% from last period',
                    changeType: 'positive'
                },
                {
                    title: 'Learning Improvements',
                    value: data.learning_improvements || '17',
                    change: '+5 this week',
                    changeType: 'positive'
                }
            ];
            
            metricsGrid.innerHTML = metrics.map(metric => `
                <div class="metric-card">
                    <div class="metric-title">${metric.title}</div>
                    <div class="metric-value">${metric.value}</div>
                    <div class="metric-change ${metric.changeType}">${metric.change}</div>
                </div>
            `).join('');
        }
        
        function updatePerformanceChart(data) {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            // Demo chart data (replace with actual API data)
            const chartData = {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                datasets: [
                    {
                        label: 'Accuracy Rate',
                        data: [65.2, 68.7, 71.3, 73.5],
                        borderColor: '#ffd700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Confidence Score',
                        data: [72.1, 74.5, 75.8, 76.2],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4
                    }
                ]
            };
            
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#ffffff'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }
        
        async function loadStrategyPerformance(timePeriod) {
            const strategyContainer = document.getElementById('strategyPerformance');
            
            // Demo strategy data (replace with actual API calls)
            const strategies = [
                { name: 'Technical Analysis', accuracy: 74.2, trades: 342, winRate: 67.8 },
                { name: 'Sentiment Analysis', accuracy: 69.1, trades: 289, winRate: 63.2 },
                { name: 'Pattern Recognition', accuracy: 71.5, trades: 256, winRate: 65.4 },
                { name: 'Momentum Strategy', accuracy: 68.9, trades: 198, winRate: 61.7 }
            ];
            
            strategyContainer.innerHTML = strategies.map(strategy => `
                <div class="strategy-card">
                    <div class="strategy-name">${strategy.name}</div>
                    <div class="strategy-stats">
                        <div class="stat">
                            <div class="stat-label">Accuracy</div>
                            <div class="stat-value">${strategy.accuracy}%</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Win Rate</div>
                            <div class="stat-value">${strategy.winRate}%</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Total Trades</div>
                            <div class="stat-value">${strategy.trades}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Status</div>
                            <div class="stat-value positive">Active</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        async function loadLearningInsights() {
            const insightsContainer = document.getElementById('learningInsights');
            
            // Demo insights data (replace with actual API call)
            const insights = [
                "RSI indicator shows 15% improved accuracy after parameter optimization",
                "Market regime detection enhanced prediction accuracy by 8.2%",
                "Feature importance analysis revealed MACD as top performer for XAUUSD",
                "Model retraining completed: +3.5% accuracy improvement detected",
                "Volatility-based position sizing reduced maximum drawdown by 12%"
            ];
            
            insightsContainer.innerHTML = `
                <div style="space-y: 15px;">
                    ${insights.map(insight => `
                        <div style="padding: 12px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; margin-bottom: 10px;">
                            💡 ${insight}
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        function refreshDashboard() {
            loadDashboardData();
        }
        
        function exportData() {
            const timePeriod = document.getElementById('timePeriod').value;
            window.open(`/dashboard/api/export/predictions?days=${timePeriod}&format=csv`, '_blank');
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboardData();
            
            // Refresh data every 5 minutes
            setInterval(loadDashboardData, 5 * 60 * 1000);
        });
        
        // Event listeners
        document.getElementById('timePeriod').addEventListener('change', loadDashboardData);
        document.getElementById('symbol').addEventListener('change', loadDashboardData);
    </script>
</body>
</html>
    """
    return render_template_string(dashboard_html)

# Additional utility endpoints

@dashboard_bp.route('/api/learning/trigger-retraining', methods=['POST'])
def trigger_model_retraining():
    """Manually trigger model retraining"""
    try:
        strategy = request.json.get('strategy')
        force = request.json.get('force', False)
        
        result = asyncio.run(learning_engine.trigger_manual_retraining(
            strategy=strategy,
            force_retrain=force
        ))
        
        return jsonify({
            'success': True,
            'data': result,
            'message': f'Retraining triggered for {strategy or "all strategies"}'
        })
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/system/health-check', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        # Check all system components
        checks = {
            'prediction_tracker': prediction_tracker is not None,
            'learning_engine': learning_engine is not None,
            'backtest_engine': backtest_engine is not None,
            'database_connection': prediction_tracker.check_database_health() if prediction_tracker else False,
        }
        
        all_healthy = all(checks.values())
        
        return jsonify({
            'success': True,
            'healthy': all_healthy,
            'checks': checks,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200 if all_healthy else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    # Example of how to integrate with Flask app
    from flask import Flask
    from prediction_tracker import PredictionTracker
    from learning_engine import LearningEngine
    from backtesting_framework import BacktestEngine, HistoricalDataManager
    
    app = Flask(__name__)
    
    # Initialize services
    tracker = PredictionTracker()
    engine = LearningEngine(tracker)
    data_manager = HistoricalDataManager()
    bt_engine = BacktestEngine(tracker, data_manager)
    
    # Initialize dashboard services
    initialize_dashboard_services(tracker, engine, bt_engine)
    
    # Register blueprint
    app.register_blueprint(dashboard_bp)
    
    print("Dashboard API endpoints available:")
    print("- GET  /dashboard/                           - Main dashboard page")
    print("- GET  /dashboard/api/performance/summary    - Performance summary")
    print("- GET  /dashboard/api/performance/trends     - Performance trends")
    print("- GET  /dashboard/api/learning/insights      - Learning insights")
    print("- GET  /dashboard/api/system/status          - System status")
    print("- POST /dashboard/api/backtesting/run        - Run backtest")
    
    app.run(debug=True, port=5001)
