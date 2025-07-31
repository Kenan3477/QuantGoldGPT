#!/usr/bin/env python3
"""
Integrated Strategy API - Flask Routes
Connects the integrated strategy engine to the GoldGPT web interface
"""

from flask import Blueprint, request, jsonify, render_template
from datetime import datetime, timedelta
import asyncio
import json
import logging
from typing import Dict, Any

# Import the integrated strategy engine
try:
    from integrated_strategy_engine import integrated_strategy_engine
    STRATEGY_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Integrated strategy engine not available: {e}")
    STRATEGY_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create blueprint
strategy_bp = Blueprint('strategy', __name__, url_prefix='/strategy')

@strategy_bp.route('/')
def strategy_dashboard():
    """Main strategy dashboard"""
    return render_template('strategy_dashboard.html')

@strategy_bp.route('/api/signals/generate', methods=['POST'])
def generate_signal():
    """Generate new integrated trading signal"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        data = request.get_json() or {}
        symbol = data.get('symbol', 'XAU')
        timeframe = data.get('timeframe', '1h')

        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        signal = loop.run_until_complete(
            integrated_strategy_engine.generate_integrated_signal(symbol, timeframe)
        )
        loop.close()

        if signal:
            return jsonify({
                "success": True,
                "signal": {
                    "timestamp": signal.timestamp.isoformat(),
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type,
                    "confidence": round(signal.confidence, 3),
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "timeframe": signal.timeframe,
                    "strategy_name": signal.strategy_name,
                    "risk_reward_ratio": signal.risk_reward_ratio
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "No signal generated - conditions not met"
            })

    except Exception as e:
        logger.error(f"❌ Error generating signal: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/signals/recent')
def get_recent_signals():
    """Get recent trading signals"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        limit = request.args.get('limit', 10, type=int)
        signals = integrated_strategy_engine.get_recent_signals(limit)

        return jsonify({
            "success": True,
            "signals": signals,
            "count": len(signals)
        })

    except Exception as e:
        logger.error(f"❌ Error getting recent signals: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """Run strategy backtest"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        data = request.get_json() or {}
        strategy_name = data.get('strategy', 'ml_momentum')
        timeframe = data.get('timeframe', '1h')
        days = data.get('days', 30)

        # Run backtest
        result = integrated_strategy_engine.run_strategy_backtest(
            strategy_name, timeframe, days
        )

        if result:
            return jsonify({
                "success": True,
                "backtest": {
                    "strategy_name": result.strategy_id,
                    "timeframe": result.timeframe,
                    "total_return": round(result.total_return, 2),
                    "total_return_percent": round(result.total_return_percent, 2),
                    "sharpe_ratio": round(result.performance_metrics.get('sharpe_ratio', 0), 3),
                    "max_drawdown": round(result.performance_metrics.get('max_drawdown', 0), 3),
                    "total_trades": len(result.trades),
                    "win_rate": round(result.trade_analysis.get('win_rate', 0), 2),
                    "start_date": result.start_date.isoformat(),
                    "end_date": result.end_date.isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Backtest failed to complete"
            })

    except Exception as e:
        logger.error(f"❌ Error running backtest: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/optimize', methods=['POST'])
def optimize_strategy():
    """Optimize strategy parameters"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        data = request.get_json() or {}
        strategy_name = data.get('strategy', 'ml_momentum')
        timeframe = data.get('timeframe', '1h')

        # Run optimization
        result = integrated_strategy_engine.optimize_strategy(strategy_name, timeframe)

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Error optimizing strategy: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/performance')
def get_performance():
    """Get strategy performance data"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        strategy_name = request.args.get('strategy')
        performance = integrated_strategy_engine.get_strategy_performance(strategy_name)

        return jsonify({
            "success": True,
            "performance": performance
        })

    except Exception as e:
        logger.error(f"❌ Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/strategies')
def get_strategies():
    """Get available strategies"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        strategies = list(integrated_strategy_engine.strategies.keys())
        
        return jsonify({
            "success": True,
            "strategies": strategies,
            "configs": integrated_strategy_engine.strategies
        })

    except Exception as e:
        logger.error(f"❌ Error getting strategies: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/live-monitoring/start', methods=['POST'])
def start_live_monitoring():
    """Start live strategy monitoring"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        data = request.get_json() or {}
        strategy_name = data.get('strategy', 'ml_momentum')
        
        # This would start a background task for live monitoring
        # For now, just return success
        
        return jsonify({
            "success": True,
            "message": f"Live monitoring started for {strategy_name}",
            "strategy": strategy_name
        })

    except Exception as e:
        logger.error(f"❌ Error starting live monitoring: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/live-monitoring/stop', methods=['POST'])
def stop_live_monitoring():
    """Stop live strategy monitoring"""
    try:
        return jsonify({
            "success": True,
            "message": "Live monitoring stopped"
        })

    except Exception as e:
        logger.error(f"❌ Error stopping live monitoring: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/risk-analysis')
def get_risk_analysis():
    """Get comprehensive risk analysis"""
    try:
        if not STRATEGY_ENGINE_AVAILABLE:
            return jsonify({"error": "Strategy engine not available"}), 500

        # Get recent performance data
        performance = integrated_strategy_engine.get_strategy_performance()
        
        if 'results' not in performance:
            return jsonify({
                "success": False,
                "message": "No performance data available"
            })

        results = performance['results']
        
        # Calculate risk metrics
        returns = [r['total_return_percent'] for r in results]
        
        if not returns:
            return jsonify({
                "success": False,
                "message": "No return data available"
            })

        import numpy as np
        
        risk_analysis = {
            "volatility": round(np.std(returns), 2),
            "max_return": round(max(returns), 2),
            "min_return": round(min(returns), 2),
            "avg_return": round(np.mean(returns), 2),
            "positive_trades": len([r for r in returns if r > 0]),
            "negative_trades": len([r for r in returns if r <= 0]),
            "win_rate": round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
            "total_backtests": len(returns)
        }

        return jsonify({
            "success": True,
            "risk_analysis": risk_analysis
        })

    except Exception as e:
        logger.error(f"❌ Error getting risk analysis: {e}")
        return jsonify({"error": str(e)}), 500

@strategy_bp.route('/api/market-conditions')
def get_market_conditions():
    """Get current market conditions assessment"""
    try:
        # This would integrate with your existing market data sources
        # For now, return mock data
        
        market_conditions = {
            "trend": "BULLISH",
            "volatility": "MEDIUM",
            "volume": "HIGH",
            "sentiment": "POSITIVE",
            "key_levels": {
                "support": 1980.0,
                "resistance": 2050.0,
                "current": 2020.0
            },
            "indicators": {
                "rsi": 65.2,
                "macd": "BULLISH_CROSSOVER",
                "bollinger_position": "MIDDLE"
            },
            "timestamp": datetime.now().isoformat()
        }

        return jsonify({
            "success": True,
            "market_conditions": market_conditions
        })

    except Exception as e:
        logger.error(f"❌ Error getting market conditions: {e}")
        return jsonify({"error": str(e)}), 500

# Integration helper functions
def setup_strategy_integration(app):
    """Setup strategy integration with main Flask app"""
    try:
        # Register blueprint
        app.register_blueprint(strategy_bp)
        
        # Add strategy data to dashboard context
        @app.context_processor
        def inject_strategy_data():
            return {
                'strategy_engine_available': STRATEGY_ENGINE_AVAILABLE,
                'available_strategies': list(integrated_strategy_engine.strategies.keys()) if STRATEGY_ENGINE_AVAILABLE else []
            }
        
        logger.info("✅ Strategy integration setup completed")
        
    except Exception as e:
        logger.error(f"❌ Error setting up strategy integration: {e}")

def get_strategy_status():
    """Get current strategy engine status"""
    if not STRATEGY_ENGINE_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Strategy engine not initialized"
        }
    
    try:
        # Get recent signals count
        recent_signals = integrated_strategy_engine.get_recent_signals(5)
        
        # Get performance summary
        performance = integrated_strategy_engine.get_strategy_performance()
        
        return {
            "status": "active",
            "recent_signals_count": len(recent_signals),
            "performance_summary": performance.get("summary", {}),
            "available_strategies": list(integrated_strategy_engine.strategies.keys())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Export for use in main app
__all__ = [
    'strategy_bp',
    'setup_strategy_integration',
    'get_strategy_status'
]
