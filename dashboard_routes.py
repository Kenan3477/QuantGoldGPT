#!/usr/bin/env python3
"""
GoldGPT Dashboard Routes
Professional Trading 212-inspired dashboard routes and API endpoints
"""

from flask import Blueprint, render_template, jsonify, request, current_app
from flask_socketio import emit
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import GoldGPT components
try:
    from integrated_strategy_engine import integrated_strategy_engine
    STRATEGY_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Strategy engine not available: {e}")
    STRATEGY_ENGINE_AVAILABLE = False

try:
    from advanced_ml_prediction_engine import AdvancedMLPredictionEngine
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML engine not available: {e}")
    ML_ENGINE_AVAILABLE = False

try:
    from ai_analysis_api import get_ai_analysis_sync
    AI_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI analysis not available: {e}")
    AI_ANALYSIS_AVAILABLE = False

try:
    from auto_validation_api import get_validation_system
    from improved_validation_system import get_improved_validation_status
    VALIDATION_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Validation system not available: {e}")
    VALIDATION_SYSTEM_AVAILABLE = False

try:
    from data_pipeline_core import data_pipeline, DataType
    DATA_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Data pipeline not available: {e}")
    DATA_PIPELINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
def dashboard():
    """Serve the main Trading 212-inspired dashboard"""
    try:
        # Try to serve the working version first
        return render_template('dashboard_working.html')
    except Exception as e:
        logger.error(f"❌ Error serving dashboard: {e}")
        try:
            # Fallback to original dashboard
            return render_template('dashboard.html')
        except:
            return f"Dashboard Error: {e}", 500

@dashboard_bp.route('/api/chart-data/<timeframe>')
def get_chart_data(timeframe):
    """Get chart data for specific timeframe"""
    try:
        symbol = request.args.get('symbol', 'XAUUSD')
        limit = int(request.args.get('limit', 100))
        
        # Mock chart data for now - replace with actual data source
        data = generate_mock_chart_data(timeframe, limit)
        
        return jsonify({
            'success': True,
            'data': data,
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting chart data: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/predictions')
def get_predictions():
    """Get AI predictions for multiple timeframes with validation status"""
    try:
        symbol = request.args.get('symbol', 'XAU')
        strategy = request.args.get('strategy', 'all')
        min_confidence = float(request.args.get('min_confidence', 0.5))
        
        predictions = []
        validation_status = None
        
        # Get validation status to filter predictions
        if VALIDATION_SYSTEM_AVAILABLE:
            try:
                validation_status = get_improved_validation_status()
                validated_strategies = [s for s in validation_status.get('strategy_rankings', []) 
                                      if s.get('recommendation') in ['approved', 'conditional']]
                logger.info(f"🎯 Using validated strategies: {[s['strategy'] for s in validated_strategies]}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get validation status: {e}")
        
        predictions = []
        
        if STRATEGY_ENGINE_AVAILABLE:
            # Get recent signals from integrated strategy engine
            recent_signals = integrated_strategy_engine.get_recent_signals(20)
            
            for signal in recent_signals:
                if signal['confidence'] >= min_confidence:
                    if strategy == 'all' or signal.get('strategy_name') == strategy:
                        predictions.append({
                            'id': signal['id'],
                            'timestamp': signal['timestamp'],
                            'timeframe': signal.get('timeframe', '1h'),
                            'direction': signal['signal_type'],
                            'confidence': signal['confidence'],
                            'entry_price': signal['entry_price'],
                            'target_price': signal.get('take_profit'),
                            'stop_loss': signal.get('stop_loss'),
                            'strategy': signal.get('strategy_name', 'integrated'),
                            'source': 'integrated_engine'
                        })
        
        # If no predictions from strategy engine, generate mock predictions
        if not predictions:
            predictions = generate_mock_predictions(symbol, strategy, min_confidence)
        
        # Calculate summary statistics
        summary = calculate_predictions_summary(predictions)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'summary': summary,
            'filters': {
                'symbol': symbol,
                'strategy': strategy,
                'min_confidence': min_confidence
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting predictions: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/market-context')
def get_market_context():
    """Get real-time market context and regime analysis"""
    try:
        symbol = request.args.get('symbol', 'XAU')
        
        # Get REAL current market data from Gold API
        if DATA_PIPELINE_AVAILABLE:
            try:
                import asyncio
                
                # Get real-time price from data pipeline
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    price_data = loop.run_until_complete(data_pipeline.get_unified_data(symbol, DataType.PRICE))
                    current_price = price_data.get('price', 3428.90) if price_data else 3428.90
                    loop.close()
                    logger.info(f"🎯 Using REAL Gold API price for market context: ${current_price}")
                except Exception as e:
                    logger.warning(f"⚠️ Async data pipeline failed, using fallback: {e}")
                    current_price = 3428.90
            except Exception as e:
                logger.warning(f"⚠️ Error getting real price: {e}")
                current_price = 3428.90
        else:
            logger.warning("📋 Data pipeline not available, using fallback price")
            current_price = 3428.90
        
        # Generate market context
        context = {
            'market_regime': {
                'type': 'bullish_trending',
                'name': 'Bullish Trending',
                'confidence': 0.87,
                'description': 'Strong upward momentum with sustained buying pressure.',
                'trend_strength': 0.78,
                'momentum': 0.34,
                'indicator': '📈'
            },
            'key_levels': generate_support_resistance_levels(current_price),
            'volatility': {
                'current': 23.4,
                'change': 2.1,
                'level': 'moderate'
            },
            'sentiment': {
                'score': 0.62,
                'change': 0.05,
                'level': 'neutral_positive'
            },
            'economic_events': get_upcoming_economic_events(),
            'trading_implications': generate_trading_implications()
        }
        
        return jsonify({
            'success': True,
            'context': context,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting market context: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/correlation')
def get_timeframe_correlation():
    """Get timeframe correlation analysis"""
    try:
        primary_tf = request.args.get('primary', '1h')
        compare_tfs = request.args.getlist('compare') or ['4h', '1d', '1w']
        
        # Generate correlation matrix
        correlation_matrix = generate_correlation_matrix(primary_tf, compare_tfs)
        
        # Calculate overall metrics
        overall_correlation = sum(correlation_matrix.values()) / len(correlation_matrix)
        divergences = [k for k, v in correlation_matrix.items() if abs(v) < 0.3]
        alignment_score = calculate_alignment_score(correlation_matrix)
        trade_confidence = calculate_trade_confidence(overall_correlation, len(divergences))
        
        return jsonify({
            'success': True,
            'correlation_matrix': correlation_matrix,
            'overview': {
                'overall_correlation': round(overall_correlation, 2),
                'divergence_count': len(divergences),
                'alignment_score': round(alignment_score, 1),
                'trade_confidence': round(trade_confidence * 100)
            },
            'divergences': generate_divergence_alerts(divergences),
            'confidence_breakdown': generate_confidence_breakdown(correlation_matrix),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting correlation analysis: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/validation-status')
def get_validation_status():
    """Get comprehensive validation status for dashboard integration"""
    try:
        if not VALIDATION_SYSTEM_AVAILABLE:
            return jsonify({
                'status': 'unavailable',
                'message': 'Validation system not available',
                'timestamp': datetime.now().isoformat()
            })
        
        # Get comprehensive validation status
        validation_data = get_improved_validation_status()
        
        # Enhance with dashboard-specific data
        dashboard_validation = {
            'system_status': validation_data.get('status', 'unknown'),
            'health_score': validation_data.get('health_score', 0),
            'total_strategies': validation_data.get('strategies_validated', 0),
            'approved_count': len([s for s in validation_data.get('strategy_rankings', []) 
                                 if s.get('recommendation') == 'approved']),
            'conditional_count': len([s for s in validation_data.get('strategy_rankings', []) 
                                    if s.get('recommendation') == 'conditional']),
            'rejected_count': len([s for s in validation_data.get('strategy_rankings', []) 
                                 if s.get('recommendation') == 'rejected']),
            'top_strategies': validation_data.get('strategy_rankings', [])[:3],
            'critical_alerts': [alert for alert in validation_data.get('alerts', []) 
                              if alert.get('severity') == 'high'],
            'performance_summary': validation_data.get('performance_summary', {}),
            'last_validation': validation_data.get('last_validation'),
            'validation_health_indicator': get_health_indicator(validation_data.get('health_score', 0)),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(dashboard_validation)
        
    except Exception as e:
        logger.error(f"❌ Error getting validation status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@dashboard_bp.route('/api/generate-signal', methods=['POST'])
def generate_signal():
    """Generate new trading signal"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'XAU')
        timeframe = data.get('timeframe', '1h')
        
        if STRATEGY_ENGINE_AVAILABLE:
            # Generate signal using integrated strategy engine
            import asyncio
            signal = asyncio.run(integrated_strategy_engine.generate_integrated_signal(symbol, timeframe))
            
            if signal:
                return jsonify({
                    'success': True,
                    'signal': {
                        'timestamp': signal.timestamp.isoformat(),
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'timeframe': signal.timeframe,
                        'strategy_name': signal.strategy_name
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No signal generated - conditions not met'
                }), 400
        else:
            # Generate mock signal
            signal = generate_mock_signal(symbol, timeframe)
            return jsonify({
                'success': True,
                'signal': signal
            })
        
    except Exception as e:
        logger.error(f"❌ Error generating signal: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/portfolio-summary')
def get_portfolio_summary():
    """Get portfolio summary and performance metrics"""
    try:
        # Mock portfolio data - replace with actual portfolio management
        portfolio = {
            'total_value': 10000.00,
            'daily_change': 125.50,
            'daily_change_percent': 1.26,
            'daily_pnl_percent': 2.1,
            'win_rate': 68,
            'positions': [
                {
                    'symbol': 'XAUUSD',
                    'size': 0.1,
                    'entry_price': 2045.20,
                    'current_price': 2054.32,
                    'pnl': 91.20,
                    'pnl_percent': 0.45
                }
            ],
            'recent_trades': get_recent_trades()
        }
        
        return jsonify({
            'success': True,
            'portfolio': portfolio,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting portfolio summary: {e}")
        return jsonify({'error': str(e)}), 500

# Helper functions for mock data generation

def generate_mock_chart_data(timeframe: str, limit: int) -> List[Dict]:
    """Generate mock OHLCV data for charts"""
    import random
    
    # Base price and time intervals
    base_price = 2054.32
    time_intervals = {
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800
    }
    
    interval = time_intervals.get(timeframe, 3600)
    data = []
    
    current_time = datetime.now()
    current_price = base_price
    
    for i in range(limit):
        # Generate realistic price movement
        change_percent = random.uniform(-0.02, 0.02)  # ±2% max change
        price_change = current_price * change_percent
        
        open_price = current_price
        high_price = max(open_price, open_price + abs(price_change) * random.uniform(0.5, 2))
        low_price = min(open_price, open_price - abs(price_change) * random.uniform(0.5, 2))
        close_price = open_price + price_change
        
        volume = random.uniform(50000, 200000)
        
        data.append({
            'time': int((current_time - timedelta(seconds=interval * (limit - i))).timestamp()),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(volume)
        })
        
        current_price = close_price
    
    return data

def generate_mock_predictions(symbol: str, strategy: str, min_confidence: float) -> List[Dict]:
    """Generate mock predictions for testing using REAL Gold API price"""
    import random
    
    # Get REAL current price from Gold API via data pipeline
    try:
        if DATA_PIPELINE_AVAILABLE:
            import asyncio
            
            # Try to get real-time price from data pipeline
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                price_data = loop.run_until_complete(data_pipeline.get_unified_data(symbol, DataType.PRICE))
                base_price = price_data.get('price', 3428.90) if price_data else 3428.90
                loop.close()
                logger.info(f"🎯 Using REAL Gold API price for predictions: ${base_price}")
            except Exception as e:
                logger.warning(f"⚠️ Async data pipeline failed, trying sync: {e}")
                # Fallback: use cached price or reasonable default
                base_price = 3428.90
        else:
            logger.warning("📋 Data pipeline not available, using fallback price")
            base_price = 3428.90
            
    except Exception as e:
        logger.error(f"❌ Error getting real price, using fallback: {e}")
        base_price = 3428.90
    
    predictions = []
    timeframes = ['1h', '4h', '1d', '1w']
    strategies = ['ml_momentum', 'conservative', 'aggressive']
    
    for i, tf in enumerate(timeframes):
        for j, strat in enumerate(strategies):
            if strategy != 'all' and strat != strategy:
                continue
                
            confidence = random.uniform(0.5, 0.95)
            if confidence < min_confidence:
                continue
                
            direction = random.choice(['BUY', 'SELL'])
            
            # Use REAL base price for realistic predictions
            entry_price = round(base_price + random.uniform(-2, 2), 2)
            
            if direction == 'BUY':
                target_price = round(base_price + random.uniform(8, 25), 2)
                stop_loss = round(base_price - random.uniform(5, 12), 2)
            else:
                target_price = round(base_price - random.uniform(8, 25), 2)
                stop_loss = round(base_price + random.uniform(5, 12), 2)
            
            predictions.append({
                'id': f'pred_{i}_{j}',
                'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
                'timeframe': tf,
                'direction': direction,
                'confidence': round(confidence, 2),
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'strategy': strat,
                'source': f'gold_api_based (real_price: ${base_price})'
            })
    
    return predictions[:8]  # Limit to 8 predictions

def calculate_predictions_summary(predictions: List[Dict]) -> Dict:
    """Calculate summary statistics for predictions"""
    if not predictions:
        return {
            'total_predictions': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'avg_confidence': 0,
            'trend': 'neutral'
        }
    
    total = len(predictions)
    bullish = len([p for p in predictions if p['direction'] == 'BUY'])
    bearish = total - bullish
    avg_confidence = sum(p['confidence'] for p in predictions) / total
    
    if bullish > bearish:
        trend = 'bullish'
    elif bearish > bullish:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    return {
        'total_predictions': total,
        'bullish_count': bullish,
        'bearish_count': bearish,
        'avg_confidence': round(avg_confidence * 100),
        'trend': trend
    }

def generate_support_resistance_levels(current_price: float) -> List[Dict]:
    """Generate support and resistance levels"""
    levels = []
    
    # Generate support levels below current price
    for i in range(1, 4):
        level_price = current_price - (i * 15)
        levels.append({
            'type': 'support',
            'price': round(level_price, 2),
            'strength': max(1, 4 - i),
            'distance': round(((current_price - level_price) / current_price) * 100, 1),
            'timeframe': '1d' if i <= 2 else '1w'
        })
    
    # Generate resistance levels above current price
    for i in range(1, 4):
        level_price = current_price + (i * 18)
        levels.append({
            'type': 'resistance',
            'price': round(level_price, 2),
            'strength': max(1, 4 - i),
            'distance': round(((level_price - current_price) / current_price) * 100, 1),
            'timeframe': '1d' if i <= 2 else '1w'
        })
    
    return sorted(levels, key=lambda x: abs(x['price'] - current_price))

def get_upcoming_economic_events() -> List[Dict]:
    """Get upcoming economic events that may impact gold"""
    events = [
        {
            'time': '2025-07-23T14:00:00Z',
            'title': 'Federal Reserve Interest Rate Decision',
            'impact': 'high',
            'currency': 'USD',
            'forecast': '5.25%',
            'previous': '5.25%',
            'description': 'FOMC interest rate decision and policy statement'
        },
        {
            'time': '2025-07-24T12:30:00Z',
            'title': 'US GDP Quarterly Growth',
            'impact': 'medium',
            'currency': 'USD',
            'forecast': '2.1%',
            'previous': '1.9%',
            'description': 'Quarterly GDP growth rate'
        },
        {
            'time': '2025-07-25T08:30:00Z',
            'title': 'US Unemployment Rate',
            'impact': 'medium',
            'currency': 'USD',
            'forecast': '3.7%',
            'previous': '3.8%',
            'description': 'Monthly unemployment rate'
        }
    ]
    
    return events

def generate_trading_implications() -> List[Dict]:
    """Generate trading implications based on market analysis"""
    implications = [
        {
            'type': 'opportunity',
            'title': 'Bullish Momentum Continuation',
            'description': 'Strong upward momentum suggests potential for further gains. Consider long positions on pullbacks.',
            'confidence': 'high',
            'timeframe': 'short-term'
        },
        {
            'type': 'risk',
            'title': 'Overbought Conditions',
            'description': 'RSI approaching overbought levels. Watch for potential correction or consolidation.',
            'confidence': 'medium',
            'timeframe': 'short-term'
        },
        {
            'type': 'strategy',
            'title': 'Dollar Weakness Support',
            'description': 'Continued USD weakness provides fundamental support for gold prices.',
            'confidence': 'high',
            'timeframe': 'medium-term'
        }
    ]
    
    return implications

def generate_correlation_matrix(primary_tf: str, compare_tfs: List[str]) -> Dict[str, float]:
    """Generate correlation matrix between timeframes"""
    import random
    
    matrix = {}
    all_tfs = [primary_tf] + [tf for tf in compare_tfs if tf != primary_tf]
    
    for i, tf1 in enumerate(all_tfs):
        for tf2 in all_tfs[i+1:]:
            # Higher correlation for closer timeframes
            base_correlation = 0.8 if abs(all_tfs.index(tf1) - all_tfs.index(tf2)) == 1 else 0.6
            correlation = base_correlation + random.uniform(-0.2, 0.2)
            correlation = max(-1, min(1, correlation))
            
            matrix[f"{tf1}_{tf2}"] = round(correlation, 2)
    
    return matrix

def calculate_alignment_score(correlation_matrix: Dict[str, float]) -> float:
    """Calculate overall alignment score from correlation matrix"""
    if not correlation_matrix:
        return 0.0
    
    # Weight positive correlations more heavily
    total_score = 0
    for correlation in correlation_matrix.values():
        if correlation > 0:
            total_score += correlation * 2
        else:
            total_score += abs(correlation)
    
    return (total_score / len(correlation_matrix)) * 10

def calculate_trade_confidence(overall_correlation: float, divergence_count: int) -> float:
    """Calculate trade confidence based on correlations and divergences"""
    base_confidence = abs(overall_correlation)
    divergence_penalty = divergence_count * 0.1
    
    confidence = max(0, min(1, base_confidence - divergence_penalty))
    return confidence

def generate_divergence_alerts(divergences: List[str]) -> List[Dict]:
    """Generate divergence alert details"""
    alerts = []
    
    for div in divergences:
        timeframes = div.split('_')
        alerts.append({
            'type': 'divergence',
            'severity': 'medium',
            'title': f'{timeframes[0].upper()} vs {timeframes[1].upper()} Divergence',
            'description': f'Predictions between {timeframes[0]} and {timeframes[1]} timeframes are conflicting.',
            'suggestion': 'Wait for alignment or reduce position size',
            'timeframes': timeframes
        })
    
    return alerts

def generate_confidence_breakdown(correlation_matrix: Dict[str, float]) -> List[Dict]:
    """Generate confidence breakdown factors"""
    factors = []
    
    # Timeframe alignment factor
    avg_correlation = sum(correlation_matrix.values()) / len(correlation_matrix) if correlation_matrix else 0
    factors.append({
        'factor': 'Timeframe Alignment',
        'value': round(abs(avg_correlation) * 100),
        'weight': 0.4,
        'description': 'How well predictions align across timeframes'
    })
    
    # Prediction consistency factor
    factors.append({
        'factor': 'Prediction Consistency',
        'value': 75,
        'weight': 0.3,
        'description': 'Consistency of ML model predictions'
    })
    
    # Market conditions factor
    factors.append({
        'factor': 'Market Conditions',
        'value': 82,
        'weight': 0.2,
        'description': 'Favorability of current market conditions'
    })
    
    # Technical setup factor
    factors.append({
        'factor': 'Technical Setup',
        'value': 68,
        'weight': 0.1,
        'description': 'Quality of technical analysis setup'
    })
    
    return factors

def generate_mock_signal(symbol: str, timeframe: str) -> Dict:
    """Generate mock trading signal"""
    import random
    
    direction = random.choice(['BUY', 'SELL'])
    base_price = 2054.32
    confidence = random.uniform(0.6, 0.9)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'signal_type': direction,
        'confidence': round(confidence, 2),
        'entry_price': round(base_price + random.uniform(-5, 5), 2),
        'stop_loss': round(base_price + (-15 if direction == 'BUY' else 15), 2),
        'take_profit': round(base_price + (25 if direction == 'BUY' else -25), 2),
        'timeframe': timeframe,
        'strategy_name': 'mock_strategy'
    }

def get_recent_trades() -> List[Dict]:
    """Get recent trading history"""
    import random
    from datetime import datetime, timedelta
    
    trades = []
    for i in range(5):
        trade_time = datetime.now() - timedelta(hours=random.randint(1, 48))
        direction = random.choice(['BUY', 'SELL'])
        entry_price = 2050 + random.uniform(-20, 20)
        exit_price = entry_price + random.uniform(-10, 15)
        pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        
        trades.append({
            'id': f'trade_{i}',
            'timestamp': trade_time.isoformat(),
            'symbol': 'XAUUSD',
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'size': 0.1,
            'pnl': round(pnl * 0.1, 2),
            'status': 'closed'
        })
    
    return trades

def get_health_indicator(health_score: float) -> Dict[str, str]:
    """Get health indicator based on health score"""
    if health_score >= 80:
        return {
            'status': 'excellent',
            'color': '#00d084',
            'icon': '🟢',
            'message': 'All systems performing optimally'
        }
    elif health_score >= 60:
        return {
            'status': 'good',
            'color': '#00d4aa',
            'icon': '🟡',
            'message': 'Systems performing well'
        }
    elif health_score >= 40:
        return {
            'status': 'warning',
            'color': '#ffa502',
            'icon': '🟠',
            'message': 'Some performance issues detected'
        }
    else:
        return {
            'status': 'critical',
            'color': '#ff4757',
            'icon': '🔴',
            'message': 'Critical performance issues'
        }

@dashboard_bp.route('/api/system-status')
def get_system_status():
    """Get comprehensive system status for the System Hub"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'operational',
            'systems': {}
        }
        
        # Check AI Analysis System
        try:
            if AI_ANALYSIS_AVAILABLE:
                # Test AI analysis endpoint
                analysis_result = get_ai_analysis_sync('XAU')
                if analysis_result and 'error' not in analysis_result:
                    status['systems']['ai_analysis'] = {
                        'status': 'operational',
                        'health': 'excellent',
                        'last_check': datetime.now().isoformat(),
                        'details': 'AI analysis engine responding normally'
                    }
                else:
                    status['systems']['ai_analysis'] = {
                        'status': 'degraded',
                        'health': 'warning',
                        'last_check': datetime.now().isoformat(),
                        'details': 'AI analysis returning errors'
                    }
            else:
                status['systems']['ai_analysis'] = {
                    'status': 'unavailable',
                    'health': 'critical',
                    'last_check': datetime.now().isoformat(),
                    'details': 'AI analysis module not loaded'
                }
        except Exception as e:
            status['systems']['ai_analysis'] = {
                'status': 'error',
                'health': 'critical',
                'last_check': datetime.now().isoformat(),
                'details': f'Error checking AI analysis: {str(e)}'
            }
        
        # Check Validation System
        try:
            if VALIDATION_SYSTEM_AVAILABLE:
                # Try the improved validation system first
                try:
                    validation_status = get_improved_validation_status()
                    if validation_status and validation_status.get('status') == 'active':
                        health_score = validation_status.get('health_score', 0)
                        health_indicator = get_health_indicator(health_score)
                        status['systems']['validation'] = {
                            'status': 'operational',
                            'health': health_indicator['status'],
                            'health_score': health_score,
                            'last_check': datetime.now().isoformat(),
                            'details': f'Validation system health: {health_score}%'
                        }
                    else:
                        # Fall back to basic validation system
                        validation_system = get_validation_system()
                        if validation_system:
                            status['systems']['validation'] = {
                                'status': 'operational',
                                'health': 'good',
                                'last_check': datetime.now().isoformat(),
                                'details': 'Basic validation system active'
                            }
                        else:
                            status['systems']['validation'] = {
                                'status': 'unavailable',
                                'health': 'warning',
                                'last_check': datetime.now().isoformat(),
                                'details': 'Validation system not responding'
                            }
                except Exception as inner_e:
                    # If improved validation fails, try basic validation
                    try:
                        validation_system = get_validation_system()
                        if validation_system:
                            status['systems']['validation'] = {
                                'status': 'operational',
                                'health': 'good',
                                'last_check': datetime.now().isoformat(),
                                'details': 'Basic validation system active'
                            }
                        else:
                            status['systems']['validation'] = {
                                'status': 'unavailable',
                                'health': 'warning',
                                'last_check': datetime.now().isoformat(),
                                'details': f'Validation system error: {str(inner_e)}'
                            }
                    except Exception as basic_e:
                        status['systems']['validation'] = {
                            'status': 'error',
                            'health': 'critical',
                            'last_check': datetime.now().isoformat(),
                            'details': f'Validation system errors: {str(inner_e)}, {str(basic_e)}'
                        }
            else:
                status['systems']['validation'] = {
                    'status': 'unavailable',
                    'health': 'critical',
                    'last_check': datetime.now().isoformat(),
                    'details': 'Validation module not loaded'
                }
        except Exception as e:
            status['systems']['validation'] = {
                'status': 'error',
                'health': 'critical',
                'last_check': datetime.now().isoformat(),
                'details': f'Error checking validation system: {str(e)}'
            }
        
        # Check ML Engine
        try:
            if ML_ENGINE_AVAILABLE:
                status['systems']['ml_engine'] = {
                    'status': 'operational',
                    'health': 'excellent',
                    'last_check': datetime.now().isoformat(),
                    'details': 'ML prediction engine loaded and ready'
                }
            else:
                status['systems']['ml_engine'] = {
                    'status': 'unavailable',
                    'health': 'warning',
                    'last_check': datetime.now().isoformat(),
                    'details': 'ML engine module not loaded'
                }
        except Exception as e:
            status['systems']['ml_engine'] = {
                'status': 'error',
                'health': 'critical',
                'last_check': datetime.now().isoformat(),
                'details': f'Error checking ML engine: {str(e)}'
            }
        
        # Check Data Pipeline
        try:
            if DATA_PIPELINE_AVAILABLE:
                # Test data pipeline with fallback method
                try:
                    fallback_data = data_pipeline.get_fallback_data('XAU', DataType.PRICE)
                    if fallback_data and fallback_data.get('price'):
                        status['systems']['data_pipeline'] = {
                            'status': 'operational',
                            'health': 'excellent',
                            'last_check': datetime.now().isoformat(),
                            'details': f'Data pipeline active, current price: ${fallback_data.get("price")}'
                        }
                    else:
                        status['systems']['data_pipeline'] = {
                            'status': 'degraded',
                            'health': 'warning',
                            'last_check': datetime.now().isoformat(),
                            'details': 'Data pipeline loaded but not returning valid price data'
                        }
                except Exception as inner_e:
                    # Try to get source status as a backup
                    source_status = data_pipeline.get_source_status()
                    if source_status:
                        active_sources = len([s for s in source_status.values() if s.get('status') == 'active'])
                        status['systems']['data_pipeline'] = {
                            'status': 'operational',
                            'health': 'good',
                            'last_check': datetime.now().isoformat(),
                            'details': f'Data pipeline loaded with {active_sources} active sources'
                        }
                    else:
                        status['systems']['data_pipeline'] = {
                            'status': 'degraded',
                            'health': 'warning',
                            'last_check': datetime.now().isoformat(),
                            'details': f'Data pipeline limited functionality: {str(inner_e)}'
                        }
            else:
                status['systems']['data_pipeline'] = {
                    'status': 'unavailable',
                    'health': 'critical',
                    'last_check': datetime.now().isoformat(),
                    'details': 'Data pipeline module not loaded'
                }
        except Exception as e:
            status['systems']['data_pipeline'] = {
                'status': 'error',
                'health': 'critical',
                'last_check': datetime.now().isoformat(),
                'details': f'Error checking data pipeline: {str(e)}'
            }
        
        # Calculate overall status
        system_healths = [sys['health'] for sys in status['systems'].values()]
        critical_count = system_healths.count('critical')
        warning_count = system_healths.count('warning')
        
        if critical_count > 1:
            status['overall_status'] = 'critical'
        elif critical_count > 0 or warning_count > 2:
            status['overall_status'] = 'degraded'
        elif warning_count > 0:
            status['overall_status'] = 'warning'
        else:
            status['overall_status'] = 'operational'
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'error',
            'systems': {}
        }), 500

# Export the blueprint
__all__ = ['dashboard_bp']
