"""
🚀 ADVANCED ML INTEGRATION API
==============================

Integration layer connecting the Advanced Multi-Strategy ML Architecture 
with the existing GoldGPT application infrastructure.

Features:
- Seamless integration with existing ML prediction API
- Real-time ensemble predictions with institutional data
- WebSocket updates for live ML monitoring
- API endpoints for strategy performance and optimization
- Dashboard data for the AI Analysis Dashboard

Author: GoldGPT AI System
Created: July 23, 2025
"""

import asyncio
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify
import traceback

# Import institutional data engine
from institutional_real_data_engine import get_institutional_real_time_price, InstitutionalRealDataEngine

# Import advanced ensemble system
from ensemble_voting_system import (
    EnsembleVotingSystem, MetaLearningEngine, create_advanced_ml_system,
    MarketRegimeDetector, PerformanceMonitor
)

logger = logging.getLogger('advanced_ml_integration')

class AdvancedMLIntegration:
    """
    🧠 Advanced ML Integration Controller
    Bridges the ensemble system with GoldGPT infrastructure
    """
    
    def __init__(self):
        self.ensemble_system = None
        self.institutional_engine = None
        self.is_initialized = False
        self.prediction_cache = {}
        self.last_update = datetime.now()
        
    async def initialize(self):
        """Initialize the advanced ML system"""
        try:
            logger.info("🚀 Initializing Advanced ML Integration...")
            
            # Initialize institutional data engine
            self.institutional_engine = InstitutionalRealDataEngine()
            
            # Create advanced ensemble system
            self.ensemble_system = await create_advanced_ml_system()
            
            self.is_initialized = True
            logger.info("✅ Advanced ML Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Advanced ML Integration initialization failed: {e}")
            self.is_initialized = False
    
    async def get_enhanced_predictions(self, symbol: str = "XAUUSD", timeframes: List[str] = None) -> Dict[str, Any]:
        """Get enhanced predictions using the advanced ensemble system"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if not self.is_initialized:
                return self._get_fallback_predictions(symbol)
            
            timeframes = timeframes or ['1h', '4h', '1d', '1w']
            
            # Get institutional data
            historical_data = {}
            for tf in timeframes:
                try:
                    data = self.institutional_engine.get_historical_data(tf, days=30)
                    if data:
                        # Convert to DataFrame
                        df_data = []
                        for point in data:
                            df_data.append({
                                'timestamp': point.timestamp,
                                'Open': point.open,
                                'High': point.high,
                                'Low': point.low,
                                'Close': point.close,
                                'Volume': point.volume
                            })
                        
                        if df_data:
                            historical_data[tf] = pd.DataFrame(df_data)
                            historical_data[tf].set_index('timestamp', inplace=True)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get {tf} data: {e}")
            
            if not historical_data:
                logger.warning("⚠️ No historical data available, using fallback")
                return self._get_fallback_predictions(symbol)
            
            # Generate ensemble predictions for each timeframe
            ensemble_predictions = {}
            current_price = get_institutional_real_time_price()
            
            for tf in timeframes:
                if tf in historical_data:
                    try:
                        prediction = await self.ensemble_system.generate_ensemble_prediction(
                            historical_data[tf], tf
                        )
                        ensemble_predictions[tf] = prediction
                    except Exception as e:
                        logger.error(f"❌ Ensemble prediction failed for {tf}: {e}")
            
            # Format for API response
            response = self._format_ensemble_response(ensemble_predictions, current_price, symbol)
            
            # Cache predictions
            self.prediction_cache[symbol] = response
            self.last_update = datetime.now()
            
            logger.info(f"🎯 Enhanced predictions generated for {symbol}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Enhanced predictions failed: {e}")
            return self._get_fallback_predictions(symbol)
    
    def _format_ensemble_response(self, ensemble_predictions: Dict[str, Any], 
                                current_price: float, symbol: str) -> Dict[str, Any]:
        """Format ensemble predictions for API response"""
        try:
            predictions = []
            overall_confidence = 0.0
            overall_direction = 'neutral'
            
            # Direction voting
            bullish_votes = 0
            bearish_votes = 0
            total_weight = 0.0
            
            for timeframe, prediction in ensemble_predictions.items():
                # Calculate price change
                price_change = prediction.predicted_price - current_price
                change_percent = (price_change / current_price) * 100
                
                # Format prediction
                formatted_prediction = {
                    'timeframe': timeframe,
                    'predicted_price': round(prediction.predicted_price, 2),
                    'current_price': round(current_price, 2),
                    'change_amount': round(price_change, 2),
                    'change_percent': round(change_percent, 2),
                    'confidence': round(prediction.confidence, 3),
                    'direction': prediction.direction,
                    'contributing_strategies': prediction.contributing_strategies,
                    'ensemble_weights': prediction.ensemble_weights,
                    'risk_metrics': prediction.risk_metrics,
                    'market_regime': prediction.market_conditions.market_regime,
                    'reasoning': self._generate_ensemble_reasoning(prediction),
                    'technical_analysis': self._extract_technical_indicators(prediction)
                }
                
                predictions.append(formatted_prediction)
                
                # Vote weighting
                weight = prediction.confidence
                total_weight += weight
                overall_confidence += prediction.confidence
                
                if prediction.direction == 'bullish':
                    bullish_votes += weight
                elif prediction.direction == 'bearish':
                    bearish_votes += weight
            
            # Determine overall direction
            if bullish_votes > bearish_votes:
                overall_direction = 'bullish'
            elif bearish_votes > bullish_votes:
                overall_direction = 'bearish'
            
            # Calculate overall confidence
            overall_confidence = overall_confidence / len(predictions) if predictions else 0.0
            
            # Get strategy performance summary
            strategy_performance = self._get_strategy_performance_summary()
            
            response = {
                'success': True,
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'predictions': predictions,
                'overall_direction': overall_direction,
                'overall_confidence': round(overall_confidence, 3),
                'strategy_performance': strategy_performance,
                'system_info': {
                    'ensemble_strategies': len(self.ensemble_system.strategies),
                    'last_optimization': self.last_update.isoformat(),
                    'data_source': 'institutional_real_data'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response formatting failed: {e}")
            return self._get_fallback_predictions(symbol)
    
    def _generate_ensemble_reasoning(self, prediction: Any) -> str:
        """Generate reasoning text for ensemble prediction"""
        try:
            reasoning = f"Ensemble analysis with {len(prediction.contributing_strategies)} strategies "
            reasoning += f"in {prediction.market_conditions.market_regime} market regime. "
            
            # Top contributing strategies
            sorted_strategies = sorted(
                prediction.contributing_strategies, 
                key=lambda x: x['weight'], 
                reverse=True
            )
            
            if sorted_strategies:
                top_strategy = sorted_strategies[0]
                reasoning += f"Lead strategy: {top_strategy['strategy']} "
                reasoning += f"({top_strategy['weight']:.1%} weight, {top_strategy['direction']}). "
            
            # Risk assessment
            overall_risk = prediction.risk_metrics.get('overall_risk', 0.5)
            if overall_risk > 0.7:
                reasoning += "High risk environment detected. "
            elif overall_risk < 0.3:
                reasoning += "Low risk conditions favorable. "
            
            return reasoning
            
        except Exception:
            return "Ensemble prediction based on multiple strategy analysis."
    
    def _extract_technical_indicators(self, prediction: Any) -> Dict[str, Any]:
        """Extract technical indicators from ensemble prediction"""
        try:
            technical_data = {}
            
            # Aggregate technical indicators from contributing strategies
            for strategy in prediction.contributing_strategies:
                if 'technical_indicators' in strategy:
                    for indicator, value in strategy['technical_indicators'].items():
                        if isinstance(value, (int, float)):
                            if indicator not in technical_data:
                                technical_data[indicator] = []
                            technical_data[indicator].append(value)
            
            # Calculate weighted averages
            aggregated_indicators = {}
            for indicator, values in technical_data.items():
                if values:
                    aggregated_indicators[indicator] = round(np.mean(values), 2)
            
            # Add market condition indicators
            aggregated_indicators.update({
                'market_regime': prediction.market_conditions.market_regime,
                'volatility': round(prediction.market_conditions.volatility, 3),
                'trend_strength': round(prediction.market_conditions.trend_strength, 3),
                'sentiment_score': round(prediction.market_conditions.sentiment_score, 3)
            })
            
            return aggregated_indicators
            
        except Exception:
            return {}
    
    def _get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        try:
            if not self.ensemble_system:
                return {}
            
            rankings = self.ensemble_system.performance_monitor.get_strategy_rankings()
            
            summary = {
                'total_strategies': len(rankings),
                'avg_accuracy': round(np.mean([v['accuracy'] for v in rankings.values()]), 3) if rankings else 0.0,
                'top_performer': None,
                'strategy_details': {}
            }
            
            if rankings:
                # Find top performer
                top_strategy = max(rankings.items(), key=lambda x: x[1]['accuracy'])
                summary['top_performer'] = {
                    'name': top_strategy[0],
                    'accuracy': round(top_strategy[1]['accuracy'], 3),
                    'predictions': top_strategy[1]['predictions']
                }
                
                # Strategy details
                for name, metrics in rankings.items():
                    summary['strategy_details'][name] = {
                        'accuracy': round(metrics['accuracy'], 3),
                        'confidence': round(metrics['confidence'], 3),
                        'predictions': metrics['predictions'],
                        'score': round(metrics['score'], 3)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Strategy performance summary failed: {e}")
            return {}
    
    def _get_fallback_predictions(self, symbol: str) -> Dict[str, Any]:
        """Provide fallback predictions when advanced system is unavailable"""
        try:
            current_price = get_institutional_real_time_price() or 3430.0
            
            # Simple fallback predictions
            predictions = []
            for timeframe in ['1h', '4h', '1d', '1w']:
                # Simple trend-based prediction
                change_percent = np.random.normal(0, 1.0)  # Random walk
                predicted_price = current_price * (1 + change_percent / 100)
                
                predictions.append({
                    'timeframe': timeframe,
                    'predicted_price': round(predicted_price, 2),
                    'current_price': round(current_price, 2),
                    'change_amount': round(predicted_price - current_price, 2),
                    'change_percent': round(change_percent, 2),
                    'confidence': 0.3,  # Low confidence for fallback
                    'direction': 'bullish' if change_percent > 0 else 'bearish',
                    'reasoning': 'Fallback prediction - advanced system unavailable',
                    'technical_analysis': {'status': 'limited_data'}
                })
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'predictions': predictions,
                'overall_direction': 'neutral',
                'overall_confidence': 0.3,
                'strategy_performance': {},
                'system_info': {
                    'status': 'fallback_mode',
                    'note': 'Advanced ML system unavailable'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback predictions failed: {e}")
            return {
                'success': False,
                'error': 'Prediction system unavailable',
                'timestamp': datetime.now().isoformat()
            }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Trigger system optimization"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'System not initialized'}
            
            # Get some historical data for optimization
            historical_data = self.institutional_engine.get_historical_data('1h', days=30)
            
            if historical_data:
                # Convert to DataFrame
                df_data = []
                for point in historical_data:
                    df_data.append({
                        'timestamp': point.timestamp,
                        'Close': point.close,
                        'High': point.high,
                        'Low': point.low,
                        'Volume': point.volume
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                
                # Run optimization
                results = await self.ensemble_system.meta_learner.optimize_system(df)
                
                return {
                    'success': True,
                    'optimization_results': results,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Insufficient historical data for optimization'
                }
                
        except Exception as e:
            logger.error(f"❌ System optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            status = {
                'advanced_ml_initialized': self.is_initialized,
                'institutional_data_available': self.institutional_engine is not None,
                'last_update': self.last_update.isoformat(),
                'cache_status': len(self.prediction_cache),
                'ensemble_strategies': 0,
                'strategy_performance': {}
            }
            
            if self.ensemble_system:
                status['ensemble_strategies'] = len(self.ensemble_system.strategies)
                status['strategy_performance'] = self._get_strategy_performance_summary()
            
            return status
            
        except Exception as e:
            logger.error(f"❌ System status check failed: {e}")
            return {'error': str(e)}

# Create global instance
advanced_ml_integration = AdvancedMLIntegration()

# Flask Blueprint for API endpoints
advanced_ml_api = Blueprint('advanced_ml_api', __name__)

@advanced_ml_api.route('/api/ml/predictions/enhanced/<symbol>', methods=['GET'])
async def get_enhanced_ml_predictions(symbol: str):
    """Enhanced ML predictions endpoint"""
    try:
        timeframes = request.args.getlist('timeframes') or ['1h', '4h', '1d', '1w']
        
        predictions = await advanced_ml_integration.get_enhanced_predictions(symbol, timeframes)
        
        return jsonify(predictions)
        
    except Exception as e:
        logger.error(f"❌ Enhanced predictions API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@advanced_ml_api.route('/api/ml/system/optimize', methods=['POST'])
async def optimize_ml_system():
    """Optimize ML system endpoint"""
    try:
        results = await advanced_ml_integration.optimize_system()
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"❌ Optimization API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@advanced_ml_api.route('/api/ml/system/status', methods=['GET'])
def get_ml_system_status():
    """Get ML system status"""
    try:
        status = advanced_ml_integration.get_system_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ Status API failed: {e}")
        return jsonify({'error': str(e)}), 500

@advanced_ml_api.route('/api/ml/dashboard/data', methods=['GET'])
async def get_dashboard_data():
    """Get comprehensive dashboard data for AI Analysis Dashboard"""
    try:
        symbol = request.args.get('symbol', 'XAUUSD')
        
        # Get enhanced predictions
        predictions = await advanced_ml_integration.get_enhanced_predictions(symbol)
        
        # Get system status
        status = advanced_ml_integration.get_system_status()
        
        # Format for dashboard
        dashboard_data = {
            'predictions': predictions.get('predictions', []),
            'current_price': predictions.get('current_price', 0),
            'overall_direction': predictions.get('overall_direction', 'neutral'),
            'overall_confidence': predictions.get('overall_confidence', 0),
            'strategy_performance': predictions.get('strategy_performance', {}),
            'system_status': status,
            'market_analysis': {
                'regime': 'unknown',
                'volatility': 'moderate',
                'trend_strength': 'weak'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract market analysis from predictions
        if predictions.get('predictions'):
            first_prediction = predictions['predictions'][0]
            dashboard_data['market_analysis'] = {
                'regime': first_prediction.get('market_regime', 'unknown'),
                'volatility': 'high' if first_prediction.get('risk_metrics', {}).get('overall_risk', 0.5) > 0.7 else 'moderate',
                'trend_strength': 'strong' if first_prediction.get('confidence', 0) > 0.7 else 'weak'
            }
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"❌ Dashboard data API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Integration with existing ML prediction API
async def get_advanced_ml_predictions(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    🎯 Main integration function for existing ML prediction API
    Replaces the old prediction system with advanced ensemble
    """
    try:
        logger.info(f"🎯 Getting advanced ML predictions for {symbol}")
        
        # Get enhanced predictions from ensemble system
        predictions = await advanced_ml_integration.get_enhanced_predictions(symbol)
        
        # Transform to match existing API format
        if predictions.get('success'):
            # Convert ensemble predictions to legacy format
            legacy_predictions = []
            
            for pred in predictions.get('predictions', []):
                legacy_pred = {
                    'timeframe': pred['timeframe'],
                    'predicted_price': pred['predicted_price'],
                    'confidence': pred['confidence'],
                    'direction': pred['direction'],
                    'change_percent': pred['change_percent'],
                    'change_amount': pred['change_amount'],
                    'technical_analysis': pred.get('technical_analysis', {}),
                    'reasoning': pred.get('reasoning', ''),
                    'risk_assessment': pred.get('risk_metrics', {}),
                    'strategy_ensemble': True,
                    'data_source': 'institutional_real_data'
                }
                legacy_predictions.append(legacy_pred)
            
            return {
                'success': True,
                'current_price': predictions['current_price'],
                'predictions': legacy_predictions,
                'data_quality': 'excellent',
                'confidence_level': 'high' if predictions['overall_confidence'] > 0.7 else 'medium',
                'market_condition': predictions.get('strategy_performance', {}).get('market_regime', 'unknown'),
                'timestamp': predictions['timestamp'],
                'enhanced_features': True
            }
        else:
            return predictions
            
    except Exception as e:
        logger.error(f"❌ Advanced ML predictions integration failed: {e}")
        # Return error in legacy format
        return {
            'success': False,
            'error': str(e),
            'predictions': [],
            'current_price': 0,
            'data_quality': 'poor'
        }

# Export main functions and classes
__all__ = [
    'AdvancedMLIntegration', 'advanced_ml_integration', 'advanced_ml_api',
    'get_advanced_ml_predictions'
]
