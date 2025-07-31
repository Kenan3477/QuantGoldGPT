#!/usr/bin/env python3
"""
GoldGPT Flask API Integration for Data Pipeline
Connects the comprehensive data pipeline to the existing Flask application
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Blueprint, jsonify, request
from functools import wraps

# Import our data pipeline components
from data_integration_engine import DataIntegrationEngine, DataManager
from data_sources_config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (will be initialized when needed)
data_integration_engine: Optional[DataIntegrationEngine] = None
data_manager: Optional[DataManager] = None

def async_route(f):
    """Decorator to handle async routes in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(f(*args, **kwargs))
        except Exception as e:
            logger.error(f"Async route error: {e}")
            return jsonify({'error': str(e), 'status': 'error'}), 500
        finally:
            loop.close()
    return wrapper

def get_data_pipeline():
    """Get or initialize data pipeline components"""
    global data_integration_engine, data_manager
    
    if data_integration_engine is None:
        data_integration_engine = DataIntegrationEngine()
        data_manager = DataManager(data_integration_engine)
    
    return data_integration_engine, data_manager

# Create Blueprint for data pipeline APIs
data_pipeline_bp = Blueprint('data_pipeline', __name__, url_prefix='/api/data-pipeline')

@data_pipeline_bp.route('/health', methods=['GET'])
@async_route
async def health_check():
    """Health check endpoint for the data pipeline"""
    try:
        integration_engine, manager = get_data_pipeline()
        health_status = await manager.health_check()
        
        return jsonify({
            'status': 'healthy' if health_status.get('status') == 'healthy' else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'pipeline_health': health_status,
            'components': {
                'data_integration_engine': 'active',
                'data_manager': 'active',
                'cache_system': 'active'
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/unified-dataset', methods=['GET'])
@async_route
async def get_unified_dataset():
    """Get the unified dataset with all features"""
    try:
        integration_engine, manager = get_data_pipeline()
        
        # Check for force refresh parameter
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        dataset = await manager.get_ml_ready_dataset(force_refresh=force_refresh)
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset,
            'feature_count': len(dataset.get('features', {})),
            'data_quality': dataset.get('data_quality', {})
        })
    except Exception as e:
        logger.error(f"Dataset fetch failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/features', methods=['GET'])
@async_route
async def get_features():
    """Get feature vector and feature names for ML models"""
    try:
        integration_engine, manager = get_data_pipeline()
        
        dataset = await manager.get_ml_ready_dataset()
        feature_vector = manager.get_feature_vector(dataset)
        feature_names = manager.get_feature_names(dataset)
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'names': feature_names,
                'values': feature_vector.tolist(),
                'count': len(feature_names)
            },
            'dataset_timestamp': dataset.get('timestamp'),
            'data_quality_score': dataset.get('data_quality', {}).get('overall_score', 0)
        })
    except Exception as e:
        logger.error(f"Features fetch failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/price-data', methods=['GET'])
@async_route
async def get_price_data():
    """Get candlestick price data"""
    try:
        integration_engine, _ = get_data_pipeline()
        
        # Get timeframes from query parameters
        timeframes = request.args.getlist('timeframes')
        if not timeframes:
            timeframes = ['1h']
        
        candlestick_data = await integration_engine.candlestick_fetcher.fetch_candlestick_data(timeframes)
        
        # Convert to JSON-serializable format
        price_data = []
        for candle in candlestick_data:
            price_data.append({
                'timestamp': candle.timestamp.isoformat(),
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'timeframe': candle.timeframe
            })
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'price_data': price_data,
            'count': len(price_data),
            'timeframes': timeframes
        })
    except Exception as e:
        logger.error(f"Price data fetch failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/news-analysis', methods=['GET'])
@async_route
async def get_news_analysis():
    """Get news data with sentiment analysis"""
    try:
        integration_engine, _ = get_data_pipeline()
        
        hours_back = int(request.args.get('hours_back', 24))
        news_data = await integration_engine.news_fetcher.fetch_news_data(hours_back)
        
        # Convert to JSON-serializable format and calculate aggregates
        news_items = []
        sentiment_scores = []
        relevance_scores = []
        
        for news_item in news_data:
            news_items.append({
                'timestamp': news_item.timestamp.isoformat(),
                'title': news_item.title,
                'content': news_item.content[:200] + "..." if len(news_item.content) > 200 else news_item.content,
                'source': news_item.source,
                'sentiment_score': news_item.sentiment_score,
                'relevance_score': news_item.relevance_score,
                'url': news_item.url
            })
            sentiment_scores.append(news_item.sentiment_score)
            relevance_scores.append(news_item.relevance_score)
        
        # Calculate aggregate metrics
        aggregates = {
            'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'positive_news_count': len([s for s in sentiment_scores if s > 0.1]),
            'negative_news_count': len([s for s in sentiment_scores if s < -0.1]),
            'neutral_news_count': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
        }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'news_data': news_items,
            'count': len(news_items),
            'aggregates': aggregates,
            'hours_analyzed': hours_back
        })
    except Exception as e:
        logger.error(f"News analysis failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/economic-indicators', methods=['GET'])
@async_route
async def get_economic_indicators():
    """Get economic indicators data"""
    try:
        integration_engine, _ = get_data_pipeline()
        
        economic_data = await integration_engine.economic_fetcher.fetch_economic_indicators()
        
        # Convert to JSON-serializable format
        indicators = []
        for indicator in economic_data:
            indicators.append({
                'timestamp': indicator.timestamp.isoformat(),
                'indicator_name': indicator.indicator_name,
                'value': indicator.value,
                'country': indicator.country,
                'impact_level': indicator.impact_level,
                'source': indicator.source
            })
        
        # Group by impact level
        high_impact = [i for i in indicators if i['impact_level'] == 'high']
        medium_impact = [i for i in indicators if i['impact_level'] == 'medium']
        low_impact = [i for i in indicators if i['impact_level'] == 'low']
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'economic_indicators': indicators,
            'count': len(indicators),
            'by_impact': {
                'high': high_impact,
                'medium': medium_impact,
                'low': low_impact
            }
        })
    except Exception as e:
        logger.error(f"Economic indicators fetch failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/technical-analysis', methods=['GET'])
@async_route
async def get_technical_analysis():
    """Get technical analysis indicators"""
    try:
        integration_engine, _ = get_data_pipeline()
        
        # First get price data
        candlestick_data = await integration_engine.candlestick_fetcher.fetch_candlestick_data(['1d'])
        
        # Calculate technical indicators
        technical_indicators = integration_engine.technical_analyzer.calculate_technical_indicators(candlestick_data)
        
        # Convert to JSON-serializable format
        indicators = []
        bullish_signals = 0
        bearish_signals = 0
        
        for indicator in technical_indicators:
            indicator_data = {
                'timestamp': indicator.timestamp.isoformat(),
                'indicator_name': indicator.indicator_name,
                'value': indicator.value,
                'signal': indicator.signal,
                'timeframe': indicator.timeframe
            }
            indicators.append(indicator_data)
            
            # Count signals
            if indicator.signal == 'bullish':
                bullish_signals += 1
            elif indicator.signal == 'bearish':
                bearish_signals += 1
        
        # Calculate overall market sentiment
        total_signals = bullish_signals + bearish_signals
        market_sentiment = 'neutral'
        if total_signals > 0:
            if bullish_signals > bearish_signals * 1.5:
                market_sentiment = 'bullish'
            elif bearish_signals > bullish_signals * 1.5:
                market_sentiment = 'bearish'
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'technical_indicators': indicators,
            'count': len(indicators),
            'signal_summary': {
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'market_sentiment': market_sentiment,
                'consensus_strength': abs(bullish_signals - bearish_signals) / total_signals if total_signals > 0 else 0
            }
        })
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/ml-prediction-data', methods=['GET'])
@async_route
async def get_ml_prediction_data():
    """Get comprehensive data specifically formatted for ML predictions"""
    try:
        integration_engine, manager = get_data_pipeline()
        
        # Get the full unified dataset
        dataset = await manager.get_ml_ready_dataset()
        
        # Extract components for ML model
        features = dataset.get('features', {})
        feature_vector = manager.get_feature_vector(dataset)
        feature_names = manager.get_feature_names(dataset)
        
        # Organize features by category
        feature_categories = {
            'price_features': {k: v for k, v in features.items() if 'price' in k or 'volume' in k or 'volatility' in k},
            'technical_features': {k: v for k, v in features.items() if 'tech' in k or any(indicator in k for indicator in ['sma', 'ema', 'rsi', 'macd'])},
            'sentiment_features': {k: v for k, v in features.items() if 'news' in k or 'sentiment' in k},
            'economic_features': {k: v for k, v in features.items() if 'econ' in k or 'usd' in k},
            'time_features': {k: v for k, v in features.items() if any(time_word in k for time_word in ['hour', 'day', 'month', 'weekend', 'market'])}
        }
        
        # Calculate feature importance scores (simplified)
        feature_importance = {}
        for name in feature_names:
            if 'price' in name or 'sentiment' in name:
                feature_importance[name] = 0.9  # High importance
            elif 'technical' in name or 'econ' in name:
                feature_importance[name] = 0.7  # Medium importance
            else:
                feature_importance[name] = 0.5  # Low importance
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'ml_data': {
                'feature_vector': feature_vector.tolist(),
                'feature_names': feature_names,
                'feature_categories': feature_categories,
                'feature_importance': feature_importance,
                'data_quality_score': dataset.get('data_quality', {}).get('overall_score', 0),
                'total_features': len(feature_names),
                'dataset_timestamp': dataset.get('timestamp'),
                'validation_timestamp': dataset.get('validation_timestamp')
            },
            'metadata': {
                'raw_data_counts': dataset.get('raw_data', {}),
                'pipeline_performance': dataset.get('data_quality', {}),
                'cache_status': 'active'
            }
        })
    except Exception as e:
        logger.error(f"ML prediction data fetch failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/cache/status', methods=['GET'])
def get_cache_status():
    """Get cache system status"""
    try:
        integration_engine, _ = get_data_pipeline()
        
        # Simple cache status check
        cache_info = {
            'cache_active': True,
            'cache_type': 'SQLite',
            'cache_location': integration_engine.cache.db_path if integration_engine else 'unknown',
            'last_cleanup': 'unknown'  # Would track in production
        }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'cache_status': cache_info
        })
    except Exception as e:
        logger.error(f"Cache status check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/cache/clear', methods=['POST'])
@async_route
async def clear_cache():
    """Clear the data pipeline cache"""
    try:
        integration_engine, _ = get_data_pipeline()
        
        # Clear expired cache entries
        await integration_engine.cleanup_cache()
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@data_pipeline_bp.route('/config', methods=['GET'])
def get_pipeline_config():
    """Get data pipeline configuration"""
    try:
        # Get enabled data sources
        enabled_sources = config.get_enabled_sources()
        
        config_info = {
            'enabled_sources': len(enabled_sources),
            'price_sources': len(config.get_price_sources()),
            'news_sources': len(config.get_news_sources()),
            'economic_sources': len(config.get_economic_sources()),
            'cache_ttl_config': {
                'price_data_1h': 3600,
                'news_data': 3600,
                'economic_data': 14400,
                'technical_indicators': 300
            },
            'feature_categories': [
                'price_features', 'technical_features', 'sentiment_features',
                'economic_features', 'time_features'
            ]
        }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'configuration': config_info
        })
    except Exception as e:
        logger.error(f"Config fetch failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Integration helper functions for existing GoldGPT APIs

async def get_enhanced_ml_prediction_data():
    """Enhanced ML prediction data for existing ML prediction system"""
    try:
        integration_engine, manager = get_data_pipeline()
        dataset = await manager.get_ml_ready_dataset()
        
        # Format for existing ML system compatibility
        ml_data = {
            'timestamp': datetime.now().isoformat(),
            'price_data': {
                'current_price': dataset.get('features', {}).get('current_price', 2000.0),
                'price_change': dataset.get('features', {}).get('price_change', 0.0),
                'volatility': dataset.get('features', {}).get('volatility_10d', 0.0)
            },
            'sentiment_data': {
                'news_sentiment': dataset.get('features', {}).get('news_sentiment_avg', 0.0),
                'market_sentiment': 'neutral'  # Would calculate from technical indicators
            },
            'technical_data': {
                'rsi': dataset.get('features', {}).get('tech_rsi_14', 50.0),
                'macd': dataset.get('features', {}).get('tech_macd', 0.0)
            },
            'economic_data': {
                'usd_strength': dataset.get('features', {}).get('econ_usd_index', 100.0),
                'interest_rates': dataset.get('features', {}).get('econ_fed_funds_rate', 5.0)
            },
            'confidence_score': dataset.get('data_quality', {}).get('overall_score', 0.5)
        }
        
        return ml_data
    except Exception as e:
        logger.error(f"Enhanced ML data fetch failed: {e}")
        return None

def init_data_pipeline_for_app(app):
    """Initialize data pipeline for Flask app"""
    try:
        # Register the blueprint
        app.register_blueprint(data_pipeline_bp)
        
        # Initialize components
        global data_integration_engine, data_manager
        data_integration_engine = DataIntegrationEngine()
        data_manager = DataManager(data_integration_engine)
        
        logger.info("Data pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize data pipeline: {e}")
        return False

# Cleanup function
def cleanup_data_pipeline():
    """Cleanup data pipeline resources"""
    global data_integration_engine
    if data_integration_engine:
        data_integration_engine.close()
        logger.info("Data pipeline cleaned up")

# Export the blueprint and helper functions
__all__ = [
    'data_pipeline_bp', 
    'get_enhanced_ml_prediction_data', 
    'init_data_pipeline_for_app', 
    'cleanup_data_pipeline'
]
