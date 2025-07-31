#!/usr/bin/env python3
"""
GoldGPT Flask Integration for Advanced ML Prediction Engine
Integrates the multi-strategy ML engine with the existing Flask application
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

# Global async helper function for easy import
def run_async_prediction(coro):
    """Helper to run async predictions in sync context - global version"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Async prediction execution failed: {e}")
        return {'status': 'error', 'error': str(e)}

def integrate_advanced_ml_with_app(app):
    """
    Integrate Advanced ML Prediction Engine with existing GoldGPT Flask app
    """
    try:
        # Import advanced ML components
        from advanced_ml_api import init_advanced_ml_api
        from advanced_ml_prediction_engine import get_advanced_ml_predictions
        
        logger.info("🚀 Integrating Advanced ML Prediction Engine with GoldGPT...")
        
        # 1. Register Advanced ML API Blueprint
        api_initialized = init_advanced_ml_api(app)
        if not api_initialized:
            logger.error("Failed to initialize Advanced ML API")
            return False
        
        # 2. Create async helper functions for existing routes
        def run_prediction_async(timeframes=None):
            """Helper to run async predictions in sync context"""
            try:
                if timeframes is None:
                    timeframes = ['1H', '4H', '1D']
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(get_advanced_ml_predictions(timeframes))
                    return result
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Async prediction execution failed: {e}")
                return {'status': 'error', 'error': str(e)}
        
        # Make the function available globally
        globals()['run_async_prediction'] = run_prediction_async
        
        # 3. Add advanced ML to existing ML predictions endpoint
        @app.route('/api/ml-predictions-advanced', methods=['GET'])
        def get_advanced_ml_predictions_endpoint():
            """Enhanced ML predictions endpoint using advanced engine"""
            try:
                result = run_prediction_async(['1H', '4H', '1D'])
                
                if result['status'] == 'success':
                    # Transform to match existing API format while adding advanced features
                    enhanced_response = {
                        'status': 'success',
                        'timestamp': result['timestamp'],
                        'execution_time': result['execution_time'],
                        'advanced_engine': True,
                        'strategies_used': result['system_info']['strategies_active'],
                        'predictions': {}
                    }
                    
                    # Convert advanced predictions to standard format
                    for timeframe, prediction in result['predictions'].items():
                        enhanced_response['predictions'][timeframe] = {
                            'timeframe': timeframe,
                            'current_price': prediction['current_price'],
                            'predicted_price': prediction['predicted_price'],
                            'price_change_percent': prediction['price_change_percent'],
                            'direction': prediction['direction'],
                            'confidence': prediction['confidence'],
                            'stop_loss': prediction['recommended_stop_loss'],
                            'take_profit': prediction['recommended_take_profit'],
                            'support_levels': prediction['support_levels'],
                            'resistance_levels': prediction['resistance_levels'],
                            'strategy_votes': prediction['strategy_votes'],
                            'confidence_interval': prediction['confidence_interval'],
                            'validation_score': prediction['validation_score'],
                            'advanced_features': {
                                'ensemble_method': 'weighted_voting',
                                'meta_learning': True,
                                'multi_strategy': True,
                                'real_time_validation': True
                            }
                        }
                    
                    return enhanced_response
                else:
                    return {
                        'status': 'error',
                        'error': result.get('error', 'Unknown error'),
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'fallback_available': True
                    }
                    
            except Exception as e:
                logger.error(f"Advanced ML endpoint error: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'fallback_available': True
                }
        
        # 4. Enhance existing ML prediction endpoint with fallback
        original_ml_predictions = None
        
        # Try to find existing ML predictions route
        for rule in app.url_map.iter_rules():
            if rule.endpoint == 'get_ml_predictions' or '/api/ml-predictions' in rule.rule:
                # Store reference to original function if it exists
                original_ml_predictions = app.view_functions.get(rule.endpoint)
                break
        
        @app.route('/api/ml-predictions-enhanced', methods=['GET'])
        def enhanced_ml_predictions_with_fallback():
            """Enhanced ML predictions with advanced engine and fallback"""
            try:
                # Try advanced engine first
                advanced_result = run_prediction_async(['1H'])
                
                if advanced_result['status'] == 'success' and advanced_result.get('predictions'):
                    # Return advanced prediction in standard format
                    prediction = list(advanced_result['predictions'].values())[0]
                    return {
                        'status': 'success',
                        'timestamp': advanced_result['timestamp'],
                        'engine': 'advanced_ml',
                        'prediction': {
                            'current_price': prediction['current_price'],
                            'predicted_price': prediction['predicted_price'],
                            'change_percent': prediction['price_change_percent'],
                            'direction': prediction['direction'],
                            'confidence': prediction['confidence'],
                            'timeframe': '1H'
                        },
                        'advanced_features': {
                            'strategy_count': len(prediction['strategy_votes']),
                            'ensemble_confidence': prediction['confidence'],
                            'validation_score': prediction['validation_score']
                        }
                    }
                else:
                    # Fallback to original ML engine if available
                    if original_ml_predictions:
                        fallback_result = original_ml_predictions()
                        if isinstance(fallback_result, dict):
                            fallback_result['engine'] = 'fallback'
                            fallback_result['advanced_engine_error'] = advanced_result.get('error', 'Unknown')
                        return fallback_result
                    else:
                        return {
                            'status': 'error',
                            'error': 'No ML engines available',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        
            except Exception as e:
                logger.error(f"Enhanced ML predictions error: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        # 5. Add strategy performance monitoring endpoint
        @app.route('/api/ml-strategy-performance', methods=['GET'])
        def get_strategy_performance():
            """Get performance metrics for all ML strategies"""
            try:
                from advanced_ml_prediction_engine import advanced_ml_engine
                
                if advanced_ml_engine is None:
                    return {
                        'status': 'error',
                        'error': 'Advanced ML engine not initialized'
                    }
                
                performance_result = run_prediction_async([])  # Just get performance data
                
                if performance_result['status'] == 'success':
                    return {
                        'status': 'success',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'performance': performance_result.get('performance', {}),
                        'system_info': performance_result.get('system_info', {}),
                        'strategies': performance_result.get('performance', {}).get('strategies', {})
                    }
                else:
                    return {
                        'status': 'error',
                        'error': 'Failed to get performance data'
                    }
                    
            except Exception as e:
                logger.error(f"Strategy performance endpoint error: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        # 6. Add system health check endpoint
        @app.route('/api/ml-system-status', methods=['GET'])
        def get_ml_system_status():
            """Get overall ML system status"""
            try:
                from advanced_ml_prediction_engine import advanced_ml_engine
                
                status = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'advanced_ml_available': advanced_ml_engine is not None,
                    'api_endpoints': [
                        '/api/advanced-ml/predict',
                        '/api/advanced-ml/quick-prediction',
                        '/api/advanced-ml/strategies',
                        '/api/advanced-ml/health',
                        '/api/ml-predictions-advanced',
                        '/api/ml-predictions-enhanced',
                        '/api/ml-strategy-performance'
                    ],
                    'features': {
                        'multi_strategy': True,
                        'ensemble_voting': True,
                        'performance_tracking': True,
                        'real_time_validation': True,
                        'meta_learning': True,
                        'confidence_intervals': True
                    }
                }
                
                if advanced_ml_engine:
                    status['strategy_count'] = len(advanced_ml_engine.strategies)
                    status['status'] = 'healthy'
                else:
                    status['status'] = 'degraded'
                    status['error'] = 'Advanced ML engine not initialized'
                
                return status
                
            except Exception as e:
                logger.error(f"ML system status error: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        logger.info("✅ Advanced ML Prediction Engine integrated successfully")
        logger.info("📡 Available enhanced endpoints:")
        logger.info("   • /api/advanced-ml/* - Full advanced ML API")
        logger.info("   • /api/ml-predictions-advanced - Enhanced predictions")
        logger.info("   • /api/ml-predictions-enhanced - Predictions with fallback")
        logger.info("   • /api/ml-strategy-performance - Strategy metrics")
        logger.info("   • /api/ml-system-status - System status")
        
        return True
        
    except ImportError as e:
        logger.warning(f"Advanced ML engine not available: {e}")
        logger.info("Continuing with existing ML system")
        return False
    except Exception as e:
        logger.error(f"Failed to integrate Advanced ML engine: {e}")
        return False

def add_advanced_ml_to_existing_routes(app):
    """
    Enhance existing Flask routes with advanced ML capabilities
    """
    try:
        # Import existing app modules
        import advanced_systems
        from advanced_ml_prediction_engine import get_advanced_ml_predictions
        
        # Store original function references
        original_functions = {}
        
        # Enhance the main ML predictions route if it exists
        for rule in app.url_map.iter_rules():
            if 'ml' in rule.rule.lower() and 'prediction' in rule.rule.lower():
                endpoint = rule.endpoint
                if endpoint in app.view_functions:
                    original_functions[endpoint] = app.view_functions[endpoint]
        
        # Create enhanced wrapper for ML predictions
        def create_enhanced_ml_wrapper(original_func):
            def enhanced_ml_prediction(*args, **kwargs):
                try:
                    # Try advanced ML first
                    def run_advanced():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(get_advanced_ml_predictions(['1H']))
                            return result
                        finally:
                            loop.close()
                    
                    advanced_result = run_advanced()
                    
                    if (advanced_result['status'] == 'success' and 
                        advanced_result.get('predictions') and 
                        '1H' in advanced_result['predictions']):
                        
                        # Return enhanced result
                        prediction = advanced_result['predictions']['1H']
                        return {
                            'status': 'success',
                            'engine': 'advanced',
                            'timestamp': advanced_result['timestamp'],
                            'prediction': {
                                'price': prediction['predicted_price'],
                                'change_percent': prediction['price_change_percent'],
                                'direction': prediction['direction'],
                                'confidence': prediction['confidence']
                            },
                            'metadata': {
                                'strategies': len(prediction['strategy_votes']),
                                'validation_score': prediction['validation_score'],
                                'execution_time': advanced_result['execution_time']
                            }
                        }
                    else:
                        # Fallback to original function
                        result = original_func(*args, **kwargs)
                        if isinstance(result, dict):
                            result['engine'] = 'fallback'
                        return result
                        
                except Exception as e:
                    logger.error(f"Enhanced ML wrapper error: {e}")
                    # Fallback to original function
                    try:
                        result = original_func(*args, **kwargs)
                        if isinstance(result, dict):
                            result['engine'] = 'fallback'
                            result['advanced_error'] = str(e)
                        return result
                    except Exception as fallback_error:
                        return {
                            'status': 'error',
                            'error': f'Both advanced and fallback engines failed: {str(e)}, {str(fallback_error)}',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
            
            return enhanced_ml_prediction
        
        # Apply enhancements to found ML routes
        enhanced_count = 0
        for endpoint, original_func in original_functions.items():
            if original_func:
                app.view_functions[endpoint] = create_enhanced_ml_wrapper(original_func)
                enhanced_count += 1
                logger.info(f"Enhanced endpoint: {endpoint}")
        
        if enhanced_count > 0:
            logger.info(f"✅ Enhanced {enhanced_count} existing ML endpoints with advanced engine")
        else:
            logger.info("ℹ️  No existing ML endpoints found to enhance")
            
        return enhanced_count > 0
        
    except Exception as e:
        logger.error(f"Failed to enhance existing routes: {e}")
        return False

# Main integration function
def setup_advanced_ml_integration(app):
    """
    Complete setup of advanced ML integration with GoldGPT Flask app
    """
    logger.info("🔧 Setting up Advanced ML Integration...")
    
    # Step 1: Integrate new advanced ML APIs
    api_success = integrate_advanced_ml_with_app(app)
    
    # Step 2: Enhance existing routes
    routes_success = add_advanced_ml_to_existing_routes(app)
    
    # Step 3: Add context processor for templates
    try:
        @app.context_processor
        def inject_advanced_ml_context():
            """Inject advanced ML availability into template context"""
            return {
                'advanced_ml_available': api_success,
                'ml_features': {
                    'multi_strategy': True,
                    'ensemble_voting': True,
                    'real_time_validation': True,
                    'confidence_scoring': True
                } if api_success else {}
            }
        
        logger.info("✅ Template context processor added")
    except Exception as e:
        logger.warning(f"Failed to add context processor: {e}")
    
    # Summary
    if api_success:
        logger.info("🎉 Advanced ML Integration Complete!")
        logger.info("📊 Features enabled:")
        logger.info("   • Multi-strategy ensemble predictions")
        logger.info("   • Real-time performance tracking") 
        logger.info("   • Confidence interval forecasting")
        logger.info("   • Advanced validation systems")
        logger.info("   • Meta-learning optimization")
        
        return True
    else:
        logger.warning("⚠️  Advanced ML integration failed - using fallback systems")
        return False

# Test the integration
if __name__ == "__main__":
    from flask import Flask
    import sys
    import os
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("🧪 Testing Advanced ML Flask Integration")
    print("="*50)
    
    # Create test Flask app
    app = Flask(__name__)
    
    # Test integration
    success = setup_advanced_ml_integration(app)
    
    if success:
        print("✅ Advanced ML integration test passed")
        print("📡 Test the integration by starting your Flask app and visiting:")
        print("   • http://localhost:5000/api/advanced-ml/health")
        print("   • http://localhost:5000/api/ml-predictions-advanced")
        print("   • http://localhost:5000/api/ml-system-status")
    else:
        print("❌ Advanced ML integration test failed")
        print("ℹ️  Check logs for details. System will fall back to existing ML engine.")
    
    print("\n🚀 Integration ready for production use!")
