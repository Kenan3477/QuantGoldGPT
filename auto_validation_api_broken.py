#!/usr/bin/env python3
"""
🌐 GOLDGPT AUTO VALIDATION API INTEGRATION
Advanced Flask API for automated strategy validation system
"""

import logging
import threading
import asyncio
from datetime import datetime
from flask import Blueprint, jsonify, request, render_template

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Blueprint for auto validation API
auto_validation_bp = Blueprint('auto_validation', __name__, url_prefix='/api/auto-validation')

# Global validation system instance
validation_system = None

# Check if improved validation system is available
try:
    from improved_validation_system import ImprovedValidationSystem, run_improved_validation_batch, get_improved_validation_status
    IMPROVED_VALIDATION_AVAILABLE = True
    logger.info("✅ Improved validation system available")
except ImportError as e:
    logger.warning(f"⚠️ Improved validation system not available: {e}")
    IMPROVED_VALIDATION_AVAILABLE = False

def get_validation_system():
    """Get or create validation system instance"""
    global validation_system
    if validation_system is None:
        try:
            from auto_strategy_validation_system import AutoStrategyValidationSystem
            validation_system = AutoStrategyValidationSystem()
            logger.info("✅ Auto validation system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize validation system: {e}")
            validation_system = None
    return validation_system

@auto_validation_bp.route('/status', methods=['GET'])
def get_validation_status():
    """Get current validation system status"""
    try:
        if IMPROVED_VALIDATION_AVAILABLE:
            # Use improved validation system
            status = get_improved_validation_status()
        else:
            # Use original system
            system = get_validation_system()
            status = system.get_auto_validation_status()
        
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Status endpoint failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500==============================

Flask API endpoints to integrate auto strategy validation system 
with the existing GoldGPT web application.

Author: GoldGPT Development Team
Created: July 23, 2025
Status: PRODUCTION READY
"""

from flask import Blueprint, request, jsonify, render_template
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import threading
import json

# Import the auto validation system
try:
    from improved_validation_system import improved_validation_system, run_improved_validation, get_improved_validation_status
    IMPROVED_VALIDATION_AVAILABLE = True
    logger.info("✅ Using improved validation system")
except ImportError as e:
    logger.warning(f"⚠️ Improved validation system not available: {e}")
    IMPROVED_VALIDATION_AVAILABLE = False
    
    # Fallback to original system
    try:
        from auto_strategy_validation_system import AutoStrategyValidationSystem
        logger.info("📊 Using original validation system as fallback")
    except ImportError:
        logger.error("❌ No validation system available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('auto_validation_api')

# Create Flask blueprint
auto_validation_bp = Blueprint('auto_validation', __name__, url_prefix='/api/auto-validation')

# Global validation system instance
validation_system = None
system_lock = threading.Lock()

def get_validation_system():
    """Get or create validation system instance"""
    global validation_system
    
    with system_lock:
        if IMPROVED_VALIDATION_AVAILABLE:
            # Use improved validation system
            return improved_validation_system
        else:
            # Fallback to original system
            if validation_system is None:
                validation_system = AutoStrategyValidationSystem()
                # Start the system
                validation_system.start_auto_validation()
                logger.info("🚀 Auto validation system started via API")
            return validation_system

@auto_validation_bp.route('/status', methods=['GET'])
def get_validation_status():
    """Get current auto validation system status"""
    try:
        system = get_validation_system()
        status = system.get_auto_validation_status()
        
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Status endpoint failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/validate-all', methods=['POST'])
def trigger_validation():
    """Trigger validation of all strategies"""
    try:
        if IMPROVED_VALIDATION_AVAILABLE:
            # Use improved validation system
            def run_improved_validation():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(run_improved_validation_batch())
                    logger.info(f"✅ Improved validation completed: {len(results)} strategies validated")
                except Exception as e:
                    logger.error(f"❌ Improved validation failed: {e}")
                finally:
                    loop.close()
            
            # Start validation in background
            validation_thread = threading.Thread(target=run_improved_validation, daemon=True)
            validation_thread.start()
            
        else:
            # Use original system
            system = get_validation_system()
            
            # Run validation in background thread to avoid blocking
            def run_validation():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results = loop.run_until_complete(system.validate_all_strategies())
                    logger.info(f"✅ Manual validation completed: {len(results)} strategies validated")
                except Exception as e:
                    logger.error(f"❌ Manual validation failed: {e}")
                finally:
                    loop.close()
            
            validation_thread = threading.Thread(target=run_validation, daemon=True)
            validation_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Validation started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Validation trigger failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
            except Exception as e:
                logger.error(f"❌ Manual validation failed: {e}")
            finally:
                loop.close()
        
        validation_thread = threading.Thread(target=run_validation, daemon=True)
        validation_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Validation started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Validation trigger failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/rankings', methods=['GET'])
def get_strategy_rankings():
    """Get current strategy rankings"""
    try:
        system = get_validation_system()
        rankings = system.get_strategy_rankings()
        
        return jsonify({
            'success': True,
            'rankings': rankings,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Rankings endpoint failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/strategy/<strategy_name>', methods=['GET'])
def get_strategy_details(strategy_name):
    """Get detailed information about a specific strategy"""
    try:
        system = get_validation_system()
        
        # Get validation results for the strategy
        results = system.validation_results.get(strategy_name)
        
        if results:
            return jsonify({
                'success': True,
                'strategy': {
                    'name': results.strategy_name,
                    'type': results.strategy_type,
                    'last_validation': results.validation_timestamp.isoformat(),
                    'performance': results.backtest_performance,
                    'risk_metrics': results.risk_metrics,
                    'regime_performance': results.regime_performance,
                    'confidence': results.confidence_score,
                    'recommendation': results.recommendation,
                    'optimization_suggestions': results.optimization_suggestions,
                    'next_validation': results.next_validation_time.isoformat()
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Strategy {strategy_name} not found',
                'timestamp': datetime.now().isoformat()
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Strategy details failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/alerts', methods=['GET'])
def get_risk_alerts():
    """Get current risk alerts"""
    try:
        system = get_validation_system()
        status = system.get_auto_validation_status()
        alerts = status.get('risk_alerts', [])
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'alert_count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Alerts endpoint failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/performance-summary', methods=['GET'])
def get_performance_summary():
    """Get performance summary across all strategies"""
    try:
        system = get_validation_system()
        status = system.get_auto_validation_status()
        performance = status.get('performance_summary', {})
        
        return jsonify({
            'success': True,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Performance summary failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/dashboard')
def auto_validation_dashboard():
    """Render auto validation dashboard"""
    try:
        return render_template('auto_validation_dashboard.html')
    except Exception as e:
        logger.error(f"❌ Dashboard render failed: {e}")
        return f"Dashboard error: {e}", 500

# Integration function for main app
def register_auto_validation_api(app):
    """Register auto validation API with Flask app"""
    try:
        app.register_blueprint(auto_validation_bp)
        logger.info("🌐 Auto validation API registered with Flask app")
        
        # Add integration endpoint to main app routes
        @app.route('/auto-validation')
        def auto_validation_page():
            """Main auto validation page"""
            return render_template('auto_validation_dashboard.html')
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to register auto validation API: {e}")
        return False

def shutdown_validation_system():
    """Shutdown validation system (call on app shutdown)"""
    global validation_system
    
    with system_lock:
        if validation_system and validation_system.is_running:
            validation_system.stop_auto_validation()
            validation_system = None
            logger.info("🛑 Auto validation system shutdown")

# For testing the API directly
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    register_auto_validation_api(app)
    
    print("🌐 Testing Auto Validation API...")
    print("Available endpoints:")
    print("  GET  /api/auto-validation/status")
    print("  POST /api/auto-validation/validate-all") 
    print("  GET  /api/auto-validation/rankings")
    print("  GET  /api/auto-validation/alerts")
    print("  GET  /api/auto-validation/performance-summary")
    print("  GET  /auto-validation (dashboard)")
    
    try:
        app.run(debug=True, port=5001, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        shutdown_validation_system()
    except Exception as e:
        print(f"❌ Server error: {e}")
        shutdown_validation_system()
