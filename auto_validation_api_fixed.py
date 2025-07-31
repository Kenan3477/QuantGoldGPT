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
system_lock = threading.Lock()

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
            with system_lock:
                system = get_validation_system()
                if system is None:
                    return jsonify({
                        'error': 'Validation system not available',
                        'status': 'offline',
                        'timestamp': datetime.now().isoformat()
                    }), 503
                
                status = {
                    'status': 'running' if system.is_running else 'stopped',
                    'strategies_validated': len(system.strategy_results),
                    'last_validation': system.last_validation_time.isoformat() if system.last_validation_time else None,
                    'auto_validation_enabled': system.auto_validation_enabled,
                    'health_score': system.system_health_score,
                    'timestamp': datetime.now().isoformat()
                }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
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

@auto_validation_bp.route('/rankings', methods=['GET'])
def get_strategy_rankings():
    """Get strategy performance rankings"""
    try:
        if IMPROVED_VALIDATION_AVAILABLE:
            # Use improved validation system
            rankings = get_improved_validation_status().get('strategy_rankings', [])
        else:
            # Use original system
            system = get_validation_system()
            if system is None:
                return jsonify({'error': 'Validation system not available'}), 503
            
            rankings = []
            for strategy_name, result in system.strategy_results.items():
                rankings.append({
                    'strategy': strategy_name,
                    'confidence': result.get('confidence_score', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'win_rate': result.get('win_rate', 0),
                    'recommendation': result.get('recommendation', 'unknown'),
                    'last_updated': result.get('timestamp', '')
                })
            
            # Sort by confidence score
            rankings.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'rankings': rankings,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Rankings fetch failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/alerts', methods=['GET'])
def get_validation_alerts():
    """Get current validation alerts"""
    try:
        if IMPROVED_VALIDATION_AVAILABLE:
            # Use improved validation system
            alerts = get_improved_validation_status().get('alerts', [])
        else:
            # Use original system
            system = get_validation_system()
            if system is None:
                return jsonify({'error': 'Validation system not available'}), 503
            
            alerts = []
            for strategy_name, result in system.strategy_results.items():
                if result.get('recommendation') == 'rejected':
                    alerts.append({
                        'type': 'performance_warning',
                        'strategy': strategy_name,
                        'message': f"Strategy {strategy_name} performance below threshold",
                        'severity': 'high' if result.get('confidence_score', 0) < 0.3 else 'medium',
                        'timestamp': result.get('timestamp', '')
                    })
        
        return jsonify({
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Alerts fetch failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@auto_validation_bp.route('/performance-summary', methods=['GET'])
def get_performance_summary():
    """Get overall performance summary"""
    try:
        if IMPROVED_VALIDATION_AVAILABLE:
            # Use improved validation system
            summary = get_improved_validation_status().get('performance_summary', {})
        else:
            # Use original system
            system = get_validation_system()
            if system is None:
                return jsonify({'error': 'Validation system not available'}), 503
            
            total_strategies = len(system.strategy_results)
            approved_strategies = sum(1 for r in system.strategy_results.values() 
                                    if r.get('recommendation') == 'approved')
            
            avg_confidence = sum(r.get('confidence_score', 0) for r in system.strategy_results.values()) / max(total_strategies, 1)
            avg_sharpe = sum(r.get('sharpe_ratio', 0) for r in system.strategy_results.values()) / max(total_strategies, 1)
            
            summary = {
                'total_strategies': total_strategies,
                'approved_strategies': approved_strategies,
                'approval_rate': approved_strategies / max(total_strategies, 1),
                'average_confidence': avg_confidence,
                'average_sharpe_ratio': avg_sharpe,
                'system_health': system.system_health_score
            }
        
        return jsonify({
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Performance summary failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

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
        if validation_system and hasattr(validation_system, 'is_running') and validation_system.is_running:
            if hasattr(validation_system, 'stop_auto_validation'):
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
