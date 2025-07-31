"""
🔗 PHASE 3: INTEGRATION SYSTEM
===============================

Phase3Integration - Seamless integration of all Phase 3 components
Connects PredictionValidator, AdvancedLearningEngine, OutcomeTracker, and PerformanceAnalytics

Author: GoldGPT AI System
Created: July 23, 2025
"""

import logging
import threading
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, Blueprint
from prediction_validator import PredictionValidator
from advanced_learning_engine import AdvancedLearningEngine
from outcome_tracker import OutcomeTracker
from performance_analytics import (
    PerformanceAnalytics, 
    initialize_performance_analytics, 
    start_performance_analytics,
    analytics_bp
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('phase3_integration')

class Phase3Integration:
    """
    Complete Phase 3 Self-Learning Prediction Validation System Integration
    Manages all components and their interactions
    """
    
    def __init__(self):
        self.prediction_validator = None
        self.learning_engine = None
        self.outcome_tracker = None
        self.performance_analytics = None
        self.is_initialized = False
        self.is_running = False
        logger.info("🔗 Phase 3 Integration initialized")
    
    def initialize_all_components(self) -> bool:
        """Initialize all Phase 3 components"""
        try:
            logger.info("🚀 Initializing Phase 3 components...")
            
            # Initialize PredictionValidator
            self.prediction_validator = PredictionValidator()
            logger.info("✅ PredictionValidator initialized")
            
            # Initialize AdvancedLearningEngine
            self.learning_engine = AdvancedLearningEngine()
            logger.info("✅ AdvancedLearningEngine initialized")
            
            # Initialize OutcomeTracker
            self.outcome_tracker = OutcomeTracker()
            logger.info("✅ OutcomeTracker initialized")
            
            # Initialize PerformanceAnalytics
            self.performance_analytics = initialize_performance_analytics(
                self.prediction_validator,
                self.learning_engine,
                self.outcome_tracker
            )
            logger.info("✅ PerformanceAnalytics initialized")
            
            self.is_initialized = True
            logger.info("🎉 All Phase 3 components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 3 initialization failed: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all Phase 3 background services"""
        if not self.is_initialized:
            logger.error("❌ Cannot start services - components not initialized")
            return False
        
        try:
            logger.info("🚀 Starting Phase 3 services...")
            
            # Start PredictionValidator service
            self.prediction_validator.start_validation_service()
            logger.info("✅ PredictionValidator service started")
            
            # Start AdvancedLearningEngine service
            self.learning_engine.start_learning_service()
            logger.info("✅ AdvancedLearningEngine service started")
            
            # Start OutcomeTracker service
            self.outcome_tracker.start_analysis_service()
            logger.info("✅ OutcomeTracker service started")
            
            # Start PerformanceAnalytics service
            start_performance_analytics()
            logger.info("✅ PerformanceAnalytics service started")
            
            self.is_running = True
            logger.info("🎉 All Phase 3 services started successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 3 services startup failed: {e}")
            return False
    
    def stop_all_services(self):
        """Stop all Phase 3 services"""
        try:
            logger.info("⏹️ Stopping Phase 3 services...")
            
            if self.prediction_validator:
                self.prediction_validator.stop_validation_service()
                logger.info("✅ PredictionValidator service stopped")
            
            if self.learning_engine:
                self.learning_engine.stop_learning_service()
                logger.info("✅ AdvancedLearningEngine service stopped")
            
            if self.outcome_tracker:
                self.outcome_tracker.stop_analysis_service()
                logger.info("✅ OutcomeTracker service stopped")
            
            if self.performance_analytics:
                self.performance_analytics.stop_analytics_service()
                logger.info("✅ PerformanceAnalytics service stopped")
            
            self.is_running = False
            logger.info("🎉 All Phase 3 services stopped successfully!")
            
        except Exception as e:
            logger.error(f"❌ Phase 3 services shutdown failed: {e}")
    
    def submit_prediction_for_learning(self, prediction_data: Dict[str, Any]) -> bool:
        """Submit a prediction to the self-learning system"""
        if not self.is_initialized:
            logger.error("❌ Cannot submit prediction - Phase 3 not initialized")
            return False
        
        try:
            # Submit to PredictionValidator
            prediction_id = self.prediction_validator.store_prediction(prediction_data)
            
            logger.info(f"✅ Prediction {prediction_id} submitted to Phase 3 system")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to submit prediction: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive Phase 3 system status"""
        try:
            status = {
                'phase3_initialized': self.is_initialized,
                'phase3_running': self.is_running,
                'components': {
                    'prediction_validator': {
                        'initialized': self.prediction_validator is not None,
                        'running': self.prediction_validator.is_running if self.prediction_validator else False,
                        'total_predictions': self._get_validator_stats()
                    },
                    'learning_engine': {
                        'initialized': self.learning_engine is not None,
                        'running': self.learning_engine.is_running if self.learning_engine else False,
                        'learning_cycles': self._get_learning_stats()
                    },
                    'outcome_tracker': {
                        'initialized': self.outcome_tracker is not None,
                        'running': self.outcome_tracker.is_running if self.outcome_tracker else False,
                        'tracked_outcomes': self._get_tracker_stats()
                    },
                    'performance_analytics': {
                        'initialized': self.performance_analytics is not None,
                        'running': self.performance_analytics.is_running if self.performance_analytics else False,
                        'current_metrics': self._get_analytics_stats()
                    }
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}
    
    def _get_validator_stats(self) -> int:
        """Get PredictionValidator statistics"""
        try:
            if self.prediction_validator:
                recent_metrics = self.prediction_validator.get_recent_performance(days=7)
                return recent_metrics.get('total_predictions', 0)
            return 0
        except:
            return 0
    
    def _get_learning_stats(self) -> int:
        """Get AdvancedLearningEngine statistics"""
        try:
            if self.learning_engine:
                learning_progress = self.learning_engine.get_learning_progress(days=7)
                return learning_progress.get('total_cycles', 0)
            return 0
        except:
            return 0
    
    def _get_tracker_stats(self) -> int:
        """Get OutcomeTracker statistics"""
        try:
            if self.outcome_tracker:
                analysis = self.outcome_tracker.get_comprehensive_analysis(days=7)
                return analysis.get('summary_statistics', {}).get('total_outcomes', 0)
            return 0
        except:
            return 0
    
    def _get_analytics_stats(self) -> Optional[Dict[str, Any]]:
        """Get PerformanceAnalytics statistics"""
        try:
            if self.performance_analytics and self.performance_analytics.current_metrics:
                return {
                    'overall_accuracy': self.performance_analytics.current_metrics.overall_accuracy,
                    'prediction_volume': self.performance_analytics.current_metrics.prediction_volume
                }
            return None
        except:
            return None
    
    def get_learning_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive learning insights from all components"""
        if not self.is_initialized:
            return {'error': 'Phase 3 not initialized'}
        
        try:
            insights = {
                'validation_metrics': self.prediction_validator.get_recent_performance(days),
                'learning_progress': self.learning_engine.get_learning_progress(days),
                'outcome_analysis': self.outcome_tracker.get_comprehensive_analysis(days),
                'performance_dashboard': self.performance_analytics.get_dashboard_data(f'{days*24}h'),
                'generated_at': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to get learning insights: {e}")
            return {'error': str(e)}

# Global Phase 3 integration instance
phase3_system = Phase3Integration()

def initialize_phase3() -> bool:
    """Initialize the complete Phase 3 system"""
    return phase3_system.initialize_all_components()

def start_phase3() -> bool:
    """Start the complete Phase 3 system"""
    return phase3_system.start_all_services()

def stop_phase3():
    """Stop the complete Phase 3 system"""
    phase3_system.stop_all_services()

def submit_prediction(prediction_data: Dict[str, Any]) -> bool:
    """Submit a prediction to Phase 3 learning system"""
    return phase3_system.submit_prediction_for_learning(prediction_data)

def get_phase3_status() -> Dict[str, Any]:
    """Get Phase 3 system status"""
    return phase3_system.get_system_status()

def get_phase3_insights(days: int = 7) -> Dict[str, Any]:
    """Get Phase 3 learning insights"""
    return phase3_system.get_learning_insights(days)

# Create Flask Blueprint for Phase 3 API
phase3_bp = Blueprint('phase3', __name__)

@phase3_bp.route('/api/phase3/status')
def api_get_status():
    """API endpoint for Phase 3 status"""
    from flask import jsonify
    return jsonify(get_phase3_status())

@phase3_bp.route('/api/phase3/insights')
def api_get_insights():
    """API endpoint for Phase 3 insights"""
    from flask import jsonify, request
    days = request.args.get('days', 7, type=int)
    return jsonify(get_phase3_insights(days))

@phase3_bp.route('/api/phase3/submit-prediction', methods=['POST'])
def api_submit_prediction():
    """API endpoint for submitting predictions"""
    from flask import jsonify, request
    try:
        prediction_data = request.get_json()
        success = submit_prediction(prediction_data)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def register_phase3_with_flask(app: Flask):
    """Register Phase 3 blueprints with Flask app"""
    app.register_blueprint(phase3_bp)
    app.register_blueprint(analytics_bp)
    logger.info("✅ Phase 3 blueprints registered with Flask")

if __name__ == "__main__":
    print("🔗 Phase 3 Self-Learning Prediction Validation System")
    print("=" * 55)
    
    # Initialize Phase 3
    if initialize_phase3():
        print("✅ Phase 3 initialized successfully!")
        
        # Start Phase 3 services
        if start_phase3():
            print("✅ Phase 3 services started successfully!")
            print("\n🎉 Phase 3 is now fully operational!")
            print("\nComponents running:")
            print("  • PredictionValidator - Autonomous prediction validation")
            print("  • AdvancedLearningEngine - Daily learning cycles")
            print("  • OutcomeTracker - Comprehensive outcome analysis")
            print("  • PerformanceAnalytics - Real-time dashboard")
            
            try:
                input("\nPress Enter to stop Phase 3 system...")
            except KeyboardInterrupt:
                pass
            finally:
                stop_phase3()
                print("✅ Phase 3 system stopped")
        else:
            print("❌ Failed to start Phase 3 services")
    else:
        print("❌ Failed to initialize Phase 3")
