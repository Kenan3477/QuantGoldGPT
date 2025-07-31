#!/usr/bin/env python3
"""
GoldGPT App.py Integration Guide
Code snippets to integrate the learning system with the existing Flask application
"""

# ====================================================================================
# STEP 1: Add these imports to the top of your app.py (after existing imports)
# ====================================================================================

# Add these imports after your existing imports in app.py:
"""
# Learning System Integration
try:
    from learning_system_integration import integrate_learning_system_with_app, learning_system
    LEARNING_SYSTEM_AVAILABLE = True
    print("✅ Learning system integration available")
except ImportError as e:
    print(f"⚠️ Learning system not available: {e}")
    LEARNING_SYSTEM_AVAILABLE = False
    learning_system = None
"""

# ====================================================================================
# STEP 2: Initialize learning system (add after Flask app creation)
# ====================================================================================

# Add this after your app = Flask(__name__) line:
"""
# Initialize Learning System
if LEARNING_SYSTEM_AVAILABLE:
    try:
        learning_system_integration = integrate_learning_system_with_app(app)
        print("✅ Learning system integrated successfully")
    except Exception as e:
        print(f"⚠️ Learning system integration failed: {e}")
        learning_system_integration = None
else:
    learning_system_integration = None
"""

# ====================================================================================
# STEP 3: Enhance existing AI analysis endpoint
# ====================================================================================

# Replace your existing /api/analysis endpoint with this enhanced version:
"""
@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    try:
        # Get existing analysis
        analysis_result = get_ai_analysis_sync()
        
        # Enhance with learning system tracking
        if learning_system_integration and isinstance(analysis_result, dict):
            # Track predictions if they exist
            if 'predictions' in analysis_result:
                for prediction in analysis_result['predictions']:
                    try:
                        # Convert prediction format for tracking
                        tracking_data = {
                            'symbol': prediction.get('symbol', 'XAUUSD'),
                            'timeframe': prediction.get('timeframe', '1H'),
                            'strategy': prediction.get('strategy', 'ai_analysis'),
                            'direction': prediction.get('prediction', 'neutral'),
                            'confidence': prediction.get('confidence', 0.5),
                            'predicted_price': prediction.get('target_price', 0.0),
                            'current_price': prediction.get('current_price', 0.0),
                            'features': prediction.get('analysis_points', []),
                            'indicators': prediction.get('technical_analysis', {}),
                            'market_context': prediction.get('market_context', {})
                        }
                        
                        # Track the prediction
                        tracking_id = learning_system_integration.track_prediction(tracking_data)
                        prediction['learning_tracking_id'] = tracking_id
                        
                    except Exception as e:
                        print(f"⚠️ Failed to track prediction: {e}")
            
            # Add learning insights to response
            recent_insights = learning_system_integration.get_learning_insights(limit=3)
            analysis_result['learning_insights'] = recent_insights
            
            # Add performance summary
            performance = learning_system_integration.get_performance_summary(days=7)
            analysis_result['recent_performance'] = performance
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"❌ Analysis endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': 'Analysis temporarily unavailable',
            'details': str(e)
        }), 500
"""

# ====================================================================================
# STEP 4: Add validation webhook (add as new endpoint)
# ====================================================================================

# Add this new endpoint to automatically validate predictions:
"""
@app.route('/api/validate-predictions', methods=['POST'])
def validate_predictions_endpoint():
    '''Endpoint to validate predictions when market data becomes available'''
    try:
        if not learning_system_integration:
            return jsonify({'error': 'Learning system not available'}), 503
        
        data = request.get_json()
        results = []
        
        for validation in data.get('validations', []):
            tracking_id = validation.get('tracking_id')
            actual_price = validation.get('actual_price')
            
            if tracking_id and actual_price is not None:
                result = learning_system_integration.validate_prediction(tracking_id, actual_price)
                results.append({
                    'tracking_id': tracking_id,
                    'validation_result': result
                })
        
        return jsonify({
            'success': True,
            'validated_count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
"""

# ====================================================================================
# STEP 5: Add learning system status to WebSocket updates
# ====================================================================================

# Enhance your existing WebSocket price update function:
"""
def broadcast_price_update():
    '''Enhanced price update with learning system status'''
    try:
        # Your existing price fetching logic here
        price_data = get_current_gold_price()  # Your existing function
        
        # Add learning system status
        if learning_system_integration:
            try:
                health = learning_system_integration.health_check()
                price_data['learning_system'] = {
                    'status': health.get('overall_status', 'unknown'),
                    'recent_accuracy': health.get('metrics', {}).get('recent_accuracy', 0.0),
                    'total_predictions': health.get('metrics', {}).get('total_predictions_7d', 0)
                }
            except:
                price_data['learning_system'] = {'status': 'unavailable'}
        
        # Emit to all connected clients
        socketio.emit('price_update', price_data, broadcast=True)
        
    except Exception as e:
        print(f"❌ Price update error: {e}")
"""

# ====================================================================================
# STEP 6: Add learning system routes to navigation
# ====================================================================================

# Add this route to provide learning system dashboard access:
"""
@app.route('/learning')
def learning_dashboard():
    '''Redirect to learning system dashboard'''
    return redirect('/dashboard/')

@app.route('/api/learning-status')
def learning_status():
    '''Get learning system status for frontend'''
    try:
        if not learning_system_integration:
            return jsonify({
                'available': False,
                'message': 'Learning system not initialized'
            })
        
        health = learning_system_integration.health_check()
        performance = learning_system_integration.get_performance_summary(days=1)
        insights = learning_system_integration.get_learning_insights(limit=5)
        
        return jsonify({
            'available': True,
            'health': health,
            'recent_performance': performance,
            'recent_insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        })
"""

# ====================================================================================
# STEP 7: Frontend Integration (add to your main.js or equivalent)
# ====================================================================================

# Add this JavaScript to your frontend:
"""
// Learning System Frontend Integration
class LearningSystemIntegration {
    constructor() {
        this.isAvailable = false;
        this.checkAvailability();
    }
    
    async checkAvailability() {
        try {
            const response = await fetch('/api/learning-status');
            const data = await response.json();
            this.isAvailable = data.available;
            
            if (this.isAvailable) {
                this.displayLearningStatus(data);
            }
        } catch (error) {
            console.warn('Learning system not available:', error);
        }
    }
    
    displayLearningStatus(data) {
        // Add learning system status to dashboard
        const learningStatus = document.createElement('div');
        learningStatus.className = 'learning-system-status';
        learningStatus.innerHTML = `
            <div class="learning-header">
                <h3>🧠 AI Learning System</h3>
                <span class="status ${data.health.overall_status}">${data.health.overall_status}</span>
            </div>
            <div class="learning-metrics">
                <div class="metric">
                    <span>Recent Accuracy:</span>
                    <span>${(data.recent_performance.accuracy_rate * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span>Predictions Today:</span>
                    <span>${data.recent_performance.total_predictions}</span>
                </div>
            </div>
            <a href="/dashboard/" class="learning-dashboard-link">View Full Dashboard →</a>
        `;
        
        // Add to main dashboard
        const dashboard = document.querySelector('.dashboard-container');
        if (dashboard) {
            dashboard.appendChild(learningStatus);
        }
    }
    
    async validatePrediction(trackingId, actualPrice) {
        if (!this.isAvailable) return null;
        
        try {
            const response = await fetch('/api/validate-predictions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    validations: [{
                        tracking_id: trackingId,
                        actual_price: actualPrice
                    }]
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Prediction validation failed:', error);
            return null;
        }
    }
}

// Initialize learning system integration
const learningSystem = new LearningSystemIntegration();
"""

# ====================================================================================
# STEP 8: CSS Styles for Learning System (add to your main.css)
# ====================================================================================

# Add these CSS styles:
"""
/* Learning System Styles */
.learning-system-status {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.learning-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.learning-header h3 {
    color: #ffd700;
    margin: 0;
}

.status {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    text-transform: uppercase;
}

.status.healthy {
    background: rgba(0, 255, 136, 0.2);
    color: #00ff88;
}

.status.degraded {
    background: rgba(255, 215, 0, 0.2);
    color: #ffd700;
}

.learning-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-bottom: 15px;
}

.metric {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
}

.learning-dashboard-link {
    display: inline-block;
    background: linear-gradient(45deg, #ffd700, #ffed4e);
    color: #1a1a2e;
    padding: 8px 16px;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9em;
    transition: transform 0.2s ease;
}

.learning-dashboard-link:hover {
    transform: translateY(-2px);
}
"""

# ====================================================================================
# STEP 9: Complete Integration Instructions
# ====================================================================================

print("""
🎯 GOLDGPT LEARNING SYSTEM INTEGRATION COMPLETE!

To integrate the learning system with your existing app.py:

1. 📥 IMPORT INTEGRATION:
   Add the imports from STEP 1 to the top of your app.py

2. 🚀 INITIALIZE SYSTEM:
   Add the initialization code from STEP 2 after your Flask app creation

3. 🔄 ENHANCE ENDPOINTS:
   Replace your /api/analysis endpoint with the enhanced version from STEP 3

4. ✅ ADD VALIDATION:
   Add the validation webhook from STEP 4 as a new endpoint

5. 📡 UPDATE WEBSOCKET:
   Enhance your WebSocket broadcasts with learning status from STEP 5

6. 🎨 FRONTEND INTEGRATION:
   Add the JavaScript code from STEP 7 to your frontend

7. 💅 ADD STYLES:
   Add the CSS styles from STEP 8 to your stylesheet

🏆 FEATURES YOU'LL GET:
✅ Automatic prediction tracking for all AI analysis calls
✅ Learning insights displayed in main dashboard  
✅ Performance metrics in real-time
✅ Comprehensive learning dashboard at /dashboard/
✅ Advanced backtesting capabilities
✅ Continuous model improvement
✅ Prediction validation system

🔗 KEY ENDPOINTS:
• /dashboard/ - Main learning dashboard
• /api/learning/health - System health check
• /api/learning-status - Status for frontend
• /api/validate-predictions - Manual validation

🎮 USAGE:
1. Your existing AI analysis will automatically be tracked
2. Access the learning dashboard at http://localhost:5000/dashboard/
3. Monitor system health at http://localhost:5000/api/learning/health
4. View learning insights in the main application interface

The system will start learning from your predictions immediately and provide
continuous improvements to your AI trading strategies! 🚀
""")

if __name__ == "__main__":
    print("GoldGPT Learning System Integration Guide")
    print("Follow the steps above to integrate with your existing app.py")
