"""
ML Dashboard Test Route
Simple test page to verify ML Dashboard API integration
"""

from flask import Blueprint, render_template_string
import logging

# Create test blueprint
ml_test_bp = Blueprint('ml_test', __name__)

@ml_test_bp.route('/ml-dashboard-test')
def ml_dashboard_test():
    """Test page for ML Dashboard APIs"""
    
    test_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Dashboard API Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #1a1a1a;
                color: #ffffff;
            }
            .test-section {
                background: #2a2a2a;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                border: 1px solid #3a3a3a;
            }
            .test-button {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            }
            .test-button:hover {
                background: #2563eb;
            }
            .test-result {
                background: #1a1a1a;
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
                max-height: 400px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-success { background: #10b981; }
            .status-error { background: #ef4444; }
            .status-warning { background: #f59e0b; }
            h1 { color: #3b82f6; }
            h2 { color: #10b981; }
        </style>
    </head>
    <body>
        <h1>🧠 ML Dashboard API Test</h1>
        <p>Test the ML Dashboard API endpoints and data manager functionality.</p>
        
        <div class="test-section">
            <h2>API Endpoint Tests</h2>
            <button class="test-button" onclick="testPredictions()">Test Predictions API</button>
            <button class="test-button" onclick="testAccuracy()">Test Accuracy API</button>
            <button class="test-button" onclick="testPerformance()">Test Performance API</button>
            <button class="test-button" onclick="testHealth()">Test Health API</button>
            <button class="test-button" onclick="testAllAPIs()">Test All APIs</button>
            
            <div id="api-results" class="test-result">
                Click a button above to test the API endpoints...
            </div>
        </div>
        
        <div class="test-section">
            <h2>ML Data Manager Test</h2>
            <button class="test-button" onclick="initDataManager()">Initialize Data Manager</button>
            <button class="test-button" onclick="testDataManagerMethods()">Test Data Manager Methods</button>
            <button class="test-button" onclick="checkDataManagerStatus()">Check Status</button>
            
            <div id="datamanager-results" class="test-result">
                Initialize the data manager to begin testing...
            </div>
        </div>
        
        <div class="test-section">
            <h2>Live Data Display</h2>
            <div id="live-status">
                <p><span class="status-indicator status-warning"></span>Status: Initializing...</p>
            </div>
            
            <h3>Current Predictions:</h3>
            <div id="live-predictions" class="test-result">Loading...</div>
            
            <h3>Accuracy Metrics:</h3>
            <div id="live-accuracy" class="test-result">Loading...</div>
            
            <h3>Performance Data:</h3>
            <div id="live-performance" class="test-result">Loading...</div>
        </div>

        <script>
            let dataManager = null;
            
            // Utility functions
            function log(elementId, message, isError = false) {
                const element = document.getElementById(elementId);
                const timestamp = new Date().toLocaleTimeString();
                const color = isError ? '#ef4444' : '#10b981';
                element.innerHTML += `<div style="color: ${color};">[${timestamp}] ${message}</div>`;
                element.scrollTop = element.scrollHeight;
            }
            
            function clearLog(elementId) {
                document.getElementById(elementId).innerHTML = '';
            }
            
            // API Tests
            async function testPredictions() {
                clearLog('api-results');
                log('api-results', 'Testing ML Predictions API...');
                
                try {
                    const response = await fetch('/api/ml-predictions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ timeframes: ['15m', '1h', '4h', '24h'] })
                    });
                    
                    const data = await response.json();
                    log('api-results', `✅ Predictions API response: ${JSON.stringify(data, null, 2)}`);
                    
                } catch (error) {
                    log('api-results', `❌ Predictions API error: ${error.message}`, true);
                }
            }
            
            async function testAccuracy() {
                clearLog('api-results');
                log('api-results', 'Testing ML Accuracy API...');
                
                try {
                    const response = await fetch('/api/ml-accuracy?timeframe=7d');
                    const data = await response.json();
                    log('api-results', `✅ Accuracy API response: ${JSON.stringify(data, null, 2)}`);
                    
                } catch (error) {
                    log('api-results', `❌ Accuracy API error: ${error.message}`, true);
                }
            }
            
            async function testPerformance() {
                clearLog('api-results');
                log('api-results', 'Testing ML Performance API...');
                
                try {
                    const response = await fetch('/api/ml-performance');
                    const data = await response.json();
                    log('api-results', `✅ Performance API response: ${JSON.stringify(data, null, 2)}`);
                    
                } catch (error) {
                    log('api-results', `❌ Performance API error: ${error.message}`, true);
                }
            }
            
            async function testHealth() {
                clearLog('api-results');
                log('api-results', 'Testing ML Health API...');
                
                try {
                    const response = await fetch('/api/ml-health');
                    const data = await response.json();
                    log('api-results', `✅ Health API response: ${JSON.stringify(data, null, 2)}`);
                    
                } catch (error) {
                    log('api-results', `❌ Health API error: ${error.message}`, true);
                }
            }
            
            async function testAllAPIs() {
                clearLog('api-results');
                log('api-results', 'Testing all ML APIs...');
                
                await testPredictions();
                await new Promise(resolve => setTimeout(resolve, 500));
                await testAccuracy();
                await new Promise(resolve => setTimeout(resolve, 500));
                await testPerformance();
                await new Promise(resolve => setTimeout(resolve, 500));
                await testHealth();
                
                log('api-results', '✅ All API tests completed!');
            }
            
            // Data Manager Tests
            async function initDataManager() {
                clearLog('datamanager-results');
                log('datamanager-results', 'Initializing ML Data Manager...');
                
                try {
                    // Check if script is loaded
                    if (typeof MLDataManager === 'undefined') {
                        log('datamanager-results', '❌ MLDataManager class not found. Loading script...', true);
                        
                        // Try to load the script
                        const script = document.createElement('script');
                        script.src = '/static/js/ml_data_manager.js';
                        script.onload = () => {
                            log('datamanager-results', '✅ MLDataManager script loaded');
                            initDataManager(); // Retry
                        };
                        script.onerror = () => {
                            log('datamanager-results', '❌ Failed to load MLDataManager script', true);
                        };
                        document.head.appendChild(script);
                        return;
                    }
                    
                    dataManager = new MLDataManager();
                    
                    // Setup event listeners
                    dataManager.on('ready', (data) => {
                        log('datamanager-results', `✅ Data Manager ready: ${JSON.stringify(data)}`);
                        updateLiveStatus('success', 'Ready');
                        updateLiveData();
                    });
                    
                    dataManager.on('error', (data) => {
                        log('datamanager-results', `❌ Data Manager error: ${JSON.stringify(data)}`, true);
                        updateLiveStatus('error', 'Error');
                    });
                    
                    dataManager.on('predictions_updated', (data) => {
                        log('datamanager-results', '📊 Predictions updated');
                        updateLiveData();
                    });
                    
                    log('datamanager-results', '✅ ML Data Manager initialized successfully');
                    
                } catch (error) {
                    log('datamanager-results', `❌ Data Manager initialization error: ${error.message}`, true);
                }
            }
            
            async function testDataManagerMethods() {
                if (!dataManager) {
                    log('datamanager-results', '❌ Data Manager not initialized', true);
                    return;
                }
                
                log('datamanager-results', 'Testing Data Manager methods...');
                
                try {
                    // Test status
                    const status = dataManager.getConnectionStatus();
                    log('datamanager-results', `📊 Connection Status: ${JSON.stringify(status, null, 2)}`);
                    
                    // Test data retrieval
                    const predictions = dataManager.getPredictions();
                    log('datamanager-results', `📈 Current Predictions: ${JSON.stringify(predictions, null, 2)}`);
                    
                    // Test refresh
                    log('datamanager-results', '🔄 Testing refresh...');
                    await dataManager.refreshAll();
                    log('datamanager-results', '✅ Refresh completed');
                    
                } catch (error) {
                    log('datamanager-results', `❌ Method test error: ${error.message}`, true);
                }
            }
            
            function checkDataManagerStatus() {
                if (!dataManager) {
                    log('datamanager-results', '❌ Data Manager not initialized', true);
                    return;
                }
                
                const status = dataManager.getStatus();
                log('datamanager-results', `📊 Data Manager Status: ${JSON.stringify(status, null, 2)}`);
            }
            
            // Live data updates
            function updateLiveStatus(type, text) {
                const statusEl = document.getElementById('live-status');
                const indicatorClass = `status-${type}`;
                statusEl.innerHTML = `<p><span class="status-indicator ${indicatorClass}"></span>Status: ${text}</p>`;
            }
            
            function updateLiveData() {
                if (!dataManager) return;
                
                // Update predictions
                const predictions = dataManager.getPredictions();
                document.getElementById('live-predictions').innerHTML = predictions ? 
                    JSON.stringify(predictions, null, 2) : 'No predictions available';
                
                // Update accuracy
                const accuracy = dataManager.getAccuracyMetrics();
                document.getElementById('live-accuracy').innerHTML = accuracy ? 
                    JSON.stringify(accuracy, null, 2) : 'No accuracy data available';
                
                // Update performance
                const performance = dataManager.getPerformanceData();
                document.getElementById('live-performance').innerHTML = performance ? 
                    JSON.stringify(performance, null, 2) : 'No performance data available';
            }
            
            // Auto-refresh live data every 30 seconds
            setInterval(() => {
                if (dataManager) {
                    updateLiveData();
                }
            }, 30000);
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', () => {
                log('api-results', '🚀 ML Dashboard Test Page loaded. Click buttons to test functionality.');
                updateLiveStatus('warning', 'Initializing...');
            });
        </script>
    </body>
    </html>
    """
    
    return test_html

def register_ml_test_routes(app):
    """Register ML test routes with Flask app"""
    try:
        app.register_blueprint(ml_test_bp)
        logging.info("✅ ML Test routes registered")
    except Exception as e:
        logging.error(f"❌ Failed to register ML Test routes: {e}")

if __name__ == "__main__":
    print("ML Dashboard Test Route - Use /ml-dashboard-test to access the test page")
