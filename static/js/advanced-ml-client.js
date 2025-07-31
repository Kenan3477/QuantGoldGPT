/**
 * Advanced ML Prediction Client for GoldGPT
 * Professional JavaScript client for ML API with WebSocket support
 */

class AdvancedMLClient {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || '/api/ml-predictions';
        this.websocketUrl = options.websocketUrl || window.location.origin;
        this.socket = null;
        this.apiKey = options.apiKey || null;
        this.retryAttempts = options.retryAttempts || 3;
        this.retryDelay = options.retryDelay || 1000;
        
        // Event callbacks
        this.callbacks = {
            onPredictionUpdate: [],
            onValidationUpdate: [],
            onModelUpdate: [],
            onError: [],
            onConnect: [],
            onDisconnect: []
        };
        
        // Data cache
        this.cache = {
            predictions: {},
            accuracy: null,
            features: null,
            status: null,
            lastUpdate: null
        };
        
        // Initialize
        this.initializeWebSocket();
        
        console.log('✅ Advanced ML Client initialized');
    }
    
    // =======================
    // WebSocket Management
    // =======================
    
    initializeWebSocket() {
        try {
            this.socket = io(this.websocketUrl);
            
            this.socket.on('connect', () => {
                console.log('🔌 Connected to ML prediction WebSocket');
                this.emit('onConnect');
            });
            
            this.socket.on('disconnect', () => {
                console.log('🔌 Disconnected from ML prediction WebSocket');
                this.emit('onDisconnect');
            });
            
            this.socket.on('new_predictions', (data) => {
                console.log('📊 Received new predictions:', data);
                this.handlePredictionUpdate(data);
            });
            
            this.socket.on('validation_update', (data) => {
                console.log('✅ Received validation update:', data);
                this.emit('onValidationUpdate', data);
            });
            
            this.socket.on('model_update', (data) => {
                console.log('🤖 Received model update:', data);
                this.emit('onModelUpdate', data);
            });
            
            this.socket.on('periodic_update', (data) => {
                this.updateCache(data);
                this.emit('onPredictionUpdate', data);
            });
            
            this.socket.on('error', (error) => {
                console.error('❌ WebSocket error:', error);
                this.emit('onError', error);
            });
            
        } catch (error) {
            console.error('❌ Failed to initialize WebSocket:', error);
            this.emit('onError', error);
        }
    }
    
    subscribeToUpdates(timeframes = ['1h', '4h', '24h']) {
        if (this.socket && this.socket.connected) {
            this.socket.emit('subscribe_predictions', { timeframes });
        }
    }
    
    requestPredictionUpdate(timeframe = '1h') {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_prediction_update', { timeframe });
        }
    }
    
    // =======================
    // API Methods
    // =======================
    
    async getPredictions(timeframe) {
        const url = `${this.baseUrl}/${timeframe}`;
        return await this.makeRequest('GET', url);
    }
    
    async getAllPredictions() {
        const url = `${this.baseUrl}/all`;
        return await this.makeRequest('GET', url);
    }
    
    async getAccuracyStats(options = {}) {
        const params = new URLSearchParams();
        if (options.days) params.append('days', options.days);
        if (options.strategy) params.append('strategy', options.strategy);
        if (options.timeframe) params.append('timeframe', options.timeframe);
        
        const url = `${this.baseUrl}/accuracy?${params.toString()}`;
        return await this.makeRequest('GET', url);
    }
    
    async refreshPredictions(timeframes = null) {
        const url = `${this.baseUrl}/refresh`;
        const body = timeframes ? { timeframes } : {};
        return await this.makeRequest('POST', url, body);
    }
    
    async getFeatureImportance(options = {}) {
        const params = new URLSearchParams();
        if (options.strategy) params.append('strategy', options.strategy);
        if (options.days) params.append('days', options.days);
        
        const url = `${this.baseUrl}/features?${params.toString()}`;
        return await this.makeRequest('GET', url);
    }
    
    async getSystemStatus() {
        const url = `${this.baseUrl}/status`;
        return await this.makeRequest('GET', url);
    }
    
    async getConfig() {
        const url = `${this.baseUrl}/config`;
        return await this.makeRequest('GET', url);
    }
    
    async updateConfig(newConfig) {
        const url = `${this.baseUrl}/config`;
        return await this.makeRequest('POST', url, newConfig);
    }
    
    // =======================
    // Utility Methods
    // =======================
    
    async makeRequest(method, url, body = null, attempt = 1) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (this.apiKey) {
                options.headers['Authorization'] = `Bearer ${this.apiKey}`;
            }
            
            if (body) {
                options.body = JSON.stringify(body);
            }
            
            const response = await fetch(url, options);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data;
            
        } catch (error) {
            console.error(`❌ Request failed (attempt ${attempt}):`, error);
            
            if (attempt < this.retryAttempts) {
                console.log(`🔄 Retrying in ${this.retryDelay}ms...`);
                await this.delay(this.retryDelay);
                return await this.makeRequest(method, url, body, attempt + 1);
            }
            
            this.emit('onError', error);
            throw error;
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // =======================
    // Event Management
    // =======================
    
    on(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        }
    }
    
    off(event, callback) {
        if (this.callbacks[event]) {
            const index = this.callbacks[event].indexOf(callback);
            if (index > -1) {
                this.callbacks[event].splice(index, 1);
            }
        }
    }
    
    emit(event, data = null) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`❌ Error in ${event} callback:`, error);
                }
            });
        }
    }
    
    // =======================
    // Cache Management
    // =======================
    
    updateCache(data) {
        this.cache.lastUpdate = new Date();
        
        if (data.predictions) {
            this.cache.predictions = { ...this.cache.predictions, ...data.predictions };
        }
        
        if (data.accuracy_stats) {
            this.cache.accuracy = data.accuracy_stats;
        }
        
        if (data.feature_importance) {
            this.cache.features = data.feature_importance;
        }
        
        if (data.status) {
            this.cache.status = data.status;
        }
    }
    
    getCachedData(type) {
        return this.cache[type];
    }
    
    handlePredictionUpdate(data) {
        if (data.predictions) {
                this.callbacks.onPrediction.forEach(callback => {
                    try {
                        callback(data);
                    } catch (e) {
                        console.error('Error in prediction callback:', e);
                    }
                });
            } else {
                this.callbacks.onError.forEach(callback => {
                    try {
                        callback(data);
                    } catch (e) {
                        console.error('Error in error callback:', e);
                    }
                });
            }
        });
        
        // Handle strategy performance updates
        this.socket.on('strategy_performance', (data) => {
            console.log('📈 Received Strategy Performance:', data);
            this.strategyPerformance = data;
            
            if (data.success) {
                this.callbacks.onPerformance.forEach(callback => {
                    try {
                        callback(data);
                    } catch (e) {
                        console.error('Error in performance callback:', e);
                    }
                });
            }
        });
    }
    
    // API Methods
    async checkSystemAvailability() {
        try {
            const response = await fetch('/api/ml-system-status');
            const data = await response.json();
            
            console.log('🔍 ML System Status:', data);
            
            if (data.advanced_ml_available) {
                console.log('✅ Advanced ML System Available');
                console.log(`📊 Strategies: ${data.strategy_count}`);
            } else {
                console.log('⚠️ Advanced ML System Not Available');
            }
            
            return data;
        } catch (error) {
            console.error('❌ Failed to check system availability:', error);
            return { advanced_ml_available: false, error: error.message };
        }
    }
    
    async getPredictions(timeframes = ['1H', '4H', '1D']) {
        try {
            const response = await fetch('/api/advanced-ml/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ timeframes })
            });
            
            const data = await response.json();
            console.log('📊 Advanced ML Predictions:', data);
            
            if (data.status === 'success') {
                this.lastPrediction = data;
                return data;
            } else {
                throw new Error(data.error || 'Failed to get predictions');
            }
        } catch (error) {
            console.error('❌ Failed to get predictions:', error);
            
            // Fallback to enhanced ML endpoint
            try {
                const fallbackResponse = await fetch('/api/ml-predictions-enhanced');
                const fallbackData = await fallbackResponse.json();
                console.log('🔄 Using fallback ML system:', fallbackData);
                return fallbackData;
            } catch (fallbackError) {
                console.error('❌ Fallback also failed:', fallbackError);
                throw error;
            }
        }
    }
    
    async getQuickPrediction() {
        try {
            const response = await fetch('/api/advanced-ml/quick-prediction');
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('⚡ Quick Prediction:', data);
                return data;
            } else {
                throw new Error(data.error || 'Failed to get quick prediction');
            }
        } catch (error) {
            console.error('❌ Failed to get quick prediction:', error);
            throw error;
        }
    }
    
    async getStrategyPerformance() {
        try {
            const response = await fetch('/api/ml-strategy-performance');
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('📈 Strategy Performance:', data);
                this.strategyPerformance = data;
                return data;
            } else {
                throw new Error(data.error || 'Failed to get strategy performance');
            }
        } catch (error) {
            console.error('❌ Failed to get strategy performance:', error);
            throw error;
        }
    }
    
    async getConfidenceAnalysis() {
        try {
            const response = await fetch('/api/advanced-ml/confidence-analysis');
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('🎯 Confidence Analysis:', data);
                return data;
            } else {
                throw new Error(data.error || 'Failed to get confidence analysis');
            }
        } catch (error) {
            console.error('❌ Failed to get confidence analysis:', error);
            throw error;
        }
    }
    
    // Real-time WebSocket Methods
    requestRealtimePrediction(timeframe = '1H') {
        if (this.isConnected && this.socket) {
            console.log(`📡 Requesting real-time prediction for ${timeframe}`);
            this.socket.emit('request_advanced_ml_prediction', { timeframe });
        } else {
            console.warn('⚠️ WebSocket not connected, falling back to HTTP request');
            return this.getPredictions([timeframe]);
        }
    }
    
    requestStrategyPerformance() {
        if (this.isConnected && this.socket) {
            console.log('📡 Requesting real-time strategy performance');
            this.socket.emit('request_strategy_performance', {});
        } else {
            console.warn('⚠️ WebSocket not connected, falling back to HTTP request');
            return this.getStrategyPerformance();
        }
    }
    
    // Event Handlers
    onPrediction(callback) {
        if (typeof callback === 'function') {
            this.callbacks.onPrediction.push(callback);
        }
    }
    
    onPerformance(callback) {
        if (typeof callback === 'function') {
            this.callbacks.onPerformance.push(callback);
        }
    }
    
    onError(callback) {
        if (typeof callback === 'function') {
            this.callbacks.onError.push(callback);
        }
    }
    
    // UI Integration Methods
    updatePredictionDisplay(containerId, prediction) {
        const container = document.getElementById(containerId);
        if (!container || !prediction || !prediction.success) return;
        
        const pred = prediction.prediction;
        const changePercent = pred.price_change_percent;
        const isPositive = changePercent > 0;
        
        container.innerHTML = `
            <div class="advanced-ml-prediction">
                <div class="prediction-header">
                    <h4>Advanced ML Prediction (${prediction.timeframe})</h4>
                    <span class="engine-badge">${prediction.engine}</span>
                </div>
                
                <div class="prediction-content">
                    <div class="price-prediction">
                        <div class="current-price">
                            <label>Current Price:</label>
                            <span class="price">$${pred.current_price.toFixed(2)}</span>
                        </div>
                        <div class="predicted-price">
                            <label>Predicted Price:</label>
                            <span class="price">$${pred.predicted_price.toFixed(2)}</span>
                        </div>
                        <div class="price-change ${isPositive ? 'positive' : 'negative'}">
                            <label>Change:</label>
                            <span>${isPositive ? '+' : ''}${changePercent.toFixed(2)}%</span>
                        </div>
                    </div>
                    
                    <div class="prediction-details">
                        <div class="direction-confidence">
                            <div class="direction ${pred.direction}">
                                <label>Direction:</label>
                                <span class="direction-value">${pred.direction.toUpperCase()}</span>
                            </div>
                            <div class="confidence">
                                <label>Confidence:</label>
                                <span class="confidence-value">${(pred.confidence * 100).toFixed(1)}%</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
                                </div>
                            </div>
                        </div>
                        
                        ${pred.confidence_interval ? `
                            <div class="confidence-interval">
                                <label>95% Confidence Interval:</label>
                                <span>$${pred.confidence_interval.lower.toFixed(2)} - $${pred.confidence_interval.upper.toFixed(2)}</span>
                            </div>
                        ` : ''}
                        
                        <div class="risk-management">
                            <div class="stop-loss">
                                <label>Stop Loss:</label>
                                <span class="stop-loss-value">$${pred.stop_loss.toFixed(2)}</span>
                            </div>
                            <div class="take-profit">
                                <label>Take Profit:</label>
                                <span class="take-profit-value">$${pred.take_profit.toFixed(2)}</span>
                            </div>
                        </div>
                        
                        ${pred.strategy_votes && Object.keys(pred.strategy_votes).length > 0 ? `
                            <div class="strategy-votes">
                                <label>Strategy Votes:</label>
                                <div class="votes-container">
                                    ${Object.entries(pred.strategy_votes).map(([strategy, weight]) => `
                                        <div class="strategy-vote">
                                            <span class="strategy-name">${strategy}:</span>
                                            <span class="strategy-weight">${(weight * 100).toFixed(1)}%</span>
                                            <div class="weight-bar">
                                                <div class="weight-fill" style="width: ${weight * 100}%"></div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
                
                <div class="prediction-meta">
                    <div class="execution-time">
                        <span>Generated in ${prediction.execution_time?.toFixed(2) || '0.00'}s</span>
                    </div>
                    <div class="timestamp">
                        <span>${new Date(prediction.timestamp).toLocaleString()}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    updateStrategyPerformanceDisplay(containerId, performance) {
        const container = document.getElementById(containerId);
        if (!container || !performance || !performance.success) return;
        
        const strategies = performance.performance.strategies || {};
        
        container.innerHTML = `
            <div class="strategy-performance">
                <div class="performance-header">
                    <h4>Strategy Performance</h4>
                    <span class="total-strategies">${Object.keys(strategies).length} Strategies Active</span>
                </div>
                
                <div class="strategies-list">
                    ${Object.entries(strategies).map(([name, data]) => `
                        <div class="strategy-item">
                            <div class="strategy-name">${name}</div>
                            <div class="strategy-metrics">
                                <div class="metric">
                                    <label>Weight:</label>
                                    <span class="weight-value">${(data.weight * 100).toFixed(1)}%</span>
                                    <div class="weight-bar">
                                        <div class="weight-fill" style="width: ${data.weight * 100}%"></div>
                                    </div>
                                </div>
                                <div class="metric">
                                    <label>Accuracy:</label>
                                    <span class="accuracy-value">${(data.accuracy_score * 100).toFixed(1)}%</span>
                                </div>
                                <div class="metric">
                                    <label>Predictions:</label>
                                    <span class="prediction-count">${data.prediction_count}</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
}

// Initialize global Advanced ML client
let advancedMLClient = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 DOM ready, initializing Advanced ML Client...');
    advancedMLClient = new AdvancedMLClient();
    
    // Make it globally accessible
    window.advancedMLClient = advancedMLClient;
    
    // Set up automatic prediction updates every 30 seconds
    setInterval(() => {
        if (advancedMLClient.isConnected) {
            advancedMLClient.requestRealtimePrediction('1H');
        }
    }, 30000); // 30 seconds
    
    // Set up performance updates every 2 minutes
    setInterval(() => {
        if (advancedMLClient.isConnected) {
            advancedMLClient.requestStrategyPerformance();
        }
    }, 120000); // 2 minutes
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedMLClient;
}
