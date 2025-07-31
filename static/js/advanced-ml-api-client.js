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
            if (typeof io !== 'undefined') {
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
            }
            
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
            // Group predictions by timeframe
            const predictionsByTimeframe = {};
            
            data.predictions.forEach(pred => {
                if (!predictionsByTimeframe[pred.timeframe]) {
                    predictionsByTimeframe[pred.timeframe] = [];
                }
                predictionsByTimeframe[pred.timeframe].push(pred);
            });
            
            this.updateCache({ predictions: predictionsByTimeframe });
        }
        
        this.emit('onPredictionUpdate', data);
    }
    
    // =======================
    // Convenience Methods
    // =======================
    
    async getLatestPredictions(timeframe = '1h') {
        try {
            // Try cache first
            const cached = this.getCachedData('predictions')[timeframe];
            if (cached && this.cache.lastUpdate && 
                (Date.now() - this.cache.lastUpdate.getTime()) < 60000) {
                return { success: true, predictions: cached, source: 'cache' };
            }
            
            // Fetch from API
            const response = await this.getPredictions(timeframe);
            return { ...response, source: 'api' };
            
        } catch (error) {
            console.error('❌ Failed to get latest predictions:', error);
            return { success: false, error: error.message, source: 'error' };
        }
    }
    
    formatPrediction(prediction) {
        return {
            strategy: prediction.strategy_name,
            timeframe: prediction.timeframe,
            direction: prediction.direction,
            confidence: Math.round(prediction.confidence * 100),
            currentPrice: prediction.current_price,
            predictedPrice: prediction.predicted_price,
            priceChange: prediction.price_change,
            priceChangePercent: prediction.price_change_percent,
            supportLevel: prediction.support_level,
            resistanceLevel: prediction.resistance_level,
            stopLoss: prediction.stop_loss,
            takeProfit: prediction.take_profit,
            timestamp: prediction.timestamp,
            accuracy: prediction.accuracy_score
        };
    }
    
    getDirectionIcon(direction) {
        const icons = {
            'bullish': '📈',
            'bearish': '📉',
            'neutral': '➖'
        };
        return icons[direction] || '❓';
    }
    
    formatCurrency(amount, decimals = 2) {
        return `$${amount.toFixed(decimals)}`;
    }
    
    formatPercentage(percent, decimals = 2) {
        const sign = percent >= 0 ? '+' : '';
        return `${sign}${percent.toFixed(decimals)}%`;
    }
    
    destroy() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        // Clear callbacks
        Object.keys(this.callbacks).forEach(key => {
            this.callbacks[key] = [];
        });
        
        console.log('✅ Advanced ML Client destroyed');
    }
}

// =======================
// Global Instance
// =======================

// Create global instance
window.AdvancedMLClient = AdvancedMLClient;

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {
    if (!window.goldGPTMLClient) {
        window.goldGPTMLClient = new AdvancedMLClient();
    }
});

console.log('✅ Advanced ML Client module loaded');
