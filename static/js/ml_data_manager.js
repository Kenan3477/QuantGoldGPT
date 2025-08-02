/**
 * MLDataManager - Advanced ML Data Management System
 * Handles API communication, WebSocket updates, data processing, and error management
 */

class MLDataManager {
    constructor() {
        this.apiBaseUrl = '/api';
        this.websocket = null;
        this.eventListeners = new Map();
        this.cache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second base delay
        this.isConnected = false;
        this.loadingStates = new Set();
        
        // Data stores
        this.predictions = new Map();
        this.accuracyMetrics = null;
        this.performanceData = null;
        this.featureImportance = null;
        
        console.log('🔄 MLDataManager initializing...');
        this.initialize();
    }

    /**
     * Initialize the data manager
     */
    async initialize() {
        try {
            await this.setupWebSocketConnection();
            await this.loadInitialData();
            this.startPeriodicUpdates();
            
            console.log('✅ MLDataManager initialized successfully');
            this.emit('ready', { timestamp: new Date().toISOString() });
            
        } catch (error) {
            console.error('❌ MLDataManager initialization failed:', error);
            this.emit('error', { type: 'initialization', error: error.message });
        }
    }

    /**
     * Setup WebSocket connection for real-time updates
     */
    async setupWebSocketConnection() {
        try {
            // Check if Socket.IO is available
            if (typeof io === 'undefined') {
                console.warn('⚠️ Socket.IO not available, using polling mode');
                return;
            }

            // Connect to WebSocket
            this.websocket = io('/ml-updates', {
                transports: ['websocket', 'polling'],
                timeout: 5000,
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000
            });

            // Setup event handlers
            this.websocket.on('connect', () => {
                console.log('🔗 ML WebSocket connected');
                this.isConnected = true;
                this.emit('connected', { timestamp: new Date().toISOString() });
            });

            this.websocket.on('disconnect', () => {
                console.log('🔌 ML WebSocket disconnected');
                this.isConnected = false;
                this.emit('disconnected', { timestamp: new Date().toISOString() });
            });

            // ML-specific events
            this.websocket.on('ml_prediction_update', (data) => {
                this.handlePredictionUpdate(data);
            });

            this.websocket.on('ml_accuracy_update', (data) => {
                this.handleAccuracyUpdate(data);
            });

            this.websocket.on('ml_performance_update', (data) => {
                this.handlePerformanceUpdate(data);
            });

            this.websocket.on('ml_error', (data) => {
                this.handleMLError(data);
            });

            // Connection timeout fallback
            setTimeout(() => {
                if (!this.isConnected) {
                    console.warn('⚠️ WebSocket connection timeout, using HTTP polling');
                }
            }, 5000);

        } catch (error) {
            console.error('❌ WebSocket setup failed:', error);
        }
    }

    /**
     * Load initial data from API
     */
    async loadInitialData() {
        console.log('📊 Loading initial ML data...');
        
        const loadPromises = [
            this.fetchPredictions(['15m', '1h', '4h', '24h']).catch(e => console.error('Predictions load failed:', e)),
            this.fetchAccuracyMetrics('7d').catch(e => console.error('Accuracy load failed:', e)),
            this.fetchPerformanceData().catch(e => console.error('Performance load failed:', e)),
            this.fetchFeatureImportance().catch(e => console.error('Features load failed:', e))
        ];

        try {
            const results = await Promise.allSettled(loadPromises);
            console.log('📊 Initial ML data loading results:', results);
            
            // Force emit events for initial data even if cached
            const predictions = this.getPredictions();
            if (predictions && Object.keys(predictions).length > 0) {
                this.emit('predictions_updated', { predictions });
            }
            
            const accuracy = this.getAccuracyMetrics();
            if (accuracy) {
                this.emit('accuracy_updated', { metrics: accuracy });
            }
            
            const performance = this.getPerformanceData();
            if (performance) {
                this.emit('performance_updated', { performance });
            }
            
            const features = this.getFeatureImportance();
            if (features) {
                this.emit('features_updated', { features });
            }
            
            console.log('✅ Initial ML data loaded and events emitted');
        } catch (error) {
            console.error('❌ Failed to load initial data:', error);
        }
    }

    /**
     * Start periodic data updates
     */
    startPeriodicUpdates() {
        // Update predictions every 30 seconds
        setInterval(() => {
            if (!this.isConnected) {
                this.fetchPredictions(['15m', '1h', '4h', '24h']);
            }
        }, 30000);

        // Update accuracy metrics every 5 minutes
        setInterval(() => {
            if (!this.isConnected) {
                this.fetchAccuracyMetrics('7d');
            }
        }, 300000);

        // Update performance data every 2 minutes
        setInterval(() => {
            if (!this.isConnected) {
                this.fetchPerformanceData();
            }
        }, 120000);
    }

    /**
     * Fetch ML predictions from API
     */
    async fetchPredictions(timeframes, forceRefresh = false) {
        const cacheKey = `predictions_${timeframes.join('_')}`;
        
        // Check cache first
        if (!forceRefresh && this.isCacheValid(cacheKey)) {
            const cachedData = this.cache.get(cacheKey);
            this.processPredictions(cachedData.data);
            return cachedData.data;
        }

        this.setLoadingState('predictions', true);

        try {
            const response = await this.makeApiRequest('/ml-predictions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ timeframes })
            });

            if (response.success) {
                const predictions = response.predictions || response;
                this.cache.set(cacheKey, {
                    data: predictions,
                    timestamp: Date.now()
                });

                this.processPredictions(predictions);
                this.emit('predictions_updated', { predictions, timeframes });
                
                return predictions;
            } else {
                throw new Error(response.error || 'Failed to fetch predictions');
            }

        } catch (error) {
            console.error('❌ Failed to fetch predictions:', error);
            this.emit('error', { type: 'predictions', error: error.message });
            
            // Return cached data if available
            const cachedData = this.cache.get(cacheKey);
            if (cachedData) {
                return cachedData.data;
            }
            
            // Generate fallback predictions
            return this.generateFallbackPredictions(timeframes);
            
        } finally {
            this.setLoadingState('predictions', false);
        }
    }

    /**
     * Fetch accuracy metrics from API
     */
    async fetchAccuracyMetrics(timeframe = '7d', forceRefresh = false) {
        const cacheKey = `accuracy_${timeframe}`;
        
        if (!forceRefresh && this.isCacheValid(cacheKey)) {
            const cachedData = this.cache.get(cacheKey);
            this.processAccuracyMetrics(cachedData.data);
            return cachedData.data;
        }

        this.setLoadingState('accuracy', true);

        try {
            const response = await this.makeApiRequest(`/ml-accuracy?timeframe=${timeframe}`);

            if (response.success) {
                const metrics = response.metrics || response;
                this.cache.set(cacheKey, {
                    data: metrics,
                    timestamp: Date.now()
                });

                this.processAccuracyMetrics(metrics);
                this.emit('accuracy_updated', { metrics, timeframe });
                
                return metrics;
            } else {
                throw new Error(response.error || 'Failed to fetch accuracy metrics');
            }

        } catch (error) {
            console.error('❌ Failed to fetch accuracy metrics:', error);
            this.emit('error', { type: 'accuracy', error: error.message });
            
            return this.generateFallbackAccuracy(timeframe);
            
        } finally {
            this.setLoadingState('accuracy', false);
        }
    }

    /**
     * Fetch performance data from API
     */
    async fetchPerformanceData(forceRefresh = false) {
        const cacheKey = 'performance';
        
        if (!forceRefresh && this.isCacheValid(cacheKey)) {
            const cachedData = this.cache.get(cacheKey);
            this.processPerformanceData(cachedData.data);
            return cachedData.data;
        }

        this.setLoadingState('performance', true);

        try {
            const response = await this.makeApiRequest('/ml-performance');

            if (response.success) {
                const performance = response.performance || response;
                this.cache.set(cacheKey, {
                    data: performance,
                    timestamp: Date.now()
                });

                this.processPerformanceData(performance);
                this.emit('performance_updated', { performance });
                
                return performance;
            } else {
                throw new Error(response.error || 'Failed to fetch performance data');
            }

        } catch (error) {
            console.error('❌ Failed to fetch performance data:', error);
            this.emit('error', { type: 'performance', error: error.message });
            
            return this.generateFallbackPerformance();
            
        } finally {
            this.setLoadingState('performance', false);
        }
    }

    /**
     * Fetch feature importance data
     */
    async fetchFeatureImportance(forceRefresh = false) {
        const cacheKey = 'feature_importance';
        
        if (!forceRefresh && this.isCacheValid(cacheKey)) {
            const cachedData = this.cache.get(cacheKey);
            this.processFeatureImportance(cachedData.data);
            return cachedData.data;
        }

        this.setLoadingState('features', true);

        try {
            // Feature importance is usually part of predictions
            const predictions = await this.fetchPredictions(['1h'], forceRefresh);
            
            if (predictions && predictions['1h'] && predictions['1h'].features) {
                const features = predictions['1h'].features;
                this.cache.set(cacheKey, {
                    data: features,
                    timestamp: Date.now()
                });

                this.processFeatureImportance(features);
                this.emit('features_updated', { features });
                
                return features;
            }

            throw new Error('No feature importance data available');

        } catch (error) {
            console.error('❌ Failed to fetch feature importance:', error);
            this.emit('error', { type: 'features', error: error.message });
            
            return this.generateFallbackFeatures();
            
        } finally {
            this.setLoadingState('features', false);
        }
    }

    /**
     * Make API request with retry logic
     */
    async makeApiRequest(endpoint, options = {}, attempt = 1) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        
        try {
            const response = await fetch(url, {
                timeout: 10000,
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            if (attempt < this.retryAttempts) {
                console.warn(`⚠️ API request failed (attempt ${attempt}/${this.retryAttempts}):`, error.message);
                
                // Exponential backoff
                const delay = this.retryDelay * Math.pow(2, attempt - 1);
                await new Promise(resolve => setTimeout(resolve, delay));
                
                return this.makeApiRequest(endpoint, options, attempt + 1);
            }

            throw error;
        }
    }

    /**
     * Process predictions data
     */
    processPredictions(predictions) {
        if (!predictions) return;

        Object.entries(predictions).forEach(([timeframe, prediction]) => {
            if (prediction) {
                // Validate and format prediction data
                const processedPrediction = {
                    timeframe,
                    predicted_price: this.formatPrice(prediction.predicted_price),
                    current_price: this.formatPrice(prediction.current_price),
                    change_amount: this.formatChange(prediction.change_amount),
                    change_percent: this.formatPercent(prediction.change_percent),
                    confidence: this.formatConfidence(prediction.confidence),
                    direction: this.validateDirection(prediction.direction),
                    timestamp: prediction.timestamp || new Date().toISOString(),
                    features: this.processFeatures(prediction.features)
                };

                this.predictions.set(timeframe, processedPrediction);
            }
        });
    }

    /**
     * Process accuracy metrics
     */
    processAccuracyMetrics(metrics) {
        if (!metrics) return;

        this.accuracyMetrics = {
            overall_accuracy: this.formatPercent(metrics.overall_accuracy),
            direction_accuracy: this.formatPercent(metrics.direction_accuracy),
            price_accuracy: this.formatPercent(metrics.price_accuracy),
            avg_confidence: this.formatPercent(metrics.avg_confidence),
            trend: this.processTrendData(metrics.trend),
            previous: metrics.previous ? {
                overall_accuracy: this.formatPercent(metrics.previous.overall_accuracy),
                direction_accuracy: this.formatPercent(metrics.previous.direction_accuracy),
                price_accuracy: this.formatPercent(metrics.previous.price_accuracy),
                avg_confidence: this.formatPercent(metrics.previous.avg_confidence)
            } : null,
            timeframe: metrics.timeframe,
            last_updated: metrics.last_updated || new Date().toISOString()
        };
    }

    /**
     * Process performance data
     */
    processPerformanceData(performance) {
        if (!performance) return;

        this.performanceData = {
            total_predictions: this.formatNumber(performance.total_predictions),
            successful_predictions: this.formatNumber(performance.successful_predictions),
            success_rate: this.calculateSuccessRate(performance.total_predictions, performance.successful_predictions),
            avg_response_time: this.formatResponseTime(performance.avg_response_time),
            model_version: performance.model_version || 'Unknown',
            last_training: this.formatDate(performance.last_training),
            health_status: this.validateHealthStatus(performance.health_status),
            last_updated: performance.last_updated || new Date().toISOString()
        };
    }

    /**
     * Process feature importance data
     */
    processFeatureImportance(features) {
        if (!features) return;

        this.featureImportance = Object.entries(features)
            .map(([name, importance]) => ({
                name: this.formatFeatureName(name),
                importance: this.formatPercent(importance * 100),
                raw_value: importance
            }))
            .sort((a, b) => b.raw_value - a.raw_value);
    }

    /**
     * Handle WebSocket prediction updates
     */
    handlePredictionUpdate(data) {
        console.log('📡 Received prediction update:', data);
        
        if (data.timeframe && data.prediction) {
            this.predictions.set(data.timeframe, data.prediction);
            this.emit('prediction_update', data);
        }
    }

    /**
     * Handle WebSocket accuracy updates
     */
    handleAccuracyUpdate(data) {
        console.log('📡 Received accuracy update:', data);
        
        this.processAccuracyMetrics(data);
        this.emit('accuracy_update', data);
    }

    /**
     * Handle WebSocket performance updates
     */
    handlePerformanceUpdate(data) {
        console.log('📡 Received performance update:', data);
        
        this.processPerformanceData(data);
        this.emit('performance_update', data);
    }

    /**
     * Handle ML errors from WebSocket
     */
    handleMLError(data) {
        console.error('📡 Received ML error:', data);
        this.emit('ml_error', data);
    }

    /**
     * Generate fallback predictions when API fails
     */
    generateFallbackPredictions(timeframes) {
        const fallbackPredictions = {};
        const basePrice = 2000;

        timeframes.forEach(timeframe => {
            const change = (Math.random() - 0.5) * this.getTimeframeRange(timeframe);
            
            fallbackPredictions[timeframe] = {
                timeframe,
                predicted_price: basePrice + change,
                current_price: basePrice,
                change_amount: change,
                change_percent: (change / basePrice) * 100,
                confidence: 0.6 + Math.random() * 0.2,
                direction: change > 5 ? 'bullish' : change < -5 ? 'bearish' : 'neutral',
                timestamp: new Date().toISOString(),
                features: this.generateFallbackFeatures(),
                is_fallback: true
            };
        });

        return fallbackPredictions;
    }

    /**
     * Generate fallback accuracy metrics
     */
    generateFallbackAccuracy(timeframe) {
        const baseAccuracy = 70 + Math.random() * 15;
        
        return {
            overall_accuracy: baseAccuracy,
            direction_accuracy: baseAccuracy + Math.random() * 10,
            price_accuracy: baseAccuracy - Math.random() * 10,
            avg_confidence: 75 + Math.random() * 10,
            trend: this.generateFallbackTrend(),
            timeframe,
            is_fallback: true,
            last_updated: new Date().toISOString()
        };
    }

    /**
     * Generate fallback performance data
     */
    generateFallbackPerformance() {
        return {
            total_predictions: 1000 + Math.floor(Math.random() * 500),
            successful_predictions: 700 + Math.floor(Math.random() * 200),
            avg_response_time: 40 + Math.floor(Math.random() * 30),
            model_version: 'v2.1.3',
            last_training: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
            health_status: 'healthy',
            is_fallback: true,
            last_updated: new Date().toISOString()
        };
    }

    /**
     * Generate fallback features
     */
    generateFallbackFeatures() {
        const features = {
            'Technical Indicators': Math.random() * 0.3,
            'Market Sentiment': Math.random() * 0.25,
            'Volume Analysis': Math.random() * 0.2,
            'Price Action': Math.random() * 0.15,
            'Economic Factors': Math.random() * 0.1
        };

        // Normalize to sum to 1
        const total = Object.values(features).reduce((sum, val) => sum + val, 0);
        Object.keys(features).forEach(key => {
            features[key] /= total;
        });

        return features;
    }

    /**
     * Utility methods for data formatting
     */
    formatPrice(price) {
        return typeof price === 'number' ? parseFloat(price.toFixed(2)) : 0;
    }

    formatChange(change) {
        return typeof change === 'number' ? parseFloat(change.toFixed(2)) : 0;
    }

    formatPercent(percent) {
        return typeof percent === 'number' ? parseFloat(percent.toFixed(1)) : 0;
    }

    formatConfidence(confidence) {
        if (typeof confidence !== 'number') return 0.5;
        return Math.max(0, Math.min(1, confidence));
    }

    formatNumber(num) {
        return typeof num === 'number' ? Math.floor(num) : 0;
    }

    formatResponseTime(time) {
        return typeof time === 'number' ? Math.floor(time) : 0;
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        try {
            return new Date(dateString).toLocaleDateString();
        } catch {
            return 'Unknown';
        }
    }

    validateDirection(direction) {
        const validDirections = ['bullish', 'bearish', 'neutral'];
        return validDirections.includes(direction) ? direction : 'neutral';
    }

    validateHealthStatus(status) {
        const validStatuses = ['healthy', 'warning', 'error'];
        return validStatuses.includes(status) ? status : 'warning';
    }

    formatFeatureName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    getTimeframeRange(timeframe) {
        const ranges = {
            '15m': 20,
            '1h': 40,
            '4h': 80,
            '24h': 150
        };
        return ranges[timeframe] || 30;
    }

    calculateSuccessRate(total, successful) {
        if (!total || total === 0) return 0;
        return Math.round((successful / total) * 100);
    }

    processTrendData(trend) {
        if (!Array.isArray(trend)) return [];
        
        return trend.map(item => ({
            date: item.date,
            overall_accuracy: this.formatPercent(item.overall_accuracy),
            direction_accuracy: this.formatPercent(item.direction_accuracy)
        }));
    }

    processFeatures(features) {
        if (!features || typeof features !== 'object') return {};
        
        const processed = {};
        Object.entries(features).forEach(([key, value]) => {
            if (typeof value === 'number') {
                processed[key] = value;
            }
        });
        
        return processed;
    }

    generateFallbackTrend() {
        const trend = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            
            trend.push({
                date: date.toISOString().split('T')[0],
                overall_accuracy: 70 + Math.random() * 15,
                direction_accuracy: 75 + Math.random() * 15
            });
        }
        return trend;
    }

    /**
     * Cache management methods
     */
    isCacheValid(key) {
        const cached = this.cache.get(key);
        if (!cached) return false;
        
        return (Date.now() - cached.timestamp) < this.cacheTimeout;
    }

    clearCache(pattern = null) {
        if (pattern) {
            for (const [key] of this.cache) {
                if (key.includes(pattern)) {
                    this.cache.delete(key);
                }
            }
        } else {
            this.cache.clear();
        }
    }

    /**
     * Loading state management
     */
    setLoadingState(operation, isLoading) {
        if (isLoading) {
            this.loadingStates.add(operation);
        } else {
            this.loadingStates.delete(operation);
        }
        
        this.emit('loading_state_changed', {
            operation,
            isLoading,
            activeOperations: Array.from(this.loadingStates)
        });
    }

    isLoading(operation = null) {
        if (operation) {
            return this.loadingStates.has(operation);
        }
        return this.loadingStates.size > 0;
    }

    /**
     * Event system methods
     */
    on(eventName, callback) {
        if (!this.eventListeners.has(eventName)) {
            this.eventListeners.set(eventName, []);
        }
        this.eventListeners.get(eventName).push(callback);
    }

    off(eventName, callback) {
        if (this.eventListeners.has(eventName)) {
            const callbacks = this.eventListeners.get(eventName);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(eventName, data = {}) {
        if (this.eventListeners.has(eventName)) {
            this.eventListeners.get(eventName).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Public API methods
     */
    
    // Get current predictions
    getPredictions(timeframe = null) {
        if (timeframe) {
            return this.predictions.get(timeframe);
        }
        return Object.fromEntries(this.predictions);
    }

    // Get accuracy metrics
    getAccuracyMetrics() {
        return this.accuracyMetrics;
    }

    // Get performance data
    getPerformanceData() {
        return this.performanceData;
    }

    // Get feature importance
    getFeatureImportance() {
        return this.featureImportance;
    }

    // Force refresh all data
    async refreshAll() {
        console.log('🔄 Refreshing all ML data...');
        
        await Promise.allSettled([
            this.fetchPredictions(['15m', '1h', '4h', '24h'], true),
            this.fetchAccuracyMetrics('7d', true),
            this.fetchPerformanceData(true),
            this.fetchFeatureImportance(true)
        ]);
        
        this.emit('refresh_complete', { timestamp: new Date().toISOString() });
    }

    // Get connection status
    getConnectionStatus() {
        return {
            websocket: this.isConnected,
            loading: this.isLoading(),
            loadingOperations: Array.from(this.loadingStates),
            cacheSize: this.cache.size,
            lastUpdate: this.performanceData?.last_updated || null
        };
    }

    // Cleanup
    destroy() {
        if (this.websocket) {
            this.websocket.disconnect();
        }
        
        this.cache.clear();
        this.eventListeners.clear();
        this.loadingStates.clear();
        
        console.log('🧹 MLDataManager destroyed');
    }
}

// Global instance
window.MLDataManager = new MLDataManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLDataManager;
}

console.log('📊 MLDataManager loaded');
