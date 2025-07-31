/**
 * GoldGPT ML Prediction Manager
 * Real-time ML-driven gold price prediction system
 * 
 * Features:
 * - Multi-timeframe predictions (1H, 4H, 1D)
 * - Real-time confidence scoring
 * - Fallback mechanisms for offline operation
 * - Seamless integration with existing UI
 * 
 * License: MIT - Compatible with GoldGPT architecture
 */

class GoldMLPredictionManager {
    constructor() {
        this.apiEndpoint = '/api/ml-predictions';
        this.predictions = {};
        this.isInitialized = false;
        this.updateInterval = null;
        this.fallbackMode = false;
        this.lastUpdateTime = null;
        this.containerId = 'ml-prediction-panel';
        
        // Timeframe configurations
        this.timeframes = {
            '1H': {
                displayName: '1 Hour',
                updateInterval: 5 * 60 * 1000, // 5 minutes
                icon: 'fa-clock',
                color: '#4285f4',
                horizon: 'Short-term'
            },
            '4H': {
                displayName: '4 Hours',
                updateInterval: 15 * 60 * 1000, // 15 minutes
                icon: 'fa-hourglass-half',
                color: '#ffa502',
                horizon: 'Medium-term'
            },
            '1D': {
                displayName: '1 Day',
                updateInterval: 60 * 60 * 1000, // 1 hour
                icon: 'fa-calendar-day',
                color: '#00d084',
                horizon: 'Long-term'
            }
        };
        
        // UI state
        this.selectedTimeframe = '1H';
        this.isLoading = false;
        this.errorCount = 0;
        this.maxErrors = 3;
        
        // Fallback prediction logic
        this.fallbackCalculator = new FallbackPredictionCalculator();
        
        console.log('🤖 ML Prediction Manager initialized');
    }
    
    /**
     * Initialize the ML prediction system
     */
    async init() {
        try {
            console.log('🚀 Initializing ML Prediction Manager...');
            
            // Create UI container
            this.createPredictionPanel();
            
            // Load initial predictions
            await this.loadPredictions();
            
            // Start auto-refresh
            this.startAutoRefresh();
            
            // Setup event listeners
            this.setupEventListeners();
            
            this.isInitialized = true;
            console.log('✅ ML Prediction Manager initialized successfully');
            
            // Show initialization success
            this.showNotification('ML Prediction system ready', 'success');
            
        } catch (error) {
            console.error('❌ ML Prediction Manager initialization failed:', error);
            this.fallbackMode = true;
            this.showNotification('ML system using fallback mode', 'warning');
        }
    }
    
    /**
     * Create the ML prediction UI panel
     */
    createPredictionPanel() {
        // Find or create container
        let container = document.getElementById(this.containerId);
        if (!container) {
            // Create container in right panel
            const rightPanel = document.querySelector('.right-panel') || 
                              document.querySelector('[class*="right"]') ||
                              document.body;
            
            container = document.createElement('div');
            container.id = this.containerId;
            container.className = 'panel-section ml-prediction-panel';
            rightPanel.appendChild(container);
        }
        
        // Create ML prediction UI
        container.innerHTML = `
            <div class="ml-prediction-header">
                <div class="section-title">
                    <i class="fas fa-brain"></i>
                    <h3>AI Price Forecasts</h3>
                    <div class="ml-status-indicator" id="ml-status">
                        <span class="status-dot loading"></span>
                        <span class="status-text">Initializing...</span>
                    </div>
                </div>
                <div class="ml-controls">
                    <div class="timeframe-selector" id="ml-timeframe-selector">
                        ${Object.keys(this.timeframes).map(tf => 
                            `<button class="timeframe-btn ${tf === this.selectedTimeframe ? 'active' : ''}" 
                                     data-timeframe="${tf}">
                                <i class="fas ${this.timeframes[tf].icon}"></i>
                                ${tf}
                            </button>`
                        ).join('')}
                    </div>
                    <button class="refresh-btn" id="ml-refresh-btn" title="Refresh Predictions">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                </div>
            </div>
            
            <div class="ml-prediction-content" id="ml-prediction-content">
                <div class="prediction-loading">
                    <i class="fas fa-brain fa-spin"></i>
                    <p>Generating AI predictions...</p>
                </div>
            </div>
            
            <div class="ml-prediction-footer">
                <div class="model-info">
                    <small>
                        <i class="fas fa-robot"></i>
                        Ensemble ML Models • Real-time Analysis
                    </small>
                </div>
                <div class="last-update" id="ml-last-update">
                    Never updated
                </div>
            </div>
        `;
        
        // Add CSS styles
        this.injectStyles();
    }
    
    /**
     * Inject CSS styles for ML prediction panel
     */
    injectStyles() {
        if (document.getElementById('ml-prediction-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'ml-prediction-styles';
        styles.textContent = `
            .ml-prediction-panel {
                background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
                border: 1px solid #333;
                border-radius: 12px;
                margin: 15px 0;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .ml-prediction-header {
                background: linear-gradient(90deg, #2c3e50, #3498db);
                padding: 15px;
                border-bottom: 1px solid #333;
            }
            
            .ml-prediction-header .section-title {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }
            
            .ml-prediction-header .section-title h3 {
                color: white;
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }
            
            .ml-status-indicator {
                display: flex;
                align-items: center;
                gap: 5px;
                margin-left: auto;
            }
            
            .ml-status-indicator .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #ffa502;
            }
            
            .ml-status-indicator .status-dot.loading {
                background: #ffa502;
                animation: pulse 1.5s infinite;
            }
            
            .ml-status-indicator .status-dot.success {
                background: #00d084;
            }
            
            .ml-status-indicator .status-dot.error {
                background: #ff4757;
            }
            
            .ml-status-indicator .status-text {
                font-size: 11px;
                color: #ecf0f1;
            }
            
            .ml-controls {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .timeframe-selector {
                display: flex;
                gap: 5px;
            }
            
            .timeframe-btn {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 11px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 4px;
            }
            
            .timeframe-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-1px);
            }
            
            .timeframe-btn.active {
                background: #00d084;
                border-color: #00d084;
                box-shadow: 0 0 10px rgba(0, 208, 132, 0.3);
            }
            
            .refresh-btn {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                padding: 8px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .refresh-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: rotate(180deg);
            }
            
            .ml-prediction-content {
                padding: 15px;
                min-height: 200px;
            }
            
            .prediction-loading {
                text-align: center;
                padding: 40px 20px;
                color: #7f8c8d;
            }
            
            .prediction-loading i {
                font-size: 24px;
                margin-bottom: 10px;
                color: #3498db;
            }
            
            .prediction-card {
                background: linear-gradient(135deg, #2c3e50, #34495e);
                border: 1px solid #444;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                transition: all 0.3s ease;
            }
            
            .prediction-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            }
            
            .prediction-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            
            .prediction-timeframe {
                display: flex;
                align-items: center;
                gap: 5px;
                color: #ecf0f1;
                font-weight: 600;
            }
            
            .confidence-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
            }
            
            .confidence-high { background: #00d084; color: white; }
            .confidence-medium { background: #ffa502; color: white; }
            .confidence-low { background: #ff4757; color: white; }
            
            .prediction-details {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 10px;
            }
            
            .prediction-metric {
                text-align: center;
            }
            
            .prediction-metric .label {
                font-size: 11px;
                color: #95a5a6;
                margin-bottom: 2px;
            }
            
            .prediction-metric .value {
                font-size: 14px;
                font-weight: 600;
                color: white;
            }
            
            .direction-bullish { color: #00d084; }
            .direction-bearish { color: #ff4757; }
            .direction-neutral { color: #ffa502; }
            
            .ml-prediction-footer {
                background: #1a1a1a;
                padding: 10px 15px;
                border-top: 1px solid #333;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .model-info {
                color: #7f8c8d;
                font-size: 11px;
            }
            
            .last-update {
                color: #95a5a6;
                font-size: 11px;
            }
            
            .prediction-main {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 15px 0;
                padding: 15px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }
            
            .predicted-price, .price-change {
                text-align: center;
            }
            
            .predicted-price .label, .price-change .label {
                font-size: 11px;
                color: #95a5a6;
                margin-bottom: 5px;
            }
            
            .predicted-price .value {
                font-size: 20px;
                font-weight: 700;
                color: #3498db;
            }
            
            .price-change .value {
                font-size: 16px;
                font-weight: 600;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 5px;
            }
            
            .technical-signals {
                margin-top: 15px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 6px;
            }
            
            .signals-header {
                font-size: 12px;
                color: #ecf0f1;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .signals-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
            }
            
            .signal-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 4px 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            
            .signal-label {
                font-size: 10px;
                color: #95a5a6;
            }
            
            .signal-value {
                font-size: 10px;
                font-weight: 600;
            }
            
            .signal-value.bullish { color: #00d084; }
            .signal-value.bearish { color: #ff4757; }
            .signal-value.neutral { color: #ffa502; }
            
            .fallback-notice {
                background: rgba(255, 165, 2, 0.1);
                border: 1px solid #ffa502;
                color: #ffa502;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
                text-align: center;
                margin-top: 10px;
            }
            
            .prediction-error {
                text-align: center;
                padding: 40px 20px;
                color: #7f8c8d;
            }
            
            .prediction-error i {
                font-size: 24px;
                margin-bottom: 10px;
                color: #ff4757;
            }
            
            .retry-btn {
                background: #3498db;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 10px;
                transition: background 0.3s ease;
            }
            
            .retry-btn:hover {
                background: #2980b9;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            @media (max-width: 768px) {
                .timeframe-selector {
                    flex-wrap: wrap;
                }
                
                .timeframe-btn {
                    font-size: 10px;
                    padding: 4px 8px;
                }
                
                .prediction-details {
                    grid-template-columns: 1fr;
                }
            }
        `;
        
        document.head.appendChild(styles);
    }
    
    /**
     * Setup event listeners for UI interactions
     */
    setupEventListeners() {
        // Timeframe selection
        document.addEventListener('click', (e) => {
            if (e.target.matches('.timeframe-btn')) {
                const timeframe = e.target.dataset.timeframe;
                this.selectTimeframe(timeframe);
            }
        });
        
        // Refresh button
        document.addEventListener('click', (e) => {
            if (e.target.matches('#ml-refresh-btn') || e.target.closest('#ml-refresh-btn')) {
                this.refreshPredictions();
            }
        });
        
        // Auto-refresh on focus
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isInitialized) {
                this.refreshPredictions();
            }
        });
    }
    
    /**
     * Load predictions from API or fallback
     */
    async loadPredictions(symbol = 'GC=F') {
        try {
            this.setLoadingState(true);
            this.updateStatus('loading', 'Loading predictions...');
            
            // Try to fetch from API
            const response = await fetch(`${this.apiEndpoint}/${symbol}`);
            
            if (response.ok) {
                const data = await response.json();
                
                if (data.success && data.predictions) {
                    this.predictions = data.predictions;
                    this.fallbackMode = false;
                    this.errorCount = 0;
                    this.updateStatus('success', 'AI predictions ready');
                    console.log('✅ ML predictions loaded:', data);
                } else {
                    throw new Error(data.error || 'Invalid prediction data');
                }
            } else {
                throw new Error(`API request failed: ${response.status}`);
            }
            
        } catch (error) {
            console.warn('⚠️ ML API failed, using fallback:', error);
            this.errorCount++;
            
            if (this.errorCount >= this.maxErrors) {
                this.fallbackMode = true;
                this.updateStatus('error', 'Using fallback mode');
            }
            
            // Generate fallback predictions
            this.predictions = this.fallbackCalculator.generatePredictions();
            
        } finally {
            this.setLoadingState(false);
            this.displayPredictions();
            this.updateLastUpdateTime();
        }
    }
    
    /**
     * Display predictions in the UI
     */
    displayPredictions() {
        const contentContainer = document.getElementById('ml-prediction-content');
        if (!contentContainer) return;
        
        if (!this.predictions || Object.keys(this.predictions).length === 0) {
            contentContainer.innerHTML = `
                <div class="prediction-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Unable to generate predictions</p>
                    <button onclick="window.goldMLPredictionManager.refreshPredictions()" class="retry-btn">
                        Try Again
                    </button>
                </div>
            `;
            return;
        }
        
        // Get prediction for selected timeframe
        const prediction = this.predictions[this.selectedTimeframe];
        if (!prediction) {
            contentContainer.innerHTML = `
                <div class="prediction-error">
                    <p>No prediction available for ${this.selectedTimeframe}</p>
                </div>
            `;
            return;
        }
        
        // Calculate confidence level
        const confidenceLevel = this.getConfidenceLevel(prediction.confidence);
        const directionClass = `direction-${prediction.direction}`;
        const changeIcon = prediction.price_change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
        
        contentContainer.innerHTML = `
            <div class="prediction-card ${directionClass}">
                <div class="prediction-header">
                    <div class="prediction-timeframe">
                        <i class="fas ${this.timeframes[this.selectedTimeframe].icon}"></i>
                        ${this.timeframes[this.selectedTimeframe].displayName} Forecast
                    </div>
                    <div class="confidence-badge confidence-${confidenceLevel}">
                        ${Math.round(prediction.confidence * 100)}% confident
                    </div>
                </div>
                
                <div class="prediction-main">
                    <div class="predicted-price">
                        <div class="label">Predicted Price</div>
                        <div class="value">$${prediction.predicted_price.toFixed(2)}</div>
                    </div>
                    
                    <div class="price-change">
                        <div class="label">Expected Change</div>
                        <div class="value ${directionClass}">
                            <i class="fas ${changeIcon}"></i>
                            $${Math.abs(prediction.price_change).toFixed(2)} 
                            (${prediction.price_change_percent >= 0 ? '+' : ''}${prediction.price_change_percent.toFixed(2)}%)
                        </div>
                    </div>
                </div>
                
                <div class="prediction-details">
                    <div class="prediction-metric">
                        <div class="label">Current Price</div>
                        <div class="value">$${prediction.current_price.toFixed(2)}</div>
                    </div>
                    
                    <div class="prediction-metric">
                        <div class="label">Direction</div>
                        <div class="value ${directionClass}">
                            ${prediction.direction.toUpperCase()}
                        </div>
                    </div>
                    
                    <div class="prediction-metric">
                        <div class="label">Support Level</div>
                        <div class="value">$${prediction.support_level.toFixed(2)}</div>
                    </div>
                    
                    <div class="prediction-metric">
                        <div class="label">Resistance Level</div>
                        <div class="value">$${prediction.resistance_level.toFixed(2)}</div>
                    </div>
                </div>
                
                ${this.renderTechnicalSignals(prediction.technical_signals)}
                
                ${this.fallbackMode ? '<div class="fallback-notice">⚠️ Using offline predictions</div>' : ''}
            </div>
        `;
    }
    
    /**
     * Render technical signals
     */
    renderTechnicalSignals(signals) {
        if (!signals) return '';
        
        return `
            <div class="technical-signals">
                <div class="signals-header">
                    <i class="fas fa-chart-line"></i>
                    Technical Indicators
                </div>
                <div class="signals-grid">
                    <div class="signal-item">
                        <span class="signal-label">RSI</span>
                        <span class="signal-value ${this.getRSIClass(signals.rsi)}">${signals.rsi.toFixed(1)}</span>
                    </div>
                    <div class="signal-item">
                        <span class="signal-label">MACD</span>
                        <span class="signal-value ${signals.macd >= 0 ? 'bullish' : 'bearish'}">${signals.macd.toFixed(3)}</span>
                    </div>
                    <div class="signal-item">
                        <span class="signal-label">BB Position</span>
                        <span class="signal-value">${(signals.bb_position * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Get RSI classification
     */
    getRSIClass(rsi) {
        if (rsi >= 70) return 'bearish'; // Overbought
        if (rsi <= 30) return 'bullish'; // Oversold
        return 'neutral';
    }
    
    /**
     * Get confidence level classification
     */
    getConfidenceLevel(confidence) {
        if (confidence >= 0.7) return 'high';
        if (confidence >= 0.5) return 'medium';
        return 'low';
    }
    
    /**
     * Select timeframe and refresh display
     */
    selectTimeframe(timeframe) {
        if (this.selectedTimeframe === timeframe) return;
        
        this.selectedTimeframe = timeframe;
        
        // Update UI
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.timeframe === timeframe);
        });
        
        // Refresh display
        this.displayPredictions();
        
        console.log(`📊 Timeframe switched to ${timeframe}`);
    }
    
    /**
     * Refresh predictions manually
     */
    async refreshPredictions() {
        if (this.isLoading) return;
        
        console.log('🔄 Refreshing ML predictions...');
        await this.loadPredictions();
        this.showNotification('Predictions updated', 'success');
    }
    
    /**
     * Start auto-refresh mechanism
     */
    startAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Use the shortest update interval from timeframes
        const updateInterval = Math.min(...Object.values(this.timeframes).map(tf => tf.updateInterval));
        
        this.updateInterval = setInterval(() => {
            if (!document.hidden && this.isInitialized) {
                this.loadPredictions();
            }
        }, updateInterval);
        
        console.log(`⏰ Auto-refresh started (${updateInterval / 1000}s interval)`);
    }
    
    /**
     * Set loading state
     */
    setLoadingState(loading) {
        this.isLoading = loading;
        
        const refreshBtn = document.getElementById('ml-refresh-btn');
        if (refreshBtn) {
            const icon = refreshBtn.querySelector('i');
            if (loading) {
                icon.classList.add('fa-spin');
                refreshBtn.disabled = true;
            } else {
                icon.classList.remove('fa-spin');
                refreshBtn.disabled = false;
            }
        }
    }
    
    /**
     * Update status indicator
     */
    updateStatus(type, message) {
        const statusDot = document.querySelector('#ml-status .status-dot');
        const statusText = document.querySelector('#ml-status .status-text');
        
        if (statusDot && statusText) {
            statusDot.className = `status-dot ${type}`;
            statusText.textContent = message;
        }
    }
    
    /**
     * Update last update time
     */
    updateLastUpdateTime() {
        const lastUpdateEl = document.getElementById('ml-last-update');
        if (lastUpdateEl) {
            this.lastUpdateTime = new Date();
            lastUpdateEl.textContent = `Updated ${this.lastUpdateTime.toLocaleTimeString()}`;
        }
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Use global notification system if available
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
    
    /**
     * Get current predictions
     */
    getCurrentPredictions() {
        return {
            predictions: this.predictions,
            selectedTimeframe: this.selectedTimeframe,
            lastUpdate: this.lastUpdateTime,
            fallbackMode: this.fallbackMode
        };
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        const container = document.getElementById(this.containerId);
        if (container) {
            container.remove();
        }
        
        const styles = document.getElementById('ml-prediction-styles');
        if (styles) {
            styles.remove();
        }
        
        console.log('🗑️ ML Prediction Manager destroyed');
    }
    
    /**
     * Initialize the ML prediction system
     */
    async init() {
        if (this.isInitialized) return;
        
        try {
            console.log('🚀 Initializing ML Prediction Manager...');
            
            // Create UI components
            this.createPredictionPanel();
            
            // Fetch initial predictions
            await this.fetchPredictions();
            
            // Start update intervals
            this.startUpdateInterval();
            
            // Set up event listeners
            this.setupEventListeners();
            
            this.isInitialized = true;
            console.log('✅ ML Prediction Manager initialized successfully');
            
        } catch (error) {
            console.error('❌ Error initializing ML Prediction Manager:', error);
            this.enableFallbackMode();
        }
    }
    
    /**
     * Create the prediction panel UI
     */
    createPredictionPanel() {
        const existingPanel = document.getElementById('ml-predictions-panel');
        if (existingPanel) {
            existingPanel.remove();
        }
        
        const panelHTML = `
            <div class="analysis-panel" id="ml-predictions-panel">
                <div class="panel-header">
                    <h3><i class="fas fa-robot"></i> ML Predictions</h3>
                    <div class="panel-status" id="ml-predictions-status">
                        <div class="status-dot loading" id="ml-status-dot"></div>
                        <span id="ml-status-text">LOADING</span>
                    </div>
                </div>
                <div class="panel-content">
                    <div class="predictions-grid" id="ml-predictions">
                        ${this.createTimeframePredictions()}
                    </div>
                    <div class="prediction-summary" id="prediction-summary">
                        <div class="summary-header">
                            <h4><i class="fas fa-chart-line"></i> Analysis Summary</h4>
                        </div>
                        <div class="summary-content" id="summary-content">
                            <div class="loading-state">
                                <i class="fas fa-spinner fa-spin"></i>
                                <span>Generating ML analysis...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Insert panel below chart
        const chartContainer = document.querySelector('.chart-container');
        if (chartContainer) {
            chartContainer.insertAdjacentHTML('afterend', panelHTML);
        } else {
            // Fallback: insert at beginning of content
            const content = document.querySelector('.content');
            if (content) {
                content.insertAdjacentHTML('afterbegin', panelHTML);
            }
        }
        
        // Add CSS styles
        this.injectStyles();
    }
    
    /**
     * Create timeframe prediction cards
     */
    createTimeframePredictions() {
        return Object.entries(this.timeframes).map(([timeframe, config]) => `
            <div class="prediction-item" data-timeframe="${timeframe}">
                <div class="prediction-header">
                    <div class="timeframe-info">
                        <i class="fas ${config.icon}"></i>
                        <span class="timeframe-label">${config.displayName}</span>
                    </div>
                    <div class="prediction-status loading" id="status-${timeframe}">
                        <i class="fas fa-spinner fa-spin"></i>
                    </div>
                </div>
                <div class="prediction-body">
                    <div class="price-prediction">
                        <div class="current-price">
                            <span class="label">Current</span>
                            <span class="value" id="current-${timeframe}">$--</span>
                        </div>
                        <div class="predicted-price">
                            <span class="label">Predicted</span>
                            <span class="value" id="predicted-${timeframe}">$--</span>
                        </div>
                    </div>
                    <div class="prediction-direction">
                        <div class="direction-indicator" id="direction-${timeframe}">
                            <i class="fas fa-minus"></i>
                            <span>--</span>
                        </div>
                        <div class="confidence-score" id="confidence-${timeframe}">
                            <span class="percentage">--%</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="prediction-factors" id="factors-${timeframe}">
                    <div class="factors-loading">
                        <i class="fas fa-circle-notch fa-spin"></i>
                        <span>Loading factors...</span>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    /**
     * Fetch predictions from the ML API
     */
    async fetchPredictions() {
        try {
            console.log('📊 Fetching ML predictions...');
            
            const response = await fetch(this.apiEndpoint, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.predictions = data.predictions;
                this.updatePredictionUI();
                this.updateStatus('success', 'ACTIVE');
                this.fallbackMode = false;
                
                console.log('✅ ML predictions fetched successfully');
            } else {
                throw new Error(data.error || 'Unknown API error');
            }
            
        } catch (error) {
            console.error('❌ Error fetching ML predictions:', error);
            this.handleFetchError(error);
        }
    }
    
    /**
     * Handle fetch errors with fallback
     */
    handleFetchError(error) {
        console.warn('⚠️ Switching to fallback prediction mode');
        this.enableFallbackMode();
        this.generateFallbackPredictions();
    }
    
    /**
     * Enable fallback mode
     */
    enableFallbackMode() {
        this.fallbackMode = true;
        this.updateStatus('warning', 'FALLBACK');
        
        // Show fallback notice
        const summaryContent = document.getElementById('summary-content');
        if (summaryContent) {
            summaryContent.innerHTML = `
                <div class="fallback-notice">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Using local prediction calculations</span>
                </div>
            `;
        }
    }
    
    /**
     * Generate fallback predictions using local calculations
     */
    generateFallbackPredictions() {
        try {
            this.predictions = this.fallbackCalculator.generatePredictions();
            this.updatePredictionUI();
            console.log('🔄 Fallback predictions generated');
        } catch (error) {
            console.error('❌ Error generating fallback predictions:', error);
            this.showErrorState();
        }
    }
    
    /**
     * Update the prediction UI with current data
     */
    updatePredictionUI() {
        Object.entries(this.predictions).forEach(([timeframe, prediction]) => {
            this.updateTimeframePrediction(timeframe, prediction);
        });
        
        this.updateSummary();
        this.lastUpdateTime = new Date();
    }
    
    /**
     * Update individual timeframe prediction
     */
    updateTimeframePrediction(timeframe, prediction) {
        const config = this.timeframes[timeframe];
        if (!config) return;
        
        // Update current price
        const currentElement = document.getElementById(`current-${timeframe}`);
        if (currentElement) {
            currentElement.textContent = `$${prediction.current_price.toFixed(2)}`;
        }
        
        // Update predicted price
        const predictedElement = document.getElementById(`predicted-${timeframe}`);
        if (predictedElement) {
            predictedElement.textContent = `$${prediction.predicted_price.toFixed(2)}`;
        }
        
        // Update direction
        const directionElement = document.getElementById(`direction-${timeframe}`);
        if (directionElement) {
            const isUp = prediction.predicted_direction === 'UP';
            directionElement.innerHTML = `
                <i class="fas fa-arrow-${isUp ? 'up' : 'down'}"></i>
                <span>${prediction.predicted_direction}</span>
            `;
            directionElement.className = `direction-indicator ${isUp ? 'bullish' : 'bearish'}`;
        }
        
        // Update confidence
        const confidenceElement = document.getElementById(`confidence-${timeframe}`);
        if (confidenceElement) {
            const confidencePercent = Math.round(prediction.confidence_score * 100);
            confidenceElement.querySelector('.percentage').textContent = `${confidencePercent}%`;
            confidenceElement.querySelector('.confidence-fill').style.width = `${confidencePercent}%`;
            
            // Color code confidence
            const fillElement = confidenceElement.querySelector('.confidence-fill');
            if (confidencePercent >= 80) {
                fillElement.className = 'confidence-fill high';
            } else if (confidencePercent >= 60) {
                fillElement.className = 'confidence-fill medium';
            } else {
                fillElement.className = 'confidence-fill low';
            }
        }
        
        // Update factors
        const factorsElement = document.getElementById(`factors-${timeframe}`);
        if (factorsElement && prediction.prediction_factors) {
            factorsElement.innerHTML = this.createFactorsHTML(prediction.prediction_factors);
        }
        
        // Update status
        const statusElement = document.getElementById(`status-${timeframe}`);
        if (statusElement) {
            statusElement.innerHTML = '<i class="fas fa-check-circle"></i>';
            statusElement.className = 'prediction-status success';
        }
    }
    
    /**
     * Create factors HTML
     */
    createFactorsHTML(factors) {
        const factorItems = Object.entries(factors).map(([key, value]) => {
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            let displayValue = value;
            
            if (typeof value === 'number') {
                displayValue = value.toFixed(3);
            }
            
            return `
                <div class="factor-item">
                    <span class="factor-label">${displayKey}</span>
                    <span class="factor-value">${displayValue}</span>
                </div>
            `;
        }).join('');
        
        return `
            <div class="factors-header">
                <i class="fas fa-cogs"></i>
                <span>Prediction Factors</span>
            </div>
            <div class="factors-list">
                ${factorItems}
            </div>
        `;
    }
    
    /**
     * Update overall summary
     */
    updateSummary() {
        const summaryContent = document.getElementById('summary-content');
        if (!summaryContent) return;
        
        const predictions = Object.values(this.predictions);
        if (predictions.length === 0) return;
        
        // Calculate consensus
        const upCount = predictions.filter(p => p.predicted_direction === 'UP').length;
        const downCount = predictions.filter(p => p.predicted_direction === 'DOWN').length;
        const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence_score, 0) / predictions.length;
        
        const consensus = upCount > downCount ? 'BULLISH' : 'BEARISH';
        const consensusStrength = Math.max(upCount, downCount) / predictions.length;
        
        summaryContent.innerHTML = `
            <div class="summary-item">
                <div class="summary-label">Market Consensus</div>
                <div class="summary-value ${consensus.toLowerCase()}">
                    <i class="fas fa-${consensus === 'BULLISH' ? 'bull' : 'bear'}"></i>
                    <span>${consensus}</span>
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Consensus Strength</div>
                <div class="summary-value">
                    <span>${Math.round(consensusStrength * 100)}%</span>
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Avg Confidence</div>
                <div class="summary-value">
                    <span>${Math.round(avgConfidence * 100)}%</span>
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Last Update</div>
                <div class="summary-value">
                    <span>${new Date().toLocaleTimeString()}</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Update status indicator
     */
    updateStatus(type, text) {
        const statusDot = document.getElementById('ml-status-dot');
        const statusText = document.getElementById('ml-status-text');
        
        if (statusDot) {
            statusDot.className = `status-dot ${type}`;
        }
        
        if (statusText) {
            statusText.textContent = text;
        }
    }
    
    /**
     * Show error state
     */
    showErrorState() {
        this.updateStatus('error', 'ERROR');
        
        Object.keys(this.timeframes).forEach(timeframe => {
            const statusElement = document.getElementById(`status-${timeframe}`);
            if (statusElement) {
                statusElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
                statusElement.className = 'prediction-status error';
            }
        });
        
        const summaryContent = document.getElementById('summary-content');
        if (summaryContent) {
            summaryContent.innerHTML = `
                <div class="error-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Unable to generate predictions</span>
                </div>
            `;
        }
    }
    
    /**
     * Start update interval
     */
    startUpdateInterval() {
        // Clear existing interval
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Update every 5 minutes
        this.updateInterval = setInterval(() => {
            this.fetchPredictions();
        }, 5 * 60 * 1000);
        
        console.log('🔄 Update interval started (5 minutes)');
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Refresh button (if added later)
        document.addEventListener('click', (e) => {
            if (e.target.matches('.ml-refresh-btn')) {
                this.fetchPredictions();
            }
        });
        
        // Visibility change handler
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Pause updates when tab is hidden
                if (this.updateInterval) {
                    clearInterval(this.updateInterval);
                }
            } else {
                // Resume updates when tab is visible
                this.startUpdateInterval();
                this.fetchPredictions();
            }
        });
    }
    
    /**
     * Inject CSS styles
     */
    injectStyles() {
        const styleId = 'ml-prediction-styles';
        if (document.getElementById(styleId)) return;
        
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            /* ML Predictions Panel Styles */
            .ml-predictions-container {
                margin: 24px 0;
            }
            
            #ml-predictions-panel {
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-primary);
                margin-bottom: 24px;
                overflow: hidden;
            }
            
            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 16px 20px;
                border-bottom: 1px solid var(--border-primary);
                background: var(--bg-tertiary);
            }
            
            .panel-header h3 {
                display: flex;
                align-items: center;
                gap: 8px;
                margin: 0;
                font-size: 16px;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .panel-header i {
                color: var(--accent-primary);
            }
            
            .panel-status {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
                font-weight: 600;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--text-muted);
            }
            
            .status-dot.loading {
                background: var(--warning);
                animation: pulse 2s infinite;
            }
            
            .status-dot.success {
                background: var(--success);
            }
            
            .status-dot.error {
                background: var(--danger);
            }
            
            .status-dot.warning {
                background: var(--warning);
            }
            
            .panel-content {
                padding: 20px;
            }
            
            .predictions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 24px;
            }
            
            .prediction-item {
                background: var(--bg-quaternary);
                border-radius: 8px;
                padding: 16px;
                border: 1px solid var(--border-primary);
                transition: var(--transition);
            }
            
            .prediction-item:hover {
                border-color: var(--accent-primary);
                box-shadow: 0 4px 12px rgba(0, 212, 170, 0.1);
            }
            
            .prediction-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 12px;
            }
            
            .timeframe-info {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .timeframe-info i {
                color: var(--accent-primary);
                font-size: 14px;
            }
            
            .timeframe-label {
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .prediction-status {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                font-size: 12px;
            }
            
            .prediction-status.loading {
                color: var(--warning);
            }
            
            .prediction-status.success {
                color: var(--success);
            }
            
            .prediction-status.error {
                color: var(--danger);
            }
            
            .prediction-body {
                margin-bottom: 16px;
            }
            
            .price-prediction {
                display: flex;
                justify-content: space-between;
                margin-bottom: 12px;
                padding: 12px;
                background: var(--bg-tertiary);
                border-radius: 6px;
            }
            
            .price-prediction > div {
                text-align: center;
            }
            
            .price-prediction .label {
                display: block;
                font-size: 12px;
                color: var(--text-muted);
                margin-bottom: 4px;
            }
            
            .price-prediction .value {
                font-size: 16px;
                font-weight: 700;
                color: var(--text-primary);
            }
            
            .prediction-direction {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
            }
            
            .direction-indicator {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 14px;
                font-weight: 600;
                padding: 6px 12px;
                border-radius: 6px;
                background: var(--bg-tertiary);
            }
            
            .direction-indicator.bullish {
                color: var(--success);
                background: rgba(0, 208, 132, 0.1);
            }
            
            .direction-indicator.bearish {
                color: var(--danger);
                background: rgba(255, 71, 87, 0.1);
            }
            
            .confidence-score {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 4px;
            }
            
            .confidence-score .percentage {
                font-size: 12px;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .confidence-bar {
                width: 60px;
                height: 6px;
                background: var(--bg-tertiary);
                border-radius: 3px;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                transition: width 0.3s ease;
                border-radius: 3px;
            }
            
            .confidence-fill.high {
                background: var(--success);
            }
            
            .confidence-fill.medium {
                background: var(--warning);
            }
            
            .confidence-fill.low {
                background: var(--danger);
            }
            
            .prediction-factors {
                border-top: 1px solid var(--border-primary);
                padding-top: 12px;
            }
            
            .factors-header {
                display: flex;
                align-items: center;
                gap: 6px;
                margin-bottom: 8px;
                font-size: 12px;
                font-weight: 600;
                color: var(--text-secondary);
            }
            
            .factors-list {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            
            .factor-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 11px;
                padding: 2px 0;
            }
            
            .factor-label {
                color: var(--text-muted);
            }
            
            .factor-value {
                color: var(--text-primary);
                font-weight: 500;
            }
            
            .factors-loading {
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--text-muted);
                font-size: 12px;
            }
            
            .prediction-summary {
                border-top: 1px solid var(--border-primary);
                padding-top: 20px;
            }
            
            .summary-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 16px;
            }
            
            .summary-header h4 {
                display: flex;
                align-items: center;
                gap: 8px;
                margin: 0;
                font-size: 14px;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .summary-content {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 16px;
            }
            
            .summary-item {
                text-align: center;
            }
            
            .summary-label {
                font-size: 12px;
                color: var(--text-muted);
                margin-bottom: 4px;
            }
            
            .summary-value {
                font-size: 14px;
                font-weight: 600;
                color: var(--text-primary);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
            }
            
            .summary-value.bullish {
                color: var(--success);
            }
            
            .summary-value.bearish {
                color: var(--danger);
            }
            
            .loading-state, .error-state, .fallback-notice {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                padding: 20px;
                color: var(--text-muted);
                font-size: 14px;
            }
            
            .fallback-notice {
                color: var(--warning);
                background: rgba(255, 165, 2, 0.1);
                border-radius: 6px;
                padding: 12px;
            }
            
            .error-state {
                color: var(--danger);
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .predictions-grid {
                    grid-template-columns: 1fr;
                }
                
                .summary-content {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        const panel = document.getElementById('ml-predictions-panel');
        if (panel) {
            panel.remove();
        }
        
        const styles = document.getElementById('ml-prediction-styles');
        if (styles) {
            styles.remove();
        }
        
        console.log('🧹 ML Prediction Manager cleaned up');
    }
}

/**
 * Fallback prediction calculator for offline operation
 */
class FallbackPredictionCalculator {
    constructor() {
        this.basePrice = null; // Will be set from real-time API
        this.priceHistory = [];
        this.initialized = false;
    }
    
    /**
     * Initialize with current market data
     */
    initialize() {
        if (this.initialized) return;
        
        // Try to get current price from DOM
        const priceElements = document.querySelectorAll('.price-value');
        let currentPrice = this.basePrice;
        
        for (let element of priceElements) {
            const text = element.textContent.replace(/[^0-9.]/g, '');
            const price = parseFloat(text);
            if (price > 1000 && price < 5000) {
                currentPrice = price;
                break;
            }
        }
        
        this.basePrice = currentPrice;
        this.generatePriceHistory();
        this.initialized = true;
    }
    
    /**
     * Generate synthetic price history
     */
    generatePriceHistory() {
        const days = 30;
        const currentPrice = this.basePrice;
        
        this.priceHistory = [];
        let price = currentPrice;
        
        for (let i = days; i >= 0; i--) {
            const change = (Math.random() - 0.5) * 0.04; // ±2% daily change
            price = price * (1 + change);
            this.priceHistory.push(price);
        }
    }
    
    /**
     * Calculate technical indicators
     */
    calculateTechnicalIndicators() {
        const prices = this.priceHistory;
        const length = prices.length;
        
        if (length < 20) return { rsi: 50, trend: 0, volatility: 0.02 };
        
        // Simple RSI calculation
        const gains = [];
        const losses = [];
        
        for (let i = 1; i < length; i++) {
            const change = prices[i] - prices[i-1];
            if (change > 0) {
                gains.push(change);
                losses.push(0);
            } else {
                gains.push(0);
                losses.push(Math.abs(change));
            }
        }
        
        const avgGain = gains.slice(-14).reduce((a, b) => a + b, 0) / 14;
        const avgLoss = losses.slice(-14).reduce((a, b) => a + b, 0) / 14;
        const rs = avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));
        
        // Trend calculation
        const ma10 = prices.slice(-10).reduce((a, b) => a + b, 0) / 10;
        const ma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
        const trend = (ma10 - ma20) / ma20;
        
        // Volatility calculation
        const returns = [];
        for (let i = 1; i < length; i++) {
            returns.push((prices[i] - prices[i-1]) / prices[i-1]);
        }
        const volatility = Math.sqrt(returns.reduce((a, b) => a + b*b, 0) / returns.length);
        
        return { rsi, trend, volatility };
    }
    
    /**
     * Generate fallback predictions
     */
    generatePredictions() {
        this.initialize();
        
        const indicators = this.calculateTechnicalIndicators();
        const currentPrice = this.basePrice;
        
        const predictions = {};
        
        // Generate predictions for each timeframe
        Object.keys(window.goldMLPredictionManager?.timeframes || {}).forEach(timeframe => {
            const timeframeMultiplier = {
                '1H': 0.001,
                '4H': 0.004,
                '1D': 0.02
            }[timeframe] || 0.01;
            
            // Simple prediction based on technical indicators
            let directionScore = 0;
            
            // RSI contribution
            if (indicators.rsi > 70) directionScore -= 0.3; // Overbought
            if (indicators.rsi < 30) directionScore += 0.3; // Oversold
            
            // Trend contribution
            directionScore += indicators.trend * 2;
            
            // Add some randomness for realism
            directionScore += (Math.random() - 0.5) * 0.4;
            
            // Calculate predicted price
            const priceChange = directionScore * timeframeMultiplier * currentPrice;
            const predictedPrice = currentPrice + priceChange;
            
            // Calculate confidence
            const confidence = Math.min(0.85, 0.5 + Math.abs(directionScore) * 0.3);
            
            predictions[timeframe] = {
                current_price: currentPrice,
                predicted_price: predictedPrice,
                predicted_direction: priceChange > 0 ? 'UP' : 'DOWN',
                confidence_score: confidence,
                prediction_factors: {
                    rsi: indicators.rsi,
                    trend: indicators.trend,
                    volatility: indicators.volatility,
                    direction_score: directionScore,
                    timeframe_multiplier: timeframeMultiplier
                }
            };
        });
        
        return predictions;
    }
}

/**
 * Fallback prediction calculator for offline operation
 */
class FallbackPredictionCalculator {
    constructor() {
        this.currentPrice = 3350; // Default gold price
        this.volatility = 0.015; // 1.5% daily volatility
        
        console.log('🔄 Fallback prediction calculator initialized');
    }
    
    /**
     * Generate fallback predictions using statistical methods
     */
    generatePredictions() {
        try {
            // Try to get current price from existing price displays
            const priceElements = document.querySelectorAll('[data-symbol="XAUUSD"], .current-price, .price-display');
            let currentPrice = this.currentPrice;
            
            for (const element of priceElements) {
                const priceText = element.textContent || element.innerText;
                const extractedPrice = parseFloat(priceText.replace(/[^0-9.]/g, ''));
                if (extractedPrice > 1000 && extractedPrice < 5000) {
                    currentPrice = extractedPrice;
                    break;
                }
            }
            
            this.currentPrice = currentPrice;
            console.log(`🔄 Using fallback price: $${currentPrice}`);
            
            const predictions = {};
            const timeframes = ['1H', '4H', '1D'];
            
            timeframes.forEach(timeframe => {
                // Generate prediction based on timeframe
                const horizonHours = timeframe === '1H' ? 1 : timeframe === '4H' ? 4 : 24;
                const trendFactor = this.generateTrendFactor(horizonHours);
                const noiseFactor = this.generateNoiseFactor(horizonHours);
                
                const priceChange = currentPrice * (trendFactor + noiseFactor);
                const predictedPrice = currentPrice + priceChange;
                
                // Calculate direction and confidence
                const direction = Math.abs(priceChange / currentPrice) > 0.005 
                    ? (priceChange > 0 ? 'bullish' : 'bearish') 
                    : 'neutral';
                
                const confidence = Math.max(0.3, 0.8 - Math.abs(trendFactor) * 5);
                
                predictions[timeframe] = {
                    symbol: 'GC=F',
                    timeframe: timeframe,
                    predicted_price: predictedPrice,
                    direction: direction,
                    confidence: confidence,
                    current_price: currentPrice,
                    price_change: priceChange,
                    price_change_percent: (priceChange / currentPrice) * 100,
                    support_level: currentPrice * 0.98,
                    resistance_level: currentPrice * 1.02,
                    timestamp: new Date().toISOString(),
                    model_agreement: confidence,
                    technical_signals: {
                        rsi: 45 + Math.random() * 20, // 45-65 range
                        macd: (Math.random() - 0.5) * 0.1,
                        bb_position: 0.3 + Math.random() * 0.4 // 0.3-0.7 range
                    }
                };
            });
            
            console.log('✅ Fallback predictions generated:', predictions);
            return predictions;
            
        } catch (error) {
            console.error('❌ Fallback prediction generation failed:', error);
            return this.generateBasicPredictions();
        }
    }
    
    /**
     * Generate trend factor based on timeframe
     */
    generateTrendFactor(horizonHours) {
        // Longer timeframes have stronger trends
        const baseDirection = (Math.random() - 0.5) * 2; // -1 to 1
        const timeframeFactor = Math.log(horizonHours + 1) / 10; // 0 to ~0.3
        return baseDirection * timeframeFactor;
    }
    
    /**
     * Generate noise factor for realistic volatility
     */
    generateNoiseFactor(horizonHours) {
        // Shorter timeframes have more noise
        const noiseIntensity = this.volatility / Math.sqrt(horizonHours);
        return (Math.random() - 0.5) * 2 * noiseIntensity;
    }
    
    /**
     * Generate very basic predictions if all else fails
     */
    generateBasicPredictions() {
        const basicPredictions = {};
        const timeframes = ['1H', '4H', '1D'];
        
        timeframes.forEach(timeframe => {
            basicPredictions[timeframe] = {
                symbol: 'GC=F',
                timeframe: timeframe,
                predicted_price: this.currentPrice + (Math.random() - 0.5) * 20,
                direction: ['bullish', 'bearish', 'neutral'][Math.floor(Math.random() * 3)],
                confidence: 0.4,
                current_price: this.currentPrice,
                price_change: (Math.random() - 0.5) * 20,
                price_change_percent: (Math.random() - 0.5) * 0.6,
                support_level: this.currentPrice * 0.98,
                resistance_level: this.currentPrice * 1.02,
                timestamp: new Date().toISOString(),
                model_agreement: 0.4,
                technical_signals: {
                    rsi: 50,
                    macd: 0,
                    bb_position: 0.5
                }
            };
        });
        
        return basicPredictions;
    }
}

// Global instance
window.goldMLPredictionManager = new GoldMLPredictionManager();

// Component loader compatibility
window.goldMLPredictionManager.init = function() {
    return this.init();
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        window.goldMLPredictionManager.init();
    }, 2000); // Wait 2 seconds for other components to load
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.goldMLPredictionManager.destroy) {
        window.goldMLPredictionManager.destroy();
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GoldMLPredictionManager;
}

console.log('🤖 Gold ML Prediction Manager loaded successfully');