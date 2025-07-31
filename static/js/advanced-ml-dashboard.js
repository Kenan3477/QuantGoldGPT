/**
 * GoldGPT Advanced ML Dashboard - Interactive Visualization System
 * Trading212-inspired interface with real-time ML predictions and analytics
 */

class AdvancedMLDashboard {
    constructor() {
        this.apiClient = null;
        this.socket = null;
        this.charts = {};
        this.currentTimeframe = 'all';
        this.currentStrategy = 'all';
        this.isConnected = false;
        this.predictionData = {};
        this.historicalData = {};
        this.learningData = {};
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Advanced ML Dashboard...');
        
        // Initialize API client
        if (typeof AdvancedMLClient !== 'undefined') {
            this.apiClient = new AdvancedMLClient();
            await this.apiClient.connect();
        }

        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize WebSocket connection
        this.initializeWebSocket();
        
        // Initialize charts
        this.initializeCharts();
        
        // Load initial data
        await this.loadInitialData();
        
        // Setup real-time updates
        this.setupRealTimeUpdates();
        
        // Hide loading overlay
        this.hideLoading();
        
        console.log('✅ Advanced ML Dashboard initialized successfully');
    }

    setupEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.refreshAllData();
        });

        // Settings button
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.showSettingsModal();
        });

        // Timeframe selector
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectTimeframe(e.target.dataset.timeframe);
            });
        });

        // Strategy selector
        document.querySelectorAll('.strategy-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectStrategy(e.target.dataset.strategy);
            });
        });

        // Chart control buttons
        document.getElementById('showSupportResistance')?.addEventListener('click', () => {
            this.toggleSupportResistance();
        });

        document.getElementById('showConfidenceIntervals')?.addEventListener('click', () => {
            this.toggleConfidenceIntervals();
        });

        // Timeline slider
        document.getElementById('timelineSlider')?.addEventListener('input', (e) => {
            this.updateTimeline(e.target.value);
        });

        // Example tabs
        document.querySelectorAll('.example-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.showExamples(e.target.dataset.tab);
            });
        });

        // Modal close
        document.getElementById('modalClose')?.addEventListener('click', () => {
            this.hideModal();
        });

        // Click outside modal to close
        document.getElementById('predictionModal')?.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                this.hideModal();
            }
        });
    }

    initializeWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('🔌 WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus('connected', 'Connected');
                
                // Subscribe to prediction updates
                this.socket.emit('subscribe_predictions', {
                    timeframes: ['15min', '30min', '1h', '4h', '24h', '7d']
                });
            });

            this.socket.on('disconnect', () => {
                console.log('🔌 WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('error', 'Disconnected');
            });

            this.socket.on('new_predictions', (data) => {
                console.log('📊 Received new predictions:', data);
                this.handleNewPredictions(data);
            });

            this.socket.on('prediction_update', (data) => {
                console.log('🔄 Prediction update received:', data);
                this.handlePredictionUpdate(data);
            });

            this.socket.on('model_retrained', (data) => {
                console.log('🤖 Model retrained:', data);
                this.handleModelRetrained(data);
            });

            this.socket.on('accuracy_update', (data) => {
                console.log('📈 Accuracy update:', data);
                this.handleAccuracyUpdate(data);
            });

            // Advanced dashboard specific events
            this.socket.on('dashboard_data_update', (data) => {
                console.log('📊 Dashboard data update:', data);
                this.handleDashboardDataUpdate(data);
            });

            this.socket.on('live_prediction_update', (data) => {
                console.log('🔮 Live prediction update:', data);
                this.handleLivePredictionUpdate(data);
            });

            this.socket.on('learning_update', (data) => {
                console.log('🧠 Learning update:', data);
                this.handleLearningUpdate(data);
            });

            this.socket.on('advanced_ml_prediction', (data) => {
                console.log('🤖 Advanced ML prediction:', data);
                this.handleAdvancedMLPrediction(data);
            });

        } catch (error) {
            console.error('❌ WebSocket initialization failed:', error);
            this.updateConnectionStatus('error', 'Connection Failed');
        }
    }

    initializeCharts() {
        // Initialize prediction chart
        this.initializePredictionChart();
        
        // Initialize accuracy trend chart
        this.initializeAccuracyChart();
        
        // Initialize strategy comparison chart
        this.initializeStrategyChart();
        
        // Initialize feature evolution chart
        this.initializeFeatureChart();
    }

    initializePredictionChart() {
        const ctx = document.getElementById('predictionChart');
        if (!ctx) return;

        this.charts.prediction = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Current Price',
                    data: [],
                    borderColor: '#0066cc',
                    backgroundColor: 'rgba(0, 102, 204, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Predicted Price',
                    data: [],
                    borderColor: '#00b386',
                    backgroundColor: 'rgba(0, 179, 134, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'start'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#333',
                        bodyColor: '#666',
                        borderColor: '#dee2e6',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    initializeAccuracyChart() {
        const ctx = document.getElementById('accuracyTrendChart');
        if (!ctx) return;

        this.charts.accuracy = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Overall Accuracy',
                    data: [],
                    borderColor: '#0066cc',
                    backgroundColor: 'rgba(0, 102, 204, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    initializeStrategyChart() {
        const ctx = document.getElementById('strategyComparisonChart');
        if (!ctx) return;

        this.charts.strategy = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Technical', 'Sentiment', 'Macro', 'Pattern', 'Momentum'],
                datasets: [{
                    label: 'Accuracy (%)',
                    data: [],
                    borderColor: '#0066cc',
                    backgroundColor: 'rgba(0, 102, 204, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: '#0066cc',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initializeFeatureChart() {
        const ctx = document.getElementById('featureEvolutionChart');
        if (!ctx) return;

        this.charts.feature = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Feature Importance',
                    data: [],
                    backgroundColor: [
                        '#0066cc',
                        '#00b386',
                        '#ffa502',
                        '#ff4757',
                        '#6c757d'
                    ],
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                animation: {
                    duration: 1200,
                    delay: (context) => context.dataIndex * 100
                }
            }
        });
    }

    async loadInitialData() {
        try {
            console.log('📊 Loading initial dashboard data...');

            // Load system status
            await this.updateSystemStatus();

            // Load predictions for all timeframes
            await this.loadPredictions();

            // Load accuracy statistics
            await this.loadAccuracyStats();

            // Load feature importance
            await this.loadFeatureImportance();

            // Load learning examples
            await this.loadLearningExamples();

            // Initialize advanced features
            await this.initializeAdvancedFeatures();

            console.log('✅ Initial data loaded successfully');

        } catch (error) {
            console.error('❌ Failed to load initial data:', error);
            this.showError('Failed to load dashboard data. Please refresh the page.');
        }
    }

    async updateSystemStatus() {
        try {
            if (!this.apiClient) return;

            const response = await this.apiClient.getSystemStatus();
            if (response.success) {
                const status = response.status;
                
                // Update overview cards
                document.getElementById('activeStrategies').textContent = 
                    status.config?.prediction_timeframes?.length || '--';
                
                document.getElementById('lastUpdate').textContent = 
                    this.formatTime(status.timestamp);
                
                document.getElementById('predictionsToday').textContent = 
                    status.total_predictions_today || 0;

                // Update system status indicator
                if (status.ml_engine_available && status.scheduler_running) {
                    this.updateConnectionStatus('connected', 'System Online');
                } else {
                    this.updateConnectionStatus('warning', 'Partial Service');
                }
            }
        } catch (error) {
            console.error('❌ Failed to update system status:', error);
            this.updateConnectionStatus('error', 'System Error');
        }
    }

    async loadPredictions() {
        try {
            if (!this.apiClient) return;

            const response = await this.apiClient.getAllPredictions();
            if (response.success) {
                this.predictionData = response.predictions || {};
                this.renderPredictionCards();
                this.updatePredictionChart();
            }
        } catch (error) {
            console.error('❌ Failed to load predictions:', error);
        }
    }

    async loadAccuracyStats() {
        try {
            if (!this.apiClient) return;

            const response = await this.apiClient.getAccuracyStats();
            if (response.success) {
                const stats = response.accuracy_stats || {};
                
                // Update overall accuracy
                const overallAccuracy = stats.overall_accuracy || 0;
                document.getElementById('overallAccuracy').textContent = 
                    overallAccuracy.toFixed(1) + '%';
                
                const change = stats.accuracy_change || 0;
                const changeElement = document.getElementById('accuracyChange');
                changeElement.textContent = (change >= 0 ? '+' : '') + change.toFixed(1) + '%';
                changeElement.className = 'metric-change ' + (change >= 0 ? 'positive' : 'negative');

                // Update accuracy chart
                if (this.charts.accuracy && stats.historical) {
                    this.updateAccuracyChart(stats.historical);
                }
            }
        } catch (error) {
            console.error('❌ Failed to load accuracy stats:', error);
        }
    }

    async loadFeatureImportance() {
        try {
            if (!this.apiClient) return;

            const response = await this.apiClient.getFeatureImportance();
            if (response.success) {
                const features = response.feature_importance || {};
                this.renderFeatureImportance(features);
                
                if (this.charts.feature) {
                    this.updateFeatureChart(features);
                }
            }
        } catch (error) {
            console.error('❌ Failed to load feature importance:', error);
        }
    }

    async loadLearningExamples() {
        try {
            // This would typically come from a dedicated API endpoint
            const examples = {
                winning: [
                    {
                        title: "Bullish Breakout Prediction",
                        result: "+2.3%",
                        description: "Successfully predicted gold breakout above $3,400 resistance using technical analysis and momentum indicators.",
                        type: "profit"
                    },
                    {
                        title: "Economic News Impact",
                        result: "+1.8%",
                        description: "Correctly anticipated price drop following unexpected inflation data using sentiment analysis.",
                        type: "profit"
                    }
                ],
                learning: [
                    {
                        title: "Geopolitical Event Surprise",
                        result: "Learning",
                        description: "Failed to predict sudden price spike due to unexpected geopolitical developments. Model now includes additional news sentiment factors.",
                        type: "learning"
                    },
                    {
                        title: "Weekend Gap Adjustment",
                        result: "Improved",
                        description: "Initial weekend gap predictions were inaccurate. Model retrained with enhanced gap analysis algorithms.",
                        type: "learning"
                    }
                ]
            };

            this.renderLearningExamples(examples);
        } catch (error) {
            console.error('❌ Failed to load learning examples:', error);
        }
    }

    async initializeAdvancedFeatures() {
        try {
            console.log('🚀 Initializing advanced dashboard features...');

            // Initialize prediction confidence tracking
            this.initializeConfidenceTracking();

            // Setup advanced chart overlays
            this.setupChartOverlays();

            // Initialize smart notifications
            this.initializeSmartNotifications();

            // Setup keyboard shortcuts
            this.setupKeyboardShortcuts();

            // Initialize data export features
            this.initializeDataExport();

            // Start performance monitoring
            this.startPerformanceMonitoring();

            console.log('✅ Advanced features initialized');

        } catch (error) {
            console.error('❌ Failed to initialize advanced features:', error);
        }
    }

    initializeConfidenceTracking() {
        // Track prediction confidence changes over time
        this.confidenceHistory = new Map();
        
        // Setup confidence threshold alerts
        this.confidenceThresholds = {
            high: 0.85,
            medium: 0.70,
            low: 0.55
        };
    }

    setupChartOverlays() {
        // Add support/resistance level overlays
        this.supportResistanceLevels = {
            support: [3380, 3350, 3320],
            resistance: [3420, 3450, 3480]
        };

        // Initialize moving average overlays
        this.movingAverages = {
            sma20: [],
            sma50: [],
            ema12: []
        };
    }

    initializeSmartNotifications() {
        // Setup intelligent notification system
        this.notificationQueue = [];
        this.notificationSettings = {
            highConfidence: true,
            priceAlerts: true,
            accuracyChanges: true,
            systemAlerts: true
        };

        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission().then(permission => {
                console.log(`📱 Notification permission: ${permission}`);
            });
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+R: Refresh data
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                this.refreshAllData();
            }
            
            // Ctrl+E: Export data
            if (e.ctrlKey && e.key === 'e') {
                e.preventDefault();
                this.exportDashboardData();
            }

            // Numbers 1-6: Switch timeframes
            if (e.key >= '1' && e.key <= '6' && !e.ctrlKey && !e.altKey) {
                const timeframes = ['15min', '30min', '1h', '4h', '24h', '7d'];
                const index = parseInt(e.key) - 1;
                if (index < timeframes.length) {
                    this.selectTimeframe(timeframes[index]);
                }
            }

            // Escape: Close modals
            if (e.key === 'Escape') {
                this.hideModal();
            }
        });

        console.log('⌨️ Keyboard shortcuts initialized (Ctrl+R, Ctrl+E, 1-6, Esc)');
    }

    initializeDataExport() {
        this.exportFormats = ['json', 'csv', 'pdf'];
        console.log('📊 Data export functionality initialized');
    }

    startPerformanceMonitoring() {
        // Monitor dashboard performance metrics
        this.performanceMetrics = {
            loadTime: 0,
            apiResponseTimes: [],
            chartRenderTimes: [],
            memoryUsage: 0
        };

        // Setup performance observer
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.entryType === 'navigation') {
                        this.performanceMetrics.loadTime = entry.loadEventEnd - entry.loadEventStart;
                    }
                }
            });
            observer.observe({ entryTypes: ['navigation'] });
        }

        console.log('📈 Performance monitoring started');
    }

    renderPredictionCards() {
        const container = document.getElementById('predictionsContainer');
        if (!container) return;

        container.innerHTML = '';

        const timeframesToShow = this.currentTimeframe === 'all' ? 
            Object.keys(this.predictionData) : [this.currentTimeframe];

        timeframesToShow.forEach(timeframe => {
            const predictions = this.predictionData[timeframe] || [];
            
            predictions.forEach(prediction => {
                const card = this.createPredictionCard(prediction, timeframe);
                container.appendChild(card);
            });
        });

        // Add animation classes
        container.querySelectorAll('.prediction-card').forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('fade-in');
            }, index * 100);
        });
    }

    createPredictionCard(prediction, timeframe) {
        const card = document.createElement('div');
        card.className = `prediction-card ${prediction.direction?.toLowerCase() || 'neutral'}`;
        
        const direction = prediction.direction || 'NEUTRAL';
        const confidence = prediction.confidence || 0;
        const currentPrice = prediction.current_price || 0;
        const targetPrice = prediction.target_price || 0;
        const features = prediction.key_features || [];

        // Determine direction icon and color
        let directionIcon = 'fas fa-arrows-alt-h';
        let directionClass = 'neutral';
        
        if (direction === 'BULLISH') {
            directionIcon = 'fas fa-arrow-trend-up';
            directionClass = 'bullish';
        } else if (direction === 'BEARISH') {
            directionIcon = 'fas fa-arrow-trend-down';
            directionClass = 'bearish';
        }

        card.innerHTML = `
            <div class="prediction-header">
                <div class="prediction-timeframe">${timeframe.toUpperCase()}</div>
                <div class="prediction-confidence">${(confidence * 100).toFixed(1)}%</div>
            </div>
            
            <div class="prediction-direction">
                <i class="direction-icon ${directionIcon} ${directionClass}"></i>
                <span class="direction-text">${direction}</span>
            </div>
            
            <div class="prediction-price">
                <div class="price-target">
                    <div class="price-label">Current</div>
                    <div class="price-value">$${currentPrice.toFixed(2)}</div>
                </div>
                <div class="price-target">
                    <div class="price-label">Target</div>
                    <div class="price-value">$${targetPrice.toFixed(2)}</div>
                </div>
            </div>
            
            <div class="prediction-features">
                <div class="features-label">Key Features</div>
                <div class="feature-tags">
                    ${features.slice(0, 3).map(feature => 
                        `<span class="feature-tag">${feature}</span>`
                    ).join('')}
                </div>
            </div>
        `;

        // Add click event to show detailed modal
        card.addEventListener('click', () => {
            this.showPredictionDetails(prediction, timeframe);
        });

        return card;
    }

    renderFeatureImportance(features) {
        const container = document.getElementById('featureList');
        if (!container) return;

        container.innerHTML = '';

        const sortedFeatures = Object.entries(features)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8); // Show top 8 features

        sortedFeatures.forEach(([name, importance], index) => {
            const item = document.createElement('div');
            item.className = 'feature-item';
            
            item.innerHTML = `
                <div class="feature-name">${name}</div>
                <div class="feature-bar-container">
                    <div class="feature-bar" style="width: ${importance * 100}%"></div>
                </div>
                <div class="feature-value">${(importance * 100).toFixed(1)}%</div>
            `;

            // Animate bar width
            setTimeout(() => {
                const bar = item.querySelector('.feature-bar');
                bar.style.width = `${importance * 100}%`;
            }, index * 150);

            container.appendChild(item);
        });
    }

    renderLearningExamples(examples) {
        const container = document.getElementById('examplesContent');
        if (!container) return;

        const currentTab = document.querySelector('.example-tab.active')?.dataset.tab || 'winning';
        const examplesData = examples[currentTab] || [];

        container.innerHTML = '';

        examplesData.forEach(example => {
            const item = document.createElement('div');
            item.className = `example-item ${example.type}`;
            
            item.innerHTML = `
                <div class="example-header">
                    <div class="example-title">${example.title}</div>
                    <div class="example-result ${example.type}">${example.result}</div>
                </div>
                <div class="example-description">${example.description}</div>
            `;

            container.appendChild(item);
        });
    }

    selectTimeframe(timeframe) {
        this.currentTimeframe = timeframe;
        
        // Update button states
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.timeframe === timeframe);
        });

        // Re-render predictions
        this.renderPredictionCards();
        
        console.log(`📊 Selected timeframe: ${timeframe}`);
    }

    selectStrategy(strategy) {
        this.currentStrategy = strategy;
        
        // Update button states
        document.querySelectorAll('.strategy-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.strategy === strategy);
        });

        // Update strategy metrics display
        this.updateStrategyMetrics(strategy);
        
        console.log(`🎯 Selected strategy: ${strategy}`);
    }

    updateStrategyMetrics(strategy) {
        const container = document.getElementById('strategyMetrics');
        if (!container) return;

        // Mock strategy performance data
        const strategyData = {
            all: {
                'Overall Accuracy': '87.5%',
                'Win Rate': '72.3%',
                'Avg Confidence': '84.2%',
                'Total Predictions': '1,247'
            },
            technical: {
                'Accuracy': '89.2%',
                'RSI Success': '91.5%',
                'MA Crossover': '85.7%',
                'Support/Resistance': '92.1%'
            },
            sentiment: {
                'Accuracy': '83.6%',
                'News Impact': '78.9%',
                'Social Sentiment': '88.2%',
                'Market Fear/Greed': '85.4%'
            },
            macro: {
                'Accuracy': '91.3%',
                'Economic Data': '93.7%',
                'Central Bank': '89.8%',
                'Inflation Metrics': '90.2%'
            },
            pattern: {
                'Accuracy': '85.8%',
                'Chart Patterns': '87.4%',
                'Candlestick': '84.1%',
                'Trend Analysis': '86.9%'
            }
        };

        const metrics = strategyData[strategy] || strategyData.all;
        
        container.innerHTML = Object.entries(metrics)
            .map(([name, value]) => `
                <div class="strategy-metric">
                    <div class="strategy-metric-name">${name}</div>
                    <div class="strategy-metric-value">${value}</div>
                </div>
            `).join('');
    }

    showPredictionDetails(prediction, timeframe) {
        const modal = document.getElementById('predictionModal');
        const title = document.getElementById('modalTitle');
        const body = document.getElementById('modalBody');
        
        if (!modal || !title || !body) return;

        title.textContent = `${timeframe.toUpperCase()} Prediction Details`;
        
        body.innerHTML = `
            <div class="prediction-detail-content">
                <div class="detail-section">
                    <h4><i class="fas fa-chart-line"></i> Prediction Overview</h4>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <span class="detail-label">Direction:</span>
                            <span class="detail-value ${prediction.direction?.toLowerCase()}">${prediction.direction || 'NEUTRAL'}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Confidence:</span>
                            <span class="detail-value">${((prediction.confidence || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Current Price:</span>
                            <span class="detail-value">$${(prediction.current_price || 0).toFixed(2)}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Target Price:</span>
                            <span class="detail-value">$${(prediction.target_price || 0).toFixed(2)}</span>
                        </div>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h4><i class="fas fa-brain"></i> AI Reasoning</h4>
                    <div class="reasoning-content">
                        <p>${prediction.reasoning || 'Advanced AI analysis based on multiple technical and fundamental factors including price action, volume patterns, market sentiment, and economic indicators.'}</p>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h4><i class="fas fa-weight-scale"></i> Key Features</h4>
                    <div class="features-detailed">
                        ${(prediction.key_features || []).map(feature => `
                            <div class="feature-detailed">
                                <span class="feature-name">${feature}</span>
                                <div class="feature-impact">High Impact</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="detail-section">
                    <h4><i class="fas fa-exclamation-triangle"></i> Risk Factors</h4>
                    <div class="risk-factors">
                        <div class="risk-item">
                            <i class="fas fa-chart-line"></i>
                            <span>Market volatility may affect prediction accuracy</span>
                        </div>
                        <div class="risk-item">
                            <i class="fas fa-newspaper"></i>
                            <span>Unexpected news events could impact price movement</span>
                        </div>
                        <div class="risk-item">
                            <i class="fas fa-clock"></i>
                            <span>Prediction confidence may decrease over time</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        modal.classList.add('show');
    }

    hideModal() {
        const modal = document.getElementById('predictionModal');
        if (modal) {
            modal.classList.remove('show');
        }
    }

    updatePredictionChart() {
        if (!this.charts.prediction) return;

        // Mock chart data - in real implementation, this would come from API
        const now = new Date();
        const labels = [];
        const currentData = [];
        const predictedData = [];

        for (let i = 24; i >= 0; i--) {
            const time = new Date(now - i * 60 * 60 * 1000);
            labels.push(time);
            
            // Mock price data with some realistic variation
            const basePrice = 3400;
            const variation = Math.sin(i / 4) * 20 + Math.random() * 10 - 5;
            currentData.push(basePrice + variation);
            
            if (i <= 6) { // Predicted data for next 6 hours
                const predictedVariation = Math.sin(i / 3) * 15 + Math.random() * 8 - 4;
                predictedData.push(basePrice + variation + predictedVariation);
            } else {
                predictedData.push(null);
            }
        }

        this.charts.prediction.data.labels = labels;
        this.charts.prediction.data.datasets[0].data = currentData;
        this.charts.prediction.data.datasets[1].data = predictedData;
        
        this.charts.prediction.update('active');
    }

    updateAccuracyChart(historicalData) {
        if (!this.charts.accuracy) return;

        const labels = historicalData.map(d => d.date);
        const accuracyData = historicalData.map(d => d.accuracy);

        this.charts.accuracy.data.labels = labels;
        this.charts.accuracy.data.datasets[0].data = accuracyData;
        
        this.charts.accuracy.update('active');
    }

    updateFeatureChart(features) {
        if (!this.charts.feature) return;

        const sortedFeatures = Object.entries(features)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);

        const labels = sortedFeatures.map(([name]) => name);
        const data = sortedFeatures.map(([, importance]) => importance);

        this.charts.feature.data.labels = labels;
        this.charts.feature.data.datasets[0].data = data;
        
        this.charts.feature.update('active');
    }

    handleNewPredictions(data) {
        // Animate prediction cards update
        const container = document.getElementById('predictionsContainer');
        if (container) {
            container.classList.add('updating');
            
            setTimeout(() => {
                this.predictionData = { ...this.predictionData, ...data.predictions };
                this.renderPredictionCards();
                container.classList.remove('updating');
                
                // Show notification
                this.showNotification('New predictions available', 'success');
            }, 300);
        }
    }

    handlePredictionUpdate(data) {
        // Update specific prediction card
        console.log('🔄 Updating prediction:', data);
        
        if (data.timeframe && this.predictionData[data.timeframe]) {
            // Find and update the specific prediction
            const predictions = this.predictionData[data.timeframe];
            const index = predictions.findIndex(p => p.id === data.prediction_id);
            
            if (index !== -1) {
                predictions[index] = { ...predictions[index], ...data.updates };
                this.renderPredictionCards();
            }
        }
    }

    handleModelRetrained(data) {
        console.log('🤖 Model retrained:', data);
        
        // Update system status
        this.showNotification(`${data.model_type} model retrained - Accuracy: ${data.new_accuracy}%`, 'info');
        
        // Refresh accuracy stats
        this.loadAccuracyStats();
    }

    handleAccuracyUpdate(data) {
        console.log('📈 Accuracy updated:', data);
        
        // Update overview card
        document.getElementById('overallAccuracy').textContent = 
            (data.overall_accuracy || 0).toFixed(1) + '%';
        
        // Update accuracy chart if needed
        if (data.historical) {
            this.updateAccuracyChart(data.historical);
        }
    }

    handleDashboardDataUpdate(data) {
        console.log('📊 Dashboard data update:', data);
        
        if (data.success && data.predictions) {
            this.updatePredictionsGrid(data.predictions.multi_timeframe);
        }
        
        if (data.performance) {
            this.updatePerformanceOverview(data.performance);
        }
        
        if (data.analysis) {
            this.updateMarketAnalysis(data.analysis);
        }
    }

    handleLivePredictionUpdate(data) {
        console.log('🔮 Live prediction update:', data);
        
        if (data.success && data.prediction) {
            // Update live prediction display
            this.updateLivePrediction(data.prediction, data.timeframe);
            
            // Show notification for high-confidence predictions
            if (data.prediction.confidence > 0.9) {
                this.showNotification(
                    `High-confidence ${data.prediction.direction} prediction for ${data.timeframe}`,
                    'success'
                );
            }
        }
    }

    handleLearningUpdate(data) {
        console.log('🧠 Learning update:', data);
        
        if (data.success && data.learning_progress) {
            this.updateLearningProgress(data.learning_progress);
            
            // Update learning dashboard if visible
            this.updateLearningDashboard(data.learning_progress);
        }
    }

    handleAdvancedMLPrediction(data) {
        console.log('🤖 Advanced ML prediction:', data);
        
        if (data.success && data.predictions) {
            // Update predictions display
            this.updateAdvancedPredictions(data.predictions);
            
            // Update confidence metrics
            if (data.confidence_metrics) {
                this.updateConfidenceMetrics(data.confidence_metrics);
            }
        } else if (data.error) {
            this.showNotification(`Prediction error: ${data.error}`, 'error');
        }
    }

    updateLivePrediction(prediction, timeframe) {
        // Update the live prediction card
        const predictionCard = document.querySelector(`[data-timeframe="${timeframe}"]`);
        if (predictionCard) {
            const directionElement = predictionCard.querySelector('.direction');
            const confidenceElement = predictionCard.querySelector('.confidence');
            const priceElement = predictionCard.querySelector('.target-price');
            
            if (directionElement) {
                directionElement.textContent = prediction.direction.toUpperCase();
                directionElement.className = `direction ${prediction.direction}`;
            }
            
            if (confidenceElement) {
                confidenceElement.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
            }
            
            if (priceElement) {
                priceElement.textContent = `$${prediction.target_price.toFixed(2)}`;
            }
            
            // Add visual feedback
            predictionCard.classList.add('updated');
            setTimeout(() => predictionCard.classList.remove('updated'), 2000);
        }
    }

    updateLearningProgress(progress) {
        // Update learning metrics
        document.getElementById('currentIteration')?.textContent = 
            progress.current_iteration?.toLocaleString() || 'N/A';
        
        document.getElementById('trainingAccuracy')?.textContent = 
            `${(progress.training_accuracy * 100).toFixed(2)}%` || 'N/A';
        
        document.getElementById('validationAccuracy')?.textContent = 
            `${(progress.validation_accuracy * 100).toFixed(2)}%` || 'N/A';
        
        document.getElementById('learningRate')?.textContent = 
            progress.learning_rate?.toFixed(4) || 'N/A';
        
        // Update improvements list
        if (progress.recent_improvements) {
            this.updateImprovementsList(progress.recent_improvements);
        }
    }

    updateImprovementsList(improvements) {
        const list = document.getElementById('improvementsList');
        if (list && improvements) {
            list.innerHTML = improvements.map(improvement => `
                <div class="improvement-item">
                    <span class="metric">${improvement.metric}</span>
                    <span class="value positive">${improvement.improvement}</span>
                </div>
            `).join('');
        }
    }

    updateAdvancedPredictions(predictions) {
        // Update advanced predictions display
        if (predictions && Array.isArray(predictions)) {
            predictions.forEach(prediction => {
                this.updatePredictionCard(prediction);
            });
        }
    }

    updateConfidenceMetrics(metrics) {
        // Update confidence display
        if (metrics.overall) {
            document.getElementById('overallConfidence')?.textContent = 
                `${(metrics.overall * 100).toFixed(1)}%`;
        }
        
        // Update timeframe-specific confidence
        if (metrics.by_timeframe) {
            Object.keys(metrics.by_timeframe).forEach(timeframe => {
                const element = document.querySelector(`[data-timeframe="${timeframe}"] .confidence-metric`);
                if (element) {
                    element.textContent = `${(metrics.by_timeframe[timeframe] * 100).toFixed(1)}%`;
                }
            });
        }
    }

    setupRealTimeUpdates() {
        // Update system status every 30 seconds
        setInterval(() => {
            if (this.isConnected) {
                this.updateSystemStatus();
            }
        }, 30000);

        // Update timestamp display every second
        setInterval(() => {
            const now = new Date();
            document.getElementById('lastUpdate').textContent = 
                this.formatTime(now.toISOString());
        }, 1000);
    }

    async refreshAllData() {
        const button = document.getElementById('refreshBtn');
        if (button) {
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            button.disabled = true;
        }

        try {
            await Promise.all([
                this.updateSystemStatus(),
                this.loadPredictions(),
                this.loadAccuracyStats(),
                this.loadFeatureImportance()
            ]);
            
            this.showNotification('Dashboard data refreshed successfully', 'success');
            
        } catch (error) {
            console.error('❌ Failed to refresh data:', error);
            this.showNotification('Failed to refresh data', 'error');
        } finally {
            if (button) {
                button.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                button.disabled = false;
            }
        }
    }

    updateConnectionStatus(status, text) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (indicator && statusText) {
            indicator.className = `status-indicator ${status}`;
            statusText.textContent = text;
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Show with animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    showError(message) {
        console.error(message);
        this.showNotification(message, 'error');
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }

    formatTime(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('en-US', { 
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        } catch {
            return '--:--:--';
        }
    }

    updateTimeline(value) {
        // Update timeline info based on slider value
        const timelineTime = document.getElementById('timelineTime');
        const timelinePrice = document.getElementById('timelinePrice');
        const timelinePrediction = document.getElementById('timelinePrediction');
        
        if (timelineTime && timelinePrice && timelinePrediction) {
            const percentage = value / 100;
            const hoursAgo = Math.round((1 - percentage) * 24);
            
            timelineTime.textContent = hoursAgo === 0 ? 'Live' : `${hoursAgo}h ago`;
            timelinePrice.textContent = `$${(3400 + Math.random() * 20 - 10).toFixed(2)}`;
            timelinePrediction.textContent = ['Bullish', 'Bearish', 'Neutral'][Math.floor(Math.random() * 3)];
        }
    }

    toggleSupportResistance() {
        const button = document.getElementById('showSupportResistance');
        if (button) {
            button.classList.toggle('active');
            // Logic to show/hide S/R levels on chart would go here
            console.log('🔲 Toggled Support/Resistance levels');
        }
    }

    toggleConfidenceIntervals() {
        const button = document.getElementById('showConfidenceIntervals');
        if (button) {
            button.classList.toggle('active');
            // Logic to show/hide confidence intervals would go here
            console.log('📊 Toggled Confidence Intervals');
        }
    }

    showExamples(tab) {
        // Update tab states
        document.querySelectorAll('.example-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tab);
        });
        
        // Re-load examples for selected tab
        this.loadLearningExamples();
    }

    showSettingsModal() {
        // Implementation for settings modal
        console.log('⚙️ Settings modal would open here');
        this.showNotification('Settings feature coming soon', 'info');
    }

    exportDashboardData() {
        try {
            const exportData = {
                timestamp: new Date().toISOString(),
                predictions: this.predictionData,
                accuracy_stats: this.historicalData,
                feature_importance: this.learningData,
                dashboard_settings: {
                    timeframe: this.currentTimeframe,
                    strategy: this.currentStrategy
                }
            };

            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `goldgpt-dashboard-${new Date().toISOString().split('T')[0]}.json`;
            link.click();

            this.showNotification('Dashboard data exported successfully', 'success');
            
        } catch (error) {
            console.error('❌ Export failed:', error);
            this.showNotification('Export failed. Please try again.', 'error');
        }
    }

    showSmartNotification(title, message, type = 'info', actions = []) {
        // Enhanced notification with actions and persistence
        if ('Notification' in window && Notification.permission === 'granted') {
            const notification = new Notification(title, {
                body: message,
                icon: '/static/images/goldgpt-icon.png',
                tag: `goldgpt-${type}-${Date.now()}`,
                requireInteraction: type === 'error',
                actions: actions
            });

            notification.onclick = () => {
                window.focus();
                notification.close();
            };

            // Auto-close after delay (except errors)
            if (type !== 'error') {
                setTimeout(() => notification.close(), 5000);
            }
        }
        
        // Fallback to in-app notification
        this.showNotification(`${title}: ${message}`, type);
    }

    trackPredictionAccuracy(predictionId, actualOutcome) {
        // Track prediction accuracy for learning
        const prediction = this.findPredictionById(predictionId);
        if (prediction) {
            const accuracy = this.calculateAccuracy(prediction, actualOutcome);
            
            this.confidenceHistory.set(predictionId, {
                prediction: prediction,
                outcome: actualOutcome,
                accuracy: accuracy,
                timestamp: new Date().toISOString()
            });

            // Update model learning data
            this.updateModelLearning(prediction, accuracy);
        }
    }

    calculateAccuracy(prediction, actualOutcome) {
        const directionCorrect = (
            (prediction.direction === 'BULLISH' && actualOutcome.priceChange > 0) ||
            (prediction.direction === 'BEARISH' && actualOutcome.priceChange < 0) ||
            (prediction.direction === 'NEUTRAL' && Math.abs(actualOutcome.priceChange) < 0.5)
        );

        const priceAccuracy = 1 - (Math.abs(prediction.target_price - actualOutcome.final_price) / prediction.current_price);
        
        return {
            direction_correct: directionCorrect,
            price_accuracy: Math.max(0, priceAccuracy),
            overall_score: directionCorrect ? (0.7 + 0.3 * Math.max(0, priceAccuracy)) : (0.3 * Math.max(0, priceAccuracy))
        };
    }

    updateModelLearning(prediction, accuracy) {
        // Send learning data back to ML model
        if (this.apiClient) {
            this.apiClient.sendLearningData({
                prediction_id: prediction.id,
                accuracy_data: accuracy,
                features_used: prediction.key_features,
                timestamp: new Date().toISOString()
            }).catch(error => {
                console.error('❌ Failed to send learning data:', error);
            });
        }
    }

    findPredictionById(id) {
        for (const timeframe in this.predictionData) {
            const predictions = this.predictionData[timeframe] || [];
            const found = predictions.find(p => p.id === id);
            if (found) return found;
        }
        return null;
    }

    enableAdvancedChartFeatures() {
        // Add support/resistance levels to charts
        if (this.charts.prediction) {
            this.addSupportResistanceLevels();
            this.addMovingAverages();
            this.addConfidenceBands();
        }
    }

    addSupportResistanceLevels() {
        const chart = this.charts.prediction;
        const supportColor = 'rgba(0, 179, 134, 0.3)';
        const resistanceColor = 'rgba(255, 71, 87, 0.3)';

        // Add support levels
        this.supportResistanceLevels.support.forEach((level, index) => {
            chart.data.datasets.push({
                label: `Support ${index + 1}`,
                data: chart.data.labels.map(() => level),
                borderColor: supportColor,
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderDash: [10, 5],
                pointRadius: 0,
                fill: false
            });
        });

        // Add resistance levels
        this.supportResistanceLevels.resistance.forEach((level, index) => {
            chart.data.datasets.push({
                label: `Resistance ${index + 1}`,
                data: chart.data.labels.map(() => level),
                borderColor: resistanceColor,
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderDash: [10, 5],
                pointRadius: 0,
                fill: false
            });
        });
    }

    addConfidenceBands() {
        const chart = this.charts.prediction;
        const predictions = chart.data.datasets[1]?.data || [];
        
        if (predictions.length > 0) {
            const upperBand = predictions.map(price => price ? price * 1.02 : null);
            const lowerBand = predictions.map(price => price ? price * 0.98 : null);

            chart.data.datasets.push({
                label: 'Confidence Upper',
                data: upperBand,
                borderColor: 'rgba(0, 179, 134, 0.2)',
                backgroundColor: 'rgba(0, 179, 134, 0.1)',
                fill: '+1',
                borderWidth: 0,
                pointRadius: 0
            });

            chart.data.datasets.push({
                label: 'Confidence Lower',
                data: lowerBand,
                borderColor: 'rgba(0, 179, 134, 0.2)',
                backgroundColor: 'rgba(0, 179, 134, 0.1)',
                fill: false,
                borderWidth: 0,
                pointRadius: 0
            });
        }
    }

    setupAdvancedWebSocketHandlers() {
        if (this.socket) {
            // Handle advanced prediction updates
            this.socket.on('prediction_confidence_update', (data) => {
                this.handleConfidenceUpdate(data);
            });

            this.socket.on('market_volatility_alert', (data) => {
                this.handleVolatilityAlert(data);
            });

            this.socket.on('model_retraining_complete', (data) => {
                this.handleModelRetrainingComplete(data);
            });

            this.socket.on('accuracy_milestone', (data) => {
                this.handleAccuracyMilestone(data);
            });
        }
    }

    handleConfidenceUpdate(data) {
        console.log('📊 Confidence update received:', data);
        
        // Update prediction confidence in real-time
        const prediction = this.findPredictionById(data.prediction_id);
        if (prediction) {
            prediction.confidence = data.new_confidence;
            this.renderPredictionCards();
            
            // Show notification if confidence changed significantly
            const confidenceChange = Math.abs(data.new_confidence - data.old_confidence);
            if (confidenceChange > 0.1) {
                this.showSmartNotification(
                    'Confidence Update',
                    `Prediction confidence ${data.new_confidence > data.old_confidence ? 'increased' : 'decreased'} to ${(data.new_confidence * 100).toFixed(1)}%`,
                    'info'
                );
            }
        }
    }

    handleVolatilityAlert(data) {
        console.log('⚠️ Volatility alert:', data);
        
        this.showSmartNotification(
            'Market Volatility Alert',
            `${data.level} volatility detected. Current level: ${data.volatility_index}`,
            'warning',
            [
                { action: 'view', title: 'View Details' },
                { action: 'dismiss', title: 'Dismiss' }
            ]
        );
    }

    handleModelRetrainingComplete(data) {
        console.log('🤖 Model retraining complete:', data);
        
        this.showSmartNotification(
            'AI Model Updated',
            `${data.model_type} model retrained with ${data.improvement}% accuracy improvement`,
            'success'
        );
        
        // Refresh accuracy stats to reflect improvements
        this.loadAccuracyStats();
    }

    handleAccuracyMilestone(data) {
        console.log('🎯 Accuracy milestone reached:', data);
        
        this.showSmartNotification(
            'Accuracy Milestone',
            `New ${data.timeframe} accuracy record: ${data.accuracy}%`,
            'success'
        );
        
        // Add celebration animation
        this.addCelebrationAnimation();
    }

    addCelebrationAnimation() {
        // Add confetti or celebration effect
        const celebration = document.createElement('div');
        celebration.className = 'celebration-animation';
        celebration.innerHTML = '🎉';
        document.body.appendChild(celebration);
        
        setTimeout(() => {
            celebration.remove();
        }, 3000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mlDashboard = new AdvancedMLDashboard();
});

// Add notification styles dynamically
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        padding: 12px 16px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateX(100%);
        transition: transform 0.3s ease, opacity 0.3s ease;
        opacity: 0;
        max-width: 300px;
    }
    
    .notification.show {
        transform: translateX(0);
        opacity: 1;
    }
    
    .notification.success { background: #00b386; }
    .notification.error { background: #ff4757; }
    .notification.warning { background: #ffa502; }
    .notification.info { background: #0066cc; }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .prediction-detail-content {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    
    .detail-section h4 {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #0066cc;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .detail-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    .detail-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 4px;
    }
    
    .detail-label {
        font-weight: 500;
        color: #6c757d;
    }
    
    .detail-value {
        font-weight: 600;
        color: #212529;
    }
    
    .detail-value.bullish { color: #00b386; }
    .detail-value.bearish { color: #ff4757; }
    
    .reasoning-content {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #0066cc;
    }
    
    .features-detailed {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .feature-detailed {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 4px;
    }
    
    .feature-impact {
        font-size: 0.8rem;
        color: #00b386;
        font-weight: 500;
    }
    
    .risk-factors {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .risk-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 4px;
        color: #856404;
    }
    
    .predictions-container.updating {
        opacity: 0.7;
        transform: scale(0.98);
        transition: all 0.3s ease;
    }
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

console.log('✅ Advanced ML Dashboard script loaded successfully');
